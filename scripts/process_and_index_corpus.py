import json
import sys
from pathlib import Path
import argparse
import torch
import queue
import threading
import concurrent.futures
from typing import List, Tuple
from tqdm import tqdm

parent_root = Path(__file__).resolve().parent.parent
if str(parent_root) not in sys.path:
    sys.path.append(str(parent_root))

from config.settings import settings
from config.logging_config import logger, setup_logging
from src.dataset_builder.parent_child_chunker import AdaptiveChunker
from src.storage.parent_store import ParentStore
from src.storage.weaviate_client import WeaviateChildStore
from src.embeddings.medcpt_embedder import MedCPTEmbedder


# Sentinel value to signal downstream stages that the upstream stage is done
_SENTINEL = None

class CorpusIndexer:
    """Class used for loading, chunking, embedding and saving data in batches."""
    def __init__(self, data_path: str, db_path: str):
        self.data_path = data_path
        self.weaviate = WeaviateChildStore()
        self.parent_store = ParentStore(db_path)

        logger.info("Loading MedCPT-Article-Encoder...")
        self.embedder = MedCPTEmbedder(mode='article')

    def reset_databases(self):
        """Drop existing Weaviate collection and recreate them. Reset SQlite table."""
        logger.info("Resetting Weaviate child collection...")
        self.weaviate.delete_collection()
        self.weaviate.create_collection()

        logger.info("Resetting parent store (SQLite)...")
        self.parent_store.conn.execute("DROP TABLE IF EXISTS parent_chunks")
        self.parent_store._create_table()
        logger.info("All databases reset successfully!")

    ### STAGE 1: READ JSONL IN BATCHES (CPU - BOUND)
    def _stage_reader_chunker(
        self,
        chunk_queue: queue.Queue,
        batch_size: int,
        limit: int | None,
        max_workers: int,
        pbar: tqdm
    ):
        """Read JSONL line by line, chunk each batch concurrently and push into chunk_queue"""
        batch: List[Tuple[str, str, str]] = []
        processed_count = 0

        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if limit is not None and processed_count >= limit:
                        break
                    try:
                        doc = json.loads(line)
                        pmid = doc.get("pmid", "")
                        if not pmid: continue
                        title = doc.get("title", "")
                        abstract = doc.get("abstractText", "")
                        batch.append((pmid, title, abstract))
                        
                        processed_count += 1
                        pbar.update(1)

                        if len(batch) >= batch_size:
                            parents, children = self._chunk_batch(batch, max_workers)
                            chunk_queue.put((parents, children))
                            batch = []
                    except json.JSONDecodeError:
                        continue
            
            # Flush remaining articles in the last incomplete batch
            if batch:
                parents, children = self._chunk_batch(batch, max_workers)
                chunk_queue.put((parents, children))
        finally:
            # Signal the next stage that no more data is coming
            chunk_queue.put(_SENTINEL)
            logger.info(f"Reading and chunking data completed. Read {processed_count} articles.")

    def _chunk_batch(
        self,
        batch: List[Tuple[str, str, str]],
        max_workers: int
    ) -> Tuple[list, list]:
        """Run AdaptiveChunker (for embedding articles) on a batch using a thread pool"""
        batch_parents = []
        batch_children = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(AdaptiveChunker.process_article, pmid, title, abs_text)
                for pmid, title, abs_text in batch
            ]
            for future in concurrent.futures.as_completed(futures):   # Process completed futures
                res = future.result()
                batch_parents.extend(res["parents"])
                for p_id, child_texts in res["children"].items():
                    pmid = p_id.split("_p")[0]
                    for c_text in child_texts:
                        batch_children.append(
                            {"parent_id": p_id, "pmid": pmid, "text": c_text}
                        )

        return batch_parents, batch_children
    
    ### STAGE 2: EMBEDDING (GPU - BOUND)
    def _stage_embedder(
        self,
        chunk_queue: queue.Queue,
        embed_queue: queue.Queue
    ):
        """Take chunked batches from chunk_queue, embed them and then push them to embed_queue."""
        try:
            while True:
                item = chunk_queue.get()
                if item is _SENTINEL:
                    break

                parents, children = item
                if not children:
                    chunk_queue.task_done()
                    continue

                texts_to_embed = [c["text"] for c in children]
                embeddings = self.embedder.embed_texts(texts_to_embed, batch_size=512)

                embed_queue.put((parents, children, embeddings))
                torch.cuda.empty_cache()
                chunk_queue.task_done()       # Signals that the item has been processed successfully
        finally:
            embed_queue.put(_SENTINEL)
            logger.info("Embedding child chunks completed!")

    ### STAGE 3: WRITE DATA TO DATABASE (I/O BOUND)
    def _stage_writer(self, embed_queue: queue.Queue):
        """Take embedded batches from embed_queue and persist to Weaviate and SQLite."""
        total_parents = 0
        total_children = 0

        try:
            while True:
                item = embed_queue.get()
                if item is _SENTINEL:
                    break

                parents, children, embeddings = item
                self.parent_store.insert_parents(parents)
                self.weaviate.insert_children(children, embeddings)

                total_parents += len(parents)
                total_children += len(children)
                embed_queue.task_done()
        finally:
            logger.info(
                "Storing data into database completed! "
                f"Stored {total_parents} parents, {total_children} children."
            )

    ### ORCHESTRATOR
    def process_and_index(
        self,
        max_workers: int = 6,
        batch_size: int = 1024,
        limit: int | None = None
    ):
        """Launch 3-stage pipeline: Read + Chunk + Embed + Store."""
        if limit == 0:
            logger.info("Limit is 0. Exiting ingestion immediately!")
            return
        
        # Create collection for child chunks in Weaviate 
        self.weaviate.create_collection()

        chunk_queue = queue.Queue(maxsize=3)   # Max 3 batches per queue
        embed_queue = queue.Queue(maxsize=3)

        pbar = tqdm(desc="Reading articles", unit=" docs")

        # Each stage runs in its own thread
        t1 = threading.Thread(
            target=self._stage_reader_chunker,
            args=(chunk_queue, batch_size, limit, max_workers, pbar),
            name="[Stage 1]: Reader-Chunker Process"
        )
        t2 = threading.Thread(
            target=self._stage_embedder,
            args=(chunk_queue, embed_queue),
            name="[Stage 2]: Embedder Process"
        )
        t3 = threading.Thread(
            target=self._stage_writer,
            args=(embed_queue,),       # Always parse a tuple
            name="[Stage 3]: Writer Process"
        )

        # Start to run 3 threads
        t1.start()
        t2.start()
        t3.start()

        # Wait for 3 threads to finish running
        t1.join()
        t2.join()
        t3.join()

        # Close progress bar
        pbar.close()
        logger.info("Pipeline completed successfully!")

    def close(self):
        """Clean up resources"""
        self.embedder.close()
        self.weaviate.close()
        self.parent_store.close()


def main():
    parser = argparse.ArgumentParser(
        description="Process and index BioASQ PudMed articles into Weaviate and SQLite."
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Limit number of articles to be processed. Set to 0 to only reset."
    )
    parser.add_argument(
        "--batch_size", type=int, default=1024,
        help="Batch size for pipeline streaming"
    )
    parser.add_argument(
        "--reset", action="store_true",
        help="Delete existing data and re-ingest"
    )
    parser.add_argument(
        "--workers", type=int, default=6,
        help="Max thread workers for chunking"
    )

    # Read paramaters from terminal
    args = parser.parse_args()

    indexer = CorpusIndexer(
        data_path="data/corpus/corpus.jsonl",
        db_path="vectorstore/parent_chunks.db"
    )

    if args.reset:
        indexer.reset_databases()

    if args.limit != 0:
        indexer.process_and_index(
            max_workers=args.workers,
            batch_size=args.batch_size,
            limit=args.limit
        )

    indexer.close()


if __name__ == "__main__":
    setup_logging()
    main()