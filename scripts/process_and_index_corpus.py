import sys
from pathlib import Path
import json
import argparse
import threading
from tqdm import tqdm
from typing import Optional
import queue

# Configure project root
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from config.settings import settings
from config.logging_config import logger, setup_logging
from src.dataset_builder.parent_child_chunker import ParentChildChunker
from src.storage.parent_store import ParentStore
from src.storage.weaviate_client import WeaviateChildStore
from src.embeddings.medcpt_embedder import MedCPTEmbedder


def count_lines(file_path: Path) -> int:
    """Count total lines for progress bar"""
    count = 0
    with open(file_path, 'r', encoding='utf-8') as f:
        for _ in f:
            count += 1
    return count


def producer_worker(corpus_path: Path, chunk_queue: queue.Queue, limit: Optional[int] = None):
    """Worker read articles from disk, chunking articles and push them into 'chunk_queue'"""
    # Stop process immediately if 'limit' = 0
    if limit == 0:
        chunk_queue.put(None)
        return

    chunker = ParentChildChunker(
        parent_chunk_size=settings.PARENT_CHUNK_SIZE,
        parent_chunk_overlap=settings.PARENT_CHUNK_OVERLAP,
        child_chunk_size=settings.CHILD_CHUNK_SIZE,
        child_chunk_overlap=settings.CHILD_CHUNK_OVERLAP
    )

    article_count = 0
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line in f:
            article = json.loads(line)
            parents, children = chunker.chunk_article(article)
            chunk_queue.put((parents, children))

            article_count += 1
            if limit is not None and article_count >= limit:
                break

    # Signals terminate the working process
    chunk_queue.put(None)


def storage_worker(storage_queue: queue.Queue, parent_store: ParentStore, weaviate_store: WeaviateChildStore):
    """"Worker handles implicit writing to SQLite and Weaviate (gRPC)"""
    while True:
        item = storage_queue.get()
        if item is None:
            # Confirm that a data unit has been processed
            storage_queue.task_done()
            break

        parents, children, vectors = item
        parent_store.insert_parents(parents)
        weaviate_store.insert_children(children, vectors)
        storage_queue.task_done()


def run_ingestion(
    corpus_path: Path,
    batch_size: int = 1024,
    limit: int | None = None
):
    """Stream articles from corpus → chunk → embed → store.
    Everything is processed in batches. 
    
    Flow per batch:
        1. Read batch of articles from corpus.jsonl (streaming)
        2. ParentChildChunker → parent_chunks + child_chunks
        3. Parents → SQLite (ParentStore)
        4. Children → MedCPT embedding → Weaviate
    """
    # Initialize components
    parent_store = ParentStore(settings.SQLITE_PARENT_DB_PATH)
    weaviate_store = WeaviateChildStore(
        url=settings.WEAVIATE_URL,
        grpc_port=settings.WEAVIATE_GRPC_PORT
    )
    embedder = MedCPTEmbedder('article')

    # Create Weaviate collections
    weaviate_store.create_collection()

    logger.info(f"Starting ingestion from {corpus_path}...")
    logger.info(f"Batch size: {batch_size}, Limit: {'None' if limit is None else limit}")

    # Total lines to be processed
    total_lines = count_lines(corpus_path) if limit is None else limit
    
    # Create queue to regulate each process
    chunk_queue = queue.Queue(maxsize=batch_size * 2)  # Ensure data is readily available for the next batch
    storage_queue = queue.Queue(maxsize=10)
    
    # Initialize worker threads
    producer = threading.Thread(target=producer_worker, args=(corpus_path, chunk_queue, limit))
    storer = threading.Thread(target=storage_worker, args=(storage_queue, parent_store, weaviate_store))

    # Start process, specifically 'producer_worker' and 'storage_worker' process
    producer.start()
    storer.start()

    article_count, total_parents, total_children = 0, 0, 0
    # Accumulators for current batch
    batch_parents, batch_children = [], []
    progress_bar = tqdm(total=total_lines, desc="Ingesting articles")

    while True:
        try:
            item = chunk_queue.get(timeout=30)
            if item is None: break   # Reach the end of file

            parents, children = item
            batch_parents.extend(parents)
            batch_children.extend(children)
            article_count += 1
            progress_bar.update(1)

            if len(batch_children) >= batch_size:
                # Embed chunks into vectors and put into 'storage_queue' for storaging data into database
                child_texts = [c['text'] for c in batch_children]
                vector_embeddings = embedder.embed_texts(child_texts)

                storage_queue.put((list(batch_parents), list(batch_children), vector_embeddings))

                total_parents += len(batch_parents)
                total_children += len(batch_children)
                
                # Reset after processing a batch
                batch_parents, batch_children = [], []

        except queue.Empty: continue

    # Flush remaining articles
    if batch_children:
        child_texts = [c['text'] for c in batch_children]
        vector_embeddings = embedder.embed_texts(child_texts)

        storage_queue.put((list(batch_parents), list(batch_children), vector_embeddings))

        total_parents += len(batch_parents)
        total_children += len(batch_children)
        
    # The main thread waits for all other threads to finish work
    producer.join()
    storage_queue.put(None)
    storer.join()
    progress_bar.close()

    logger.info(
        f"Ingestion completed."
        f"# Articles: {article_count}, # Parents: {total_parents}, # Children: {total_children}"
    )
    logger.info(f'SQLite parents: {parent_store.count()}')
    logger.info(f'Weaviate children: {weaviate_store.count()}')

    # Close connection
    parent_store.close()
    weaviate_store.close()


def main():
    setup_logging()
    parser = argparse.ArgumentParser(description="MedKG-RAG Data Ingestion Pipeline")
    parser.add_argument(
        "--limit", type=int, help="Limit number of articles (for testing)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=1024, help="Batch size for processing" 
    )
    parser.add_argument(
        "--reset", action="store_true", help="Delete existing data and re-ingest"
    )
    args = parser.parse_args()

    corpus_path = settings.DATA_PATH / "corpus" / "corpus.jsonl"

    if not corpus_path.exists():
        logger.error(f"Corpus file {corpus_path} not found!")
        return
    
    if args.reset:
        logger.warning("Reset all data...")
        # Delete Weaviate database
        weaviate_store = WeaviateChildStore(
            url=settings.WEAVIATE_URL,
            grpc_port=settings.WEAVIATE_GRPC_PORT
        )
        weaviate_store.delete_collection()
        weaviate_store.close()

        # Delete SQLite database
        db_path = Path(settings.SQLITE_PARENT_DB_PATH)
        if db_path.exists():
            db_path.unlink()     # Delete file SQLite which contains parent chunks 
            logger.info(f"Deleted {db_path}")

    try: 
        run_ingestion(corpus_path, batch_size=args.batch_size, limit=args.limit)
    except KeyboardInterrupt:
        logger.warning("Ingestion interrupted by user!")
    except Exception as e:
        logger.exception(f"Critical error during ingestion {e}")


if __name__ == "__main__":
    main()
