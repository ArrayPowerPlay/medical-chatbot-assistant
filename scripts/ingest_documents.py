import os
import sys
import json
import argparse
from pathlib import Path
from tqdm import tqdm

# Configure project root
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from config.settings import settings
from config.logging_config import setup_logging, logger
from src.dataset_builder.index_builder import IndexBuilder
from src.dataset_builder.contextual_chunker import ContextualChunker
from src.embeddings.medcpt_embedder import MedCPTEmbedder


def run_enrichment(input_path: Path, output_path: Path, limit: int | None = None):
    """Phase 1: Chunking and enriching chunks using LLM"""
    logger.info("Starting Phase 1: Enriching chunks")

    chunker = ContextualChunker(api_key=settings.GOOGLE_API_KEY)

    count = 0
    with open(input_path, 'r', encoding='utf-8') as f_in, \
         open(output_path, 'a', encoding='utf-8') as f_out:
        for line in tqdm(f_in, desc="Enriching articles"):
            article = json.loads(line)
            enriched_chunks = chunker.enrich_article(article)

            for chunk in enriched_chunks: 
                f_out.write(json.dumps(chunk, ensure_ascii=False) + "\n")

            count +=1
            if limit and count >= limit:
                break

    logger.info(f"Finishing Phase 1. Output saved to {output_path}!")


def run_indexing(enriched_path: Path):
    """Creating vector embeddings and save into FAISS/ElasticSearch"""
    logger.info("Starting Phase 2: Indexing")

    chunks = []
    with open(enriched_path, 'r', encoding='utf-8') as f:
        for line in f:
            chunks.append(json.loads(line))

    if not chunks:
        logger.error("No enriched chunks to be found!")
        return
    
    # Create vector embeddings
    logger.info(f"Generating embeddings for {len(chunks)} chunks")
    embedder = MedCPTEmbedder()
    texts = [c['text'] for c in chunks]
    embeddings = embedder.embed_texts(texts)

    # Create index
    logger.info("Building FAISS and Elastic Search indices...")
    builder = IndexBuilder()

    # Create index of FAISS
    builder.build_faiss(embeddings)
    builder.save_index(settings.FAISS_INDEX_PATH)

    # Create index of Elastic Search
    builder.build_elasticsearch(chunks) 

    logger.info("Phase 2 finished successfully!")


def main():
    setup_logging()
    parser = argparse.ArgumentParser(description="MedKG-RAG Data Ingestion Pipeline")
    parser.add_argument("--stage", choices=["enrich", "index", "all"], default="all",
                        help="Which stage of ingestion to run")
    parser.add_argument("--limit", type=int, help="Limit number of articles for testing")
    args = parser.parse_args()     # Read input from command line 

    raw_corpus = settings.DATA_PATH / "corpus" / "corpus.jsonl"
    processed_dir = settings.DATA_PATH / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    enriched_file = processed_dir / "enriched_chunks.jsonl"

    try:
        if args.stage in ["enrich", "all"]:
            run_enrichment(raw_corpus, enriched_file, args.limit)
        
        if args.stage in ["index", "all"]:
            run_indexing(enriched_file)
    except KeyboardInterrupt:
        logger.warning("Ingestion interrupted by user!")
    except Exception as e:
        logger.exception(f"Critical error during ingestion: {e}")


if __name__ == "__main__":
    main()