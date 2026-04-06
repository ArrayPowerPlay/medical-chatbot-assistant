import os
import sys
import json
import faiss
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Configure project root
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from config.settings import settings
from config.logging_config import setup_logging, logger
from src.embeddings.medcpt_embedder import MedCPTEmbedder

def load_faiss_metadata(path: str):
    """Load the metadata mapping for FAISS indices."""
    metadata = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            metadata.append(json.loads(line))
    return metadata

def evaluate_recall_at_k(qa_samples: list, faiss_index, metadata: list, embedder, k_list=[1, 5, 10, 20]):
    """Calculate Recall@K for a list of QA samples."""
    recalls = {k: [] for k in k_list}
    
    logger.info(f"Evaluating {len(qa_samples)} samples...")
    
    for sample in tqdm(qa_samples, desc="Evaluating Recall@K"):
        question = sample["body"]
        relevant_pmids = set(sample["relevant_pmid"])
        
        # Embed query
        query_embedding = embedder.embed_texts([question])
        faiss.normalize_L2(query_embedding)
        
        # Search FAISS
        distances, indices = faiss_index.search(query_embedding, max(k_list))
        
        # Map indices to PMIDs
        retrieved_pmids = []
        for idx in indices[0]:
            if idx != -1 and idx < len(metadata):
                retrieved_pmids.append(metadata[idx].get("pmid"))
            else:
                retrieved_pmids.append(None)
                
        # Calculate Recall@K
        for k in k_list:
            top_k_pmids = set(retrieved_pmids[:k])
            # intersection / total relevant (standard BioASQ recall is usually just "is at least one in top k?")
            # BioASQ Phase A evaluation often uses Mean Reciprocal Rank or Recall.
            # Here we follow the user's specific request: "so sánh k chunks này với pmid của relevant document thực sự"
            
            found = any(pmid in relevant_pmids for pmid in top_k_pmids if pmid)
            recalls[k].append(1 if found else 0)
            
    # Calculate averages
    results = {f"Recall@{k}": np.mean(val) for k, val in recalls.items()}
    return results

if __name__ == "__main__":
    setup_logging()
    
    # Paths
    val_data_path = settings.BASE_DIR / "data" / "val" / "val.jsonl"
    faiss_index_path = settings.FAISS_INDEX_PATH
    metadata_path = settings.FAISS_METADATA_PATH
    
    if not val_data_path.exists():
        logger.error(f"Validation data not found at {val_data_path}. Run preprocessing first.")
        sys.exit(1)
        
    if not os.path.exists(faiss_index_path):
        logger.error(f"FAISS index not found at {faiss_index_path}. Run ingestion first.")
        sys.exit(1)
        
    if not os.path.exists(metadata_path):
        logger.error(f"FAISS metadata not found at {metadata_path}. Run updated ingestion first.")
        sys.exit(1)
        
    # Load data
    logger.info("Loading validation samples...")
    qa_samples = []
    with open(val_data_path, 'r', encoding='utf-8') as f:
        for line in f:
            qa_samples.append(json.loads(line))
            
    logger.info("Loading FAISS index and metadata...")
    index = faiss.read_index(faiss_index_path)
    metadata = load_faiss_metadata(metadata_path)
    
    # Initialize Query Embedder
    # IMPORTANT: Use MedCPT-Query-Encoder for queries!
    embedder = MedCPTEmbedder(model_name=settings.QUERY_MODEL)
    
    # Evaluate
    results = evaluate_recall_at_k(qa_samples, index, metadata, embedder)
    
    # Report
    print("\n--- Retrieval Evaluation Results (FAISS) ---")
    for metric, score in results.items():
        print(f"{metric}: {score:.4f}")
