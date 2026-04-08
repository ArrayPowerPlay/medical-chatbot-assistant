from typing import Dict, List
import numpy as np
from src.storage.weaviate_client import WeaviateChildStore
from src.storage.parent_store import ParentStore

def _aggregate_to_parents(
    child_results: List[Dict],
    parent_store: ParentStore
) -> List[Dict]:
    """Aggregate child results to their parent
    Strategy: parent_score = max(child_scores) for that parent
    Then fetch parent object from sqlite"""
    # Group children by parent_id, keep max score
    parent_scores: Dict[str, float] = {}
    for child in child_results:
        pid = child["parent_id"]
        score = child["score"]
        if pid not in parent_scores or score > parent_scores[pid]:
            parent_scores[pid] = score
    
    if not parent_scores:
        return []
    
    # Fetch parent data from sqlite
    parent_ids = list(parent_scores.keys())
    parent_data = parent_store.get_parent_batch(parent_ids)   # Return Dict[str, Dict]

    # Build parent-level results sorted by 'score' desc
    results = []
    for pid, score in parent_scores.items():
        parent = parent_data.get(pid)
        if parent: 
            results.append({
                "parent_id": pid,
                "pmid": parent["pmid"],
                "text": parent["text"],
                "title": parent.get("title", ""),
                "score": score
            })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results


def vector_search(
    query_vector: np.ndarray,
    weaviate_store: WeaviateChildStore,
    parent_store: ParentStore,
    top_k: int = 20,
    child_fetch_limit: int = 60
) -> List[Dict]:
    """Semantic vector search: search child chunks, then aggregate to parent chunks.
    
    Args:
        query_vector: MedCPT query embedding
        weaviate_store: Weaviate client
        parent_store: SQLite parent store
        top_k: number of parent results to return
        child_fetch_limit: number of child chunks to fetch
    
    Returns:
        List of parent-level results: [{"parent_id", "pmid", "text", "title", "score"}]
    """
    child_results = weaviate_store.vector_search(
        query_vector=query_vector, 
        limit=child_fetch_limit
    )
    parent_results = _aggregate_to_parents(
        child_results=child_results, 
        parent_store=parent_store
    )
    return parent_results[:top_k]
    