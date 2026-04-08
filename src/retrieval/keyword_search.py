from typing import List, Dict
from src.retrieval.vector_search import _aggregate_to_parents
from src.storage.parent_store import ParentStore
from src.storage.weaviate_client import WeaviateChildStore

def keyword_search(
    query_text: str,
    weaviate_store: WeaviateChildStore,
    parent_store: ParentStore,
    top_k: int = 20,
    child_fetch_limit: int = 60
) -> List[Dict]:
    """BM25 keyword search: search child chunks, then aggregate to parent chunks.
    
    Args:
        query_text: raw query string
        weaviate_store: Weaviate client  
        parent_store: SQLite parent store
        top_k: number of parent results to return
        child_fetch_limit: number of children to fetch
    
    Returns:
        List of parent-level results: [{"parent_id", "pmid", "text", "title", "score"}]
    """
    child_results = weaviate_store.bm25_search(
        query_text=query_text,
        limit=child_fetch_limit
    )
    parent_results = _aggregate_to_parents(
        child_results=child_results,
        parent_store=parent_store
    )
    return parent_results[:top_k]