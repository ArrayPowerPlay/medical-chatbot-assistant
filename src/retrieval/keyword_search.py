from typing import List, Dict
from src.retrieval.vector_search import _aggregate_to_parents
from src.interfaces.storage import ISearchEngine, IParentStore
from config.settings import settings

async def keyword_search(
    query_text: str,
    search_engine: ISearchEngine,
    parent_store: IParentStore,
    top_k: int = settings.KEYWORD_TOP_K,
    child_fetch_limit: int = settings.CHILD_FETCH_LIMIT,
    original_query: str | None = None,
) -> List[Dict]:
    """BM25 keyword search: search child chunks, then aggregate to parent chunks.
    
    Args:
        query_text: raw query string (rewritten query)
        search_engine: Search engine client (Weaviate)
        parent_store: SQLite parent store
        top_k: number of parent results to return
        child_fetch_limit: number of children to fetch
        original_query: original user query string (optional)
    
    Returns:
        List of parent-level results: [{"parent_id", "pmid", "text", "title", "score"}]
    """
    child_results = await search_engine.bm25_search(
        query_text=query_text,
        limit=child_fetch_limit
    )
    
    if original_query and original_query.strip().lower() != query_text.strip().lower():
        original_child_results = await search_engine.bm25_search(
            query_text=original_query,
            limit=child_fetch_limit
        )
        child_results.extend(original_child_results)

    parent_results = await _aggregate_to_parents(
        child_results=child_results,
        parent_store=parent_store
    )
    return parent_results[:top_k]