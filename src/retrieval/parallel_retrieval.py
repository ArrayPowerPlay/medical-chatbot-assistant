"""
Orchestrates 3 parallel retrieval streams:
1. Vector Search
2. Keyword Search 
3. Knowledge Graph Search
"""
import asyncio
from typing import Dict, List, Tuple
import numpy as np

from config.settings import settings
from config.logging_config import logger
from src.retrieval.vector_search import vector_search
from src.retrieval.keyword_search import keyword_search
from src.interfaces.storage import ISearchEngine, IParentStore
from src.interfaces.kg import IKGSearcher


class ParallelRetriever:
    """
    Executes 3 retrieval streams in parallel: vector search, keyword search, and KG search.
    """
    def __init__(
        self,
        search_engine: ISearchEngine,
        parent_store: IParentStore,
        kg_searcher: IKGSearcher
    ):
        self.search_engine = search_engine
        self.parent_store = parent_store
        self.kg_searcher = kg_searcher

    async def retrieve(
        self,
        query_text: str,
        query_vector: np.ndarray,
        entity_article_embeddings: List[List[float]],
        intents: List[str] = ["general"],
        vector_top_k: int = settings.VECTOR_TOP_K,
        keyword_top_k: int = settings.KEYWORD_TOP_K,
        child_fetch_limit: int = settings.CHILD_FETCH_LIMIT,
        kg_top_k: int = settings.KG_TOP_K,
        kg_hop1_m: int = settings.KG_HOP1_M,
        kg_hop2_n: int = settings.KG_HOP2_N,
        kg_hop2_cap: int = settings.KG_HOP2_CAP,
        original_query_text: str | None = None,
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Runs the 3 retrieval streams in parallel.
        
        Args:
            query_text: The rewritten query text for BM25 search.
            query_vector: The query vector encoded by Query-Encoder for Vector Search and KG ranking.
            entity_article_embeddings: Entities encoded by Article-Encoder for KG anchor search.
            intents: List of query intents. Defaults to ["general"].
            top_k: Number of parent results to return for vector and keyword search. Defaults to 20.
            child_fetch_limit: Number of child chunks to fetch initially. Defaults to 60.
            original_query_text: The original user query text (optional).
            
        Returns:
            Tuple containing:
            - List of parent results from Vector Search.
            - List of parent results from Keyword Search.
            - List of linearized paths with metadata from Knowledge Graph.
        """
        logger.info(f"[Parallel Retrieval]: Starting streams for query: '{query_text}'...")

        # Ensure query_vector is converted into list of floats for KG search if it's a numpy array
        query_vector_list = query_vector.tolist() if isinstance(query_vector, np.ndarray) else query_vector

        # Create async tasks for the three streams
        # Task wraps coroutine and it is a background process
        task_vector = asyncio.create_task(
            vector_search(
                query_vector=query_vector,
                search_engine=self.search_engine,
                parent_store=self.parent_store,
                top_k=vector_top_k,
                child_fetch_limit=child_fetch_limit
            )
        )
        
        task_keyword = asyncio.create_task(
            keyword_search(
                query_text=query_text,
                search_engine=self.search_engine,
                parent_store=self.parent_store,
                top_k=keyword_top_k,
                child_fetch_limit=child_fetch_limit,
                original_query=original_query_text
            )
        )

        task_kg = asyncio.create_task(
            self.kg_searcher.search(
                entity_article_embeddings=entity_article_embeddings,
                rewritten_query_vec=query_vector_list,
                intents=intents,
                top_k=kg_top_k,
                hop1_m=kg_hop1_m,
                hop2_n=kg_hop2_n,
                hop2_cap=kg_hop2_cap,
            )
        )

        # Gather results concurrently
        # Still return results if exception occurs
        results = await asyncio.gather(task_vector, task_keyword, task_kg, return_exceptions=True)
        
        vector_results = results[0] if not isinstance(results[0], Exception) else []
        keyword_results = results[1] if not isinstance(results[1], Exception) else []
        kg_results = results[2] if not isinstance(results[2], Exception) else []

        if isinstance(results[0], Exception):
            logger.error(f"[Parallel Retrieval]: Error in Vector Search: {results[0]}")
        else:
            logger.info(f"[Parallel Retrieval]: Vector Search returned {len(vector_results)} results.")

        if isinstance(results[1], Exception):
            logger.error(f"[Parallel Retrieval]: Error in Keyword Search: {results[1]}")
        else:
            logger.info(f"[Parallel Retrieval]: Keyword Search returned {len(keyword_results)} results.")

        if isinstance(results[2], Exception):
            logger.error(f"[Parallel Retrieval]: Error in KG Search: {results[2]}")
        else:
            logger.info(f"[Parallel Retrieval]: KG Search returned {len(kg_results)} paths.")

        logger.info("[Parallel Retrieval]: Completed all streams!")
        return vector_results, keyword_results, kg_results
