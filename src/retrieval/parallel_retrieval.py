"""
Orchestrates 3 parallel retrieval streams:
1. Vector Search
2. Keyword Search 
3. Knowledge Graph Search
"""

import concurrent.futures
from typing import Dict, List, Tuple
import numpy as np

from config.logging_config import logger
from src.retrieval.vector_search import vector_search
from src.retrieval.keyword_search import keyword_search
from src.kg.kg_search import KGSearch
from src.storage.weaviate_client import WeaviateChildStore
from src.storage.parent_store import ParentStore


class ParallelRetriever:
    """
    Executes 3 retrieval streams in parallel: vector search, keyword search, and KG search.
    """
    def __init__(
        self,
        weaviate_store: WeaviateChildStore,
        parent_store: ParentStore,
        kg_searcher: KGSearch
    ):
        self.weaviate_store = weaviate_store
        self.parent_store = parent_store
        self.kg_searcher = kg_searcher

    def retrieve(
        self,
        query_text: str,
        query_vector: np.ndarray,
        entity_article_embeddings: List[List[float]],
        intents: List[str] = ["general"],
        top_k: int = 20,
        child_fetch_limit: int = 60
    ) -> Tuple[List[Dict], List[Dict], str]:
        """
        Runs the 3 retrieval streams in parallel.
        
        Args:
            query_text: The rewritten query text for BM25 search.
            query_vector: The query vector encoded by Query-Encoder for Vector Search and KG ranking.
            entity_article_embeddings: Entities encoded by Article-Encoder for KG anchor search.
            intents: List of query intents. Defaults to ["general"].
            top_k: Number of parent results to return for vector and keyword search. Defaults to 20.
            child_fetch_limit: Number of child chunks to fetch initially. Defaults to 60.
            
        Returns:
            Tuple containing:
            - List of parent results from Vector Search.
            - List of parent results from Keyword Search.
            - Linearized subgraph text from Knowledge Graph.
        """
        logger.info(f"[Parallel Retrieval]: Starting streams for query: '{query_text}'...")

        # Ensure query_vector is converted into list of floats for KG search if it's a numpy array
        query_vector_list = query_vector.tolist() if isinstance(query_vector, np.ndarray) else query_vector

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            # Stream 1: Vector Search
            future_vector = executor.submit(
                vector_search,
                query_vector=query_vector,
                weaviate_store=self.weaviate_store,
                parent_store=self.parent_store,
                top_k=top_k,
                child_fetch_limit=child_fetch_limit
            )
            
            # Stream 2: Keyword Search
            future_keyword = executor.submit(
                keyword_search,
                query_text=query_text,
                weaviate_store=self.weaviate_store,
                parent_store=self.parent_store,
                top_k=top_k,
                child_fetch_limit=child_fetch_limit
            )

            # Stream 3: KG Search
            future_kg = executor.submit(
                self.kg_searcher.search,
                entity_article_embeddings=entity_article_embeddings,
                rewritten_query_vec=query_vector_list,
                intents=intents
            )

            # Gather results
            try:
                vector_results = future_vector.result()   # Wait for the thread to complete
                logger.info(f"[Parallel Retrieval]: Vector Search returned {len(vector_results)} results.")
            except Exception as e:
                logger.error(f"[Parallel Retrieval]: Error in Vector Search: {e}")
                vector_results = []
            
            try:
                keyword_results = future_keyword.result()
                logger.info(f"[Parallel Retrieval]: Keyword Search returned {len(keyword_results)} results.")
            except Exception as e:
                logger.error(f"[Parallel Retrieval]: Error in Keyword Search: {e}")
                keyword_results = []
                
            try:
                kg_results = future_kg.result()
                logger.info(f"[Parallel Retrieval]: KG Search returned text length {len(kg_results)}.")
            except Exception as e:
                logger.error(f"[Parallel Retrieval]: Error in KG Search: {e}")
                kg_results = ""
        
        logger.info("[Parallel Retrieval]: Completed all 3 streams!")
        return vector_results, keyword_results, kg_results