from abc import ABC, abstractmethod
from typing import List, Dict, Tuple

class IReranker(ABC):
    """Interface for reranking search results."""
    
    @abstractmethod
    async def rerank(
        self,
        query: str,
        rrf_results: List[Dict],
        kg_results: List[Dict],
        top_m: int,
        top_n: int
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Reranks both text and KG passages independently based on the user query.
        
        Args:
            query: The rewritten query of the user
            rrf_results: List of dicts from RRF (text retrieval)
            kg_results: List of dictionary paths from KG search with metadata
            top_m: Number of text results to return
            top_n: Number of KG results to return
            
        Returns:
            A tuple of (ranked_text, ranked_kg), where each list is sorted independently.
        """
        pass
