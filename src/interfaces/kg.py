from abc import ABC, abstractmethod
from typing import List, Dict

class IKGSearcher(ABC):
    """Abstract interface for Knowledge Graph Search."""
    
    @abstractmethod
    async def search(
        self,
        entity_article_embeddings: List[List[float]],
        rewritten_query_vec: List[float],
        intents: List[str] = ["general"],
        top_k: int = 2,
        hop1_m: int = 3,
        hop2_n: int = 3,
        hop2_cap: int = 30,
    ) -> List[Dict]:
        """Perform a multi-hop graph traversal and return linearized paths."""
        pass
