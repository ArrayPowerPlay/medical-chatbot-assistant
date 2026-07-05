from abc import ABC, abstractmethod
from typing import List, Dict, Union
import numpy as np

class ISearchEngine(ABC):
    """Abstract interface for a child chunk search engine (e.g., Weaviate).
    Handles both dense (vector) and sparse (BM25) retrieval.
    """
    
    @abstractmethod
    async def vector_search(self, query_vector: Union[List[float], np.ndarray], limit: int) -> List[Dict]:
        """Perform semantic search using a query vector."""
        pass
        
    @abstractmethod
    async def bm25_search(self, query_text: str, limit: int) -> List[Dict]:
        """Perform lexical search using a keyword query."""
        pass

class IParentStore(ABC):
    """Abstract interface for a parent chunk storage (e.g., SQLite)."""
    
    @abstractmethod
    async def get_parent_batch(self, parent_ids: List[str]) -> Dict[str, Dict]:
        """Fetch parent metadata for a batch of parent IDs."""
        pass
