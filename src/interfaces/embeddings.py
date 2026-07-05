from abc import ABC, abstractmethod
from typing import List, Union
import numpy as np

class IEmbedder(ABC):
    """Abstract interface for Text Embedding Models."""
    
    @abstractmethod
    async def embed_texts(self, texts: Union[str, List[str]], batch_size: int = 256) -> np.ndarray:
        """Embed a list of texts into dense vectors."""
        pass
