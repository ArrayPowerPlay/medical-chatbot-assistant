from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any

class ILLMGenerator(ABC):
    """Abstract interface for LLM Answer Generation."""
    temperature: float
    max_tokens: int
    
    @abstractmethod
    async def generate_answer(
        self, 
        system_prompt: str, 
        user_prompt: str, 
        history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """Generate final answer using system/user prompts and optional history."""
        pass

    @abstractmethod
    async def generate_answer_stream(
        self, 
        system_prompt: str, 
        user_prompt: str, 
        history: Optional[List[Dict[str, str]]] = None
    ):
        """Generate final answer using system/user prompts and optional history via SSE."""
        pass


class IQueryAnalyzer(ABC):
    """Abstract interface for LLM Query Analysis (Rewriting & NER)."""
    
    @abstractmethod
    async def analyze(
        self, 
        query: str, 
        history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """Analyze query to return rewritten query, entities, and intents."""
        pass
