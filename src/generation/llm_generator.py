from groq import AsyncGroq
from config.settings import settings
from config.logging_config import logger
from typing import Dict, List, Optional
from src.interfaces.llm import ILLMGenerator


class LLMGenerator(ILLMGenerator):
    """Client for interacting with Groq API for text generation."""
    def __init__(self):
        api_key = settings.GROQ_API_KEY
        if not api_key:
            raise ValueError("GROQ_API_KEY is missing in settings or environment variables!")
        
        self.client = AsyncGroq(api_key=api_key)
        self.model_name = settings.LLM_MODEL
        self.temperature = settings.GENERATION_TEMPERATURE
        self.max_tokens = settings.GENERATION_MAX_TOKENS
    
    async def generate_answer(self, system_prompt: str, user_prompt: str, history: Optional[List[Dict[str, str]]] = None) -> str:
        """Generate a complete answer for user using system prompt, history, and user prompt."""
        try:
            messages = [{"role": "system", "content": system_prompt}]
            
            # Inject native conversation history to save tokens and improve model comprehension
            if history:
                messages.extend(history)
                
            messages.append({"role": "user", "content": user_prompt})
            
            chat_completion = await self.client.chat.completions.create(
                messages=messages, # type: ignore
                model=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            content = chat_completion.choices[0].message.content
            return content if content is not None else ""
        
        except Exception as e:
            logger.error(f"Error during LLM generation: {e}")
            return f"I apologize, but I encountered an error while generating the response. Error details: {str(e)}"
