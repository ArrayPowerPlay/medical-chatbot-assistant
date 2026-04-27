from groq import Groq
from config.settings import settings
from typing import List, Dict, Optional


class QueryRewriter:
    """
    Use LLM to rewrite user queries:
    1. Correct spelling errors
    2. Clarify abbreviations or ambiguous terms
    3. Connect to chat history
    """
    def __init__(self):
        self.client = Groq(api_key=settings.GROQ_API_KEY)
        self.model = settings.LLM_MODEL

    def rewrite(self, query: str, history: Optional[List[Dict[str, str]]] = None) -> str:
        """
        Parameters:
            query: User query
            history: Chat history - format: [{"role": "user", "content": "..."}]
        """
        system_prompt = (
            """You are a medical query optimizer for a medical RAG system. 
            Your objective is to transform raw, conversational user inputs into precise, standalone queries optimized for retrieval.

            ### RULES:
            1. Replace all pronouns with the exact clinical entities from the conversation history.
            2. Standardize brand names or layperson terms to their generic/clinical equivalents (e.g., "high blood pressure" -> "Hypertension").
            3. Strip away all conversational filler or introductory phrases (e.g., "Hi", "Can you tell me", "I want to know").
            4. Fix any typographical errors, especially in complex medical terminology.
            5. RETURN ONLY THE REWRITTEN QUERY TEXT. If the query is already optimal, return it as is."""
        )

        messages = [{"role": "system", "content": system_prompt}]

        if history:
            messages.extend(history[-5:])  # Retrieve 5 most recent messages to optimize token usage

        messages.append({"role": "user", "content": f"Rewrite this query: {query}"})

        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,         # type: ignore
                temperature=0.1,
                max_tokens=256
            )
            return completion.choices[0].message.content.strip()   # # type: ignore - choices = list of answers the model returns
        except Exception as e:
            print(f"Error in QueryRewriter {e}")
            return query                   # Return the query if error