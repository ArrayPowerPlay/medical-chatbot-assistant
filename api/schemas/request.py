"""
Pydantic request models for the chatbot API.
"""
from pydantic import BaseModel, Field
from typing import Optional


class ChatRequest(BaseModel):
    """
    Schema for the POST /api/chat request body.

    Attributes:
        question: The user's medical question in natural language.
        conversation_id: Optional UUID to continue an existing conversation.
    """
    question: str = Field(..., min_length=1, max_length=2500,
                          description="Ask anything about medical domain",
                          examples=["What are the symptoms of Type 2 Diabetes?"])
    conversation_id: Optional[str] = Field(default=None, description="UUID of an existing conversation")