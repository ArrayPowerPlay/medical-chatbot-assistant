"""
Pydantic response models for the chatbot API.
"""
from pydantic import BaseModel, Field
from typing import List


class SourceItem(BaseModel):
    """A single retrieved source (text passage or KG path)."""
    source_type: str = Field(description="'text_retrieval' or kg_retrieval.")
    content: str = Field(description="The text passage or KG path.")


class ChatResponse(BaseModel):
    """The pydantic model for a response of the LLM."""
    answer: str = Field(description="The generated answer.")
    sources: List[SourceItem] = Field(default_factory=list)
    conversation_id: str = Field(description="Conversation UUID.")


class ConversationResponse(BaseModel):
    id: str
    title: str
    created_at: str
    updated_at: str


class ConversationListResponse(BaseModel):
    conversations: List[ConversationResponse] = Field(default_factory=list)


class MessageItem(BaseModel):
    """The pydantic model for a message (of user or LLM)."""
    id: int
    role: str
    content: str
    created_at: str


class MessagePageResponse(BaseModel):
    """The pydantic model for a load of messages returned for the user UI when scrolling."""
    messages: List[MessageItem]
    has_more: bool               # has_more = True if history still has older messages


class HealthResponse(BaseModel):
    """Schema for the GET /api/health response body."""
    status: str = Field(description="Overall health status.")
    services: dict = Field(default_factory=dict) # A dict with (key, value) = (database_name, status)