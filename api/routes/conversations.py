"""
Conversation management endpoints for the chatbot API.

Endpoints:
    GET     /api/conversations        - List all conversations
    DELETE  /api/conversations        - Delete a conversation
"""
# Query: Declare query parameters in the URL
from fastapi import APIRouter, HTTPException, Request, Query
from config.settings import settings
from typing import Optional
from api.schemas.response import (
    ConversationListResponse, ConversationResponse,
    MessagePageResponse, MessageItem
)

router = APIRouter()


@router.get("/conversations", response_model=ConversationListResponse,
            summary="List all conversations", tags=["Conversations"])
def list_conversations(raw_request: Request) -> ConversationListResponse:
    """Return all conversations ordered by most recently updated."""
    conv_store = raw_request.app.state.conv_store
    rows = conv_store.list_conversations(limit=20)
    items = [ConversationResponse(**r) for r in rows]
    return ConversationListResponse(conversations=items)


@router.get("/conversations/{conv_id}/messages", response_model=MessagePageResponse,
            summary="Get paginated messages for a conversation", tags=["Conversations"])
def get_messages(
    conv_id: str,
    raw_request: Request,
    limit: int = Query(
        default=settings.MESSAGE_PAGE_SIZE,
        ge=1,
        le=100,
        description="Number of messages to return per page"
    ),
    before_id: Optional[int] = Query(
        default=None,
        description="Cursor: return messages with id < before_id (older messages)"
    )
) -> MessagePageResponse:
    """
    Retrieve a page of messages using cursor-based pagination.

    First call (no before_id): returns the newest 'limit' messages.
    Subsequent calls (with before_id): returns the next older batch.
    """
    conv_store = raw_request.state.app.conv_store
     
    if not conv_store.conversation_exists(conv_id):
        raise HTTPException(status_code=404, detail="Conversation not found.")
    
    result = conv_store.get_messages_page(
        conversation_id=conv_id,
        limit=limit,
        before_id=before_id
    )

    messages = [MessageItem(**m) for m in result["messages"]]
    return MessagePageResponse(messages=messages, has_more=result["has_more"])


@router.delete("/conversations/{conv_id}", summary="Delete a conversation", tags=["Conversations"])
def delete_conversation(conv_id: str, raw_request: Request) -> bool:
    """Delete a conversation and all its messages. Return True if succeed."""
    conv_store = raw_request.app.state.conv_store
    deleted = conv_store.delete_conversation(conv_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    return True
