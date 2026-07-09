"""
Conversation management endpoints for the chatbot API.

Endpoints:
    GET     /api/conversations        - List all conversations
    DELETE  /api/conversations        - Delete a conversation
"""
# Query: Declare query parameters in the URL
from fastapi import APIRouter, HTTPException, Request, Query, Depends
from config.settings import settings
from typing import Optional
from api.schemas.request import ConversationUpdateRequest, FeedbackRequest
from api.schemas.response import (
    ConversationListResponse, ConversationResponse,
    MessagePageResponse, MessageItem
)
from api.dependencies import get_current_user

router = APIRouter()


@router.get("/conversations", response_model=ConversationListResponse,
            summary="List all conversations", tags=["Conversations"])
def list_conversations(raw_request: Request, user: dict = Depends(get_current_user)) -> ConversationListResponse:
    """Return all conversations ordered by most recently updated."""
    conv_store = raw_request.app.state.conv_store
    rows = conv_store.list_conversations(user_id=user["id"], limit=20)
    items = [ConversationResponse(**r) for r in rows]
    return ConversationListResponse(conversations=items)


@router.get("/conversations/search", response_model=ConversationListResponse,
            summary="Search conversations", tags=["Conversations"])
def search_conversations(
    raw_request: Request,
    q: str = Query(..., min_length=1, description="Search query"),
    user: dict = Depends(get_current_user)
) -> ConversationListResponse:
    """Search conversations by title or message content."""
    conv_store = raw_request.app.state.conv_store
    rows = conv_store.search_conversations(search_query=q, user_id=user["id"])
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
    ),
    user: dict = Depends(get_current_user)
) -> MessagePageResponse:
    """
    Retrieve a page of messages using cursor-based pagination.

    First call (no before_id): returns the newest 'limit' messages.
    Subsequent calls (with before_id): returns the next older batch.
    """
    conv_store = raw_request.app.state.conv_store
     
    if not conv_store.conversation_exists(conv_id, user_id=user["id"]):
        raise HTTPException(status_code=404, detail="Conversation not found or unauthorized.")
    
    result = conv_store.get_message_page(
        conversation_id=conv_id,
        limit=limit,
        before_id=before_id
    )

    messages = [MessageItem(**m) for m in result["messages"]]
    return MessagePageResponse(messages=messages, has_more=result["has_more"])


@router.delete("/conversations/{conv_id}", summary="Delete a conversation", tags=["Conversations"])
def delete_conversation(conv_id: str, raw_request: Request, user: dict = Depends(get_current_user)) -> bool:
    """Delete a conversation and all its messages. Return True if succeed."""
    conv_store = raw_request.app.state.conv_store
    deleted = conv_store.delete_conversation(conv_id, user_id=user["id"])
    if not deleted:
        raise HTTPException(status_code=404, detail="Conversation not found or unauthorized.")
    return True


@router.put("/conversations/{conv_id}", summary="Update a conversation", tags=["Conversations"])
def update_conversation(
    conv_id: str, 
    request: ConversationUpdateRequest, 
    raw_request: Request,
    user: dict = Depends(get_current_user)
) -> bool:
    """Update title or pinned status."""
    conv_store = raw_request.app.state.conv_store
    if not conv_store.conversation_exists(conv_id, user_id=user["id"]):
        raise HTTPException(status_code=404, detail="Conversation not found or unauthorized.")
        
    conv_store.update_conversation(
        conversation_id=conv_id,
        title=request.title,
        is_pinned=request.is_pinned
    )
    return True


@router.post("/conversations/{conv_id}/messages/{msg_id}/feedback", summary="Add message feedback", tags=["Conversations"])
def add_feedback(
    conv_id: str,
    msg_id: int,
    request: FeedbackRequest,
    raw_request: Request,
    user: dict = Depends(get_current_user)
) -> bool:
    """Add like/dislike feedback to a message."""
    conv_store = raw_request.app.state.conv_store
    if not conv_store.conversation_exists(conv_id, user_id=user["id"]):
        raise HTTPException(status_code=404, detail="Conversation not found or unauthorized.")
        
    try:
        success = conv_store.add_feedback(
            message_id=msg_id,
            feedback_type=request.feedback_type,
            feedback_comment=request.feedback_comment
        )
        if not success:
            raise HTTPException(status_code=404, detail="Message not found.")
        return True
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
