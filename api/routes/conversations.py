"""
Conversation management endpoints for the chatbot API.

Endpoints:
    GET     /api/conversations        - List all conversations
    DELETE  /api/conversations        - Delete a conversation
"""
from fastapi import APIRouter, HTTPException, Request
from api.schemas.response import ConversationListResponse, ConversationResponse

router = APIRouter()


@router.get("/conversations", response_model=ConversationListResponse,
            summary="List all conversations", tags=["Conversations"])
def list_conversations(raw_request: Request) -> ConversationListResponse:
    """Return all conversations ordered by most recently updated."""
    conv_store = raw_request.app.state.conv_store
    rows = conv_store.list_conversations(limit=20)
    items = [ConversationResponse(**r) for r in rows]
    return ConversationListResponse(conversations=items)


@router.delete("/conversations/{conv_id}", summary="Delete a conversation", tags=["Conversations"])
def delete_conversation(conversation_id: str, raw_request: Request) -> bool:
    """Delete a conversation and all its messages. Return True if succeed."""
    conv_store = raw_request.app.state.conv_store
    deleted = conv_store.delete_conversation(conversation_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    return True
