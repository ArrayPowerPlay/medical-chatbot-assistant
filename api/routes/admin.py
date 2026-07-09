from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from typing import List, Dict, Any
from passlib.context import CryptContext
from api.schemas.request import PasswordUpdateRequest

from api.dependencies import get_db, verify_admin
from src.storage.conversation_store import ConversationStore

router = APIRouter()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

@router.get("/stats", response_model=Dict[str, Any])
def get_stats(db: ConversationStore = Depends(get_db), _: dict = Depends(verify_admin)):
    """Get system statistics for Admin Dashboard"""
    return db.get_admin_stats()


@router.get("/users", response_model=Dict[str, Any])
def get_users(search: str = None, role: str = None, limit: int = 20, offset: int = 0, db: ConversationStore = Depends(get_db), _: dict = Depends(verify_admin)):
    """Get all users for Admin Dashboard with pagination, search, and filtering"""
    return db.get_all_users(limit=limit, offset=offset, search=search, role=role)

@router.get("/users/{user_id}/conversations", response_model=List[Dict[str, Any]])
def get_user_conversations(user_id: int, db: ConversationStore = Depends(get_db), _: dict = Depends(verify_admin)):
    """Get all conversations of a specific user for Admin Dashboard"""
    # Verify user exists
    user = db.get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return db.list_conversations(user_id=user_id)

@router.get("/conversations/{conv_id}/messages", response_model=Dict[str, Any])
def get_conversation_messages(
    conv_id: str, 
    limit: int = 50, 
    before_id: int = None, 
    db: ConversationStore = Depends(get_db), 
    _: dict = Depends(verify_admin)
):
    """Get messages for a conversation (Read-only view for Admin)"""
    if not db.conversation_exists(conv_id):
        raise HTTPException(status_code=404, detail="Conversation not found")
    return db.get_message_page(conversation_id=conv_id, limit=limit, before_id=before_id)


@router.delete("/users/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_user(user_id: int, db: ConversationStore = Depends(get_db), admin_user: dict = Depends(verify_admin)):
    """Delete a user permanently"""
    if admin_user["id"] == user_id:
        raise HTTPException(status_code=400, detail="Cannot delete your own admin account")
        
    success = db.delete_user(user_id)
    if not success:
        raise HTTPException(status_code=404, detail="User not found")


@router.put("/users/{user_id}/password", status_code=status.HTTP_200_OK)
def update_user_password(
    user_id: int, 
    request: PasswordUpdateRequest,
    db: ConversationStore = Depends(get_db), 
    _: dict = Depends(verify_admin)
):
    """Force update a user's password"""
    hashed_password = pwd_context.hash(request.new_password)
    success = db.update_user_password(user_id, hashed_password)
    
    if not success:
        raise HTTPException(status_code=404, detail="User not found")
        
    return {"message": "Password updated successfully"}


@router.get("/feedback/bad", response_model=List[Dict[str, Any]])
def get_bad_feedback(limit: int = 20, offset: int = 0, db: ConversationStore = Depends(get_db), _: dict = Depends(verify_admin)):
    """Get recent disliked messages with comments (with pagination)"""
    return db.get_bad_feedback_messages(limit=limit, offset=offset)


@router.get("/feedback/good", response_model=List[Dict[str, Any]])
def get_good_feedback(limit: int = 20, offset: int = 0, db: ConversationStore = Depends(get_db), _: dict = Depends(verify_admin)):
    """Get recent liked messages with comments (with pagination)"""
    return db.get_good_feedback_messages(limit=limit, offset=offset)
