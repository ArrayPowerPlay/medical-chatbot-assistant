"""
Pydantic request models for the chatbot API.
"""
from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Optional
import re


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


class UserAuthRequest(BaseModel):
    username: str = Field(..., description="Username")
    password: str = Field(..., min_length=6, description="User password")


class UserRegisterRequest(BaseModel):
    username: str = Field(..., description="Username")
    password: str = Field(..., min_length=6, description="User password")
    confirm_password: str = Field(..., min_length=6, description="Confirm password")

    @field_validator('password')
    @classmethod        # This function functions in class-level, not object-level
    def validate_password_complexity(cls, v: str) -> str:
        if not re.search(r'\d', v):
            raise ValueError('Password must contain at least one number')
        if not re.search(r'[!@#$%^&*(),.?":{}|<>]', v):
            raise ValueError('Password must contain at least one special character')
        return v
        
    @model_validator(mode='after')   # mode='after' -> Run after Pydantic has validated the data
    # Apply logic check for two or more fields
    def check_passwords_match(self) -> 'UserRegisterRequest':
        if self.password != self.confirm_password:
            raise ValueError('Passwords do not match')
        return self


class GoogleAuthRequest(BaseModel):
    token: str = Field(..., description="Google OAuth2 ID token")
