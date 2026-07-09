from fastapi import APIRouter, Depends, HTTPException
from datetime import datetime, timedelta
import jwt
from passlib.context import CryptContext
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests

from config.settings import settings
from api.dependencies import get_db
from api.schemas.request import UserAuthRequest, UserRegisterRequest, GoogleAuthRequest
from api.schemas.response import AuthResponse
from src.storage.conversation_store import ConversationStore

router = APIRouter()

# bcrypt: hash algorithm
# deprecated="auto" -> auto update old hash password if bcrypt is updated to a new version
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def create_access_token(data: dict, expires_delta: timedelta):  # timedelta: represents the time difference
    """Create access token for authentication"""
    to_encode = data.copy()
    expire = datetime.utcnow() + expires_delta
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.JWT_SECRET, algorithm=settings.JWT_ALGORITHM)
    return encoded_jwt
 

@router.post("/register", response_model=AuthResponse)
def register(request: UserRegisterRequest, db: ConversationStore = Depends(get_db)):
    """Register with username and password"""
    # Check if user exists
    existing_user = db.get_user_by_username(request.username)
    if existing_user:
        raise HTTPException(status_code=400, detail="Username already registered")
        
    # Hash password
    hashed_password = pwd_context.hash(request.password)
    
    # Create user
    user_id = db.create_user_with_username(request.username, hashed_password, role="user")
    
    # Generate token
    access_token_expires = timedelta(minutes=settings.JWT_EXPIRATION_MINUTES)
    access_token = create_access_token(
        data={"sub": str(user_id), "role": "user"},
        expires_delta=access_token_expires
    )
    
    return AuthResponse(
        access_token=access_token,
        user={"id": user_id, "username": request.username, "role": "user", "question_count": 0}
    )
 

@router.post("/login", response_model=AuthResponse)
def login(request: UserAuthRequest, db: ConversationStore = Depends(get_db)):
    """Login using existing account"""
    user = db.get_user_by_username(request.username)
    if not user:
        raise HTTPException(status_code=401, detail="Incorrect username or password")
        
    if not user["password_hash"] or not pwd_context.verify(request.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Incorrect username or password")
        
    access_token_expires = timedelta(minutes=settings.JWT_EXPIRATION_MINUTES)
    access_token = create_access_token(
        data={"sub": str(user["id"]), "role": user["role"]},
        expires_delta=access_token_expires
    )
    
    return AuthResponse(
        access_token=access_token,
        user={"id": user["id"], "username": user["username"], "role": user["role"], "question_count": user["question_count"]}
    )
 

@router.post("/guest", response_model=AuthResponse)
def guest_login(db: ConversationStore = Depends(get_db)):
    """Access web without login, can only ask for maximum 10 questions"""
    import uuid
    # Create a random guest username
    guest_username = f"guest_{uuid.uuid4().hex[:8]}"   # .hex() returns hexadecimal type
    # Avoid slow bcrypt hashing for guests since they don't actually log in with this password
    hashed_password = f"dummy_{uuid.uuid4().hex}"
    
    user_id = db.create_user_with_username(guest_username, hashed_password, role="guest")
    
    access_token_expires = timedelta(minutes=settings.JWT_EXPIRATION_MINUTES)
    access_token = create_access_token(
        data={"sub": str(user_id), "role": "guest"},
        expires_delta=access_token_expires
    )
    
    return AuthResponse(
        access_token=access_token,
        user={"id": user_id, "username": guest_username, "role": "guest", "question_count": 0}
    )


@router.post("/google", response_model=AuthResponse)
def google_auth(request: GoogleAuthRequest, db: ConversationStore = Depends(get_db)):
    """Login/register using gmail"""
    try:
        # Verify Google token
        idinfo = id_token.verify_oauth2_token(
            request.token, 
            google_requests.Request(),   # send signals to Google's servers to verify the user's token
            settings.VITE_GOOGLE_CLIENT_ID
        )
        email = idinfo.get("email")
        if not email:
            raise HTTPException(status_code=400, detail="No email in Google token")
            
        # Check if user exists
        user = db.get_user_by_email(email)
        
        # Register if not exists
        if not user:
            user_id = db.create_user_with_email(email, role="user")
            if user_id is None:
                raise HTTPException(status_code=500, detail="Failed to create user")
            
            # Fetch the newly created user
            user = db.get_user_by_id(user_id)
            if not user:
                raise HTTPException(status_code=500, detail="Failed to create user")
                
        # Generate our JWT token
        access_token_expires = timedelta(minutes=settings.JWT_EXPIRATION_MINUTES)
        access_token = create_access_token(
            data={"sub": str(user["id"]), "role": user["role"]},
            expires_delta=access_token_expires
        )
        
        return AuthResponse(
            access_token=access_token,
            user={"id": user["id"], "email": user["email"], "role": user["role"], "question_count": user["question_count"]}
        )
    except ValueError:
        raise HTTPException(status_code=401, detail="Invalid Google token")
