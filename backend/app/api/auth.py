"""Authentication routes: Registration, Login, Google OAuth, Profile & Settings."""

from datetime import timedelta
import uuid
from typing import Any, Dict, Optional
from fastapi import APIRouter, Depends, HTTPException, Header, status
from pydantic import BaseModel, EmailStr
from sqlalchemy.orm import Session
from app.core.config import settings
from app.core.database import get_db
from app.core.security import create_access_token, decode_access_token, get_password_hash, verify_password
from app.db.models import User

router = APIRouter(prefix="/auth", tags=["Auth"])


# Pydantic Schemas
class UserRegister(BaseModel):
    email: EmailStr
    password: str
    full_name: Optional[str] = None


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class GoogleAuthRequest(BaseModel):
    credential: Optional[str] = None
    email: EmailStr
    name: Optional[str] = None
    avatar: Optional[str] = None


class AnonymousSessionRequest(BaseModel):
    device_id: Optional[str] = None


class UserSettingsUpdate(BaseModel):
    api_keys: Optional[Dict[str, str]] = None
    default_model: Optional[str] = None
    default_provider: Optional[str] = None
    custom_endpoint: Optional[str] = None
    custom_api_key: Optional[str] = None
    system_prompt: Optional[str] = None


def get_current_user(
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db),
) -> User:
    """Dependency to extract authenticated User from Bearer JWT token."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Please sign in with Google or create an account.",
        )

    token = authorization.split(" ")[1]
    payload = decode_access_token(token)
    if not payload or "sub" not in payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired session token. Please sign in again.",
        )

    user_id = payload["sub"]
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User account not found.")
    return user


@router.post("/anonymous")
def create_anonymous_session(data: AnonymousSessionRequest, db: Session = Depends(get_db)):
    """Create a unique isolated session per device if the user hasn't signed in yet."""
    device_suffix = data.device_id or str(uuid.uuid4())[:8]
    guest_email = f"guest_{device_suffix}@localgpt.internal"
    
    user = db.query(User).filter(User.email == guest_email).first()
    if not user:
        user = User(
            email=guest_email,
            full_name=f"Guest {device_suffix[:4]}",
            hashed_password=get_password_hash(str(uuid.uuid4())),
            avatar_url=f"https://api.dicebear.com/7.x/bottts/svg?seed={device_suffix}",
        )
        db.add(user)
        db.commit()
        db.refresh(user)

    token = create_access_token(user.id)
    return {
        "access_token": token,
        "token_type": "bearer",
        "user": {
            "id": user.id,
            "email": user.email,
            "full_name": user.full_name,
            "avatar_url": user.avatar_url,
            "is_guest": True,
        }
    }


@router.post("/register")
def register(data: UserRegister, db: Session = Depends(get_db)):
    """Register a new private user account."""
    existing = db.query(User).filter(User.email == data.email.lower()).first()
    if existing:
        raise HTTPException(status_code=400, detail="An account with this email already exists.")

    user = User(
        email=data.email.lower(),
        hashed_password=get_password_hash(data.password),
        full_name=data.full_name or data.email.split("@")[0],
        avatar_url=f"https://api.dicebear.com/7.x/initials/svg?seed={data.email}",
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    token = create_access_token(user.id)
    return {
        "access_token": token,
        "token_type": "bearer",
        "user": {
            "id": user.id,
            "email": user.email,
            "full_name": user.full_name,
            "avatar_url": user.avatar_url,
            "is_guest": False,
        }
    }


@router.post("/login")
def login(data: UserLogin, db: Session = Depends(get_db)):
    """Log in with email and password."""
    user = db.query(User).filter(User.email == data.email.lower()).first()
    if not user or not user.hashed_password:
        raise HTTPException(status_code=400, detail="Invalid email or password.")

    if not verify_password(data.password, user.hashed_password):
        raise HTTPException(status_code=400, detail="Invalid email or password.")

    token = create_access_token(user.id)
    return {
        "access_token": token,
        "token_type": "bearer",
        "user": {
            "id": user.id,
            "email": user.email,
            "full_name": user.full_name,
            "avatar_url": user.avatar_url,
            "is_guest": False,
        }
    }


@router.post("/google")
def google_auth(data: GoogleAuthRequest, db: Session = Depends(get_db)):
    """Authenticate or register user via Google Sign-In."""
    email = data.email.lower()
    user = db.query(User).filter(User.email == email).first()

    if not user:
        user = User(
            email=email,
            full_name=data.name or email.split("@")[0],
            avatar_url=data.avatar or f"https://api.dicebear.com/7.x/initials/svg?seed={email}",
            hashed_password=get_password_hash(str(uuid.uuid4())),
        )
        db.add(user)
        db.commit()
        db.refresh(user)
    else:
        if data.name and not user.full_name:
            user.full_name = data.name
        if data.avatar:
            user.avatar_url = data.avatar
        db.commit()

    token = create_access_token(user.id)
    return {
        "access_token": token,
        "token_type": "bearer",
        "user": {
            "id": user.id,
            "email": user.email,
            "full_name": user.full_name,
            "avatar_url": user.avatar_url,
            "is_guest": False,
        }
    }


@router.get("/me")
def get_profile(current_user: User = Depends(get_current_user)):
    """Get the current authenticated user profile and settings."""
    return {
        "id": current_user.id,
        "email": current_user.email,
        "full_name": current_user.full_name,
        "avatar_url": current_user.avatar_url,
        "is_guest": current_user.email.startswith("guest_"),
        "settings": current_user.get_settings(),
    }


@router.patch("/settings")
def update_settings(
    data: UserSettingsUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Update settings for the current authenticated user."""
    current_settings = current_user.get_settings()
    update_data = data.model_dump(exclude_unset=True)
    current_settings.update(update_data)
    current_user.set_settings(current_settings)
    db.commit()
    return {"status": "success", "settings": current_user.get_settings()}
