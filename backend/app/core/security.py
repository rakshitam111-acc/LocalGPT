"""Security utilities: password hashing and JWT token handling."""

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional, Union
from jose import jwt
import hashlib
import hmac
import os
from app.core.config import settings


def get_password_hash(password: str) -> str:
    """Hash a password securely using PBKDF2 with SHA256 and unique salt."""
    salt = os.urandom(16).hex()
    key = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 100000)
    return f"{salt}${key.hex()}"


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plain password against its hash."""
    if not hashed_password or "$" not in hashed_password:
        # Fallback for old colon format if any exists
        if ":" in hashed_password:
            salt, hash_val = hashed_password.split(":", 1)
            test_hash = hashlib.sha256((salt + plain_password).encode("utf-8")).hexdigest()
            return test_hash == hash_val
        return False

    salt, expected_hex = hashed_password.split("$", 1)
    key = hashlib.pbkdf2_hmac("sha256", plain_password.encode("utf-8"), salt.encode("utf-8"), 100000)
    return hmac.compare_digest(key.hex(), expected_hex)


def create_access_token(subject: Union[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """Generate signed JWT access token."""
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)

    to_encode = {"exp": expire, "sub": str(subject)}
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return encoded_jwt


def decode_access_token(token: str) -> Optional[Dict[str, Any]]:
    """Decode and validate a JWT access token."""
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        return payload
    except Exception:
        return None
