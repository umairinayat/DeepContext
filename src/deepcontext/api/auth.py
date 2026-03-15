"""
Authentication utilities for DeepContext API.

Provides JWT token creation/verification, password hashing,
API key encryption, and FastAPI dependency for protected routes.
"""

from __future__ import annotations

import base64
import hashlib
import os
from datetime import datetime, timedelta, timezone
from typing import Optional

from cryptography.fernet import Fernet
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from deepcontext.db.models.user import User

# ---------------------------------------------------------------------------
# Configuration (read from env, with sensible defaults for dev)
# ---------------------------------------------------------------------------

JWT_SECRET = os.environ.get("DEEPCONTEXT_JWT_SECRET", "dev-secret-change-in-production")
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_MINUTES = int(os.environ.get("DEEPCONTEXT_JWT_EXPIRE_MINUTES", "1440"))  # 24h default

# ---------------------------------------------------------------------------
# Password hashing
# ---------------------------------------------------------------------------

_pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(password: str) -> str:
    """Hash a plaintext password using bcrypt."""
    return _pwd_context.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    """Verify a plaintext password against a bcrypt hash."""
    return _pwd_context.verify(plain, hashed)


# ---------------------------------------------------------------------------
# JWT tokens
# ---------------------------------------------------------------------------


def create_access_token(user_id: int, username: str) -> str:
    """Create a JWT access token for a user."""
    expire = datetime.now(timezone.utc) + timedelta(minutes=JWT_EXPIRE_MINUTES)
    payload = {
        "sub": str(user_id),
        "username": username,
        "exp": expire,
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_access_token(token: str) -> dict:
    """Decode and verify a JWT token. Raises on invalid/expired."""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        if payload.get("sub") is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token: missing subject",
            )
        return payload
    except JWTError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid or expired token: {e}",
        )


# ---------------------------------------------------------------------------
# API key encryption (Fernet, derived from JWT_SECRET)
# ---------------------------------------------------------------------------


def _get_fernet() -> Fernet:
    """Derive a Fernet key from the JWT secret."""
    # Use SHA-256 of the JWT secret to get exactly 32 bytes, then base64-encode
    key_bytes = hashlib.sha256(JWT_SECRET.encode()).digest()
    fernet_key = base64.urlsafe_b64encode(key_bytes)
    return Fernet(fernet_key)


def encrypt_api_key(api_key: str) -> str:
    """Encrypt an API key for storage."""
    f = _get_fernet()
    return f.encrypt(api_key.encode()).decode()


def decrypt_api_key(encrypted: str) -> str:
    """Decrypt a stored API key."""
    f = _get_fernet()
    return f.decrypt(encrypted.encode()).decode()


# ---------------------------------------------------------------------------
# FastAPI dependency: get current authenticated user
# ---------------------------------------------------------------------------

_bearer_scheme = HTTPBearer()


class AuthenticatedUser:
    """Represents the current authenticated user extracted from JWT."""

    def __init__(self, user_id: int, username: str, deep_context_user_id: str) -> None:
        self.user_id = user_id
        self.username = username
        self.deep_context_user_id = deep_context_user_id


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(_bearer_scheme),
) -> AuthenticatedUser:
    """
    FastAPI dependency that extracts and validates the JWT from the
    Authorization: Bearer <token> header.

    Returns an AuthenticatedUser with user_id and deep_context_user_id.
    """
    payload = decode_access_token(credentials.credentials)
    user_id = int(payload["sub"])
    username = payload.get("username", "")
    return AuthenticatedUser(
        user_id=user_id,
        username=username,
        deep_context_user_id=f"user_{user_id}",
    )
