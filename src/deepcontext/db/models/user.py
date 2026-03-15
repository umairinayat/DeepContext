"""
User model for authentication and API key storage.

Stores hashed passwords and encrypted OpenRouter API keys.
The OpenRouter key is encrypted at rest using Fernet symmetric encryption
derived from the server's JWT secret.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Optional

from sqlalchemy import Boolean, DateTime, Integer, String, Text, text as sa_text
from sqlalchemy.orm import Mapped, mapped_column

from deepcontext.db.models.base import Base


class User(Base):
    """
    Registered user with credentials and API key storage.

    The user_id used throughout DeepContext (memories, graph, etc.)
    is derived from this table's `id` field as f"user_{id}".
    """

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String(128), nullable=False, unique=True, index=True)
    email: Mapped[Optional[str]] = mapped_column(String(256), nullable=True, unique=True)
    hashed_password: Mapped[str] = mapped_column(String(256), nullable=False)

    # Encrypted OpenRouter API key (Fernet-encrypted, stored as base64 text)
    encrypted_api_key: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=sa_text("CURRENT_TIMESTAMP"),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=sa_text("CURRENT_TIMESTAMP"),
    )

    @property
    def deep_context_user_id(self) -> str:
        """The user_id string used in memories, graph, etc."""
        return f"user_{self.id}"

    def __repr__(self) -> str:
        return f"<User(id={self.id}, username='{self.username}')>"
