"""
Alembic environment configuration for DeepContext.

Supports both online (direct DB connection) and offline (SQL script generation) modes.
Reads the database URL from DEEPCONTEXT_DATABASE_URL or falls back to alembic.ini.
"""

from __future__ import annotations

import os
import sys
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

# Ensure the src directory is on the path so deepcontext is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from deepcontext.db.models.base import Base  # noqa: E402
from deepcontext.db.models.memory import Memory  # noqa: E402, F401
from deepcontext.db.models.graph import Entity, Relationship, ConversationSummary  # noqa: E402, F401

# Alembic Config object — provides access to alembic.ini values
config = context.config

# Set up Python logging from alembic.ini
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Set the target metadata for autogenerate support
target_metadata = Base.metadata

# Override sqlalchemy.url from environment if available
db_url = os.environ.get("DEEPCONTEXT_DATABASE_URL", "")
if db_url:
    # Convert async URL to sync for Alembic (it uses synchronous connections)
    sync_url = db_url.replace("+asyncpg", "").replace("+aiosqlite", "")
    config.set_main_option("sqlalchemy.url", sync_url)


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    Generates SQL scripts without connecting to the database.
    Useful for reviewing migration SQL before applying.

    Usage: alembic upgrade head --sql
    """
    url = config.get_main_option("sqlalchemy.url")
    if not url:
        raise ValueError(
            "No database URL configured. Set DEEPCONTEXT_DATABASE_URL or "
            "sqlalchemy.url in alembic.ini."
        )
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    Connects to the database and applies migrations directly.

    Usage: alembic upgrade head
    """
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
