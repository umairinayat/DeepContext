<p align="center">
  <h1 align="center">DeepContext</h1>
  <p align="center">Hierarchical memory system for AI agents -- async, graph-aware, with hybrid retrieval and memory lifecycle management.</p>
</p>

<p align="center">
  <a href="https://github.com/umairinayat/DeepContext/blob/master/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License: MIT"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.11%2B-blue.svg" alt="Python 3.11+"></a>
  <a href="https://github.com/umairinayat/DeepContext"><img src="https://img.shields.io/badge/tests-110%20passed-brightgreen.svg" alt="Tests: 110 passed"></a>
  <a href="https://fastapi.tiangolo.com/"><img src="https://img.shields.io/badge/API-FastAPI-009688.svg" alt="FastAPI"></a>
  <a href="https://github.com/pgvector/pgvector"><img src="https://img.shields.io/badge/vector%20store-pgvector-orange.svg" alt="pgvector"></a>
  <a href="https://pydantic-docs.helpmanual.io/"><img src="https://img.shields.io/badge/models-Pydantic%20v2-e92063.svg" alt="Pydantic v2"></a>
</p>

---

DeepContext gives AI agents persistent, structured memory. Conversations are automatically broken into semantic facts, stored with embeddings, linked in a knowledge graph, and retrieved using a hybrid pipeline that fuses vector similarity, keyword search, and graph traversal.

## Features

- **Hierarchical Memory** -- Working, short-term, and long-term tiers inspired by human cognition
- **Memory Types** -- Semantic (facts), episodic (events), and procedural (how-to) memories
- **Knowledge Graph** -- Entities and relationships extracted from conversations, stored in PostgreSQL (no Neo4j required)
- **Hybrid Retrieval** -- Reciprocal Rank Fusion (RRF) across vector, keyword, and graph search
- **Memory Lifecycle** -- Ebbinghaus forgetting curve decay, consolidation of short-term into long-term, automatic cleanup
- **Fully Async** -- Built on SQLAlchemy async, asyncpg, and AsyncOpenAI
- **Multi-user** -- All memories scoped by `user_id`
- **REST API** -- FastAPI server with 7 endpoints
- **Pluggable LLM** -- OpenAI and OpenRouter support out of the box
- **SQLite Fallback** -- Works without PostgreSQL for development

---

## Quick Start

### Installation

```bash
git clone https://github.com/umairinayat/DeepContext.git
cd DeepContext
python -m venv .venv

# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install -e ".[all]"
```

### Configuration

Create a `.env` file in the project root:

```env
DEEPCONTEXT_OPENAI_API_KEY=sk-your-key-here

# PostgreSQL (recommended for production)
# DEEPCONTEXT_DATABASE_URL=postgresql+asyncpg://user:pass@localhost:5432/deepcontext

# SQLite fallback (default, no setup needed)
# Automatically uses ~/.deepcontext/memory.db
```

All settings can be passed as environment variables with the `DEEPCONTEXT_` prefix, or directly in code.

### Basic Usage

```python
import asyncio
from deepcontext import DeepContext

async def main():
    ctx = DeepContext(openai_api_key="sk-...")
    await ctx.init()

    # Store memories from a conversation
    response = await ctx.add(
        messages=[
            {"role": "user", "content": "I'm a Python developer working at Acme Corp"},
            {"role": "assistant", "content": "Nice to meet you!"},
        ],
        user_id="user_1",
        conversation_id="conv_1",
    )
    print(f"Stored {response.memories_added} memories, found {response.entities_found} entities")

    # Search memories
    results = await ctx.search("What does the user do for work?", user_id="user_1")
    for r in results.results:
        print(f"  [{r.tier.value}] {r.text} (score: {r.score:.3f})")

    # Explore the knowledge graph
    neighbors = await ctx.get_entity_graph("user_1", "Acme Corp", depth=2)
    for n in neighbors:
        print(f"  {n['entity']} --{n['relation']}--> (depth {n['depth']})")

    # Run lifecycle maintenance (decay + consolidation + cleanup)
    stats = await ctx.run_lifecycle("user_1")
    print(f"Decayed: {stats['memories_decayed']}, Consolidated: {stats['memories_consolidated']}")

    await ctx.close()

asyncio.run(main())
```

### Interactive Demo

```bash
python examples/chat_demo.py
```

An interactive chatbot that remembers conversations across turns. Special commands:

| Command | Description |
|---------|-------------|
| `memories` | Search stored memories |
| `graph <entity>` | Show knowledge graph for an entity |
| `lifecycle` | Run decay / consolidation / cleanup |
| `exit` | Quit |

---

## Architecture

```
deepcontext/
  __init__.py               DeepContext (alias), exports
  core/
    settings.py             Configuration (pydantic-settings, env vars, .env)
    types.py                Enums, Pydantic models (facts, entities, responses)
    clients.py              OpenAI/OpenRouter async client wrapper
  memory/
    engine.py               MemoryEngine -- main orchestrator
  extraction/
    extractor.py            LLM-based fact and entity extraction
    prompts.py              Prompt templates for extraction/classification
  retrieval/
    hybrid.py               HybridRetriever (vector + keyword + graph + RRF)
  graph/
    knowledge_graph.py      Entity/relationship CRUD, BFS traversal
  lifecycle/
    manager.py              Ebbinghaus decay, consolidation, cleanup
  vectorstore/
    base.py                 Abstract vector store interface
    pgvector_store.py       pgvector implementation (SQLite cosine fallback)
  db/
    database.py             Async SQLAlchemy engine manager
    models/
      base.py               Base ORM model
      memory.py             Memory table (embeddings, tiers, types, decay)
      graph.py              Entity, Relationship, ConversationSummary tables
  api/
    server.py               FastAPI REST API
```

---

## How It Works

### Memory Pipeline

When you call `ctx.add(messages, user_id)`:

```
Conversation ──> LLM Extraction ──> Classification ──> Embedding ──> Storage
                      |                   |                             |
                      v                   v                             v
                 Facts, Entities    ADD / UPDATE /              Knowledge Graph
                 Relationships      REPLACE / NOOP                  Update
```

1. **Extraction** -- The LLM analyzes the conversation and extracts semantic facts, episodic events, entities, and relationships
2. **Classification** -- Each extracted fact is compared against existing memories. The LLM decides whether to ADD, UPDATE, REPLACE, or skip (NOOP)
3. **Embedding** -- New/updated facts are embedded using the configured embedding model
4. **Storage** -- Memories are stored with their embeddings, tier (short-term), type, importance, and confidence scores
5. **Graph Update** -- Extracted entities and relationships are upserted into the knowledge graph
6. **Auto-consolidation** -- If short-term memory count exceeds the threshold, consolidation is triggered

### Hybrid Retrieval

When you call `ctx.search(query, user_id)`:

```
Query ──> Embed ──> Vector Search (0.6) ──┐
  |                                        ├──> RRF Fusion ──> Scoring ──> Results
  ├─────> Keyword Search (0.25) ──────────┤
  |                                        |
  └─────> Graph Expansion (0.15) ─────────┘
```

1. **Vector search** -- Query is embedded and compared via cosine similarity (pgvector or Python fallback)
2. **Keyword search** -- PostgreSQL `tsvector` full-text search (ILIKE fallback on SQLite)
3. **Graph expansion** -- Entities mentioned in the query are found, their graph neighbors are traversed, and memories referencing those entities are boosted
4. **RRF fusion** -- Results from all three strategies are combined using Reciprocal Rank Fusion (weights: vector 0.6, keyword 0.25, graph 0.15)
5. **Scoring** -- Final score applies importance, recency decay, confidence, and access-count boost
6. **Access tracking** -- Each returned memory's access count and timestamp are updated

### Memory Lifecycle

When you call `ctx.run_lifecycle(user_id)`:

```
Short-term Memories ──> Decay (Ebbinghaus) ──> Consolidation (LLM merge) ──> Long-term
                              |                        |
                              v                        v
                        Deactivate               Group by entity
                       (importance < 0.05)     overlap (Union-Find)
```

1. **Decay** -- Ebbinghaus forgetting curve: `R = e^(-0.693 * days / effective_half_life)`. Frequently accessed memories decay slower. Memories below 0.05 importance are deactivated
2. **Consolidation** -- When short-term memory count >= threshold (default 20), memories are grouped by entity overlap (Union-Find), each group is merged by the LLM into a long-term fact, and source memories are deactivated
3. **Cleanup** -- Remaining low-importance non-long-term memories are soft-deleted

---

## REST API

Start the server:

```bash
uvicorn deepcontext.api.server:app --reload
```

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/memory/add` | Extract and store memories from messages |
| `POST` | `/memory/search` | Hybrid search across memories |
| `PUT` | `/memory/update` | Update a memory's text and re-embed |
| `DELETE` | `/memory/delete` | Soft-delete a memory |
| `POST` | `/graph/neighbors` | Get knowledge graph neighborhood |
| `POST` | `/lifecycle/run` | Run decay + consolidation + cleanup |

### Example: Add Memory

```bash
curl -X POST http://localhost:8000/memory/add \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "I prefer Python over JavaScript"},
      {"role": "assistant", "content": "Got it, Python is your go-to!"}
    ],
    "user_id": "user_1",
    "conversation_id": "conv_1"
  }'
```

### Example: Search

```bash
curl -X POST http://localhost:8000/memory/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "programming languages",
    "user_id": "user_1",
    "limit": 5
  }'
```

---

## Configuration Reference

All settings use the `DEEPCONTEXT_` env prefix. Set them in `.env` or pass directly to `DeepContext()`.

| Setting | Default | Description |
|---------|---------|-------------|
| `database_url` | SQLite fallback | PostgreSQL connection URL (`postgresql+asyncpg://...`) |
| `llm_provider` | `openai` | `openai` or `openrouter` |
| `openai_api_key` | -- | Required for OpenAI provider |
| `openrouter_api_key` | -- | Required for OpenRouter provider |
| `llm_model` | `gpt-4o-mini` | Model for fact extraction and classification |
| `embedding_model` | `text-embedding-3-small` | Embedding model |
| `embedding_dimensions` | `1536` | Embedding vector dimensions |
| `consolidation_threshold` | `20` | Short-term memories before auto-consolidation |
| `decay_half_life_days` | `7.0` | Ebbinghaus half-life for episodic decay |
| `connection_similarity_threshold` | `0.6` | Min cosine similarity for memory connections |
| `max_connections_per_memory` | `5` | Max connections per memory node |
| `debug` | `false` | Enable debug logging |
| `auto_consolidate` | `true` | Auto-consolidate on add |

---

## Development

### Running Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

110 tests covering all subsystems. Tests use in-memory SQLite and mock LLM clients -- no API keys or database needed.

### PostgreSQL + pgvector Setup (Production)

DeepContext works with SQLite for development, but PostgreSQL with pgvector is recommended for production (native vector indexing, full-text search with `tsvector`, JSONB).

#### Windows

1. **Install PostgreSQL 15+** from https://www.postgresql.org/download/windows/ (the installer includes pgAdmin)

2. **Install pgvector** -- after PostgreSQL is installed:
   ```powershell
   # Option A: Using pgvector installer (recommended)
   # Download the latest release from https://github.com/pgvector/pgvector/releases
   # Run the .exe installer matching your PostgreSQL version

   # Option B: Build from source (requires Visual Studio Build Tools)
   git clone https://github.com/pgvector/pgvector.git
   cd pgvector
   # Set environment for your PG version, e.g.:
   set "PG_HOME=C:\Program Files\PostgreSQL\16"
   nmake /F Makefile.win install
   ```

3. **Create the database and enable pgvector**:
   ```powershell
   psql -U postgres -c "CREATE DATABASE deepcontext;"
   psql -U postgres -d deepcontext -c "CREATE EXTENSION IF NOT EXISTS vector;"
   ```

4. **Update `.env`**:
   ```env
   DEEPCONTEXT_DATABASE_URL=postgresql+asyncpg://postgres:yourpassword@localhost:5432/deepcontext
   ```

5. **Run Alembic migrations**:
   ```bash
   alembic upgrade head
   ```

#### Linux / macOS

```bash
# Ubuntu/Debian
sudo apt install postgresql-16 postgresql-16-pgvector

# macOS (Homebrew)
brew install postgresql@16 pgvector

# Create database
createdb deepcontext
psql deepcontext -c "CREATE EXTENSION IF NOT EXISTS vector;"

# Set connection URL
export DEEPCONTEXT_DATABASE_URL="postgresql+asyncpg://user:pass@localhost:5432/deepcontext"

# Run migrations
alembic upgrade head
```

### Database Migrations (Alembic)

The project uses Alembic for schema versioning. Migration files are in `alembic/versions/`.

```bash
# Apply all migrations (requires PostgreSQL connection)
alembic upgrade head

# Check current migration status
alembic current

# Generate a new migration after model changes
alembic revision --autogenerate -m "description of changes"

# Rollback one migration
alembic downgrade -1

# Generate SQL without applying (offline mode)
alembic upgrade head --sql
```

> **Note:** Alembic migrations target PostgreSQL. SQLite mode uses `Base.metadata.create_all()` at runtime and does not need Alembic.

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.11+ with full type annotations |
| ORM | SQLAlchemy 2.0 (async) |
| Vector Store | pgvector (PostgreSQL) |
| LLM | OpenAI API / OpenRouter |
| API | FastAPI + Uvicorn |
| Validation | Pydantic v2 + pydantic-settings |
| Math | NumPy |
| Migrations | Alembic |
| Testing | pytest + pytest-asyncio + httpx |

---

## License

MIT -- see [LICENSE](LICENSE) for details.

---

<p align="center">
  Built by <a href="https://github.com/umairinayat">@umairinayat</a>
</p>
