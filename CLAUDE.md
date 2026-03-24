# CLAUDE.md -- DeepContext Project Context

> This file is the single source of truth for AI assistants working on this project.
> It documents what has been built, what is planned, and the technical decisions made.

---

## Project Overview

**DeepContext** is a hierarchical memory system for AI agents. It gives agents persistent, structured memory where conversations are automatically broken into semantic facts, stored with embeddings, linked in a knowledge graph, and retrieved using a hybrid pipeline (vector + keyword + graph).

- **Repo:** https://github.com/umairinayat/DeepContext
- **Owner:** umairinayat
- **License:** MIT
- **Python:** 3.11+ (currently running 3.13.5 on Windows)

---

## What Has Been Built (Complete)

### Backend -- Python Package (`src/deepcontext/`)

| Module | File | Purpose |
|--------|------|---------|
| Core | `core/settings.py` | Pydantic-settings config, reads `.env` with `DEEPCONTEXT_` prefix |
| Core | `core/types.py` | All Pydantic models/enums: MemoryTier, MemoryType, EntityType, ToolAction, ExtractedFact, ExtractionResult, SearchResponse, AddResponse |
| Core | `core/clients.py` | LLMClients -- async OpenAI/OpenRouter wrapper for chat + embeddings |
| Database | `db/database.py` | Async SQLAlchemy engine manager |
| Database | `db/models/base.py` | DeclarativeBase |
| Database | `db/models/memory.py` | Memory ORM model with `_USE_PG` flag, `set_embedding()`/`get_embedding()` helpers for SQLite/pgvector dual mode |
| Database | `db/models/graph.py` | Entity, Relationship, ConversationSummary ORM models |
| Vector Store | `vectorstore/base.py` | Abstract BaseVectorStore |
| Vector Store | `vectorstore/pgvector_store.py` | PgVectorStore with `_search_sqlite`/`_search_pgvector` dual paths |
| Extraction | `extraction/prompts.py` | Prompt templates for LLM extraction |
| Extraction | `extraction/extractor.py` | Extractor -- LLM-based fact/entity extraction and action classification |
| Graph | `graph/knowledge_graph.py` | KnowledgeGraph -- entity/relationship CRUD, BFS traversal |
| Retrieval | `retrieval/hybrid.py` | HybridRetriever -- vector + keyword + graph search with RRF fusion |
| Lifecycle | `lifecycle/manager.py` | LifecycleManager -- Ebbinghaus decay, consolidation (Union-Find), cleanup |
| Engine | `memory/engine.py` | MemoryEngine (aliased as `DeepContext`) -- main orchestrator |
| API | `api/server.py` | FastAPI REST API with 7 endpoints + CORS + static file serving |

### REST API Endpoints

| Method | Path | Purpose | Request Body |
|--------|------|---------|-------------|
| GET | `/health` | Health check | -- |
| POST | `/memory/add` | Extract & store memories | `{messages, user_id, conversation_id}` |
| POST | `/memory/search` | Hybrid retrieval | `{query, user_id, limit, tier, memory_type, include_graph}` |
| PUT | `/memory/update` | Update memory text & re-embed | `{memory_id, text, user_id}` |
| DELETE | `/memory/delete` | Soft-delete a memory | `{memory_id, user_id}` |
| POST | `/graph/neighbors` | Knowledge graph BFS traversal | `{user_id, entity_name, depth}` |
| POST | `/lifecycle/run` | Decay + consolidation + cleanup | `{user_id}` |

### Graph Data Model

**Entities** (nodes):
```
id, user_id, name, entity_type (person|organization|technology|concept|location|event|preference|other),
attributes (JSON), mention_count, created_at, updated_at
```

**Relationships** (edges):
```
id, user_id, source_entity_id, target_entity_id, relation (string like "works_with", "prefers"),
strength (float 0-2, increases on repeated mentions), properties (JSON), created_at
```

**Graph neighbor response format:**
```json
[
  {"entity": "FastAPI", "entity_type": "technology", "relation": "built_with", "depth": 1, "strength": 0.9},
  {"entity": "pytest", "entity_type": "technology", "relation": "tested_with", "depth": 1, "strength": 0.85}
]
```

### Tests -- 188/188 Passing

| File | Tests | Covers |
|------|-------|--------|
| `test_core.py` | 23 | Enums, Pydantic models, settings |
| `test_database.py` | 12 | ORM CRUD operations |
| `test_vectorstore.py` | 16 | Cosine similarity, vector search |
| `test_graph.py` | 12 | Entity upsert, relationships, BFS traversal |
| `test_extraction.py` | 16 | JSON parsing, LLM extraction mocks |
| `test_retrieval_lifecycle.py` | 12 | RRF fusion, decay, consolidation |
| `test_api.py` | 10 | REST endpoint tests via httpx |
| `test_edge_cases.py` | 22 | Edge cases for extraction, graph, vector store |
| `test_engine_integration.py` | 16 | Full engine integration tests |
| `test_multi_user.py` | 7 | User isolation for memories, graph, lifecycle |

### Alembic Migrations

- `alembic.ini` -- config, reads `DEEPCONTEXT_DATABASE_URL`
- `alembic/env.py` -- loads ORM models, converts async URLs to sync
- `alembic/versions/2026_02_21_001_initial_schema.py` -- creates all 4 tables + pgvector extension + IVFFlat + GIN indexes

### Website (Landing Page + Dashboard) -- `website/`

A React + Vite app exists at `website/` with:
- **Home** (`/`) -- hero section, feature grid, architecture diagram, install banner
- **Docs** (`/docs`) -- documentation page
- **Demo** (`/demo`) -- interactive demo with simulated (fake) API responses for add/search/graph/lifecycle
- **Dashboard** (`/dashboard`) -- real interactive dashboard connected to the backend API:
  - **Stats** tab -- summary cards (total memories, entities, relationships) with breakdowns by tier/type
  - **Chat Input** tab -- paste conversations, extract memories via `/memory/add`
  - **Knowledge Graph** tab -- interactive force-directed graph with 2D/3D toggle, click-to-expand, color-coded entity types, responsive ResizeObserver sizing
  - **Memories** tab -- browse, search, filter by tier/type, paginate, click-to-expand detail
  - **Lifecycle** tab -- run decay/consolidation/cleanup, view results
  - User ID input, backend status indicator, refresh button
- API client: `src/api/client.js` with all endpoint functions
- Components: `Navbar`, `Footer`, `CodeBlock` (Prism.js), `StatsCards`, `ChatPanel`, `GraphViz`, `MemoryBrowser`, `LifecycleControls`
- Deployed via `gh-pages` to GitHub Pages

**Note:** The Demo page (`/demo`) still uses hardcoded/simulated responses. The Dashboard page (`/dashboard`) connects to the real backend at `http://localhost:8000`.

---

## What Needs to Be Built Next

### Phase 1: Interactive Frontend Dashboard (Priority: HIGH)

Build a real interactive frontend where users can:

1. **Paste/type chat conversations** and submit them to the backend for memory extraction
2. **Search memories** with real hybrid retrieval results
3. **Visualize the knowledge graph** as an interactive node-link diagram
4. **Run lifecycle operations** and see the results
5. **Browse stored memories** with filtering by tier/type/user

This replaces the current fake demo with a real, functional dashboard.

#### Where to Build It

Two options:

**Option A (Recommended): Extend `website/` with a new `/dashboard` route**
- Add new pages/components to the existing React app
- The landing page stays as-is for marketing
- Dashboard connects to `http://localhost:8000` (the FastAPI backend)
- Single codebase, single deploy

**Option B: Build inside `src/deepcontext/api/static/`**
- The server already has `app.mount("/static", ...)` and serves `index.html` at `/`
- Self-contained -- no separate frontend server needed
- But limited: no React, no build tooling (unless we add it)

**Decision: Go with Option A** -- extend the existing React app.

---

### Phase 2: Graph Visualization -- Technology Decision

The knowledge graph needs an interactive visualization where:
- **Nodes** = entities (colored/shaped by `entity_type`)
- **Edges** = relationships (labeled with `relation`, thickness = `strength`)
- Users can click nodes to expand neighbors (calls `/graph/neighbors`)
- Zoom, pan, hover tooltips showing attributes
- Depth control (1-5 hops)

#### Technology Comparison

| Library | Type | Rendering | Best For | Performance (1k nodes) | Interactive? | Learning Curve |
|---------|------|-----------|----------|----------------------|-------------|----------------|
| **D3.js force-directed** | Low-level | SVG | Full control, custom layouts | Good | Yes (drag, zoom, click) | High |
| **react-force-graph-2d** | React wrapper | Canvas 2D | Quick React integration, good perf | Very good | Yes | Low |
| **react-force-graph-3d** | React wrapper | Three.js/WebGL | Wow-factor 3D graphs | Good (WebGL) | Yes (orbit, click) | Low |
| **Cytoscape.js** | Dedicated graph lib | Canvas | Complex graph analytics, layouts | Excellent | Yes | Medium |
| **Sigma.js** | WebGL graph renderer | WebGL | Massive graphs (10k+ nodes) | Excellent | Yes | Medium |
| **vis-network** | Graph lib | Canvas | Simple graphs, quick setup | Good | Yes | Low |
| **Raw Three.js** | 3D engine | WebGL | Full 3D scenes | Depends on impl | Manual | Very high |
| **Raw SVG (React)** | Manual | SVG | Tiny graphs, full control | Poor at scale | Manual | Medium |

#### Recommendation: **react-force-graph-2d** (primary) + **react-force-graph-3d** (toggle)

**Why:**

1. **react-force-graph-2d** (`react-force-graph`) is the best fit:
   - Built for React -- works as a component with props
   - Uses Canvas2D (via `force-graph` under the hood) -- fast for our expected graph sizes (10-500 nodes)
   - Force-directed layout is natural for knowledge graphs
   - Built-in: zoom, pan, drag nodes, click handlers, hover tooltips
   - Node coloring by type, edge labels, link width by strength -- all supported
   - Can add a "3D mode" toggle using `react-force-graph-3d` (same API, uses Three.js)
   - npm: `react-force-graph-2d` and `react-force-graph-3d`

2. **Why NOT raw SVG:** SVG doesn't scale well beyond ~200 nodes (DOM node per element). Our graphs could grow larger.

3. **Why NOT raw Three.js:** Massive overkill. We'd spend weeks building camera controls, picking, labels, force simulation. `react-force-graph-3d` wraps Three.js already.

4. **Why NOT D3 directly:** D3 force simulation is what `react-force-graph` uses internally. Going raw D3 means manual React integration and SVG/Canvas management.

5. **Why NOT Cytoscape.js:** Better for complex graph analytics (shortest path, centrality). Overkill for visualization-only use case.

**Data format expected by react-force-graph:**
```js
{
  nodes: [
    { id: "Python", type: "technology", mentionCount: 5, val: 5 },
    { id: "FastAPI", type: "technology", mentionCount: 3, val: 3 },
    { id: "User", type: "person", mentionCount: 10, val: 10 }
  ],
  links: [
    { source: "User", target: "Python", relation: "uses", strength: 1.0 },
    { source: "Python", target: "FastAPI", relation: "built_with", strength: 0.9 }
  ]
}
```

This maps directly to our Entity/Relationship models. We need a **new API endpoint** that returns the full graph for a user (not just neighbors of one entity).

---

### Phase 3: New API Endpoints Needed

The current API is missing endpoints the frontend needs:

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/graph/full` | POST | Return ALL entities + relationships for a user (for full graph render) |
| `/memory/list` | POST | List all memories for a user with pagination, filtering |
| `/graph/entities` | POST | List all entities for a user (for entity picker dropdown) |
| `/stats` | POST | Return stats: memory count by tier/type, entity count, relationship count |

---

## Suggested Next Steps (In Order)

> **Steps 1-7 are COMPLETE.** All backend endpoints, frontend dashboard components, and integration are built and verified.

### ~~Step 1: New Backend Endpoints~~ DONE
Added `/graph/full`, `/memory/list`, `/graph/entities`, `/stats` endpoints with Pydantic request schemas. Tests pass (188/188).

### ~~Step 2: Frontend Dashboard Shell~~ DONE
Dashboard page at `/dashboard` with tabbed layout (Stats, Chat Input, Knowledge Graph, Memories, Lifecycle), user ID input, backend status indicator, refresh button.

### ~~Step 3: Chat Input Panel~~ DONE
Textarea with conversation parser (JSON or ROLE: content format), extraction results display showing facts, entities, relationships.

### ~~Step 4: Graph Visualization~~ DONE
`react-force-graph-2d` with 2D/3D toggle, click-to-expand neighbors, color-coded entity types, legend, responsive sizing via ResizeObserver.

### ~~Step 5: Memory Browser~~ DONE
List view with search, tier/type filters, pagination, click-to-expand detail view with defensive null guards.

### ~~Step 6: Lifecycle Controls~~ DONE
Run lifecycle button with results display (decayed/consolidated/cleaned counts).

### ~~Step 7: Stats Dashboard~~ DONE
Summary cards (total memories, entities, relationships) with breakdown by tier and type.

### Step 8: Polish & Deploy (REMAINING)
- Responsive design -- partially done (graph is responsive, basic media queries in CSS)
- Dark/light mode toggle -- dark theme is default, no toggle yet
- Error handling, loading states -- done in all components
- Deploy dashboard to GitHub Pages (static) pointing to a configurable backend URL
- Docker Compose for backend (FastAPI + PostgreSQL + pgvector)

---

## Technical Decisions Made

| Decision | Choice | Reason |
|----------|--------|--------|
| Database | PostgreSQL + pgvector (SQLite fallback for dev) | Vector search in-database, no separate vector DB |
| ORM | SQLAlchemy 2.0 async | Industry standard, async support |
| LLM | OpenAI API (+ OpenRouter) | Best extraction quality, easy to swap |
| Embeddings | text-embedding-3-small (1536d) | Cost-effective, good quality |
| API | FastAPI | Async, auto-docs, Pydantic integration |
| Frontend | React + Vite | Already set up in `website/` |
| Graph viz | react-force-graph-2d (+ 3d toggle) | Best React integration, Canvas perf, force-directed |
| Graph storage | PostgreSQL tables (not Neo4j) | No extra infrastructure, good enough for our scale |

---

## Known Issues / Gotchas

1. **SQLite dual-mode:** Models have `_USE_PG` flag. pgvector columns use JSON fallback on SQLite. Cosine similarity is computed in Python (not in-database) on SQLite.

2. **Pydantic-settings reads `.env`:** Tests that need to clear settings must pass `_env_file=None` to prevent `.env` file loading.

3. **LSP false positives:** The LSP reports errors for pgvector imports, SQLAlchemy column types, and OpenAI type hints. These are all runtime-correct -- the package is installed in editable mode.

4. **API key exposure:** The user's OpenAI API key was visible in `.env` during development. It should be rotated.

5. **`text()` collision:** SQLAlchemy's `text()` was shadowed by the Memory.text column. Fixed by importing as `sa_text`.

---

## File Structure

```
DeepContext/
  src/deepcontext/          # Python package (16 modules)
    core/                   # Settings, types, LLM clients
    db/                     # Database, ORM models
    vectorstore/            # Vector store abstraction
    extraction/             # LLM fact extraction
    graph/                  # Knowledge graph
    retrieval/              # Hybrid search
    lifecycle/              # Decay, consolidation, cleanup
    memory/                 # Main engine (orchestrator)
    api/                    # FastAPI REST API
  tests/                    # 188 tests across 10 files
  alembic/                  # Database migrations
  examples/                 # chat_demo.py
  website/                  # React + Vite landing page, docs, demo & dashboard
    src/
      api/                  # client.js - API client for backend
      pages/                # Home, Docs, Demo, Dashboard
      components/           # Navbar, Footer, CodeBlock, StatsCards, ChatPanel,
                            # GraphViz, MemoryBrowser, LifecycleControls
  .env                      # API keys (git-ignored)
  pyproject.toml            # Python project config
  alembic.ini               # Alembic config
  README.md                 # Project documentation
  LICENSE                   # MIT
  CLAUDE.md                 # This file
```

---

## Environment Setup

```bash
# Backend
cd DeepContext
python -m venv .venv
.venv\Scripts\activate       # Windows
pip install -e ".[dev]"
cp .env.example .env         # Add your DEEPCONTEXT_OPENAI_API_KEY

# Run tests
python -m pytest tests/ -v

# Start API server
uvicorn deepcontext.api.server:app --reload

# Frontend
cd website
npm install
npm run dev                  # Vite dev server on :5173
```

---

## Commands Cheat Sheet

```bash
# Run all tests
python -m pytest tests/ -v --tb=short

# Run specific test file
python -m pytest tests/test_graph.py -v

# Start FastAPI server
uvicorn deepcontext.api.server:app --reload --port 8000

# Start frontend dev server
cd website && npm run dev

# Build frontend for production
cd website && npm run build

# Run Alembic migration (requires PostgreSQL)
alembic upgrade head

# Check Alembic status
alembic current
```

---

## gstack

This project uses [gstack](https://github.com/garrytan/gstack.git) for browser automation, QA testing, code review, and shipping workflows.

**Important:** Always use the `/browse` skill from gstack for all web browsing. Never use `mcp__claude-in-chrome__*` tools.

### Available Skills

| Skill | Purpose |
|-------|---------|
| `/browse` | Headless browser for QA testing, site dogfooding, and web interaction |
| `/qa` | Systematic QA testing with diff-aware, full, quick, and regression modes |
| `/plan-ceo-review` | CEO/founder-mode plan review -- rethink the problem, challenge premises |
| `/plan-eng-review` | Eng manager-mode plan review -- architecture, data flow, edge cases |
| `/review` | Pre-landing PR review for SQL safety, trust boundaries, structural issues |
| `/ship` | Ship workflow: merge main, run tests, review diff, bump version, push, create PR |
| `/setup-browser-cookies` | Import cookies from your real browser into headless browse sessions |
| `/retro` | Weekly engineering retrospective with commit history and quality metrics |

### Troubleshooting

If gstack skills aren't working (e.g. browse binary not found, skills not registered), rebuild by running:

```bash
cd .claude/skills/gstack && ./setup
```

This builds the browse binary and regenerates skill docs.
