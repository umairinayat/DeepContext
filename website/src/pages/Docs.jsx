import { useState, useEffect } from 'react'
import CodeBlock from '../components/CodeBlock'

const sections = [
    { id: 'installation', label: 'Installation' },
    { id: 'quickstart', label: 'Quick Start' },
    { id: 'usage', label: 'Basic Usage' },
    { id: 'memory-types', label: 'Memory Types' },
    { id: 'api', label: 'REST API' },
    { id: 'config', label: 'Configuration' },
    { id: 'pipeline', label: 'Memory Pipeline' },
    { id: 'retrieval', label: 'Hybrid Retrieval' },
    { id: 'lifecycle', label: 'Memory Lifecycle' },
    { id: 'tech', label: 'Tech Stack' },
]

export default function Docs() {
    const [active, setActive] = useState('installation')

    useEffect(() => {
        const observer = new IntersectionObserver(
            (entries) => {
                for (const entry of entries) {
                    if (entry.isIntersecting) {
                        setActive(entry.target.id)
                    }
                }
            },
            { rootMargin: '-100px 0px -60% 0px' }
        )
        sections.forEach(s => {
            const el = document.getElementById(s.id)
            if (el) observer.observe(el)
        })
        return () => observer.disconnect()
    }, [])

    return (
        <div className="container">
            <div className="docs-layout">
                <aside className="docs-sidebar">
                    <div className="sidebar-label">Documentation</div>
                    {sections.map(s => (
                        <a
                            key={s.id}
                            href={`#${s.id}`}
                            className={active === s.id ? 'active' : ''}
                            onClick={() => setActive(s.id)}
                        >
                            {s.label}
                        </a>
                    ))}
                </aside>

                <main className="docs-content">
                    <h1>DeepContext Documentation</h1>
                    <p>Hierarchical memory system for AI agents — async, graph-aware, with hybrid retrieval.</p>

                    {/* Installation */}
                    <h2 id="installation">Installation</h2>
                    <CodeBlock language="bash" header="terminal" code={`git clone https://github.com/umairinayat/DeepContext.git
cd DeepContext
python -m venv .venv

# Windows
.venv\\Scripts\\activate
# Linux/macOS
source .venv/bin/activate

pip install -e ".[all]"`} />

                    <h3>Environment Variables</h3>
                    <p>Create a <code>.env</code> file in the project root:</p>
                    <CodeBlock language="bash" header=".env" code={`DEEPCONTEXT_OPENAI_API_KEY=sk-your-key-here

# PostgreSQL (recommended for production)
# DEEPCONTEXT_DATABASE_URL=postgresql+asyncpg://user:pass@localhost/deepcontext

# SQLite fallback (default, no setup needed)
# Automatically uses ~/.deepcontext/memory.db`} />

                    {/* Quick Start */}
                    <h2 id="quickstart">Quick Start</h2>
                    <CodeBlock code={`import asyncio
from deepcontext import DeepContext

async def main():
    ctx = DeepContext(openai_api_key="sk-...")
    await ctx.init()

    # Store memories
    response = await ctx.add(
        messages=[
            {"role": "user", "content": "I'm a Python developer at Acme Corp"},
            {"role": "assistant", "content": "Nice to meet you!"},
        ],
        user_id="user_1",
        conversation_id="conv_1",
    )
    print(f"Stored {response.memories_added} memories")
    print(f"Entities found: {response.entities_found}")

    # Search
    results = await ctx.search("What does the user do?", user_id="user_1")
    for r in results.results:
        print(f"  [{r.tier.value}] {r.text} (score: {r.score:.3f})")

    await ctx.close()

asyncio.run(main())`} />

                    {/* Basic Usage */}
                    <h2 id="usage">Basic Usage</h2>

                    <h3>Add Memories</h3>
                    <p>Pass conversation messages to extract and store semantic facts, episodic events, entities, and relationships automatically.</p>
                    <CodeBlock code={`response = await ctx.add(
    messages=[
        {"role": "user", "content": "I prefer Python over JavaScript"},
        {"role": "assistant", "content": "Got it, Python is your go-to!"},
    ],
    user_id="user_1",
    conversation_id="conv_1",
)
# response.semantic_facts → ["User prefers Python over JavaScript"]
# response.entities_found → ["Python", "JavaScript"]`} />

                    <h3>Search Memories</h3>
                    <p>Hybrid retrieval fuses vector similarity, keyword matching, and knowledge graph traversal.</p>
                    <CodeBlock code={`results = await ctx.search(
    query="programming languages",
    user_id="user_1",
    limit=10,
    tier="short_term",        # optional filter
    memory_type="semantic",   # optional filter
    include_graph=True,       # enable graph expansion
)`} />

                    <h3>Update & Delete</h3>
                    <CodeBlock code={`# Update text and re-embed
await ctx.update(memory_id=1, text="User loves Python", user_id="user_1")

# Soft-delete (sets is_active=false)
await ctx.delete(memory_id=1, user_id="user_1")`} />

                    <h3>Knowledge Graph</h3>
                    <CodeBlock code={`neighbors = await ctx.get_entity_graph("user_1", "Python", depth=2)
for n in neighbors:
    print(f"  {n['entity']} --{n['relation']}--> (depth {n['depth']})")`} />

                    <h3>Memory Lifecycle</h3>
                    <CodeBlock code={`stats = await ctx.run_lifecycle("user_1")
# stats → {"memories_decayed": 5, "memories_consolidated": 2, "memories_cleaned": 1}`} />

                    {/* Memory Types */}
                    <h2 id="memory-types">Memory Types</h2>
                    <p>DeepContext organizes memories into types and tiers, inspired by human cognitive architecture.</p>

                    <h3>Types</h3>
                    <table>
                        <thead><tr><th>Type</th><th>Description</th><th>Example</th></tr></thead>
                        <tbody>
                            <tr><td><code>semantic</code></td><td>Factual knowledge</td><td>"User is a Python developer"</td></tr>
                            <tr><td><code>episodic</code></td><td>Events & experiences</td><td>"User asked about FastAPI today"</td></tr>
                            <tr><td><code>procedural</code></td><td>How-to knowledge</td><td>"User deploys with Docker"</td></tr>
                        </tbody>
                    </table>

                    <h3>Tiers</h3>
                    <table>
                        <thead><tr><th>Tier</th><th>Behavior</th></tr></thead>
                        <tbody>
                            <tr><td><code>working</code></td><td>Active conversation context</td></tr>
                            <tr><td><code>short_term</code></td><td>Recent facts, subject to decay</td></tr>
                            <tr><td><code>long_term</code></td><td>Consolidated, persistent knowledge</td></tr>
                        </tbody>
                    </table>

                    {/* REST API */}
                    <h2 id="api">REST API</h2>
                    <p>Start the server with:</p>
                    <CodeBlock language="bash" header="terminal" code={`uvicorn deepcontext.api.server:app --reload`} />

                    <h3>Endpoints</h3>
                    <table>
                        <thead><tr><th>Method</th><th>Path</th><th>Description</th></tr></thead>
                        <tbody>
                            <tr><td><code>GET</code></td><td><code>/health</code></td><td>Health check</td></tr>
                            <tr><td><code>POST</code></td><td><code>/memory/add</code></td><td>Extract & store memories</td></tr>
                            <tr><td><code>POST</code></td><td><code>/memory/search</code></td><td>Hybrid search</td></tr>
                            <tr><td><code>PUT</code></td><td><code>/memory/update</code></td><td>Update memory text</td></tr>
                            <tr><td><code>DELETE</code></td><td><code>/memory/delete</code></td><td>Soft-delete a memory</td></tr>
                            <tr><td><code>POST</code></td><td><code>/graph/neighbors</code></td><td>Graph neighborhood</td></tr>
                            <tr><td><code>POST</code></td><td><code>/lifecycle/run</code></td><td>Run lifecycle</td></tr>
                        </tbody>
                    </table>

                    <h3>Example: Add Memory</h3>
                    <CodeBlock language="bash" header="curl" code={`curl -X POST http://localhost:8000/memory/add \\
  -H "Content-Type: application/json" \\
  -d '{
    "messages": [
      {"role": "user", "content": "I prefer Python over JavaScript"}
    ],
    "user_id": "user_1",
    "conversation_id": "conv_1"
  }'`} />

                    <h3>Example: Search</h3>
                    <CodeBlock language="bash" header="curl" code={`curl -X POST http://localhost:8000/memory/search \\
  -H "Content-Type: application/json" \\
  -d '{
    "query": "programming languages",
    "user_id": "user_1",
    "limit": 5
  }'`} />

                    {/* Configuration */}
                    <h2 id="config">Configuration</h2>
                    <p>All settings use the <code>DEEPCONTEXT_</code> env prefix. Set them in <code>.env</code> or pass directly.</p>
                    <table>
                        <thead><tr><th>Setting</th><th>Default</th><th>Description</th></tr></thead>
                        <tbody>
                            <tr><td><code>database_url</code></td><td>SQLite fallback</td><td>PostgreSQL connection URL</td></tr>
                            <tr><td><code>llm_provider</code></td><td><code>openai</code></td><td><code>openai</code> or <code>openrouter</code></td></tr>
                            <tr><td><code>openai_api_key</code></td><td>—</td><td>Required for OpenAI</td></tr>
                            <tr><td><code>llm_model</code></td><td><code>gpt-4o-mini</code></td><td>LLM for extraction</td></tr>
                            <tr><td><code>embedding_model</code></td><td><code>text-embedding-3-small</code></td><td>Embedding model</td></tr>
                            <tr><td><code>embedding_dimensions</code></td><td><code>1536</code></td><td>Vector dimensions</td></tr>
                            <tr><td><code>consolidation_threshold</code></td><td><code>20</code></td><td>Memories before auto-consolidation</td></tr>
                            <tr><td><code>decay_half_life_days</code></td><td><code>7.0</code></td><td>Ebbinghaus half-life</td></tr>
                            <tr><td><code>auto_consolidate</code></td><td><code>true</code></td><td>Auto-consolidate on add</td></tr>
                            <tr><td><code>debug</code></td><td><code>false</code></td><td>Debug logging</td></tr>
                        </tbody>
                    </table>

                    {/* Pipeline */}
                    <h2 id="pipeline">Memory Pipeline</h2>
                    <p>When you call <code>ctx.add(messages, user_id)</code>:</p>
                    <ol style={{ color: 'var(--text-secondary)', marginBottom: '1rem', paddingLeft: '1.5rem' }}>
                        <li><strong>Extraction</strong> — LLM analyzes the conversation, extracts semantic facts, episodic events, entities, and relationships</li>
                        <li><strong>Classification</strong> — Each fact is compared to existing memories. LLM decides: ADD, UPDATE, REPLACE, or NOOP</li>
                        <li><strong>Embedding</strong> — New/updated facts are embedded using the configured model</li>
                        <li><strong>Storage</strong> — Memories stored with embeddings, tier, type, importance, and confidence</li>
                        <li><strong>Graph Update</strong> — Entities and relationships upserted into the knowledge graph</li>
                        <li><strong>Auto-consolidation</strong> — Triggered if short-term count exceeds threshold</li>
                    </ol>

                    {/* Retrieval */}
                    <h2 id="retrieval">Hybrid Retrieval</h2>
                    <p>When you call <code>ctx.search(query, user_id)</code>:</p>
                    <ol style={{ color: 'var(--text-secondary)', marginBottom: '1rem', paddingLeft: '1.5rem' }}>
                        <li><strong>Vector search</strong> — Query embedded and compared via cosine similarity</li>
                        <li><strong>Keyword search</strong> — PostgreSQL tsvector full-text (ILIKE fallback on SQLite)</li>
                        <li><strong>Graph expansion</strong> — Entities found in query → graph neighbors → boost related memories</li>
                        <li><strong>RRF fusion</strong> — Reciprocal Rank Fusion (vector 0.6, keyword 0.25, graph 0.15)</li>
                        <li><strong>Scoring</strong> — Final score applies importance, recency, confidence, access-count boost</li>
                    </ol>

                    {/* Lifecycle */}
                    <h2 id="lifecycle">Memory Lifecycle</h2>
                    <p>When you call <code>ctx.run_lifecycle(user_id)</code>:</p>
                    <ol style={{ color: 'var(--text-secondary)', marginBottom: '1rem', paddingLeft: '1.5rem' }}>
                        <li><strong>Decay</strong> — Ebbinghaus curve: <code>R = e^(-0.693 × days / half_life)</code>. Frequently accessed memories decay slower. Below 0.05 → deactivated</li>
                        <li><strong>Consolidation</strong> — Short-term memories grouped by entity overlap (Union-Find), merged by LLM into long-term facts</li>
                        <li><strong>Cleanup</strong> — Remaining low-importance non-long-term memories soft-deleted</li>
                    </ol>

                    {/* Tech Stack */}
                    <h2 id="tech">Tech Stack</h2>
                    <table>
                        <thead><tr><th>Component</th><th>Technology</th></tr></thead>
                        <tbody>
                            <tr><td>Language</td><td>Python 3.11+ with full type annotations</td></tr>
                            <tr><td>ORM</td><td>SQLAlchemy 2.0 (async)</td></tr>
                            <tr><td>Vector Store</td><td>pgvector (PostgreSQL)</td></tr>
                            <tr><td>LLM</td><td>OpenAI API / OpenRouter</td></tr>
                            <tr><td>API</td><td>FastAPI + Uvicorn</td></tr>
                            <tr><td>Validation</td><td>Pydantic v2 + pydantic-settings</td></tr>
                            <tr><td>Testing</td><td>pytest + pytest-asyncio + httpx</td></tr>
                            <tr><td>Migrations</td><td>Alembic</td></tr>
                        </tbody>
                    </table>

                </main>
            </div>
        </div>
    )
}
