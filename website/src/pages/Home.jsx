import { Link } from 'react-router-dom'
import CodeBlock from '../components/CodeBlock'

const heroCode = `from deepcontext import DeepContext

async def main():
    ctx = DeepContext(openai_api_key="sk-...")
    await ctx.init()

    # Store memories from a conversation
    await ctx.add(
        messages=[
            {"role": "user", "content": "I'm a Python dev at Acme Corp"},
            {"role": "assistant", "content": "Nice to meet you!"},
        ],
        user_id="user_1",
        conversation_id="conv_1",
    )

    # Search with hybrid retrieval (vector + keyword + graph)
    results = await ctx.search("What does the user do?", user_id="user_1")
    for r in results.results:
        print(f"[{r.tier.value}] {r.text} (score: {r.score:.3f})")

    await ctx.close()`

const features = [
    {
        icon: '🧠',
        color: 'var(--accent-dim)',
        title: 'Hierarchical Memory',
        desc: 'Working, short-term, and long-term tiers inspired by human cognition. Memories evolve and consolidate over time.',
    },
    {
        icon: '🔍',
        color: 'var(--cyan-dim)',
        title: 'Hybrid Retrieval',
        desc: 'Reciprocal Rank Fusion across vector similarity, keyword search, and knowledge graph traversal.',
    },
    {
        icon: '🕸️',
        color: 'var(--purple-dim)',
        title: 'Knowledge Graph',
        desc: 'Entities and relationships auto-extracted from conversations. No separate graph database needed.',
    },
    {
        icon: '♻️',
        color: 'var(--green-dim)',
        title: 'Memory Lifecycle',
        desc: 'Ebbinghaus forgetting curve decay, automatic consolidation, and cleanup of low-value memories.',
    },
    {
        icon: '⚡',
        color: 'var(--orange-dim)',
        title: 'Fully Async',
        desc: 'Built on SQLAlchemy async, asyncpg, and AsyncOpenAI. Non-blocking from top to bottom.',
    },
    {
        icon: '🔌',
        color: 'var(--pink-dim)',
        title: 'REST API + SDK',
        desc: 'FastAPI server with 7 endpoints. Python SDK for direct integration. OpenAI or OpenRouter.',
    },
]

const archDiagram = `Conversation ──> LLM Extraction ──> Classification ──> Embedding ──> Storage
                      |                   |                             |
                      v                   v                             v
                 Facts, Entities    ADD / UPDATE /              Knowledge Graph
                 Relationships      REPLACE / NOOP                  Update

Query ──> Embed ──> Vector Search (0.6) ──┐
  |                                        ├──> RRF Fusion ──> Results
  ├─────> Keyword Search (0.25) ──────────┤
  |                                        |
  └─────> Graph Expansion (0.15) ─────────┘`

export default function Home() {
    return (
        <>
            {/* Hero */}
            <section className="hero">
                <div className="container">
                    <h1>
                        Give Your AI Agents<br />
                        <span className="gradient">Context + Memory</span>
                    </h1>
                    <p>
                        Persistent memory that grows, connects, and evolves with every conversation.
                        Graph-aware hybrid retrieval with lifecycle management.
                    </p>
                    <div className="hero-buttons">
                        <Link to="/docs" className="btn btn-primary">📖 Read the Docs</Link>
                        <Link to="/demo" className="btn btn-outline">🎯 Live Demo</Link>
                    </div>
                    <CodeBlock code={heroCode} language="python" header="quickstart.py" />
                </div>
            </section>

            {/* Features */}
            <section className="features-section">
                <div className="container">
                    <h2>Built for Production AI</h2>
                    <p className="subtitle">
                        Everything your agent needs to remember, reason, and recall — out of the box.
                    </p>
                    <div className="features-grid">
                        {features.map((f, i) => (
                            <div className="feature-card" key={i}>
                                <div className="feature-icon" style={{ background: f.color }}>
                                    {f.icon}
                                </div>
                                <h3>{f.title}</h3>
                                <p>{f.desc}</p>
                            </div>
                        ))}
                    </div>
                </div>
            </section>

            {/* Architecture */}
            <section className="arch-section">
                <div className="container">
                    <h2>How It Works</h2>
                    <p className="subtitle">
                        From conversation to stored memory — and back again on retrieval.
                    </p>
                    <div className="arch-diagram">
                        <CodeBlock code={archDiagram} language="text" header="Architecture" />
                    </div>
                </div>
            </section>

            {/* Install */}
            <section className="install-banner">
                <div className="container">
                    <h2>Get Started in Seconds</h2>
                    <div className="install-cmd">
                        <span className="dollar">$</span>
                        <code>pip install deepcontext</code>
                    </div>
                </div>
            </section>
        </>
    )
}
