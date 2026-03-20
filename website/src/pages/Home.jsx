import { Link } from 'react-router-dom'
import CodeBlock from '../components/CodeBlock'
import CopyButton from '../components/CopyButton'

/* ─── Quickstart code ─────────────────────────────────────────────────────── */
const heroCode = `from deepcontext import DeepContext
import asyncio

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

    # Search memories
    results = await ctx.search(
        query="What does the user work on?",
        user_id="user_1",
    )

asyncio.run(main())`

/* ─── Feature definitions ─────────────────────────────────────────────────── */
const features = [
    {
        color: 'var(--accent-dim)',
        iconColor: 'var(--accent-hover)',
        glow: 'rgba(99,102,241,0.08)',
        title: 'Hybrid Retrieval',
        desc: 'Vector + keyword + graph search fused with Reciprocal Rank Fusion for the highest-quality results.',
        icon: (
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round">
                <circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/>
            </svg>
        ),
    },
    {
        color: 'var(--purple-dim)',
        iconColor: 'var(--purple)',
        glow: 'rgba(168,85,247,0.08)',
        title: 'Knowledge Graph',
        desc: 'Entities and relationships automatically linked and traversable via BFS — no separate graph DB needed.',
        icon: (
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round">
                <circle cx="12" cy="5" r="2"/><circle cx="5" cy="19" r="2"/><circle cx="19" cy="19" r="2"/>
                <line x1="12" y1="7" x2="5" y2="17"/><line x1="12" y1="7" x2="19" y2="17"/>
                <line x1="5" y1="19" x2="19" y2="19"/>
            </svg>
        ),
    },
    {
        color: 'var(--cyan-dim)',
        iconColor: 'var(--cyan)',
        glow: 'rgba(6,182,212,0.08)',
        title: 'Lifecycle Management',
        desc: 'Ebbinghaus-inspired decay, automatic consolidation of duplicates, and cleanup of low-value memories.',
        icon: (
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round">
                <polyline points="23 4 23 10 17 10"/><polyline points="1 20 1 14 7 14"/>
                <path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"/>
            </svg>
        ),
    },
    {
        color: 'var(--orange-dim)',
        iconColor: 'var(--orange)',
        glow: 'rgba(249,115,22,0.08)',
        title: 'Fact Extraction',
        desc: 'LLM-powered extraction of facts, entities, and relationships directly from raw conversation messages.',
        icon: (
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round">
                <polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/>
            </svg>
        ),
    },
    {
        color: 'var(--green-dim)',
        iconColor: 'var(--green)',
        glow: 'rgba(34,197,94,0.08)',
        title: 'Embedding Storage',
        desc: 'PostgreSQL + pgvector for in-database vector search. SQLite fallback for local development.',
        icon: (
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round">
                <ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"/>
                <path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/>
            </svg>
        ),
    },
    {
        color: 'var(--pink-dim)',
        iconColor: 'var(--pink)',
        glow: 'rgba(236,72,153,0.08)',
        title: 'Multi-User Isolation',
        desc: 'Complete data separation per user. Fully async from top to bottom — SQLAlchemy async, asyncpg, AsyncOpenAI.',
        icon: (
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round">
                <path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/>
                <circle cx="9" cy="7" r="4"/>
                <path d="M23 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/>
            </svg>
        ),
    },
]

/* ─── Architecture steps ──────────────────────────────────────────────────── */
const archSteps = [
    {
        label: 'Conversation',
        desc: 'Raw chat messages',
        icon: (
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round">
                <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
            </svg>
        ),
    },
    {
        label: 'Extraction',
        desc: 'LLM fact extraction',
        icon: (
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round">
                <polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/>
            </svg>
        ),
    },
    {
        label: 'Embedding',
        desc: 'Vector encoding',
        icon: (
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round">
                <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"/>
            </svg>
        ),
    },
    {
        label: 'Storage',
        desc: 'PostgreSQL + pgvector',
        icon: (
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round">
                <ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"/>
                <path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/>
            </svg>
        ),
    },
    {
        label: 'Retrieval',
        desc: 'Hybrid search + RRF',
        icon: (
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round">
                <circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/>
            </svg>
        ),
    },
]

/* ─── Arrow SVG ───────────────────────────────────────────────────────────── */
function ArrowRight() {
    return (
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <line x1="5" y1="12" x2="19" y2="12"/><polyline points="12 5 19 12 12 19"/>
        </svg>
    )
}

/* ─── Component ───────────────────────────────────────────────────────────── */
export default function Home() {
    return (
        <>
            {/* ── Hero ─────────────────────────────────────────────────────── */}
            <section className="hero" style={{ minHeight: '100vh', display: 'flex', alignItems: 'center', paddingTop: '5rem', paddingBottom: '5rem' }}>
                {/* Animated gradient mesh blobs */}
                <div className="hero-blob hero-blob-1" />
                <div className="hero-blob hero-blob-2" />
                <div className="hero-blob hero-blob-3" />

                <div className="container" style={{ position: 'relative', zIndex: 1 }}>
                    {/* Accent badge pill */}
                    <div style={{ display: 'flex', justifyContent: 'center', marginBottom: '1.75rem' }}>
                        <span className="hero-badge">
                            <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor">
                                <circle cx="12" cy="12" r="5"/>
                            </svg>
                            Now available on PyPI
                        </span>
                    </div>

                    {/* H1 with gradient text */}
                    <h1 style={{
                        fontSize: 'clamp(2.5rem, 6vw, 4rem)',
                        fontWeight: 800,
                        lineHeight: 1.12,
                        letterSpacing: '-0.04em',
                        marginBottom: '1.25rem',
                        textAlign: 'center',
                    }}>
                        Give Your AI Agents<br />
                        <span className="gradient-text">Context&nbsp;+&nbsp;Memory</span>
                    </h1>

                    {/* Subtitle */}
                    <p className="hero-subtitle" style={{ textAlign: 'center' }}>
                        Persistent, structured memory where conversations are automatically broken into
                        semantic facts, stored with embeddings, linked in a knowledge graph, and retrieved
                        via hybrid search.
                    </p>

                    {/* CTA buttons */}
                    <div className="hero-actions">
                        <Link to="/docs" className="btn btn-primary" style={{ padding: '0.75rem 1.75rem', fontSize: '0.9rem' }}>
                            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                <path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"/><path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"/>
                            </svg>
                            Read the Docs
                        </Link>
                        <Link to="/dashboard" className="btn btn-outline" style={{ padding: '0.75rem 1.75rem', fontSize: '0.9rem' }}>
                            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                <rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/>
                                <rect x="14" y="14" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/>
                            </svg>
                            Open Dashboard
                        </Link>
                    </div>

                    {/* Install pill */}
                    <div style={{ display: 'flex', justifyContent: 'center', marginBottom: '3rem' }}>
                        <div className="install-pill">
                            <span className="install-prompt">$</span>
                            <span>pip install deepcontext</span>
                            <CopyButton text="pip install deepcontext" />
                        </div>
                    </div>

                    {/* Quickstart code block in glassmorphic card */}
                    <div className="hero-code-card">
                        <div className="code-header">
                            <div className="code-dots">
                                <span className="code-dot code-dot-red" />
                                <span className="code-dot code-dot-yellow" />
                                <span className="code-dot code-dot-green" />
                            </div>
                            <span className="code-filename">quickstart.py</span>
                            <CopyButton text={heroCode} />
                        </div>
                        <CodeBlock code={heroCode} language="python" />
                    </div>
                </div>
            </section>

            {/* ── Features ─────────────────────────────────────────────────── */}
            <section className="features-section section-alt">
                <div className="container">
                    <h2>Built for Production AI</h2>
                    <p className="section-subtitle">
                        Everything your agent needs to remember, reason, and recall — out of the box.
                    </p>

                    {/* Bento grid — 6 cards, 3-column asymmetric */}
                    <div className="features-grid">
                        {features.map((f, i) => (
                            <div
                                key={i}
                                className="feature-card card-glass card-interactive"
                                style={{ '--feature-glow': f.glow }}
                            >
                                <div
                                    className="feature-card-icon"
                                    style={{ background: f.color, color: f.iconColor }}
                                >
                                    {f.icon}
                                </div>
                                <h3>{f.title}</h3>
                                <p>{f.desc}</p>
                            </div>
                        ))}
                    </div>
                </div>
            </section>

            {/* ── Architecture ─────────────────────────────────────────────── */}
            <section className="architecture-section section-full">
                <div className="container">
                    <h2>How It Works</h2>
                    <p className="section-subtitle" style={{ textAlign: 'center', color: 'var(--text-secondary)', fontSize: '1.05rem', marginBottom: '3.5rem' }}>
                        From conversation to stored memory — and back again on retrieval.
                    </p>

                    {/* Horizontal flow diagram */}
                    <div className="architecture-flow">
                        {archSteps.map((step, i) => (
                            <div key={i} style={{ display: 'flex', alignItems: 'flex-start' }}>
                                <div className="arch-step">
                                    <div className="arch-step-icon">
                                        {step.icon}
                                    </div>
                                    <div className="arch-step-label">{step.label}</div>
                                    <div className="arch-step-desc">{step.desc}</div>
                                </div>
                                {i < archSteps.length - 1 && (
                                    <div className="arch-arrow">
                                        <ArrowRight />
                                    </div>
                                )}
                            </div>
                        ))}
                    </div>
                </div>
            </section>

            {/* ── CTA ──────────────────────────────────────────────────────── */}
            <section className="cta-section">
                <div className="container" style={{ position: 'relative', zIndex: 1 }}>
                    <h2>
                        Start building with{' '}
                        <span className="gradient-text">DeepContext</span>
                    </h2>
                    <p>Open source. MIT licensed. Up and running in minutes.</p>

                    {/* Install pill */}
                    <div style={{ display: 'flex', justifyContent: 'center', marginBottom: '2rem' }}>
                        <div className="install-pill">
                            <span className="install-prompt">$</span>
                            <span>pip install deepcontext</span>
                            <CopyButton text="pip install deepcontext" />
                        </div>
                    </div>

                    {/* Two white outline buttons */}
                    <div className="cta-actions">
                        <Link to="/docs" className="btn btn-white-outline" style={{ padding: '0.75rem 1.75rem', fontSize: '0.9rem' }}>
                            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                <path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"/><path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"/>
                            </svg>
                            Read the Docs
                        </Link>
                        <a
                            href="https://github.com/umairinayat/DeepContext"
                            target="_blank"
                            rel="noopener noreferrer"
                            className="btn btn-white-outline"
                            style={{ padding: '0.75rem 1.75rem', fontSize: '0.9rem' }}
                        >
                            <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
                                <path d="M12 0C5.37 0 0 5.37 0 12c0 5.3 3.438 9.8 8.205 11.385.6.113.82-.258.82-.577 0-.285-.01-1.04-.015-2.04-3.338.724-4.042-1.61-4.042-1.61-.546-1.385-1.335-1.755-1.335-1.755-1.087-.744.084-.729.084-.729 1.205.084 1.838 1.236 1.838 1.236 1.07 1.835 2.809 1.305 3.495.998.108-.776.417-1.305.76-1.605-2.665-.3-5.466-1.332-5.466-5.93 0-1.31.465-2.38 1.235-3.22-.135-.303-.54-1.523.105-3.176 0 0 1.005-.322 3.3 1.23.96-.267 1.98-.399 3-.405 1.02.006 2.04.138 3 .405 2.28-1.552 3.285-1.23 3.285-1.23.645 1.653.24 2.873.12 3.176.765.84 1.23 1.91 1.23 3.22 0 4.61-2.805 5.625-5.475 5.92.42.36.81 1.096.81 2.22 0 1.606-.015 2.896-.015 3.286 0 .315.21.69.825.57C20.565 21.795 24 17.295 24 12c0-6.63-5.37-12-12-12"/>
                            </svg>
                            View on GitHub
                        </a>
                    </div>
                </div>
            </section>
        </>
    )
}
