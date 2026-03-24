import { Link } from 'react-router-dom'
import { useEffect, useRef, useState } from 'react'
import CopyButton from '../components/CopyButton'

/* ─── Terminal typing animation ─────────────────────────────────────────── */
const TERMINAL_LINES = [
  '> Setting up index',
  '> Building context graph...',
  '> Resolving relations...',
  '> Embedding 1,024 facts',
  '> Hybrid retrieval ready',
]

function TerminalAnimation() {
  const [lines, setLines] = useState([])
  const [currentLine, setCurrentLine] = useState(0)
  const [currentChar, setCurrentChar] = useState(0)
  const [typing, setTyping] = useState(true)

  useEffect(() => {
    if (!typing) return
    if (currentLine >= TERMINAL_LINES.length) {
      // Restart after pause
      const timer = setTimeout(() => {
        setLines([])
        setCurrentLine(0)
        setCurrentChar(0)
      }, 3000)
      return () => clearTimeout(timer)
    }

    const line = TERMINAL_LINES[currentLine]
    if (currentChar < line.length) {
      const timer = setTimeout(() => {
        setCurrentChar(c => c + 1)
      }, 30 + Math.random() * 40)
      return () => clearTimeout(timer)
    } else {
      // Line complete, move to next
      const timer = setTimeout(() => {
        setLines(prev => [...prev, line])
        setCurrentLine(l => l + 1)
        setCurrentChar(0)
      }, 400)
      return () => clearTimeout(timer)
    }
  }, [currentLine, currentChar, typing])

  const partialLine = currentLine < TERMINAL_LINES.length
    ? TERMINAL_LINES[currentLine].slice(0, currentChar)
    : ''

  return (
    <div className="hero-terminal">
      <div className="terminal-label">///ContextEngine that works</div>
      {lines.map((line, i) => (
        <div key={i} className="terminal-line">{line}</div>
      ))}
      {currentLine < TERMINAL_LINES.length && (
        <div className="terminal-line">
          {partialLine}<span className="terminal-cursor">&nbsp;</span>
        </div>
      )}
    </div>
  )
}

/* ─── Scroll reveal hook ────────────────────────────────────────────────── */
function useScrollReveal() {
  const ref = useRef(null)
  useEffect(() => {
    const el = ref.current
    if (!el) return
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          el.classList.add('visible')
          observer.unobserve(el)
        }
      },
      { threshold: 0.1 }
    )
    observer.observe(el)
    return () => observer.disconnect()
  }, [])
  return ref
}

/* ─── Feature definitions ───────────────────────────────────────────────── */
const features = [
  {
    title: 'Recall Everything',
    desc: 'Assemble context from business data, chat sessions, documents. Remember user preferences while retrieving.',
    icon: (
      <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round">
        <ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"/>
        <path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/>
      </svg>
    ),
  },
  {
    title: 'Highest Recall Accuracy',
    desc: 'The highest accuracy context engine for your AI. Hybrid retrieval with vector + keyword + graph search fused via RRF.',
    icon: (
      <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round">
        <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/>
        <polyline points="22 4 12 14.01 9 11.01"/>
      </svg>
    ),
  },
  {
    title: 'Knowledge Graph',
    desc: 'Maps entities and relationships to provide structured reasoning alongside semantic retrieval. BFS traversal for context expansion.',
    icon: (
      <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round">
        <circle cx="12" cy="5" r="2"/><circle cx="5" cy="19" r="2"/><circle cx="19" cy="19" r="2"/>
        <line x1="12" y1="7" x2="5" y2="17"/><line x1="12" y1="7" x2="19" y2="17"/>
        <line x1="5" y1="19" x2="19" y2="19"/>
      </svg>
    ),
  },
  {
    title: 'Lifecycle Management',
    desc: 'Automated memory decay, consolidation, and cleanup. Ebbinghaus-inspired forgetting keeps context fresh and relevant.',
    icon: (
      <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round">
        <polyline points="23 4 23 10 17 10"/><polyline points="1 20 1 14 7 14"/>
        <path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"/>
      </svg>
    ),
  },
]

/* ─── Latency metric card (dark) ────────────────────────────────────────── */
const latencyCard = {
  label: 'LATENCY',
  value: '<200',
  unit: 'ms',
}

/* ─── Use cases ─────────────────────────────────────────────────────────── */
const useCases = [
  {
    number: '01',
    title: 'Build memory layers',
    desc: 'Memory is what separates general from personalized. Maintain long-term awareness across sessions, users, and workflows. Your agents retain relevant knowledge so every interaction builds on the last.',
  },
  {
    number: '02',
    title: 'Give persistent context',
    desc: 'Create structured context stores for your agents that evolve with every interaction. Ingest conversations, documents, and signals into a persistent context layer your AI can query instantly.',
  },
  {
    number: '03',
    title: 'Make agents stateful',
    desc: 'Turn stateless models into systems that understand history and progress. Agents track tasks, decisions, and evolving context to operate reliably in production.',
  },
]

/* ─── Logo marquee items ────────────────────────────────────────────────── */
const logos = ['OpenAI', 'Anthropic', 'LlamaIndex', 'LangChain', 'FastAPI', 'PostgreSQL', 'pgvector', 'SQLAlchemy']

/* ─── Reveal wrapper component ──────────────────────────────────────────── */
function Reveal({ children, className = '', stagger = false }) {
  const ref = useScrollReveal()
  return (
    <div ref={ref} className={`${stagger ? 'reveal-stagger' : 'reveal'} ${className}`}>
      {children}
    </div>
  )
}

/* ─── Component ─────────────────────────────────────────────────────────── */
export default function Home() {
  return (
    <>
      {/* ── Hero (two-column HydraDB style) ──────────────────────── */}
      <section className="hero">
        <div className="hero-left">
          <h1>
            Give your AI agents context &amp; memory they need.
          </h1>
          <div className="hero-description">
            <p>
              <strong>VectorDBs are flat.</strong> AI is stateless. DeepContext fixes that.
            </p>
            <p>
              DeepContext stores the full context, its relationships, decisions around it, and the timeline of how it has evolved.
            </p>
          </div>
        </div>

        <div className="hero-right">
          <TerminalAnimation />
          <div className="hero-cta">
            <Link to="/dashboard" className="btn btn-primary btn-lg">
              Get Started &rarr;
            </Link>
          </div>
        </div>
      </section>

      {/* ── Logo Marquee ─────────────────────────────────────────── */}
      <section className="logo-marquee-section">
        <div className="logo-marquee-label">Built with industry-standard tools</div>
        <div className="logo-marquee-wrapper">
          <div className="marquee-track">
            {[...logos, ...logos].map((logo, i) => (
              <span key={i} className="logo-marquee-item">{logo}</span>
            ))}
          </div>
        </div>
      </section>

      {/* ── Why DeepContext ──────────────────────────────────────── */}
      <section className="why-section">
        <Reveal className="container">
          <p className="section-label">///Why DeepContext</p>
          <h2>VectorDBs aren't enough.</h2>
          <p>
            <strong>VectorDBs are flat document indexes.</strong> They fail as soon as you need complex enterprise data. The project &ldquo;strawberry&rdquo; and the fruit &ldquo;strawberry&rdquo; are the same embedding, but completely different contexts. VectorDBs cannot tell the difference.
          </p>
          <p>
            Retrieval accuracy collapses at scale. They store no relationships, no decisions, no timeline &mdash; just vectors. <strong>Similarity is not relevance.</strong>
          </p>
          <p>
            DeepContext fixes this with hybrid retrieval: vector search + BM25 keyword matching + knowledge graph traversal, all fused through Reciprocal Rank Fusion. <Link to="/docs">Read the docs &rarr;</Link>
          </p>
        </Reveal>
      </section>

      {/* ── Features (HydraDB grid with + corners) ──────────────── */}
      <section className="features-section">
        <div className="container">
          <Reveal>
            <div className="features-header">
              <p className="section-label">///Features</p>
              <h2>Core Architecture</h2>
            </div>
            <p className="features-subtitle">
              End-to-end context engineering made easy. The most thoughtful way to make agents stateful with any form of data.
            </p>
          </Reveal>

          <div className="features-grid-wrapper corner-plus corner-plus-bottom">
            <Reveal stagger className="features-grid">
              {features.map((f, i) => (
                <div key={i} className="feature-card">
                  <div className="feature-card-icon">{f.icon}</div>
                  <h3>{f.title}</h3>
                  <p>{f.desc}</p>
                </div>
              ))}
              {/* Dark latency metric card */}
              <div className="feature-card feature-card-dark">
                <div className="feature-metric-label">{latencyCard.label}</div>
                <div className="feature-metric-value">
                  Always {latencyCard.value}<span> {latencyCard.unit}</span>
                </div>
              </div>
            </Reveal>
          </div>
        </div>
      </section>

      {/* ── Use Cases ────────────────────────────────────────────── */}
      <section className="usecases-section">
        <div className="container">
          <Reveal>
            <div className="usecases-header">
              <p className="section-label">///Use Cases</p>
              <h2>AI is only as good as the context it can access.</h2>
              <p>
                The context infrastructure built with hybrid retrieval: ultra low latency, highest precision recall, and relationally-aware.
              </p>
            </div>
          </Reveal>

          <Reveal stagger className="usecases-list">
            {useCases.map(uc => (
              <div key={uc.number} className="usecase-item">
                <span className="usecase-number">{uc.number}</span>
                <div className="usecase-content">
                  <h3>{uc.title}</h3>
                  <p>{uc.desc}</p>
                </div>
              </div>
            ))}
          </Reveal>
        </div>
      </section>

      {/* ── CTA ──────────────────────────────────────────────────── */}
      <section className="cta-section">
        <Reveal className="cta-inner">
          <h2>Deploy Persistent Memory in Minutes.</h2>
          <p>Scale your agents from stateless bots to persistent digital entities with a single API call.</p>

          <div style={{ display: 'flex', justifyContent: 'center', marginBottom: '1.5rem' }}>
            <div className="install-pill">
              <span className="install-prompt">$</span>
              <code style={{ background: 'none', padding: 0, color: 'var(--text-primary)' }}>pip install deepcontext</code>
              <CopyButton text="pip install deepcontext" />
            </div>
          </div>

          <div className="cta-actions">
            <Link to="/dashboard" className="btn btn-primary btn-lg">
              Open Dashboard &rarr;
            </Link>
            <Link to="/docs" className="btn btn-outline btn-lg">
              Read Docs
            </Link>
          </div>

          <div className="cta-logos">
            <span>OpenAI</span>
            <span>Anthropic</span>
            <span>LlamaIndex</span>
            <span>LangChain</span>
          </div>
        </Reveal>
      </section>
    </>
  )
}
