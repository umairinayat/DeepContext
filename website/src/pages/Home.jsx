import { useState } from "react";
import { Link } from "react-router-dom";
import CodeBlock from "../components/CodeBlock";

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

    # Hybrid search: vector + keyword + knowledge graph
    results = await ctx.search("What does the user do?", user_id="user_1")
    for r in results.results:
        print(f"[{r.tier.value}] {r.text}  (score: {r.score:.3f})")

    await ctx.close()`;

const stats = [
  { value: "3", suffix: "", label: "Memory Tiers" },
  { value: "188", suffix: "+", label: "Tests Passing" },
  { value: "7", suffix: "", label: "API Endpoints" },
  { value: "100", suffix: "%", label: "Fully Async" },
];

const features = [
  {
    icon: "🧠",
    colorClass: "feat-indigo",
    title: "Hierarchical Memory",
    desc: "Working, short-term, and long-term tiers inspired by human cognition. Memories evolve and consolidate automatically.",
  },
  {
    icon: "🔍",
    colorClass: "feat-cyan",
    title: "Hybrid Retrieval",
    desc: "Reciprocal Rank Fusion across vector similarity, BM25 keyword search, and knowledge-graph traversal in one query.",
  },
  {
    icon: "🕸️",
    colorClass: "feat-purple",
    title: "Knowledge Graph",
    desc: "Entities and relationships auto-extracted from every conversation. No separate graph database required.",
  },
  {
    icon: "♻️",
    colorClass: "feat-green",
    title: "Memory Lifecycle",
    desc: "Ebbinghaus forgetting-curve decay, automatic consolidation, and cleanup of low-value memories over time.",
  },
  {
    icon: "⚡",
    colorClass: "feat-orange",
    title: "Fully Async",
    desc: "Built on SQLAlchemy 2.0 async, asyncpg, and AsyncOpenAI — completely non-blocking from top to bottom.",
  },
  {
    icon: "🔌",
    colorClass: "feat-pink",
    title: "REST API + SDK",
    desc: "FastAPI server with 7 endpoints and a Python SDK for direct integration. Works with OpenAI or OpenRouter.",
  },
];

const steps = [
  {
    number: "01",
    icon: "💬",
    colorClass: "step-indigo",
    title: "Conversation Input",
    desc: "Pass raw conversation messages — user and assistant turns — directly to DeepContext.",
  },
  {
    number: "02",
    icon: "🤖",
    colorClass: "step-purple",
    title: "LLM Extraction",
    desc: "An LLM extracts semantic facts, episodic events, entities, and relationships from each exchange.",
  },
  {
    number: "03",
    icon: "💾",
    colorClass: "step-cyan",
    title: "Embed & Store",
    desc: "Facts are embedded with text-embedding-3-small and stored with tier, type, importance, and confidence.",
  },
  {
    number: "04",
    icon: "🔍",
    colorClass: "step-green",
    title: "Hybrid Retrieval",
    desc: "Queries fuse vector (60 %), keyword (25 %), and graph (15 %) results via Reciprocal Rank Fusion.",
  },
];

function GitHubIcon() {
  return (
    <svg
      width="16"
      height="16"
      fill="currentColor"
      viewBox="0 0 24 24"
      aria-hidden="true"
    >
      <path d="M12 0C5.37 0 0 5.37 0 12c0 5.3 3.438 9.8 8.205 11.385.6.113.82-.258.82-.577 0-.285-.01-1.04-.015-2.04-3.338.724-4.042-1.61-4.042-1.61-.546-1.387-1.333-1.756-1.333-1.756-1.09-.744.083-.729.083-.729 1.205.084 1.838 1.236 1.838 1.236 1.07 1.835 2.809 1.305 3.495.998.108-.776.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.31.468-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.3 1.23A11.509 11.509 0 0 1 12 5.803c1.02.005 2.047.138 3.006.404 2.29-1.552 3.297-1.23 3.297-1.23.653 1.652.242 2.873.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.929.43.372.823 1.102.823 2.222 0 1.606-.015 2.898-.015 3.293 0 .322.218.694.825.576C20.565 21.796 24 17.298 24 12c0-6.63-5.37-12-12-12z" />
    </svg>
  );
}

function DocsIcon() {
  return (
    <svg
      width="16"
      height="16"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      viewBox="0 0 24 24"
      aria-hidden="true"
    >
      <path d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
    </svg>
  );
}

function PlayIcon() {
  return (
    <svg
      width="16"
      height="16"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      viewBox="0 0 24 24"
      aria-hidden="true"
    >
      <polygon points="5 3 19 12 5 21 5 3" />
    </svg>
  );
}

function ArrowRightIcon() {
  return (
    <svg
      width="14"
      height="14"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      viewBox="0 0 24 24"
      aria-hidden="true"
    >
      <line x1="5" y1="12" x2="19" y2="12" />
      <polyline points="12 5 19 12 12 19" />
    </svg>
  );
}

export default function Home() {
  const [copied, setCopied] = useState(false);

  function handleCopy() {
    navigator.clipboard.writeText("pip install deepcontext");
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  }

  return (
    <>
      {/* ── Hero ─────────────────────────────────────────────── */}
      <section className="hero">
        {/* animated background */}
        <div className="hero-bg" aria-hidden="true">
          <div className="hero-orb hero-orb-1" />
          <div className="hero-orb hero-orb-2" />
          <div className="hero-orb hero-orb-3" />
          <div className="hero-grid" />
        </div>

        <div className="container hero-inner">
          {/* pill badge */}
          <div className="hero-badge">
            <span className="hero-badge-dot" />
            Open Source · MIT License · Python 3.11+
          </div>

          <h1>
            Give Your AI Agents
            <br />
            <span className="gradient">Context + Memory</span>
          </h1>

          <p className="hero-sub">
            Persistent memory that grows, connects, and evolves with every
            conversation. Graph-aware hybrid retrieval with lifecycle
            management.
          </p>

          <div className="hero-buttons">
            <Link to="/docs" className="btn btn-primary">
              <DocsIcon /> Read the Docs
            </Link>
            <Link to="/demo" className="btn btn-outline">
              <PlayIcon /> Live Demo
            </Link>
            <a
              href="https://github.com/umairinayat/DeepContext"
              target="_blank"
              rel="noreferrer"
              className="btn btn-ghost"
            >
              <GitHubIcon /> GitHub
            </a>
          </div>

          <div className="hero-code">
            <CodeBlock
              code={heroCode}
              language="python"
              header="quickstart.py"
            />
          </div>
        </div>
      </section>

      {/* ── Stats bar ────────────────────────────────────────── */}
      <section className="stats-section">
        <div className="container">
          <div className="stats-bar">
            {stats.map((s, i) => (
              <div className="stat-item" key={i}>
                <div className="stat-value">
                  {s.value}
                  <span className="stat-suffix">{s.suffix}</span>
                </div>
                <div className="stat-label">{s.label}</div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── Features ─────────────────────────────────────────── */}
      <section className="features-section">
        <div className="container">
          <div className="section-label">Features</div>
          <h2>Built for Production AI</h2>
          <p className="subtitle">
            Everything your agent needs to remember, reason, and recall — out of
            the box.
          </p>
          <div className="features-grid">
            {features.map((f, i) => (
              <div className={`feature-card ${f.colorClass}`} key={i}>
                <div className="feature-icon">{f.icon}</div>
                <h3>{f.title}</h3>
                <p>{f.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── How It Works ─────────────────────────────────────── */}
      <section className="how-section">
        <div className="container">
          <div className="section-label">Pipeline</div>
          <h2>How It Works</h2>
          <p className="subtitle">
            From raw conversation to stored, searchable memory in four steps.
          </p>
          <div className="steps-grid">
            {steps.map((step, i) => (
              <div className={`step-card ${step.colorClass}`} key={i}>
                <div className="step-number">{step.number}</div>
                <div className="step-icon-wrap">
                  <span role="img" aria-label={step.title}>
                    {step.icon}
                  </span>
                </div>
                <h3>{step.title}</h3>
                <p>{step.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── Retrieval callout ─────────────────────────────────── */}
      <section className="retrieval-section">
        <div className="container">
          <div className="retrieval-grid">
            <div className="retrieval-text">
              <div className="section-label">Retrieval</div>
              <h2>Hybrid Search That Actually Works</h2>
              <p>
                No single search strategy is perfect. DeepContext combines three
                complementary signals and merges them with Reciprocal Rank
                Fusion so the best memory always floats to the top.
              </p>
              <ul className="retrieval-list">
                <li>
                  <span className="retrieval-dot dot-indigo" />
                  <div>
                    <strong>Vector search (60 %)</strong>
                    <span>
                      Cosine similarity via pgvector — finds semantically
                      related memories even when words differ.
                    </span>
                  </div>
                </li>
                <li>
                  <span className="retrieval-dot dot-cyan" />
                  <div>
                    <strong>Keyword search (25 %)</strong>
                    <span>
                      PostgreSQL full-text / ILIKE fallback — catches exact
                      terms and acronyms.
                    </span>
                  </div>
                </li>
                <li>
                  <span className="retrieval-dot dot-purple" />
                  <div>
                    <strong>Graph expansion (15 %)</strong>
                    <span>
                      BFS on the knowledge graph — surfaces related entities you
                      didn't ask for directly.
                    </span>
                  </div>
                </li>
              </ul>
              <Link
                to="/docs#retrieval"
                className="btn btn-outline btn-sm"
                style={{ marginTop: "1.5rem", display: "inline-flex" }}
              >
                Learn more <ArrowRightIcon />
              </Link>
            </div>
            <div className="retrieval-visual">
              <div className="rrf-card">
                <div className="rrf-header">
                  <span className="rrf-dot" />
                  <span className="rrf-dot" />
                  <span className="rrf-dot" />
                  <span
                    style={{
                      marginLeft: "0.5rem",
                      fontSize: "0.75rem",
                      color: "var(--text-muted)",
                    }}
                  >
                    RRF Fusion
                  </span>
                </div>
                <div className="rrf-body">
                  <div className="rrf-row rrf-row-indigo">
                    <span className="rrf-label">Vector</span>
                    <div className="rrf-bar-wrap">
                      <div className="rrf-bar" style={{ width: "60%" }} />
                    </div>
                    <span className="rrf-pct">60%</span>
                  </div>
                  <div className="rrf-row rrf-row-cyan">
                    <span className="rrf-label">Keyword</span>
                    <div className="rrf-bar-wrap">
                      <div className="rrf-bar" style={{ width: "25%" }} />
                    </div>
                    <span className="rrf-pct">25%</span>
                  </div>
                  <div className="rrf-row rrf-row-purple">
                    <span className="rrf-label">Graph</span>
                    <div className="rrf-bar-wrap">
                      <div className="rrf-bar" style={{ width: "15%" }} />
                    </div>
                    <span className="rrf-pct">15%</span>
                  </div>
                  <div className="rrf-divider" />
                  <div className="rrf-result">
                    <span className="rrf-result-label">Final Score</span>
                    <div className="rrf-result-bar">
                      <div className="rrf-result-fill" />
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* ── Install / CTA ─────────────────────────────────────── */}
      <section className="install-banner">
        <div className="container">
          <div className="install-inner">
            <div className="section-label">Get Started</div>
            <h2>Ready in Seconds</h2>
            <p className="subtitle">
              Install via pip — SQLite out of the box, PostgreSQL when you
              scale.
            </p>

            <button
              className={`install-cmd${copied ? " install-cmd-copied" : ""}`}
              onClick={handleCopy}
              title="Click to copy"
            >
              <span className="dollar">$</span>
              <code>pip install deepcontext</code>
              <span className="copy-badge">
                {copied ? (
                  <>
                    <svg
                      width="11"
                      height="11"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="2.5"
                      viewBox="0 0 24 24"
                    >
                      <polyline points="20 6 9 17 4 12" />
                    </svg>
                    Copied!
                  </>
                ) : (
                  <>
                    <svg
                      width="11"
                      height="11"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="2"
                      viewBox="0 0 24 24"
                    >
                      <rect x="9" y="9" width="13" height="13" rx="2" />
                      <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
                    </svg>
                    Copy
                  </>
                )}
              </span>
            </button>

            <div className="install-actions">
              <Link to="/docs" className="btn btn-primary">
                <DocsIcon /> Read the Docs
              </Link>
              <Link to="/dashboard" className="btn btn-outline">
                Open Dashboard <ArrowRightIcon />
              </Link>
              <a
                href="https://github.com/umairinayat/DeepContext"
                target="_blank"
                rel="noreferrer"
                className="btn btn-ghost"
              >
                <GitHubIcon /> Star on GitHub
              </a>
            </div>
          </div>
        </div>
      </section>
    </>
  );
}
