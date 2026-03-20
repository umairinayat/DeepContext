import { useState } from 'react'
import CodeBlock from '../components/CodeBlock'

const TABS = ['add', 'search', 'graph', 'lifecycle']
const TAB_LABELS = {
    add: 'Add Memory',
    search: 'Search',
    graph: 'Graph',
    lifecycle: 'Lifecycle',
}

function DemoTab({ active, children }) {
    return active ? <div>{children}</div> : null
}

export default function Demo() {
    const [tab, setTab] = useState('add')
    const [output, setOutput] = useState('')
    const [loading, setLoading] = useState(false)

    const fakeDelay = () => new Promise(r => setTimeout(r, 800))

    async function handleAdd() {
        setLoading(true)
        await fakeDelay()
        setOutput(JSON.stringify({
            semantic_facts: [
                "User is a Python developer",
                "User works at Acme Corp",
                "User loves FastAPI and pytest"
            ],
            memories_added: 3,
            entities_found: ["Python", "Acme Corp", "FastAPI", "pytest"]
        }, null, 2))
        setLoading(false)
    }

    async function handleSearch() {
        setLoading(true)
        await fakeDelay()
        setOutput(JSON.stringify({
            query: "programming languages",
            user_id: "demo_user",
            total: 2,
            results: [
                {
                    memory_id: 1,
                    text: "User is a Python developer",
                    memory_type: "semantic",
                    tier: "short_term",
                    score: 0.912,
                    created_at: new Date().toISOString()
                },
                {
                    memory_id: 3,
                    text: "User loves FastAPI and pytest",
                    memory_type: "semantic",
                    tier: "short_term",
                    score: 0.847,
                    created_at: new Date().toISOString()
                }
            ]
        }, null, 2))
        setLoading(false)
    }

    async function handleGraph() {
        setLoading(true)
        await fakeDelay()
        setOutput(JSON.stringify([
            { entity: "Python", entity_type: "technology", relation: "uses", depth: 0, strength: 1.0 },
            { entity: "FastAPI", entity_type: "technology", relation: "built_with", depth: 1, strength: 0.9 },
            { entity: "pytest", entity_type: "technology", relation: "tested_with", depth: 1, strength: 0.85 },
            { entity: "Acme Corp", entity_type: "organization", relation: "works_at", depth: 1, strength: 0.8 },
        ], null, 2))
        setLoading(false)
    }

    async function handleLifecycle() {
        setLoading(true)
        await fakeDelay()
        setOutput(JSON.stringify({
            memories_decayed: 3,
            memories_consolidated: 1,
            memories_cleaned: 0,
            details: {
                decay: "Applied Ebbinghaus forgetting curve to 3 short-term memories",
                consolidation: "Merged 2 Python-related facts into 1 long-term memory",
                cleanup: "No memories below importance threshold"
            }
        }, null, 2))
        setLoading(false)
    }

    const handlers = { add: handleAdd, search: handleSearch, graph: handleGraph, lifecycle: handleLifecycle }

    return (
        <div className="demo-page">
            <div className="container">
                {/* Page header */}
                <div style={{ textAlign: 'center', marginBottom: '2.5rem' }}>
                    <h1>Interactive Demo</h1>
                    <p style={{ color: 'var(--text-secondary)', maxWidth: '560px', margin: '0.75rem auto 0' }}>
                        Try DeepContext's core operations — simulated responses show what the real API returns.
                    </p>
                </div>

                {/* Tab bar */}
                <div className="demo-tabs">
                    {TABS.map(t => (
                        <button
                            key={t}
                            className={`demo-tab btn btn-pill${tab === t ? ' active' : ''}`}
                            onClick={() => { setTab(t); setOutput('') }}
                        >
                            {TAB_LABELS[t]}
                        </button>
                    ))}
                </div>

                {/* Two-column content */}
                <div className="demo-content">
                    {/* Left: Input card */}
                    <div className="demo-input-card card">
                        <DemoTab active={tab === 'add'}>
                            <h3 style={{ marginBottom: '0.75rem' }}>Add Memory</h3>
                            <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: '1rem' }}>
                                Extract and store memories from a conversation turn.
                            </p>
                            <CodeBlock header={false} language="json" code={`{
  "messages": [
    {"role": "user", "content": "I'm a Python developer at Acme Corp. I love FastAPI and pytest."},
    {"role": "assistant", "content": "Great stack!"}
  ],
  "user_id": "demo_user",
  "conversation_id": "conv_1"
}`} />
                        </DemoTab>

                        <DemoTab active={tab === 'search'}>
                            <h3 style={{ marginBottom: '0.75rem' }}>Search Memories</h3>
                            <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: '1rem' }}>
                                Hybrid retrieval: vector + keyword + graph fusion.
                            </p>
                            <CodeBlock header={false} language="json" code={`{
  "query": "programming languages",
  "user_id": "demo_user",
  "limit": 10,
  "include_graph": true
}`} />
                        </DemoTab>

                        <DemoTab active={tab === 'graph'}>
                            <h3 style={{ marginBottom: '0.75rem' }}>Knowledge Graph</h3>
                            <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: '1rem' }}>
                                Explore entity relationships via BFS traversal.
                            </p>
                            <CodeBlock header={false} language="json" code={`{
  "user_id": "demo_user",
  "entity_name": "Python",
  "depth": 2
}`} />
                        </DemoTab>

                        <DemoTab active={tab === 'lifecycle'}>
                            <h3 style={{ marginBottom: '0.75rem' }}>Run Lifecycle</h3>
                            <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: '1rem' }}>
                                Decay → Consolidation → Cleanup pipeline.
                            </p>
                            <CodeBlock header={false} language="json" code={`{
  "user_id": "demo_user"
}`} />
                        </DemoTab>

                        <button
                            className="btn btn-primary btn-pill"
                            style={{ marginTop: '1rem', width: '100%', justifyContent: 'center' }}
                            onClick={handlers[tab]}
                            disabled={loading}
                        >
                            {loading ? 'Processing...' : 'Run'}
                        </button>
                    </div>

                    {/* Right: Output card */}
                    <div className="demo-output-card card">
                        <h3 style={{ marginBottom: '0.75rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                            <span style={{ color: 'var(--green)' }}>●</span> Response
                        </h3>
                        {output ? (
                            <CodeBlock header={false} language="json" code={output} />
                        ) : (
                            <div style={{
                                textAlign: 'center',
                                padding: '3rem',
                                color: 'var(--text-muted)',
                                fontSize: '0.9rem',
                            }}>
                                <div style={{ fontSize: '2rem', marginBottom: '0.5rem' }}>▶</div>
                                Click "Run" to see the response
                            </div>
                        )}
                    </div>
                </div>

                <p style={{ textAlign: 'center', color: 'var(--text-muted)', fontSize: '0.8rem', marginTop: '2rem' }}>
                    This is a client-side simulation. To use with real data, run the server locally:<br />
                    <code style={{ color: 'var(--accent-hover)' }}>uvicorn deepcontext.api.server:app --reload</code>
                </p>
            </div>
        </div>
    )
}
