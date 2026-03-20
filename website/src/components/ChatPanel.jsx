import { useState } from 'react'
import { addMemory } from '../api/client'
import ErrorBanner from './ErrorBanner'

const PLACEHOLDER = `Paste a conversation here, e.g.:

USER: I've been working with Python and FastAPI lately.
ASSISTANT: That's great! FastAPI is excellent for async APIs.
USER: Yeah, I prefer it over Flask for new projects.`

const ENTITY_TYPE_COLORS = {
  person: 'var(--pink)',
  organization: 'var(--orange)',
  technology: 'var(--accent)',
  concept: 'var(--cyan)',
  location: 'var(--green)',
  event: 'var(--purple)',
  preference: 'var(--orange)',
  other: 'var(--text-muted)',
}

function entityColor(type) {
  return ENTITY_TYPE_COLORS[type] || 'var(--accent)'
}

export default function ChatPanel({ userId }) {
  const [text, setText] = useState('')
  const [conversationId, setConversationId] = useState('default')
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  function parseMessages(raw) {
    // Try JSON array first
    try {
      const parsed = JSON.parse(raw)
      if (Array.isArray(parsed)) return parsed
    } catch { /* not JSON */ }

    // Parse "ROLE: content" lines
    const lines = raw.split('\n').filter(l => l.trim())
    const messages = []
    let current = null

    for (const line of lines) {
      const match = line.match(/^(USER|ASSISTANT|SYSTEM|user|assistant|system)\s*:\s*(.+)/i)
      if (match) {
        if (current) messages.push(current)
        current = { role: match[1].toLowerCase(), content: match[2].trim() }
      } else if (current) {
        current.content += '\n' + line.trim()
      }
    }
    if (current) messages.push(current)

    // If nothing parsed, treat entire text as a single user message
    if (messages.length === 0 && raw.trim()) {
      messages.push({ role: 'user', content: raw.trim() })
    }

    return messages
  }

  async function handleExtract() {
    if (!text.trim() || !userId) return
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      const messages = parseMessages(text)
      if (messages.length === 0) {
        setError('Could not parse any messages from the input.')
        return
      }
      const res = await addMemory(messages, userId, conversationId)
      setResult(res)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="chat-panel">
      <div className="chat-input-section">
        <div className="chat-input-row">
          <label className="input-label">Conversation ID</label>
          <input
            type="text"
            className="input-field"
            value={conversationId}
            onChange={e => setConversationId(e.target.value)}
            placeholder="default"
          />
        </div>

        <textarea
          className="chat-textarea mono"
          value={text}
          onChange={e => setText(e.target.value)}
          placeholder={PLACEHOLDER}
          rows={10}
        />

        <button
          className="btn btn-primary btn-pill"
          onClick={handleExtract}
          disabled={loading || !text.trim() || !userId}
        >
          {loading ? 'Extracting...' : 'Extract Memories'}
        </button>
      </div>

      {error && <ErrorBanner message={error} onRetry={() => setError(null)} />}

      {result && (
        <div className="chat-results">
          <div className="card">
            <div className="chat-result-header">
              <h4>Extraction Result</h4>
              <div className="chat-result-badges">
                <span className="badge badge-green">+{result.memories_added} added</span>
                {result.memories_updated > 0 && (
                  <span className="badge badge-cyan">{result.memories_updated} updated</span>
                )}
                {result.memories_replaced > 0 && (
                  <span className="badge badge-orange">{result.memories_replaced} replaced</span>
                )}
              </div>
            </div>

            {result.semantic_facts?.length > 0 && (
              <div className="chat-result-section">
                <h5 className="result-section-title">Semantic Facts</h5>
                <ul className="result-list">
                  {result.semantic_facts.map((f, i) => <li key={i}>{f}</li>)}
                </ul>
              </div>
            )}

            {result.episodic_facts?.length > 0 && (
              <div className="chat-result-section">
                <h5 className="result-section-title">Episodic Facts</h5>
                <ul className="result-list">
                  {result.episodic_facts.map((f, i) => <li key={i}>{f}</li>)}
                </ul>
              </div>
            )}

            {result.entities_found?.length > 0 && (
              <div className="chat-result-section">
                <h5 className="result-section-title">Entities Found</h5>
                <div className="entity-chips">
                  {result.entities_found.map((e, i) => {
                    const type = typeof e === 'object' ? e.type : 'other'
                    const name = typeof e === 'object' ? e.name : e
                    return (
                      <span
                        key={i}
                        className="badge"
                        style={{ background: entityColor(type), color: '#fff' }}
                      >
                        {name}
                      </span>
                    )
                  })}
                </div>
              </div>
            )}

            {result.relationships_found > 0 && (
              <div className="chat-result-section">
                <h5 className="result-section-title">Relationships</h5>
                {Array.isArray(result.relationships_found) ? (
                  result.relationships_found.map((rel, i) => (
                    <div key={i} className="relationship-item">
                      <span className="relationship-source">{rel.source}</span>
                      <span className="relationship-arrow">
                        {rel.relation}
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                          <line x1="5" y1="12" x2="19" y2="12" /><polyline points="12 5 19 12 12 19" />
                        </svg>
                      </span>
                      <span className="relationship-target">{rel.target}</span>
                      {rel.strength !== undefined && (
                        <span className="relationship-strength">{rel.strength.toFixed(2)}</span>
                      )}
                    </div>
                  ))
                ) : (
                  <span className="text-muted">{result.relationships_found} relationship{result.relationships_found !== 1 ? 's' : ''} stored</span>
                )}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
