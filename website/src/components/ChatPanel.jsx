import { useState } from 'react'
import { addMemory } from '../api/client'

const PLACEHOLDER = `Paste a conversation here, e.g.:

USER: I've been working with Python and FastAPI lately.
ASSISTANT: That's great! FastAPI is excellent for async APIs.
USER: Yeah, I prefer it over Flask for new projects.`

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
      <div className="chat-panel-header">
        <h3>Chat Input</h3>
        <div className="chat-conv-id">
          <label>Conversation ID:</label>
          <input
            type="text"
            value={conversationId}
            onChange={e => setConversationId(e.target.value)}
            placeholder="default"
          />
        </div>
      </div>

      <textarea
        className="chat-textarea"
        value={text}
        onChange={e => setText(e.target.value)}
        placeholder={PLACEHOLDER}
        rows={10}
      />

      <button
        className="btn btn-primary chat-extract-btn"
        onClick={handleExtract}
        disabled={loading || !text.trim() || !userId}
      >
        {loading ? 'Extracting...' : 'Extract Memories'}
      </button>

      {error && <div className="chat-error">Error: {error}</div>}

      {result && (
        <div className="chat-result">
          <h4>Extraction Result</h4>
          <div className="chat-result-stats">
            <span className="badge badge-green">+{result.memories_added} added</span>
            {result.memories_updated > 0 && <span className="badge badge-cyan">{result.memories_updated} updated</span>}
            {result.memories_replaced > 0 && <span className="badge badge-orange">{result.memories_replaced} replaced</span>}
          </div>

          {result.semantic_facts.length > 0 && (
            <div className="chat-result-section">
              <h5>Semantic Facts</h5>
              <ul>
                {result.semantic_facts.map((f, i) => <li key={i}>{f}</li>)}
              </ul>
            </div>
          )}

          {result.episodic_facts.length > 0 && (
            <div className="chat-result-section">
              <h5>Episodic Facts</h5>
              <ul>
                {result.episodic_facts.map((f, i) => <li key={i}>{f}</li>)}
              </ul>
            </div>
          )}

          {result.entities_found.length > 0 && (
            <div className="chat-result-section">
              <h5>Entities Found</h5>
              <div className="chat-entities">
                {result.entities_found.map((e, i) => (
                  <span key={i} className="badge badge-purple">{e}</span>
                ))}
              </div>
            </div>
          )}

          {result.relationships_found > 0 && (
            <div className="chat-result-section">
              <h5>Relationships: {result.relationships_found}</h5>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
