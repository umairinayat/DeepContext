import { useState } from 'react'
import { runLifecycle } from '../api/client'

export default function LifecycleControls({ userId }) {
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  async function handleRun() {
    if (!userId) return
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      const res = await runLifecycle(userId)
      setResult(res)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="lifecycle-panel">
      <h3>Lifecycle Management</h3>
      <p className="lifecycle-desc">
        Run decay, consolidation, and cleanup on stored memories.
        Decay reduces importance of old short-term memories.
        Consolidation merges similar memories into long-term storage.
        Cleanup removes fully decayed or low-importance memories.
      </p>

      <button
        className="btn btn-primary"
        onClick={handleRun}
        disabled={loading || !userId}
      >
        {loading ? 'Running...' : 'Run Lifecycle'}
      </button>

      {error && <div className="lifecycle-error">Error: {error}</div>}

      {result && (
        <div className="lifecycle-result">
          <h4>Lifecycle Results</h4>
          <div className="lifecycle-stats">
            <div className="lifecycle-stat">
              <span className="lifecycle-stat-value">{result.memories_decayed}</span>
              <span className="lifecycle-stat-label">Decayed</span>
            </div>
            <div className="lifecycle-stat">
              <span className="lifecycle-stat-value">{result.memories_consolidated}</span>
              <span className="lifecycle-stat-label">Consolidated</span>
            </div>
            <div className="lifecycle-stat">
              <span className="lifecycle-stat-value">{result.memories_cleaned}</span>
              <span className="lifecycle-stat-label">Cleaned Up</span>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
