import { useState } from 'react'
import { runLifecycle } from '../api/client'
import ErrorBanner from './ErrorBanner'

const DECAY_ICON = (
  <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
    <path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83"/>
  </svg>
)

const CONSOLIDATE_ICON = (
  <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
    <path d="M8 3H5a2 2 0 0 0-2 2v3M21 8V5a2 2 0 0 0-2-2h-3M3 16v3a2 2 0 0 0 2 2h3M16 21h3a2 2 0 0 0 2-2v-3"/>
  </svg>
)

const CLEANUP_ICON = (
  <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
    <polyline points="3 6 5 6 21 6"/><path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6M10 11v6M14 11v6M9 6V4a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2"/>
  </svg>
)

export default function LifecycleControls({ userId }) {
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [lastRun, setLastRun] = useState(null)

  async function handleRun() {
    if (!userId) return
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      const res = await runLifecycle(userId)
      setResult(res)
      setLastRun(new Date())
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="lifecycle-section">
      <p className="lifecycle-description">
        Run decay, consolidation, and cleanup on stored memories.
        Decay reduces importance of old short-term memories.
        Consolidation merges similar memories into long-term storage.
        Cleanup removes fully decayed or low-importance memories.
      </p>

      <button
        className="btn btn-primary btn-lg btn-pill"
        onClick={handleRun}
        disabled={loading || !userId}
      >
        {loading ? (
          <>
            <span className="spinner spinner-sm" />
            Running...
          </>
        ) : (
          'Run Lifecycle'
        )}
      </button>

      {lastRun && !loading && (
        <div className="lifecycle-last-run text-muted text-sm">
          Last run: {lastRun.toLocaleString()}
        </div>
      )}

      {error && <ErrorBanner message={error} onRetry={handleRun} />}

      {result && (
        <div className="lifecycle-results">
          <div className="card lifecycle-result-card">
            <div className="lifecycle-result-icon" style={{ color: 'var(--orange)' }}>
              {DECAY_ICON}
            </div>
            <div className="lifecycle-result-number" style={{ color: 'var(--orange)' }}>
              {result.memories_decayed}
            </div>
            <div className="lifecycle-result-label">Decayed</div>
          </div>

          <div className="card lifecycle-result-card">
            <div className="lifecycle-result-icon" style={{ color: 'var(--cyan)' }}>
              {CONSOLIDATE_ICON}
            </div>
            <div className="lifecycle-result-number" style={{ color: 'var(--cyan)' }}>
              {result.memories_consolidated}
            </div>
            <div className="lifecycle-result-label">Consolidated</div>
          </div>

          <div className="card lifecycle-result-card">
            <div className="lifecycle-result-icon" style={{ color: 'var(--green)' }}>
              {CLEANUP_ICON}
            </div>
            <div className="lifecycle-result-number" style={{ color: 'var(--green)' }}>
              {result.memories_cleaned}
            </div>
            <div className="lifecycle-result-label">Cleaned Up</div>
          </div>
        </div>
      )}
    </div>
  )
}
