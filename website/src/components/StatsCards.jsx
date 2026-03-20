import { useState, useEffect } from 'react'
import { getStats } from '../api/client'
import ProgressBar from './ProgressBar'
import { SkeletonBar, SkeletonCard } from './Skeleton'
import ErrorBanner from './ErrorBanner'
import EmptyState from './EmptyState'

const TIER_COLORS = {
  working: 'var(--accent)',
  short_term: 'var(--cyan)',
  long_term: 'var(--green)',
}

const TYPE_COLORS = {
  semantic: 'var(--accent)',
  episodic: 'var(--purple)',
  procedural: 'var(--orange)',
}

export default function StatsCards({ userId }) {
  const [stats, setStats] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  function load() {
    if (!userId) return
    setLoading(true)
    setError(null)
    getStats(userId)
      .then(setStats)
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }

  useEffect(() => {
    load()
  }, [userId])

  if (!userId) {
    return (
      <EmptyState
        message="Enter a User ID above to see memory statistics."
        actionLabel="Add Memories"
        actionTo="/dashboard/chat"
      />
    )
  }

  if (loading) {
    return (
      <div>
        <div className="stats-grid">
          <SkeletonCard height="120px" />
          <SkeletonCard height="120px" />
          <SkeletonCard height="120px" />
        </div>
        <div className="stats-breakdown" style={{ marginTop: '1.5rem' }}>
          <div className="card breakdown-card">
            <SkeletonBar height="1rem" width="40%" />
            <div style={{ marginTop: '1rem', display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
              <SkeletonBar />
              <SkeletonBar />
              <SkeletonBar />
            </div>
          </div>
          <div className="card breakdown-card">
            <SkeletonBar height="1rem" width="40%" />
            <div style={{ marginTop: '1rem', display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
              <SkeletonBar />
              <SkeletonBar />
              <SkeletonBar />
            </div>
          </div>
        </div>
      </div>
    )
  }

  if (error) {
    return <ErrorBanner message={error} onRetry={load} />
  }

  if (!stats) return null

  const hasData = stats.total_memories > 0 || stats.total_entities > 0 || stats.total_relationships > 0

  if (!hasData) {
    return (
      <EmptyState
        message="No memories found for this user. Start by adding a conversation."
        actionLabel="Add Memories"
        actionTo="/dashboard/chat"
      />
    )
  }

  const tierEntries = Object.entries(stats.by_tier || {})
  const typeEntries = Object.entries(stats.by_type || {})
  const totalByTier = tierEntries.reduce((s, [, v]) => s + v, 0)
  const totalByType = typeEntries.reduce((s, [, v]) => s + v, 0)

  return (
    <div>
      <div className="stats-grid">
        <div className="card stat-card">
          <div className="stat-card-value" style={{ color: 'var(--accent)' }}>
            {stats.total_memories}
          </div>
          <div className="stat-card-label">Total Memories</div>
        </div>
        <div className="card stat-card">
          <div className="stat-card-value" style={{ color: 'var(--green)' }}>
            {stats.total_entities}
          </div>
          <div className="stat-card-label">Entities</div>
        </div>
        <div className="card stat-card">
          <div className="stat-card-value" style={{ color: 'var(--purple)' }}>
            {stats.total_relationships}
          </div>
          <div className="stat-card-label">Relationships</div>
        </div>
      </div>

      <div className="stats-breakdown">
        <div className="card breakdown-card">
          <h4 className="breakdown-title">By Tier</h4>
          {tierEntries.map(([tier, count]) => (
            <ProgressBar
              key={tier}
              value={count}
              max={totalByTier}
              color={TIER_COLORS[tier] || 'var(--text-muted)'}
              label={tier.replace('_', ' ')}
              count={count}
            />
          ))}
        </div>
        <div className="card breakdown-card">
          <h4 className="breakdown-title">By Type</h4>
          {typeEntries.map(([type, count]) => (
            <ProgressBar
              key={type}
              value={count}
              max={totalByType}
              color={TYPE_COLORS[type] || 'var(--text-muted)'}
              label={type}
              count={count}
            />
          ))}
        </div>
      </div>
    </div>
  )
}
