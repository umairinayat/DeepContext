import { useState, useEffect } from 'react'
import { getStats } from '../api/client'

function SkeletonCards() {
  return (
    <div className="stats-cards">
      <div className="skeleton skeleton-card" />
      <div className="skeleton skeleton-card" />
      <div className="skeleton skeleton-card" />
      <div className="stats-breakdown">
        <div className="skeleton skeleton-card" style={{ height: 80 }} />
        <div className="skeleton skeleton-card" style={{ height: 80 }} />
      </div>
    </div>
  )
}

export default function StatsCards() {
  const [stats, setStats] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  useEffect(() => {
    setLoading(true)
    setError(null)
    getStats()
      .then(setStats)
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <SkeletonCards />
  if (error) return <div className="stats-error">Error: {error}</div>

  if (!stats || (stats.total_memories === 0 && stats.total_entities === 0)) {
    return (
      <div className="empty-state">
        <div className="empty-state-icon">{'\ud83d\udcca'}</div>
        <div className="empty-state-title">No data to show</div>
        <div className="empty-state-desc">
          Start by extracting memories from conversations in the Chat Input tab.
        </div>
      </div>
    )
  }

  return (
    <div className="stats-cards">
      <div className="stats-card stats-card-accent">
        <div className="stats-card-value">{stats.total_memories}</div>
        <div className="stats-card-label">Total Memories</div>
      </div>
      <div className="stats-card stats-card-green">
        <div className="stats-card-value">{stats.total_entities}</div>
        <div className="stats-card-label">Entities</div>
      </div>
      <div className="stats-card stats-card-purple">
        <div className="stats-card-value">{stats.total_relationships}</div>
        <div className="stats-card-label">Relationships</div>
      </div>

      <div className="stats-breakdown">
        <div className="stats-group">
          <h4>By Tier</h4>
          <div className="stats-badges">
            {Object.entries(stats.by_tier).map(([tier, count]) => (
              <span key={tier} className={`stats-badge tier-${tier}`}>
                {tier.replace('_', ' ')}: {count}
              </span>
            ))}
            {Object.keys(stats.by_tier).length === 0 && (
              <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>No tier data</span>
            )}
          </div>
        </div>
        <div className="stats-group">
          <h4>By Type</h4>
          <div className="stats-badges">
            {Object.entries(stats.by_type).map(([type, count]) => (
              <span key={type} className={`stats-badge type-${type}`}>
                {type}: {count}
              </span>
            ))}
            {Object.keys(stats.by_type).length === 0 && (
              <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>No type data</span>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
