import { useState, useEffect } from 'react'
import { getStats } from '../api/client'

export default function StatsCards({ userId }) {
  const [stats, setStats] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  useEffect(() => {
    if (!userId) return
    setLoading(true)
    setError(null)
    getStats(userId)
      .then(setStats)
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [userId])

  if (!userId) return <div className="stats-empty">Enter a User ID to see stats</div>
  if (loading) return <div className="stats-loading">Loading stats...</div>
  if (error) return <div className="stats-error">Error: {error}</div>
  if (!stats) return null

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
          </div>
        </div>
      </div>
    </div>
  )
}
