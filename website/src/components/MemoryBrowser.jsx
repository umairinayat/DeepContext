import { useState, useEffect } from 'react'
import { listMemories, searchMemory } from '../api/client'

export default function MemoryBrowser({ userId }) {
  const [memories, setMemories] = useState([])
  const [total, setTotal] = useState(0)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  // Filters
  const [tier, setTier] = useState('')
  const [memoryType, setMemoryType] = useState('')
  const [searchQuery, setSearchQuery] = useState('')
  const [isSearchMode, setIsSearchMode] = useState(false)
  const [page, setPage] = useState(0)
  const LIMIT = 20

  // Selected memory for detail view
  const [selected, setSelected] = useState(null)

  useEffect(() => {
    if (!userId) return
    if (isSearchMode && searchQuery.trim()) {
      doSearch()
    } else {
      doList()
    }
  }, [userId, tier, memoryType, page])

  async function doList() {
    setLoading(true)
    setError(null)
    try {
      const opts = { limit: LIMIT, offset: page * LIMIT }
      if (tier) opts.tier = tier
      if (memoryType) opts.memory_type = memoryType
      const res = await listMemories(userId, opts)
      setMemories(res.memories)
      setTotal(res.total)
      setIsSearchMode(false)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  async function doSearch() {
    if (!searchQuery.trim()) {
      doList()
      return
    }
    setLoading(true)
    setError(null)
    try {
      const opts = { limit: LIMIT }
      if (tier) opts.tier = tier
      if (memoryType) opts.memory_type = memoryType
      const res = await searchMemory(searchQuery, userId, opts)
      setMemories(
        res.results.map(r => ({
          id: r.memory_id,
          text: r.text,
          memory_type: r.memory_type,
          tier: r.tier,
          importance: r.importance,
          confidence: r.confidence,
          score: r.score,
          source_entities: r.entities || [],
          created_at: r.created_at,
        }))
      )
      setTotal(res.total)
      setIsSearchMode(true)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  function handleSearchSubmit(e) {
    e.preventDefault()
    setPage(0)
    doSearch()
  }

  const totalPages = Math.ceil(total / LIMIT)

  if (!userId) return <div className="memory-empty">Enter a User ID to browse memories</div>

  return (
    <div className="memory-browser">
      {/* Search & Filters */}
      <div className="memory-controls">
        <form className="memory-search-form" onSubmit={handleSearchSubmit}>
          <input
            type="text"
            className="memory-search-input"
            value={searchQuery}
            onChange={e => setSearchQuery(e.target.value)}
            placeholder="Search memories..."
          />
          <button type="submit" className="btn btn-primary btn-sm">Search</button>
          {isSearchMode && (
            <button type="button" className="btn btn-outline btn-sm" onClick={() => { setSearchQuery(''); setIsSearchMode(false); doList() }}>
              Clear
            </button>
          )}
        </form>

        <div className="memory-filters">
          <select value={tier} onChange={e => { setTier(e.target.value); setPage(0) }}>
            <option value="">All Tiers</option>
            <option value="working">Working</option>
            <option value="short_term">Short Term</option>
            <option value="long_term">Long Term</option>
          </select>
          <select value={memoryType} onChange={e => { setMemoryType(e.target.value); setPage(0) }}>
            <option value="">All Types</option>
            <option value="semantic">Semantic</option>
            <option value="episodic">Episodic</option>
            <option value="procedural">Procedural</option>
          </select>
        </div>
      </div>

      {/* Status */}
      <div className="memory-status">
        {loading ? 'Loading...' : `${total} memor${total === 1 ? 'y' : 'ies'} found`}
        {isSearchMode && <span className="memory-search-label"> (search results)</span>}
      </div>

      {error && <div className="memory-error">Error: {error}</div>}

      {/* Memory list */}
      <div className="memory-list">
        {memories.map(m => (
          <div
            key={m.id}
            className={`memory-item ${selected?.id === m.id ? 'memory-item-selected' : ''}`}
            onClick={() => setSelected(selected?.id === m.id ? null : m)}
          >
            <div className="memory-item-header">
              <span className={`badge tier-${m.tier || 'unknown'}`}>{(m.tier || 'unknown').replace('_', ' ')}</span>
              <span className={`badge type-${m.memory_type || 'unknown'}`}>{m.memory_type || 'unknown'}</span>
              {m.score !== undefined && <span className="badge badge-cyan">score: {m.score.toFixed(3)}</span>}
              <span className="memory-importance">imp: {m.importance?.toFixed(2) ?? 'N/A'}</span>
            </div>
            <div className="memory-item-text">{m.text}</div>

            {selected?.id === m.id && (
              <div className="memory-item-detail">
                <div><strong>ID:</strong> {m.id}</div>
                <div><strong>Confidence:</strong> {m.confidence?.toFixed(2)}</div>
                {m.source_entities?.length > 0 && (
                  <div>
                    <strong>Entities:</strong>{' '}
                    {m.source_entities.map((e, i) => <span key={i} className="badge badge-purple">{e}</span>)}
                  </div>
                )}
                <div><strong>Created:</strong> {m.created_at ? new Date(m.created_at).toLocaleString() : 'N/A'}</div>
              </div>
            )}
          </div>
        ))}
      </div>

      {/* Pagination */}
      {totalPages > 1 && !isSearchMode && (
        <div className="memory-pagination">
          <button
            className="btn btn-outline btn-sm"
            disabled={page === 0}
            onClick={() => setPage(p => p - 1)}
          >
            Previous
          </button>
          <span className="memory-page-info">Page {page + 1} of {totalPages}</span>
          <button
            className="btn btn-outline btn-sm"
            disabled={page >= totalPages - 1}
            onClick={() => setPage(p => p + 1)}
          >
            Next
          </button>
        </div>
      )}
    </div>
  )
}
