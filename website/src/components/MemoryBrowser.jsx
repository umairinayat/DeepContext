import { useState, useEffect } from 'react'
import { listMemories, searchMemory } from '../api/client'
import SlidePanel from './SlidePanel'
import ErrorBanner from './ErrorBanner'
import EmptyState from './EmptyState'
import { SkeletonRow } from './Skeleton'

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

  // Selected memory for slide panel
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

  function handleClearSearch() {
    setSearchQuery('')
    setIsSearchMode(false)
    doList()
  }

  const totalPages = Math.ceil(total / LIMIT)

  if (!userId) {
    return (
      <EmptyState
        message="Enter a User ID above to browse memories."
        actionLabel="Add Memories"
        actionTo="/dashboard/chat"
      />
    )
  }

  return (
    <div className="memory-browser">
      {/* Search & Filters */}
      <div className="memory-search-bar">
        <form className="memory-search-form" onSubmit={handleSearchSubmit}>
          <input
            type="text"
            className="input-field"
            value={searchQuery}
            onChange={e => setSearchQuery(e.target.value)}
            placeholder="Search memories..."
          />
          <button type="submit" className="btn btn-primary btn-sm btn-pill">Search</button>
          {isSearchMode && (
            <button type="button" className="btn btn-outline btn-sm btn-pill" onClick={handleClearSearch}>
              Clear
            </button>
          )}
        </form>

        <div className="memory-filters">
          <select
            className="input-field input-field-sm"
            value={tier}
            onChange={e => { setTier(e.target.value); setPage(0) }}
          >
            <option value="">All Tiers</option>
            <option value="working">Working</option>
            <option value="short_term">Short Term</option>
            <option value="long_term">Long Term</option>
          </select>
          <select
            className="input-field input-field-sm"
            value={memoryType}
            onChange={e => { setMemoryType(e.target.value); setPage(0) }}
          >
            <option value="">All Types</option>
            <option value="semantic">Semantic</option>
            <option value="episodic">Episodic</option>
            <option value="procedural">Procedural</option>
          </select>
        </div>
      </div>

      {/* Status */}
      <div className="memory-status text-muted text-sm">
        {loading
          ? 'Loading...'
          : `${total} memor${total === 1 ? 'y' : 'ies'} found`}
        {isSearchMode && <span className="memory-search-label"> &mdash; search results</span>}
      </div>

      {error && <ErrorBanner message={error} onRetry={isSearchMode ? doSearch : doList} />}

      {/* Memory list */}
      {loading ? (
        <SkeletonRow count={6} />
      ) : !error && memories.length === 0 ? (
        <EmptyState
          message={isSearchMode ? 'No memories match your search.' : 'No memories found for this user.'}
          actionLabel={isSearchMode ? 'Clear Search' : 'Add Memories'}
          onAction={isSearchMode ? handleClearSearch : undefined}
          actionTo={isSearchMode ? undefined : '/dashboard/chat'}
        />
      ) : (
        <div className="memory-list">
          {memories.map(m => (
            <div
              key={m.id}
              className={`memory-row ${selected?.id === m.id ? 'memory-row-active' : ''}`}
              onClick={() => setSelected(m)}
              role="button"
              tabIndex={0}
              onKeyDown={e => e.key === 'Enter' && setSelected(m)}
            >
              <div className="memory-row-text">{m.text}</div>
              <div className="memory-row-badges">
                <span className={`badge badge-tier-${m.tier || 'unknown'}`}>
                  {(m.tier || 'unknown').replace('_', ' ')}
                </span>
                <span className={`badge badge-type-${m.memory_type || 'unknown'}`}>
                  {m.memory_type || 'unknown'}
                </span>
              </div>
              {m.score !== undefined && (
                <span className="memory-row-score text-muted text-sm">
                  {m.score.toFixed(3)}
                </span>
              )}
            </div>
          ))}
        </div>
      )}

      {/* Pagination */}
      {totalPages > 1 && !isSearchMode && (
        <div className="memory-pagination">
          {Array.from({ length: totalPages }, (_, i) => (
            <button
              key={i}
              className={`btn btn-sm btn-pill ${i === page ? 'btn-primary' : 'btn-outline'}`}
              onClick={() => setPage(i)}
            >
              {i + 1}
            </button>
          ))}
        </div>
      )}

      {/* Slide panel for memory detail */}
      <SlidePanel
        isOpen={!!selected}
        onClose={() => setSelected(null)}
        title="Memory Detail"
      >
        {selected && (
          <div className="memory-detail">
            <div className="detail-row">
              <span className="detail-label">Text</span>
              <span className="detail-value">{selected.text}</span>
            </div>
            <div className="detail-row">
              <span className="detail-label">ID</span>
              <span className="detail-value text-muted text-sm">{selected.id}</span>
            </div>
            <div className="detail-row">
              <span className="detail-label">Tier</span>
              <span className="detail-value">
                <span className={`badge badge-tier-${selected.tier || 'unknown'}`}>
                  {(selected.tier || 'unknown').replace('_', ' ')}
                </span>
              </span>
            </div>
            <div className="detail-row">
              <span className="detail-label">Type</span>
              <span className="detail-value">
                <span className={`badge badge-type-${selected.memory_type || 'unknown'}`}>
                  {selected.memory_type || 'unknown'}
                </span>
              </span>
            </div>
            <div className="detail-row">
              <span className="detail-label">Importance</span>
              <span className="detail-value">{selected.importance?.toFixed(2) ?? 'N/A'}</span>
            </div>
            <div className="detail-row">
              <span className="detail-label">Confidence</span>
              <span className="detail-value">{selected.confidence?.toFixed(2) ?? 'N/A'}</span>
            </div>
            {selected.source_entities?.length > 0 && (
              <div className="detail-row">
                <span className="detail-label">Entities</span>
                <span className="detail-value">
                  <div className="entity-chips">
                    {selected.source_entities.map((e, i) => (
                      <span key={i} className="badge badge-purple">{e}</span>
                    ))}
                  </div>
                </span>
              </div>
            )}
            <div className="detail-row">
              <span className="detail-label">Created</span>
              <span className="detail-value text-muted text-sm">
                {selected.created_at ? new Date(selected.created_at).toLocaleString() : 'N/A'}
              </span>
            </div>

            <div className="slide-panel-footer">
              <button className="btn btn-outline btn-sm btn-pill">Edit</button>
              <button className="btn btn-danger btn-sm btn-pill">Delete</button>
            </div>
          </div>
        )}
      </SlidePanel>
    </div>
  )
}
