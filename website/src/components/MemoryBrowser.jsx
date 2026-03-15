import { useState, useEffect } from 'react'
import { listMemories, searchMemory, updateMemory, deleteMemory } from '../api/client'
import { useToast } from '../context/ToastContext'

function ConfirmDialog({ title, message, onConfirm, onCancel }) {
  return (
    <div className="confirm-overlay" onClick={onCancel}>
      <div className="confirm-dialog" onClick={e => e.stopPropagation()}>
        <h3>{title}</h3>
        <p>{message}</p>
        <div className="confirm-actions">
          <button className="btn btn-outline btn-sm" onClick={onCancel}>Cancel</button>
          <button className="btn btn-danger btn-sm" onClick={onConfirm}>Delete</button>
        </div>
      </div>
    </div>
  )
}

function SkeletonList({ count = 5 }) {
  return (
    <div className="memory-list">
      {Array.from({ length: count }, (_, i) => (
        <div key={i} className="memory-item" style={{ pointerEvents: 'none' }}>
          <div className="memory-item-header">
            <span className="skeleton" style={{ width: 60, height: 18 }} />
            <span className="skeleton" style={{ width: 60, height: 18 }} />
          </div>
          <div className="skeleton skeleton-line skeleton-line-full" />
          <div className="skeleton skeleton-line skeleton-line-medium" />
        </div>
      ))}
    </div>
  )
}

export default function MemoryBrowser() {
  const toast = useToast()
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

  // Edit state
  const [editingId, setEditingId] = useState(null)
  const [editText, setEditText] = useState('')
  const [editLoading, setEditLoading] = useState(false)

  // Delete confirmation
  const [deleteTarget, setDeleteTarget] = useState(null)
  const [deleteLoading, setDeleteLoading] = useState(false)

  useEffect(() => {
    if (isSearchMode && searchQuery.trim()) {
      doSearch()
    } else {
      doList()
    }
  }, [tier, memoryType, page])

  async function doList() {
    setLoading(true)
    setError(null)
    try {
      const opts = { limit: LIMIT, offset: page * LIMIT }
      if (tier) opts.tier = tier
      if (memoryType) opts.memory_type = memoryType
      const res = await listMemories(opts)
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
      const res = await searchMemory(searchQuery, opts)
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

  async function handleEditSave(memoryId) {
    if (!editText.trim()) return
    setEditLoading(true)
    try {
      await updateMemory(memoryId, editText.trim())
      setMemories(prev => prev.map(m => m.id === memoryId ? { ...m, text: editText.trim() } : m))
      setEditingId(null)
      setEditText('')
      toast.success('Memory updated')
    } catch (e) {
      toast.error(`Failed to update: ${e.message}`)
    } finally {
      setEditLoading(false)
    }
  }

  function handleEditStart(memory) {
    setEditingId(memory.id)
    setEditText(memory.text)
  }

  function handleEditCancel() {
    setEditingId(null)
    setEditText('')
  }

  async function handleDeleteConfirm() {
    if (!deleteTarget) return
    setDeleteLoading(true)
    try {
      await deleteMemory(deleteTarget.id)
      setMemories(prev => prev.filter(m => m.id !== deleteTarget.id))
      setTotal(prev => prev - 1)
      if (selected?.id === deleteTarget.id) setSelected(null)
      toast.success('Memory deleted')
    } catch (e) {
      toast.error(`Failed to delete: ${e.message}`)
    } finally {
      setDeleteLoading(false)
      setDeleteTarget(null)
    }
  }

  const totalPages = Math.ceil(total / LIMIT)

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
      {!loading && (
        <div className="memory-status">
          {`${total} memor${total === 1 ? 'y' : 'ies'} found`}
          {isSearchMode && <span className="memory-search-label"> (search results)</span>}
        </div>
      )}

      {error && <div className="memory-error">Error: {error}</div>}

      {/* Loading skeleton */}
      {loading && <SkeletonList />}

      {/* Empty state */}
      {!loading && !error && memories.length === 0 && (
        <div className="empty-state">
          <div className="empty-state-icon">{isSearchMode ? '\ud83d\udd0d' : '\ud83e\udde0'}</div>
          <div className="empty-state-title">
            {isSearchMode ? 'No results found' : 'No memories yet'}
          </div>
          <div className="empty-state-desc">
            {isSearchMode
              ? 'Try a different search query or adjust your filters.'
              : 'Extract some memories from the Chat Input tab to see them here.'}
          </div>
        </div>
      )}

      {/* Memory list */}
      {!loading && memories.length > 0 && (
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

                  {/* Edit form */}
                  {editingId === m.id ? (
                    <div className="memory-edit-form" onClick={e => e.stopPropagation()}>
                      <textarea
                        className="memory-edit-textarea"
                        value={editText}
                        onChange={e => setEditText(e.target.value)}
                        rows={3}
                      />
                      <div className="memory-edit-actions">
                        <button
                          className="btn btn-primary btn-sm"
                          onClick={() => handleEditSave(m.id)}
                          disabled={editLoading || !editText.trim()}
                        >
                          {editLoading ? 'Saving...' : 'Save'}
                        </button>
                        <button className="btn btn-outline btn-sm" onClick={handleEditCancel}>Cancel</button>
                      </div>
                    </div>
                  ) : (
                    <div className="memory-actions" onClick={e => e.stopPropagation()}>
                      <button className="btn-icon" onClick={() => handleEditStart(m)}>Edit</button>
                      <button className="btn-icon btn-icon-danger" onClick={() => setDeleteTarget(m)}>Delete</button>
                    </div>
                  )}
                </div>
              )}
            </div>
          ))}
        </div>
      )}

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

      {/* Delete confirmation */}
      {deleteTarget && (
        <ConfirmDialog
          title="Delete Memory"
          message={`Are you sure you want to delete this memory? This action cannot be undone.\n\n"${deleteTarget.text.slice(0, 100)}${deleteTarget.text.length > 100 ? '...' : ''}"`}
          onConfirm={handleDeleteConfirm}
          onCancel={() => setDeleteTarget(null)}
        />
      )}
    </div>
  )
}
