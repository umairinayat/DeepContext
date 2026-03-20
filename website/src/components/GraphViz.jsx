import { useState, useEffect, useCallback, useRef, useLayoutEffect } from 'react'
import { getFullGraph, getNeighbors } from '../api/client'
import ErrorBanner from './ErrorBanner'
import EmptyState from './EmptyState'

// Color map for entity types
const TYPE_COLORS = {
  person: '#EC4899',
  organization: '#F59E0B',
  technology: '#6C5CE7',
  concept: '#06B6D4',
  location: '#10B981',
  event: '#A855F7',
  preference: '#F59E0B',
  other: '#5A6178',
}

function getColor(type) {
  return TYPE_COLORS[type] || TYPE_COLORS.other
}

export default function GraphViz({ userId }) {
  const [graphData, setGraphData] = useState({ nodes: [], links: [] })
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [is3D, setIs3D] = useState(false)
  const [isFullscreen, setIsFullscreen] = useState(false)
  const [hovered, setHovered] = useState(null)
  const [GraphComponent, setGraphComponent] = useState(null)
  const graphRef = useRef()
  const containerRef = useRef()
  const [dimensions, setDimensions] = useState({ width: 800, height: 500 })

  // Responsive sizing via ResizeObserver
  useLayoutEffect(() => {
    const el = containerRef.current
    if (!el) return
    const ro = new ResizeObserver(entries => {
      for (const entry of entries) {
        const { width, height } = entry.contentRect
        if (width > 0 && height > 0) {
          setDimensions({ width: Math.floor(width), height: Math.floor(height) })
        }
      }
    })
    ro.observe(el)
    return () => ro.disconnect()
  }, [])

  // Close fullscreen on Escape key
  useEffect(() => {
    if (!isFullscreen) return
    function handleKey(e) {
      if (e.key === 'Escape') setIsFullscreen(false)
    }
    window.addEventListener('keydown', handleKey)
    return () => window.removeEventListener('keydown', handleKey)
  }, [isFullscreen])

  // Dynamically import the graph component based on 2D/3D toggle
  useEffect(() => {
    async function loadGraph() {
      if (is3D) {
        const mod = await import('react-force-graph-3d')
        setGraphComponent(() => mod.default)
      } else {
        const mod = await import('react-force-graph-2d')
        setGraphComponent(() => mod.default)
      }
    }
    loadGraph()
  }, [is3D])

  function fetchGraph() {
    if (!userId) return
    setLoading(true)
    setError(null)
    getFullGraph(userId)
      .then(data => setGraphData(data))
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }

  // Fetch graph data
  useEffect(() => {
    fetchGraph()
  }, [userId])

  // Click on node to expand neighbors
  const handleNodeClick = useCallback(async (node) => {
    if (!userId) return
    try {
      const neighbors = await getNeighbors(userId, node.id, 1)
      if (neighbors.length === 0) return

      setGraphData(prev => {
        const existingNodeIds = new Set(prev.nodes.map(n => n.id))
        const newNodes = []
        const newLinks = []

        for (const n of neighbors) {
          if (!existingNodeIds.has(n.entity)) {
            newNodes.push({
              id: n.entity,
              type: n.entity_type,
              mentionCount: 1,
              val: 1,
              attributes: {},
            })
            existingNodeIds.add(n.entity)
          }
          // Add link if it doesn't already exist
          const linkKey = `${node.id}-${n.entity}-${n.relation}`
          const exists = prev.links.some(
            l => `${l.source?.id || l.source}-${l.target?.id || l.target}-${l.relation}` === linkKey
          )
          if (!exists) {
            newLinks.push({
              source: node.id,
              target: n.entity,
              relation: n.relation,
              strength: n.strength,
            })
          }
        }

        return {
          nodes: [...prev.nodes, ...newNodes],
          links: [...prev.links, ...newLinks],
        }
      })
    } catch (e) {
      console.error('Failed to expand node:', e)
    }
  }, [userId])

  // Node canvas drawing (2D)
  const paintNode = useCallback((node, ctx, globalScale) => {
    const label = node.id
    const fontSize = Math.max(12 / globalScale, 2)
    const size = Math.sqrt(node.val || 1) * 4 + 3

    // Node circle
    ctx.beginPath()
    ctx.arc(node.x, node.y, size, 0, 2 * Math.PI)
    ctx.fillStyle = getColor(node.type)
    ctx.fill()

    if (hovered?.id === node.id) {
      ctx.strokeStyle = '#fff'
      ctx.lineWidth = 2 / globalScale
      ctx.stroke()
    }

    // Label
    ctx.font = `${fontSize}px Inter, sans-serif`
    ctx.textAlign = 'center'
    ctx.textBaseline = 'top'
    ctx.fillStyle = '#fafafa'
    ctx.fillText(label, node.x, node.y + size + 2)
  }, [hovered])

  if (!userId) {
    return (
      <EmptyState
        message="Enter a User ID above to visualize the knowledge graph."
        actionLabel="Add Memories"
        actionTo="/dashboard/chat"
      />
    )
  }

  return (
    <div className={`graph-page ${isFullscreen ? 'graph-fullscreen' : ''}`}>
      <div className="graph-toolbar">
        <button
          className={`btn btn-sm btn-pill ${!is3D ? 'btn-primary' : 'btn-outline'}`}
          onClick={() => setIs3D(false)}
        >
          2D
        </button>
        <button
          className={`btn btn-sm btn-pill ${is3D ? 'btn-primary' : 'btn-outline'}`}
          onClick={() => setIs3D(true)}
        >
          3D
        </button>
        <button
          className="btn btn-sm btn-pill btn-outline"
          onClick={() => setIsFullscreen(f => !f)}
          title={isFullscreen ? 'Exit fullscreen (Esc)' : 'Fullscreen'}
        >
          {isFullscreen ? 'Exit Fullscreen' : 'Fullscreen'}
        </button>
        <span className="graph-info text-muted text-sm">
          {graphData.nodes.length} nodes &middot; {graphData.links.length} links
        </span>
        <div className="graph-legend">
          {Object.entries(TYPE_COLORS).map(([type, color]) => (
            <span key={type} className="graph-legend-item">
              <span className="graph-legend-dot" style={{ background: color }} />
              {type}
            </span>
          ))}
        </div>
      </div>

      {error && <ErrorBanner message={error} onRetry={fetchGraph} />}

      {!loading && !error && graphData.nodes.length === 0 && (
        <EmptyState
          message="No graph data yet. Add some conversations to build the knowledge graph."
          actionLabel="Add Memories"
          actionTo="/dashboard/chat"
        />
      )}

      <div className="graph-container" ref={containerRef}>
        {loading && (
          <div className="graph-loading-overlay">
            <div className="spinner" />
            <span>Loading graph...</span>
          </div>
        )}
        {!loading && GraphComponent && graphData.nodes.length > 0 && (
          <GraphComponent
            ref={graphRef}
            graphData={graphData}
            nodeLabel={node => `${node.id} (${node.type})\nMentions: ${node.mentionCount}`}
            nodeAutoColorBy="type"
            nodeCanvasObject={!is3D ? paintNode : undefined}
            nodeColor={node => getColor(node.type)}
            nodeVal={node => node.val || 1}
            linkLabel={link => link.relation}
            linkWidth={link => Math.max(link.strength * 2, 0.5)}
            linkDirectionalArrowLength={3.5}
            linkDirectionalArrowRelPos={1}
            linkColor={() => 'rgba(255,255,255,0.2)'}
            onNodeClick={handleNodeClick}
            onNodeHover={setHovered}
            backgroundColor="transparent"
            width={dimensions.width}
            height={dimensions.height}
          />
        )}
      </div>

      {hovered && (
        <div className="graph-tooltip">
          <strong>{hovered.id}</strong>
          <span className="badge" style={{ background: getColor(hovered.type), color: '#fff' }}>
            {hovered.type}
          </span>
          <span className="text-muted text-sm">Mentions: {hovered.mentionCount}</span>
        </div>
      )}
    </div>
  )
}
