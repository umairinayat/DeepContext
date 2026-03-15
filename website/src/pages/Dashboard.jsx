import { useState, useEffect } from 'react'
import { useAuth } from '../context/AuthContext'
import StatsCards from '../components/StatsCards'
import ChatPanel from '../components/ChatPanel'
import GraphViz from '../components/GraphViz'
import MemoryBrowser from '../components/MemoryBrowser'
import LifecycleControls from '../components/LifecycleControls'
import { healthCheck } from '../api/client'

const TABS = [
  { id: 'stats', label: 'Stats' },
  { id: 'chat', label: 'Chat Input' },
  { id: 'graph', label: 'Knowledge Graph' },
  { id: 'memories', label: 'Memories' },
  { id: 'lifecycle', label: 'Lifecycle' },
]

export default function Dashboard() {
  const { user, hasApiKey } = useAuth()
  const [activeTab, setActiveTab] = useState('stats')
  const [backendStatus, setBackendStatus] = useState('checking')
  const [refreshKey, setRefreshKey] = useState(0)

  // Track which tabs have been visited (lazy mount)
  const [mountedTabs, setMountedTabs] = useState(new Set(['stats']))

  function switchTab(tabId) {
    setActiveTab(tabId)
    setMountedTabs(prev => {
      if (prev.has(tabId)) return prev
      const next = new Set(prev)
      next.add(tabId)
      return next
    })
  }

  // Check backend connectivity
  useEffect(() => {
    healthCheck()
      .then(() => setBackendStatus('connected'))
      .catch(() => setBackendStatus('disconnected'))
  }, [])

  function handleRefresh() {
    // Reset mounted tabs to force remount all
    setMountedTabs(new Set([activeTab]))
    setRefreshKey(k => k + 1)
  }

  return (
    <div className="dashboard">
      {/* Dashboard Header */}
      <div className="dashboard-header">
        <div className="dashboard-header-left">
          <h1>Dashboard</h1>
          <div className={`backend-status status-${backendStatus}`}>
            <span className="status-dot" />
            {backendStatus === 'connected' ? 'Backend Connected' :
             backendStatus === 'disconnected' ? 'Backend Offline' : 'Checking...'}
          </div>
        </div>
        <div className="dashboard-header-right">
          <span className="dashboard-user">
            {user?.username}
          </span>
          <span className={`badge ${hasApiKey ? 'badge-green' : 'badge-orange'}`}>
            {hasApiKey ? 'API Key Set' : 'No API Key'}
          </span>
          <button className="btn btn-outline btn-sm" onClick={handleRefresh}>
            Refresh
          </button>
        </div>
      </div>

      {backendStatus === 'disconnected' && (
        <div className="dashboard-warning">
          Backend is not running. Start it with: <code>uvicorn deepcontext.api.server:app --reload</code>
        </div>
      )}

      {!hasApiKey && backendStatus === 'connected' && (
        <div className="dashboard-warning dashboard-warning-info">
          No API key configured. Go to <a href="#/settings">Settings</a> to add your OpenRouter API key before using memory extraction.
        </div>
      )}

      {/* Tab Navigation */}
      <div className="dashboard-tabs">
        {TABS.map(tab => (
          <button
            key={tab.id}
            className={`dashboard-tab ${activeTab === tab.id ? 'dashboard-tab-active' : ''}`}
            onClick={() => switchTab(tab.id)}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* Tab Content - show/hide instead of conditional render to preserve state */}
      <div className="dashboard-content" key={refreshKey}>
        <div style={{ display: activeTab === 'stats' ? 'block' : 'none' }}>
          {mountedTabs.has('stats') && <StatsCards />}
        </div>
        <div style={{ display: activeTab === 'chat' ? 'block' : 'none' }}>
          {mountedTabs.has('chat') && <ChatPanel />}
        </div>
        <div style={{ display: activeTab === 'graph' ? 'block' : 'none' }}>
          {mountedTabs.has('graph') && <GraphViz />}
        </div>
        <div style={{ display: activeTab === 'memories' ? 'block' : 'none' }}>
          {mountedTabs.has('memories') && <MemoryBrowser />}
        </div>
        <div style={{ display: activeTab === 'lifecycle' ? 'block' : 'none' }}>
          {mountedTabs.has('lifecycle') && <LifecycleControls />}
        </div>
      </div>
    </div>
  )
}
