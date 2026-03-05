import { useState, useEffect } from 'react'
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
  const [userId, setUserId] = useState('default_user')
  const [activeTab, setActiveTab] = useState('stats')
  const [backendStatus, setBackendStatus] = useState('checking')
  const [refreshKey, setRefreshKey] = useState(0)

  // Check backend connectivity
  useEffect(() => {
    healthCheck()
      .then(() => setBackendStatus('connected'))
      .catch(() => setBackendStatus('disconnected'))
  }, [])

  function handleRefresh() {
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
          <div className="user-id-input">
            <label>User ID:</label>
            <input
              type="text"
              value={userId}
              onChange={e => setUserId(e.target.value)}
              placeholder="Enter user ID"
            />
          </div>
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

      {/* Tab Navigation */}
      <div className="dashboard-tabs">
        {TABS.map(tab => (
          <button
            key={tab.id}
            className={`dashboard-tab ${activeTab === tab.id ? 'dashboard-tab-active' : ''}`}
            onClick={() => setActiveTab(tab.id)}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* Tab Content */}
      <div className="dashboard-content" key={refreshKey}>
        {activeTab === 'stats' && <StatsCards userId={userId} />}
        {activeTab === 'chat' && <ChatPanel userId={userId} />}
        {activeTab === 'graph' && <GraphViz userId={userId} />}
        {activeTab === 'memories' && <MemoryBrowser userId={userId} />}
        {activeTab === 'lifecycle' && <LifecycleControls userId={userId} />}
      </div>
    </div>
  )
}
