import { useState, useEffect } from 'react'
import { Link, Outlet, useLocation } from 'react-router-dom'
import Sidebar from '../components/Sidebar'
import { healthCheck } from '../api/client'
import { useAuth } from '../context/AuthContext'

const PAGE_TITLES = {
  stats: 'Stats',
  chat: 'Chat Input',
  graph: 'Knowledge Graph',
  memories: 'Memories',
  lifecycle: 'Lifecycle',
  settings: 'Settings',
}

export default function Dashboard() {
  const { user, hasApiKey } = useAuth()
  const [backendStatus, setBackendStatus] = useState('checking')
  const [refreshKey, setRefreshKey] = useState(0)
  const location = useLocation()
  const userId = user?.username || 'current_user'

  const currentSection = location.pathname.split('/').pop() || 'stats'
  const pageTitle = PAGE_TITLES[currentSection] || 'Dashboard'

  useEffect(() => {
    healthCheck()
      .then(() => setBackendStatus('connected'))
      .catch(() => setBackendStatus('disconnected'))
  }, [])

  function handleRefresh() {
    setRefreshKey(k => k + 1)
  }

  return (
    <div className="dashboard-layout">
      <Sidebar />
      <div className="dashboard-main">
        <div className="dashboard-header">
          <h2 className="dashboard-title">{pageTitle}</h2>
          <div className="dashboard-header-right">
            <div className={`backend-status status-${backendStatus}`}>
              <span className="status-dot" />
              {backendStatus === 'connected' ? 'Connected' :
               backendStatus === 'disconnected' ? 'Disconnected' : 'Checking...'}
            </div>
            <span className="text-muted text-sm">{user?.username}</span>
            <span className={`badge ${hasApiKey ? 'badge-green' : 'badge-orange'}`}>
              {hasApiKey ? 'API Key Set' : 'No API Key'}
            </span>
            <button className="btn btn-icon btn-ghost" onClick={handleRefresh} title="Refresh">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <polyline points="23 4 23 10 17 10"/><path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"/>
              </svg>
            </button>
          </div>
        </div>

        {backendStatus === 'disconnected' && (
          <div className="error-banner" style={{ margin: '1rem 2rem 0' }}>
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/>
            </svg>
            <span>Backend offline. Start it: <code>uvicorn deepcontext.api.server:app --reload --port 8000</code></span>
          </div>
        )}

        {!hasApiKey && backendStatus === 'connected' && (
          <div className="error-banner" style={{ margin: '1rem 2rem 0' }}>
            <span>
              No API key configured. Add one in <Link to="/dashboard/settings">Settings</Link> before running extractions.
            </span>
          </div>
        )}

        <div className="dashboard-content" key={refreshKey}>
          <Outlet context={{ userId, refreshKey, backendStatus }} />
        </div>
      </div>
    </div>
  )
}
