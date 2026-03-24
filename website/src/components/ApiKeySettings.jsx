import { useState } from 'react'
import { setApiKey, deleteApiKey } from '../api/client'
import { useAuth } from '../context/AuthContext'

export default function ApiKeySettings() {
  const { user, hasApiKey, refreshProfile } = useAuth()
  const [apiKey, setApiKeyValue] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [success, setSuccess] = useState(null)

  async function handleSave(e) {
    e.preventDefault()
    if (!apiKey.trim()) return

    setLoading(true)
    setError(null)
    setSuccess(null)
    try {
      await setApiKey(apiKey.trim())
      setApiKeyValue('')
      setSuccess('API key saved successfully.')
      await refreshProfile()
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  async function handleDelete() {
    setLoading(true)
    setError(null)
    setSuccess(null)
    try {
      await deleteApiKey()
      setSuccess('API key removed.')
      await refreshProfile()
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="settings-page">
      <div className="settings-card">
        <h1>Settings</h1>

        {/* Account info */}
        <div className="settings-section">
          <h3>Account</h3>
          <div className="settings-info">
            <div className="settings-info-row">
              <span className="settings-info-label">Username</span>
              <span className="settings-info-value">{user?.username || '--'}</span>
            </div>
            {user?.email && (
              <div className="settings-info-row">
                <span className="settings-info-label">Email</span>
                <span className="settings-info-value">{user.email}</span>
              </div>
            )}
          </div>
        </div>

        {/* API Key */}
        <div className="settings-section">
          <h3>OpenRouter API Key</h3>
          <p className="settings-desc">
            Your API key is encrypted at rest and used server-side for LLM calls
            (memory extraction, embeddings). It is never exposed to the frontend.
          </p>

          <div className="settings-key-status">
            <span className={`badge ${hasApiKey ? 'badge-green' : 'badge-orange'}`}>
              {hasApiKey ? 'Key configured' : 'No key set'}
            </span>
          </div>

          <form onSubmit={handleSave} className="settings-key-form">
            <input
              type="password"
              value={apiKey}
              onChange={e => setApiKeyValue(e.target.value)}
              placeholder={hasApiKey ? 'Enter new key to replace...' : 'sk-or-v1-...'}
              className="settings-key-input"
            />
            <button
              type="submit"
              className="btn btn-primary btn-sm"
              disabled={loading || !apiKey.trim()}
            >
              {loading ? 'Saving...' : (hasApiKey ? 'Update Key' : 'Save Key')}
            </button>
            {hasApiKey && (
              <button
                type="button"
                className="btn btn-outline btn-sm settings-key-delete"
                onClick={handleDelete}
                disabled={loading}
              >
                Remove Key
              </button>
            )}
          </form>

          {error && <div className="auth-error">{error}</div>}
          {success && <div className="auth-success">{success}</div>}
        </div>
      </div>
    </div>
  )
}
