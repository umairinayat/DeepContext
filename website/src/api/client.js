/**
 * API client for DeepContext backend.
 * Connects to FastAPI server with JWT authentication.
 */

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000'

/**
 * Get the stored JWT token.
 */
function getToken() {
  return localStorage.getItem('dc_token')
}

/**
 * Make an authenticated request.
 */
async function request(path, { method = 'POST', body = null, auth = true } = {}) {
  const headers = { 'Content-Type': 'application/json' }

  if (auth) {
    const token = getToken()
    if (!token) {
      throw new Error('Not authenticated')
    }
    headers['Authorization'] = `Bearer ${token}`
  }

  const opts = { method, headers }
  if (body !== null) {
    opts.body = JSON.stringify(body)
  }

  const resp = await fetch(`${API_BASE}${path}`, opts)

  if (resp.status === 401) {
    // Token expired or invalid -- clear auth state
    localStorage.removeItem('dc_token')
    localStorage.removeItem('dc_user')
    window.dispatchEvent(new Event('dc_logout'))
    throw new Error('Session expired. Please log in again.')
  }

  if (!resp.ok) {
    const err = await resp.json().catch(() => ({ detail: resp.statusText }))
    throw new Error(err.detail || `HTTP ${resp.status}`)
  }

  return resp.json()
}

// ===========================================================================
// AUTH
// ===========================================================================

/** Register a new account. */
export async function register(username, password, email = null) {
  const body = { username, password }
  if (email) body.email = email
  return request('/auth/register', { body, auth: false })
}

/** Login and get JWT token. */
export async function login(username, password) {
  return request('/auth/login', { body: { username, password }, auth: false })
}

/** Get current user profile. */
export async function getProfile() {
  return request('/auth/me', { method: 'GET' })
}

/** Save OpenRouter API key. */
export async function setApiKey(apiKey) {
  return request('/auth/apikey', { body: { api_key: apiKey } })
}

/** Remove stored API key. */
export async function deleteApiKey() {
  return request('/auth/apikey', { method: 'DELETE' })
}

// ===========================================================================
// MEMORY
// ===========================================================================

/** Extract & store memories from conversation messages. */
export async function addMemory(messages, conversationId = 'default') {
  return request('/memory/add', { body: { messages, conversation_id: conversationId } })
}

/** Hybrid search for memories. */
export async function searchMemory(query, options = {}) {
  return request('/memory/search', { body: { query, ...options } })
}

/** List all memories with optional filtering & pagination. */
export async function listMemories(options = {}) {
  return request('/memory/list', { body: options })
}

/** Update a memory. */
export async function updateMemory(memoryId, text) {
  return request('/memory/update', { method: 'PUT', body: { memory_id: memoryId, text } })
}

/** Delete a memory. */
export async function deleteMemory(memoryId) {
  return request('/memory/delete', { method: 'DELETE', body: { memory_id: memoryId } })
}

// ===========================================================================
// GRAPH
// ===========================================================================

/** Get complete knowledge graph (nodes + links). */
export async function getFullGraph() {
  return request('/graph/full', { body: {} })
}

/** Get graph neighbors for an entity (BFS traversal). */
export async function getNeighbors(entityName, depth = 2) {
  return request('/graph/neighbors', { body: { entity_name: entityName, depth } })
}

/** List all entities. */
export async function listEntities() {
  return request('/graph/entities', { body: {} })
}

// ===========================================================================
// STATS & LIFECYCLE
// ===========================================================================

/** Get summary statistics. */
export async function getStats() {
  return request('/stats', { body: {} })
}

/** Run lifecycle maintenance (decay + consolidation + cleanup). */
export async function runLifecycle() {
  return request('/lifecycle/run', { body: {} })
}

/** Health check (unauthenticated). */
export async function healthCheck() {
  const resp = await fetch(`${API_BASE}/health`)
  return resp.json()
}
