/**
 * API client for DeepContext backend.
 * Connects to FastAPI server at configurable base URL.
 */

const API_BASE = 'http://localhost:8000'

async function request(path, body) {
  const resp = await fetch(`${API_BASE}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!resp.ok) {
    const err = await resp.json().catch(() => ({ detail: resp.statusText }))
    throw new Error(err.detail || `HTTP ${resp.status}`)
  }
  return resp.json()
}

/** Extract & store memories from conversation messages. */
export async function addMemory(messages, userId, conversationId = 'default') {
  return request('/memory/add', { messages, user_id: userId, conversation_id: conversationId })
}

/** Hybrid search for memories. */
export async function searchMemory(query, userId, options = {}) {
  return request('/memory/search', { query, user_id: userId, ...options })
}

/** List all memories with optional filtering & pagination. */
export async function listMemories(userId, options = {}) {
  return request('/memory/list', { user_id: userId, ...options })
}

/** Get complete knowledge graph (nodes + links) for a user. */
export async function getFullGraph(userId) {
  return request('/graph/full', { user_id: userId })
}

/** Get graph neighbors for an entity (BFS traversal). */
export async function getNeighbors(userId, entityName, depth = 2) {
  return request('/graph/neighbors', { user_id: userId, entity_name: entityName, depth })
}

/** List all entities for a user. */
export async function listEntities(userId) {
  return request('/graph/entities', { user_id: userId })
}

/** Get summary statistics for a user. */
export async function getStats(userId) {
  return request('/stats', { user_id: userId })
}

/** Run lifecycle maintenance (decay + consolidation + cleanup). */
export async function runLifecycle(userId) {
  return request('/lifecycle/run', { user_id: userId })
}

/** Health check. */
export async function healthCheck() {
  const resp = await fetch(`${API_BASE}/health`)
  return resp.json()
}
