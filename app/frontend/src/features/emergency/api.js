import { API_BASE_URL } from '../../services/signRecognizer'

/**
 * SignLine Emergency API client (/api/v1/emergency).
 *
 * Emergency users stay anonymous: this browser gets a random device key (kept in localStorage) that owns
 * its gestures and preferences. Each session returns an access token, kept in sessionStorage only for
 * this tab. Responder tokens are kept in sessionStorage too (they end when the tab closes).
 */
const BASE = `${API_BASE_URL}/api/v1/emergency`
const DEVICE_KEY = 'signline_em_device'
const SESSIONS_KEY = 'signline_em_sessions'
const RESPONDER_KEY = 'signline_em_responder'

function safeGet(storage, key) {
  try { return storage.getItem(key) } catch { return null }
}
function safeSet(storage, key, value) {
  try { value == null ? storage.removeItem(key) : storage.setItem(key, value) } catch { /* private mode: keep in memory only */ }
}

let memoryDevice = null
export function deviceKey() {
  let key = safeGet(localStorage, DEVICE_KEY) || memoryDevice
  if (!key) {
    const bytes = new Uint8Array(18)
    crypto.getRandomValues(bytes)
    key = `dev-${Array.from(bytes, (b) => b.toString(16).padStart(2, '0')).join('')}`
    memoryDevice = key
    safeSet(localStorage, DEVICE_KEY, key)
  }
  return key
}

const memorySessions = {}
function sessions() {
  try { return { ...memorySessions, ...JSON.parse(safeGet(sessionStorage, SESSIONS_KEY) || '{}') } } catch { return { ...memorySessions } }
}
export function rememberSession(id, token) {
  memorySessions[id] = token
  safeSet(sessionStorage, SESSIONS_KEY, JSON.stringify({ ...sessions(), [id]: token }))
  safeSet(sessionStorage, `${SESSIONS_KEY}_last`, id)
}
export function lastSessionId() {
  const id = safeGet(sessionStorage, `${SESSIONS_KEY}_last`)
  return id && sessions()[id] ? id : null
}
export function forgetLastSession() {
  safeSet(sessionStorage, `${SESSIONS_KEY}_last`, null)
}

let memoryResponder = null
export function responderAuth() {
  try { return JSON.parse(safeGet(sessionStorage, RESPONDER_KEY) || 'null') || memoryResponder } catch { return memoryResponder }
}
export function setResponderAuth(auth) {
  memoryResponder = auth
  safeSet(sessionStorage, RESPONDER_KEY, auth ? JSON.stringify(auth) : null)
}

export class ApiError extends Error {
  constructor(status, message) {
    super(message)
    this.status = status
  }
}

async function request(path, { method = 'GET', body, headers = {} } = {}) {
  let res
  try {
    res = await fetch(`${BASE}${path}`, {
      method,
      headers: { ...(body !== undefined ? { 'Content-Type': 'application/json' } : {}), ...headers },
      body: body !== undefined ? JSON.stringify(body) : undefined,
    })
  } catch {
    throw new ApiError(0, 'Cannot reach the SignLine server. Check your connection. In a real emergency in India, call 112.')
  }
  const data = await res.json().catch(() => null)
  if (!res.ok) {
    const detail = data?.detail
    throw new ApiError(res.status, typeof detail === 'string' ? detail : Array.isArray(detail) ? detail[0]?.msg : `Request failed (${res.status})`)
  }
  return data
}

const device = () => ({ 'X-Device-Key': deviceKey() })
const sessionHeaders = (id) => ({ 'X-Session-Token': sessions()[id] || '' })
const responderHeaders = () => ({ Authorization: `Bearer ${responderAuth()?.token || ''}` })

export const emergencyApi = {
  config: () => request('/config'),

  // sessions
  async createSession(mode, scenarioId = null) {
    const data = await request('/sessions', { method: 'POST', body: { mode, scenario_id: scenarioId }, headers: device() })
    rememberSession(data.session.id, data.access_token)
    return data.session
  },
  hasSession: (id) => Boolean(sessions()[id]),
  session: (id) => request(`/sessions/${id}`, { headers: sessionHeaders(id) }),
  act: (id, path, body) => request(`/sessions/${id}${path}`, { method: 'POST', body, headers: sessionHeaders(id) }),
  setLocation: (id, body) => request(`/sessions/${id}/location`, { method: 'PUT', body, headers: sessionHeaders(id) }),
  clearLocation: (id) => request(`/sessions/${id}/location`, { method: 'DELETE', headers: sessionHeaders(id) }),

  // gestures and preferences (this device)
  gestures: () => request('/gestures', { headers: device() }),
  createGesture: (body) => request('/gestures', { method: 'POST', body, headers: device() }),
  updateGesture: (id, body) => request(`/gestures/${id}`, { method: 'PATCH', body, headers: device() }),
  deleteGesture: (id) => request(`/gestures/${id}`, { method: 'DELETE', headers: device() }),
  testGesture: (id) => request(`/gestures/${id}/test`, { method: 'POST', headers: device() }),
  gestureReport: (id, body = {}) => request(`/gestures/${id}/report`, { method: 'POST', body, headers: device() }),
  contacts: (region) => request(`/contacts${region ? `?region=${encodeURIComponent(region)}` : ''}`),
  deleteGestureVideo: (id) => request(`/gestures/${id}/video`, { method: 'DELETE', headers: device() }),
  /** Upload a recorded demonstration (only with the user's recording consent). */
  async uploadGestureVideo(id, blob, durationS) {
    const form = new FormData()
    const ext = blob.type.includes('mp4') ? 'mp4' : 'webm'
    form.append('file', blob, `gesture.${ext}`)
    form.append('duration_s', String(durationS))
    form.append('recording_consent', 'true')
    let res
    try {
      res = await fetch(`${BASE}/gestures/${id}/video`, { method: 'PUT', headers: device(), body: form })
    } catch {
      throw new ApiError(0, 'The video could not be uploaded: cannot reach the SignLine server. The gesture is saved without it.')
    }
    const data = await res.json().catch(() => null)
    if (!res.ok) throw new ApiError(res.status, data?.detail || `Upload failed (${res.status})`)
    return data
  },
  /** The gesture's video as a Blob (the device key is sent as a header, so it cannot be a plain URL). */
  async gestureVideo(id) {
    let res
    try {
      res = await fetch(`${BASE}/gestures/${id}/video`, { headers: device() })
    } catch {
      throw new ApiError(0, 'The video could not be loaded: cannot reach the SignLine server.')
    }
    if (!res.ok) throw new ApiError(res.status, res.status === 404 ? 'No video saved for this gesture.' : `Video unavailable (${res.status})`)
    return res.blob()
  },
  /** Prepare a report from a gesture for the responder desk (up to "ready to share"; nothing is shared yet). */
  async requestAssistance(id, body) {
    const data = await request(`/gestures/${id}/request-assistance`, { method: 'POST', body, headers: device() })
    rememberSession(data.session.id, data.access_token)
    return data
  },
  preferences: () => request('/preferences', { headers: device() }),
  savePreferences: (body) => request('/preferences', { method: 'PUT', body, headers: device() }),

  // responders
  async login(username, password) {
    const data = await request('/responder/login', { method: 'POST', body: { username, password } })
    setResponderAuth({ token: data.token, user: data.user, expiresAt: Date.now() + data.expires_in * 1000 })
    return data.user
  },
  me: () => request('/responder/me', { headers: responderHeaders() }),
  incidents: (status) => request(`/responder/incidents${status ? `?status=${encodeURIComponent(status)}` : ''}`, { headers: responderHeaders() }),
  incident: (id) => request(`/responder/incidents/${id}`, { headers: responderHeaders() }),
  acknowledge: (id) => request(`/responder/incidents/${id}/acknowledge`, { method: 'POST', headers: responderHeaders() }),
  resolve: (id) => request(`/responder/incidents/${id}/resolve`, { method: 'POST', headers: responderHeaders() }),
  /** The video the user shared with a delivered report (responders only), as a Blob. */
  async incidentVideo(id) {
    let res
    try {
      res = await fetch(`${BASE}/responder/incidents/${id}/video`, { headers: responderHeaders() })
    } catch {
      throw new ApiError(0, 'The video could not be loaded: cannot reach the SignLine server.')
    }
    if (!res.ok) throw new ApiError(res.status, res.status === 404 ? 'No video was shared with this report.' : `Video unavailable (${res.status})`)
    return res.blob()
  },
  addNote: (id, text) => request(`/responder/incidents/${id}/notes`, { method: 'POST', body: { text }, headers: responderHeaders() }),
  analytics: () => request('/responder/analytics', { headers: responderHeaders() }),
}

/** Readable labels for provenance values (never colour alone). */
export const PROVENANCE = {
  USER_REPORTED: { label: 'Said by the user', short: 'User', cls: 'bg-sl-sky-100 text-sl-sky-700 border-sl-sky-700/30' },
  USER_CONFIRMED: { label: 'Confirmed by the user', short: 'Confirmed', cls: 'bg-sl-green-100 text-sl-green-700 border-sl-green-700/30' },
  AI_INFERRED: { label: 'System interpretation', short: 'Inferred', cls: 'bg-sl-sun-100 text-sl-sun-700 border-sl-sun-700/30' },
  SYSTEM_GENERATED: { label: 'Generated by the system', short: 'System', cls: 'bg-sl-cream-200 text-sl-ink-600 border-sl-ink-200' },
}

export const EMERGENCY_NUMBER = '112'
