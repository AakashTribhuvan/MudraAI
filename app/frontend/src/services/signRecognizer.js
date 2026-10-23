/**
 * MUDRA Sign Recognizer client
 *
 * Streams every tracked frame's landmarks to the backend over a WebSocket (/stream).
 * The server detects when a sign starts and ends (resting -> signing -> ended) and
 * classifies the whole sign once, so a word is never committed from one or two frames.
 * Only landmark coordinates are sent, never camera images.
 *
 * Configure with VITE_API_URL (backend address) and VITE_SIGN_MODEL=off to disable.
 */

export const API_BASE_URL = import.meta.env?.VITE_API_URL || 'http://localhost:8000'
const MODE = import.meta.env?.VITE_SIGN_MODEL || 'auto'
const WS_URL = API_BASE_URL.replace(/^http/, 'ws') + '/stream'
const HEALTH_TIMEOUT_MS = 2500

const round = (v) => Math.round(v * 1e4) / 1e4

export function toFrame(results, t) {
  const hands = (results.multiHandLandmarks || []).map((landmarks, i) => ({
    landmarks: landmarks.map((p) => [round(p.x), round(p.y), round(p.z ?? 0)]),
    handedness: results.multiHandedness?.[i]?.label ?? null,
    score: results.multiHandedness?.[i]?.score ?? 1
  }))
  const pose = results.poseLandmarks
    ? results.poseLandmarks.map((p) => [round(p.x), round(p.y), round(p.z ?? 0), round(p.visibility ?? 1)])
    : null
  return { type: 'frame', t, hands, pose }
}

class SignRecognizer {
  constructor() {
    this.status = MODE === 'off' ? 'disabled' : 'idle' // idle | checking | online | offline | disabled
    this.info = null
    this.error = null
    this.ws = null
    this.started = false
    this.onPrediction = null // whole-sign result when a sign ends
    this.onActivity = null // 'signing' when a sign starts, 'ended' when it is classified, 'idle' when it was too short
    this.onProvisional = null // live best guess while signing (display only)
    this.onLetter = null // a fingerspelled letter (model "auto": words and letters together)
    this.onStatusChange = null
    this.model = 'sequence' // GET /models id; sent with every session start
  }

  /** Choose the word model for the next session (GET /models ids, e.g. "sequence", "intro_static"). */
  setModel(model) {
    if (typeof model !== 'string' || !/^[a-z0-9_]+$/.test(model)) throw new Error(`Unknown recognition model: ${model}`)
    if (this.model === model) return
    this.model = model
    this.reset() // the next frame starts a new session with the chosen model
  }

  _setStatus(status, error = null) {
    this.status = status
    this.error = error
    this.onStatusChange?.({ status, info: this.info, error })
  }

  /** Checks the backend; call again to retry after starting it. */
  async connect() {
    if (MODE === 'off') {
      this._setStatus('disabled')
      return this.status
    }
    this._setStatus('checking')
    const controller = new AbortController()
    const timer = setTimeout(() => controller.abort(), HEALTH_TIMEOUT_MS)
    try {
      const health = await (await fetch(`${API_BASE_URL}/health`, { signal: controller.signal })).json()
      this.info = health
      if (health.model_loaded) this._setStatus('online')
      else this._setStatus('offline', health.error || 'The backend is running but no model is loaded.')
    } catch {
      this._setStatus('offline', `Recognition server not reachable at ${API_BASE_URL}. Start it with start_all.bat.`)
    } finally {
      clearTimeout(timer)
    }
    return this.status
  }

  _open(width, height) {
    const ws = new WebSocket(WS_URL)
    this.ws = ws
    this.started = false
    ws.onopen = () => {
      ws.send(JSON.stringify({ type: 'start', width, height, model: this.model }))
      this.started = true
    }
    ws.onmessage = (msg) => {
      const ev = JSON.parse(msg.data)
      if (ev.event === 'sign_started') this.onActivity?.('signing')
      else if (ev.event === 'provisional') this.onProvisional?.(ev.guess)
      else if (ev.event === 'letter') this.onLetter?.(ev)
      else if (ev.event === 'too_short') this.onActivity?.('idle')
      else if (ev.event === 'sign_ended') {
        this.onActivity?.('ended')
        this.onPrediction?.({ ...ev.result, stats: ev.stats })
      } else if (ev.event === 'error') this._setStatus('offline', ev.detail)
    }
    ws.onclose = () => {
      if (this.ws === ws) this.ws = null
    }
    ws.onerror = () => this._setStatus('offline', `Lost connection to ${WS_URL}.`)
  }

  /** Feed every tracker result. */
  addFrame(results) {
    if (this.status !== 'online') return
    if (!this.ws) this._open(results.width || 640, results.height || 480)
    if (!this.started || this.ws.readyState !== WebSocket.OPEN) return
    const now = performance.now()
    if (results.diag && now - (this.diagSentAt || 0) > 3000) {
      // Tracking diagnostics, kept with a saved session (MUDRA_SESSION_DIR) to explain low frame rates
      this.diagSentAt = now
      const d = results.diag
      this.ws.send(JSON.stringify({
        type: 'diag', camera_fps: d.cameraFps, tracking_fps: d.trackingFps, inference_ms: d.inferenceMs, delegate: d.delegate,
        gpu: d.gpu, integrated_gpu: d.integratedGpu, hand_miss: d.handMissRate,
        camera: d.camera, width: results.width, height: results.height
      }))
    }
    this.ws.send(JSON.stringify(toFrame(results, now / 1000)))
  }

  /** Ends the live session (camera stopped). */
  reset() {
    this.ws?.close()
    this.ws = null
    this.started = false
  }
}

export const signRecognizer = new SignRecognizer()
export default signRecognizer
