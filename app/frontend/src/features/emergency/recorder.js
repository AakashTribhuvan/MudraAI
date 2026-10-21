/**
 * Records a short gesture demonstration with the camera (getUserMedia + MediaRecorder).
 *
 * States: idle -> requesting -> preview -> recording -> recorded (retake -> preview), or unsupported / denied /
 * error. The camera is asked for only when requestCamera() is called (the user pressed "Record gesture"),
 * nothing records until start(), and recording stops by itself after MAX_SECONDS. The browser APIs are passed
 * in, so the flow is testable without a camera.
 */
export const MAX_SECONDS = 20
export const MIN_SECONDS = 1
const TYPES = ['video/webm;codecs=vp9', 'video/webm;codecs=vp8', 'video/webm', 'video/mp4']

export function pickMimeType(MediaRecorderImpl) {
  if (!MediaRecorderImpl) return null
  if (typeof MediaRecorderImpl.isTypeSupported !== 'function') return ''
  return TYPES.find((t) => MediaRecorderImpl.isTypeSupported(t)) ?? ''
}

export function cameraErrorKind(err) {
  const name = err?.name || ''
  if (name === 'NotAllowedError' || name === 'SecurityError' || name === 'PermissionDeniedError') return 'denied'
  if (name === 'NotFoundError' || name === 'OverconstrainedError' || name === 'DevicesNotFoundError') return 'no_camera'
  if (name === 'NotReadableError' || name === 'TrackStartError' || name === 'AbortError') return 'busy'
  return 'error'
}

export const RECORDER_MESSAGES = {
  unsupported: 'This browser cannot record video. You can still save the gesture without a recording.',
  denied: 'Camera permission was denied. You can allow it in the browser settings, or save the gesture without a recording.',
  no_camera: 'No camera was found. You can save the gesture without a recording.',
  busy: 'The camera is in use by another app or tab. Close it and try again, or save without a recording.',
  error: 'The camera could not be started. You can save the gesture without a recording.',
  too_short: `The recording is too short. Record at least ${MIN_SECONDS} second.`,
  record_failed: 'The recording failed. Try again, or save without a recording.',
}

export function createRecorder({ mediaDevices, MediaRecorderImpl, now = () => Date.now(), setTimer = setTimeout, clearTimer = clearTimeout, onChange = () => {} } = {}) {
  const state = { status: 'idle', stream: null, blob: null, mimeType: null, duration: 0, error: null, startedAt: 0 }
  let recorder = null
  let chunks = []
  let timer = null
  const emit = () => onChange({ ...state })
  const set = (patch) => { Object.assign(state, patch); emit() }
  const supported = () => Boolean(mediaDevices?.getUserMedia) && pickMimeType(MediaRecorderImpl) !== null

  async function requestCamera() {
    if (!supported()) return set({ status: 'unsupported', error: 'unsupported' })
    set({ status: 'requesting', error: null })
    try {
      const stream = await mediaDevices.getUserMedia({ video: { width: { ideal: 640 }, height: { ideal: 480 } }, audio: false })
      set({ status: 'preview', stream })
    } catch (err) {
      set({ status: cameraErrorKind(err) === 'denied' ? 'denied' : 'error', error: cameraErrorKind(err) })
    }
  }

  function start() {
    if (state.status !== 'preview' || !state.stream) return
    const mimeType = pickMimeType(MediaRecorderImpl)
    try {
      recorder = new MediaRecorderImpl(state.stream, mimeType ? { mimeType } : undefined)
    } catch {
      return set({ status: 'error', error: 'record_failed' })
    }
    chunks = []
    recorder.ondataavailable = (e) => { if (e.data && e.data.size) chunks.push(e.data) }
    recorder.onstop = () => {
      const duration = (now() - state.startedAt) / 1000
      const type = (recorder.mimeType || mimeType || 'video/webm').split(';')[0]
      if (!chunks.length) return set({ status: 'preview', error: 'record_failed' })
      if (duration < MIN_SECONDS) return set({ status: 'preview', error: 'too_short' })
      set({ status: 'recorded', blob: new Blob(chunks, { type }), mimeType: type, duration: Math.round(duration * 10) / 10, error: null })
    }
    recorder.start(250)
    set({ status: 'recording', startedAt: now(), blob: null, error: null })
    timer = setTimer(stop, MAX_SECONDS * 1000)
  }

  function stop() {
    if (timer) { clearTimer(timer); timer = null }
    if (state.status === 'recording' && recorder && recorder.state !== 'inactive') recorder.stop()
  }

  function retake() {
    if (state.status !== 'recorded') return
    set({ status: 'preview', blob: null, duration: 0, mimeType: null })
  }

  /** Turn the camera off (after saving, on cancel, when the page closes). */
  function release() {
    stop()
    state.stream?.getTracks?.().forEach((t) => t.stop())
    set({ stream: null, status: state.status === 'recorded' ? 'recorded' : 'idle' })
  }

  return { state, supported, requestCamera, start, stop, retake, release }
}
