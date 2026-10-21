/**
 * Meeting audio -> live transcript events { event: 'partial' | 'final', segment, text, audio_s }.
 *
 * Three ways in, all local:
 *  - streamToMudraASR(track): the track's audio as 16 kHz PCM chunks (100 ms) over one WebSocket to the backend's
 *    /asr (sherpa-onnx). Only the new chunk is sent each time, never the whole recording.
 *  - listenToRoom(room): the events of a caption room whose audio is captured elsewhere (the Chrome extension's
 *    offscreen page captures the meeting tab and streams it to /asr?room=...).
 *  - chromeOnDevice(track): Chrome's own on-device recogniser (SpeechRecognition with processLocally and a
 *    MediaStreamTrack; Chrome 140+). Chrome downloads its English pack once (install()); audio stays on the PC.
 * Each returns { stop() } and calls onEvent / onStatus.
 */

const WORKLET = `
class Pcm16 extends AudioWorkletProcessor {
  constructor() { super(); this.buf = new Int16Array(1600); this.n = 0 }
  process(inputs) {
    const ch = inputs[0] && inputs[0][0]
    if (ch) for (let i = 0; i < ch.length; i++) {
      const v = Math.max(-1, Math.min(1, ch[i]))
      this.buf[this.n++] = v < 0 ? v * 32768 : v * 32767
      if (this.n === this.buf.length) { this.port.postMessage(this.buf.buffer, [this.buf.buffer]); this.buf = new Int16Array(1600); this.n = 0 }
    }
    return true
  }
}
registerProcessor('mudra-pcm16', Pcm16)`

/** 16 kHz mono PCM chunks of a track (Chrome resamples into a 16 kHz AudioContext). */
export async function pcmChunks(track, onChunk) {
  const ctx = new AudioContext({ sampleRate: 16000 })
  const url = URL.createObjectURL(new Blob([WORKLET], { type: 'application/javascript' }))
  await ctx.audioWorklet.addModule(url)
  URL.revokeObjectURL(url)
  const src = ctx.createMediaStreamSource(new MediaStream([track]))
  const node = new AudioWorkletNode(ctx, 'mudra-pcm16')
  node.port.onmessage = (e) => onChunk(e.data)
  src.connect(node)
  // the worklet must be pulled by the graph; a muted gain keeps it running without playing anything
  const mute = ctx.createGain()
  mute.gain.value = 0
  node.connect(mute).connect(ctx.destination)
  return { context: ctx, stop: () => { src.disconnect(); node.disconnect(); ctx.close() } }
}

export function wsUrl(apiBase, path) {
  return apiBase.replace(/^http/, 'ws') + path
}

function socketEvents(ws, onEvent, onStatus, label) {
  ws.onmessage = (m) => {
    let ev
    try { ev = JSON.parse(m.data) } catch { return }
    ev.t_client = performance.now()
    if (ev.event === 'error') onStatus?.({ state: 'error', error: ev.detail })
    else if (ev.event === 'ready' || ev.event === 'listening') onStatus?.({ state: 'listening', engine: ev.engine || label })
    else onEvent?.(ev)
  }
  ws.onerror = () => onStatus?.({ state: 'error', error: 'Lost the connection to the MUDRA speech server.' })
}

export async function streamToMudraASR(track, { apiBase, room, onEvent, onStatus, WebSocketImpl = WebSocket } = {}) {
  const ws = new WebSocketImpl(wsUrl(apiBase, `/asr${room ? `?room=${encodeURIComponent(room)}` : ''}`))
  ws.binaryType = 'arraybuffer'
  socketEvents(ws, onEvent, onStatus, 'sherpa-onnx')
  const startedAt = performance.now()
  let sent = 0
  await new Promise((resolve, reject) => { ws.onopen = resolve; ws.onclose = () => reject(new Error('The speech server refused the connection.')) })
  ws.onclose = () => onStatus?.({ state: 'stopped' })
  const pcm = await pcmChunks(track, (buf) => {
    if (ws.readyState === 1) { ws.send(buf); sent += buf.byteLength / 2 }
  })
  return {
    startedAt,
    get audioSeconds() { return sent / 16000 },
    stop() {
      pcm.stop()
      if (ws.readyState === 1) { ws.send(JSON.stringify({ type: 'stop' })); setTimeout(() => ws.close(), 1500) }
    },
  }
}

export function listenToRoom(room, { apiBase, onEvent, onStatus, WebSocketImpl = WebSocket } = {}) {
  const ws = new WebSocketImpl(wsUrl(apiBase, `/captions?room=${encodeURIComponent(room)}`))
  socketEvents(ws, onEvent, onStatus, 'extension')
  ws.onclose = () => onStatus?.({ state: 'stopped' })
  return { stop: () => ws.close() }
}

const SR = () => (typeof window !== 'undefined' ? window.SpeechRecognition || window.webkitSpeechRecognition : null)

/** 'available' | 'downloadable' | 'downloading' | 'unavailable' | 'unsupported' (on-device English). */
export async function chromeOnDeviceStatus(lang = 'en-US') {
  const Rec = SR()
  if (!Rec || typeof Rec.available !== 'function') return 'unsupported'
  try { return await Rec.available({ langs: [lang], processLocally: true }) } catch { return 'unsupported' }
}

/** Must be called from a click: Chrome downloads the on-device language pack (once). */
export async function installChromeOnDevice(lang = 'en-US') {
  const Rec = SR()
  if (!Rec?.install) return false
  return Rec.install({ langs: [lang], processLocally: true })
}

export function chromeOnDevice(track, { lang = 'en-US', onEvent, onStatus } = {}) {
  const Rec = SR()
  if (!Rec) throw new Error('This browser has no speech recognition.')
  let running = true
  let session = 0
  let rec = null
  const startedAt = performance.now()
  const begin = () => {
    session++
    rec = new Rec()
    rec.lang = lang
    rec.continuous = true
    rec.interimResults = true
    rec.processLocally = true // never sends the meeting audio to a server
    rec.onstart = () => onStatus?.({ state: 'listening', engine: 'chrome-on-device' })
    rec.onerror = (e) => { if (e.error !== 'no-speech' && e.error !== 'aborted') onStatus?.({ state: 'error', error: `Chrome speech: ${e.error}` }) }
    rec.onresult = (e) => {
      for (let i = e.resultIndex; i < e.results.length; i++) {
        const r = e.results[i]
        onEvent?.({ event: r.isFinal ? 'final' : 'partial', segment: `c${session}-${i}`, text: r[0].transcript,
          audio_s: (performance.now() - startedAt) / 1000, t_client: performance.now() })
      }
    }
    rec.onend = () => { if (running) begin() } // continuous recognition stops by itself now and then
    rec.start(track)
  }
  begin()
  return { startedAt, stop() { running = false; try { rec?.abort() } catch { /* already ended */ } } }
}
