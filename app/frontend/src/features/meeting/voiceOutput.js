/**
 * Signed sentence -> speech the OTHER meeting participants hear.
 *
 * The backend makes a WAV with a local Windows voice (POST /tts); an <audio> element plays it on a chosen output
 * device with setSinkId(). With VB-Audio Virtual Cable installed, choose "CABLE Input" here and "CABLE Output" as
 * the microphone in Meet / Zoom: the meeting then sends MUDRA's voice. Playing on the normal speakers is reported
 * as such: the meeting does not hear it.
 *
 * testRoute() checks the route for real: it plays a test phrase into the chosen output and listens on the matching
 * cable recording device ("CABLE Output") for that sound. Echo guard: isEcho() tells whether a phrase heard from the
 * meeting is MUDRA's own recent speech coming back, so it is not signed back to the user.
 */

export const CABLE_INPUT = /cable input|vb-audio virtual cable|virtual cable/i
export const CABLE_OUTPUT = /cable output/i

export async function listAudioDevices(mediaDevices = navigator.mediaDevices) {
  const all = await mediaDevices.enumerateDevices()
  return {
    outputs: all.filter((d) => d.kind === 'audiooutput'),
    inputs: all.filter((d) => d.kind === 'audioinput'),
    labelled: all.some((d) => d.label), // labels appear once the page has camera or microphone permission
  }
}

/**
 * The virtual cable's playback end (MUDRA speaks into it) and recording end (the meeting's microphone).
 * VB-Cable 4.5 names its playback end "Speakers (VB-Audio Virtual Cable)" (older: "CABLE Input") and adds a
 * "CABLE In 16 Ch" variant; the stereo one is preferred. Chrome's "Default - ..." / "Communications - ..." aliases
 * are skipped so the real device id is used.
 */
export function findCable({ outputs, inputs }) {
  const real = (d) => d.deviceId !== 'default' && d.deviceId !== 'communications'
  const cableOuts = outputs.filter((d) => real(d) && CABLE_INPUT.test(d.label || ''))
  return {
    output: cableOuts.find((d) => /cable input/i.test(d.label)) || cableOuts.find((d) => !/16\s*ch/i.test(d.label)) || cableOuts[0] || null,
    input: inputs.filter(real).find((d) => CABLE_OUTPUT.test(d.label || '')) || null,
  }
}

const words = (t) => (t || '').toLowerCase().replace(/[^a-z0-9 ]+/g, ' ').split(/\s+/).filter(Boolean)

export function createVoiceOutput({ apiBase, onStatus, fetchImpl = fetch, AudioImpl = Audio, echoWindowMs = 8000 } = {}) {
  let sink = { id: '', label: 'Default speakers', cable: false }
  let monitor = false
  let queue = Promise.resolve()
  let active = new Set()
  let generation = 0
  const recent = [] // { words, at } of what MUDRA said
  const status = (s) => onStatus?.({ ...s, sink: sink.label, cable: sink.cable })

  const play = async (blob, deviceId) => {
    const url = URL.createObjectURL(blob)
    const el = new AudioImpl(url)
    active.add(el)
    try {
      if (deviceId) {
        if (typeof el.setSinkId !== 'function') throw new Error('This browser cannot choose an audio output (setSinkId).')
        await el.setSinkId(deviceId)
      }
      await el.play()
      await new Promise((resolve, reject) => { el.onended = resolve; el.onerror = () => reject(new Error('Audio playback failed')); el.onpause = resolve })
    } finally {
      active.delete(el)
      URL.revokeObjectURL(url)
    }
  }

  const synth = async (text, signal) => {
    const t0 = performance.now()
    const r = await fetchImpl(`${apiBase}/tts`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ text }), signal,
    })
    if (!r.ok) {
      let detail = `HTTP ${r.status}`
      try { detail = (await r.json()).detail || detail } catch { /* not JSON */ }
      throw new Error(`Speech server: ${detail}`)
    }
    return { blob: await r.blob(), ms: Math.round(performance.now() - t0) }
  }

  return {
    setSink(device) { sink = device ? { id: device.deviceId, label: device.label || 'Selected output', cable: CABLE_INPUT.test(device.label || '') } : { id: '', label: 'Default speakers', cable: false } },
    setMonitor(on) { monitor = Boolean(on) },
    get sink() { return sink },
    /** Speaks after anything already being spoken; resolves when done. */
    speak(text) {
      const gen = generation
      const job = queue.then(async () => {
        if (gen !== generation || !text) return
        status({ state: 'synthesizing', text })
        const { blob, ms } = await synth(text)
        if (gen !== generation) return
        recent.push({ words: words(text), at: performance.now() })
        status({ state: 'playing', text, ttsMs: ms })
        await Promise.all([play(blob, sink.id), monitor && sink.id ? play(blob, '') : null])
        status({ state: sink.cable ? 'sent' : 'local-only', text, ttsMs: ms })
      }).catch((err) => status({ state: 'error', text, error: err.message }))
      queue = job
      return job
    },
    /** True when a phrase heard from the meeting is mostly MUDRA's own recent speech. */
    isEcho(heardWords, now = performance.now()) {
      while (recent.length && now - recent[0].at > echoWindowMs) recent.shift()
      if (!heardWords.length || !recent.length) return false
      const said = new Set(recent.flatMap((r) => r.words))
      return heardWords.filter((w) => said.has(w)).length / heardWords.length >= 0.75
    },
    stop() {
      generation++
      for (const el of active) { try { el.pause() } catch { /* already stopped */ } }
      active = new Set()
      queue = Promise.resolve()
      status({ state: 'idle' })
    },
    /**
     * Plays a test phrase into the chosen output and listens on `listenDevice` (the cable's recording end).
     * Resolves { heard, peak, ttsMs } - heard means the sound really arrived where the meeting microphone listens.
     */
    async testRoute(listenDevice, { mediaDevices = navigator.mediaDevices, text = 'MUDRA voice test. One, two, three.' } = {}) {
      const { blob, ms } = await synth(text)
      if (!listenDevice) {
        await play(blob, sink.id)
        return { heard: null, peak: null, ttsMs: ms, note: 'Played; no virtual cable recording device found to check it.' }
      }
      const stream = await mediaDevices.getUserMedia({ audio: { deviceId: { exact: listenDevice.deviceId }, echoCancellation: false, noiseSuppression: false, autoGainControl: false } })
      const ctx = new AudioContext()
      const analyser = ctx.createAnalyser()
      analyser.fftSize = 2048
      ctx.createMediaStreamSource(stream).connect(analyser)
      const buf = new Float32Array(analyser.fftSize)
      let peak = 0
      let running = true
      const sample = () => {
        if (!running) return
        analyser.getFloatTimeDomainData(buf)
        let sum = 0
        for (const v of buf) sum += v * v
        peak = Math.max(peak, Math.sqrt(sum / buf.length))
        requestAnimationFrame(sample)
      }
      sample()
      try {
        await play(blob, sink.id)
        await new Promise((r) => setTimeout(r, 250))
      } finally {
        running = false
        stream.getTracks().forEach((t) => t.stop())
        ctx.close()
      }
      return { heard: peak > 0.01, peak: Math.round(peak * 1000) / 1000, ttsMs: ms }
    },
  }
}
