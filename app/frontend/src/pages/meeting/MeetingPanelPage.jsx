import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useAvatarRenderer } from '../../avatar/useAvatarRenderer'
import { useAvatarChoice } from '../../avatar/avatarCatalog'
import { API_BASE_URL } from '../../services/signRecognizer'
import { CABLE_INPUT, createVoiceOutput, findCable, listAudioDevices } from '../../features/meeting/voiceOutput'
import {
  chromeOnDevice, chromeOnDeviceStatus, installChromeOnDevice, listenToRoom, streamToMudraASR,
} from '../../features/meeting/speechSources'
import { useIncomingSigns } from '../../features/meeting/useIncomingSigns'
import { useOutgoingSpeech } from '../../features/meeting/useOutgoingSpeech'
import { initMeetAddon, initZoomApp, shareToMainStage, surfaceFromUrl } from '../../features/meeting/clientSurfaces'

const btn = 'rounded-lg border-2 border-sl-ink px-2.5 py-1 text-xs font-bold disabled:opacity-40'
const chip = (tone) => `inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[11px] font-bold ${tone}`
const TONES = { ok: 'bg-emerald-100 text-emerald-900', warn: 'bg-amber-100 text-amber-900', bad: 'bg-rose-100 text-rose-900', off: 'bg-slate-100 text-slate-600', live: 'bg-rose-600 text-white' }
const randomRoom = () => Array.from(crypto.getRandomValues(new Uint8Array(12)), (b) => b.toString(16).padStart(2, '0')).join('')
// renderer speed 1.2 = recorded pace: 1.5 is 1.25x, 2.1 is 1.75x (the most before a sign gets hard to read)
const BASE_SPEED = 1.5
const MAX_SPEED = 2.1
const isExtensionOrigin = (o) => typeof o === 'string' && o.startsWith('chrome-extension://')

/**
 * MUDRA meeting panel (route /meet-panel): the ISL avatar and live caption for what others say, and the user's
 * signs spoken into the meeting. Runs inside the Chrome extension's overlay on Google Meet / Zoom Web (?room=...:
 * the extension captures the tab audio), as a pop-out window, or on its own (share a meeting tab or the screen
 * with audio - also for the Zoom desktop app). Everything runs on this computer: local speech recognition, local
 * voices, nothing recorded or stored.
 */
export default function MeetingPanelPage() {
  const params = useMemo(() => new URLSearchParams(window.location.search), [])
  const surface = surfaceFromUrl()
  const mainStage = surface === 'meet-mainstage' // shown to everyone in the call: avatar and captions only
  const [roomFromExtension, setRoomFromExtension] = useState(params.get('room'))
  const embedded = window.parent !== window
  const testMode = import.meta.env.DEV || params.get('test') === '1'
  const hostRef = useRef(null)
  const videoRef = useRef(null)
  const { avatar } = useAvatarChoice()
  const [avatarReady, setAvatarReady] = useState(false)
  const [speed, setSpeed] = useState(BASE_SPEED)
  const renderer = useAvatarRenderer({
    containerRef: hostRef, modelPath: avatar.url, modelType: avatar.type, animationSpeed: speed, pauseDuration: 200,
    onLoadingChange: (loading) => setAvatarReady(!loading),
  })

  // ---- server and devices
  const [server, setServer] = useState({ state: 'checking' })
  useEffect(() => {
    fetch(`${API_BASE_URL}/speech/status`).then((r) => r.json())
      .then((s) => setServer({ state: 'online', ...s })).catch(() => setServer({ state: 'offline' }))
  }, [])
  const [devices, setDevices] = useState({ outputs: [], inputs: [], labelled: false })
  const refreshDevices = useCallback(() => listAudioDevices().then(setDevices).catch(() => {}), [])
  useEffect(() => { refreshDevices(); navigator.mediaDevices?.addEventListener?.('devicechange', refreshDevices) }, [refreshDevices])
  const cable = findCable(devices)

  // ---- voice out
  const [voiceStatus, setVoiceStatus] = useState({ state: 'idle' })
  const voice = useMemo(() => createVoiceOutput({ apiBase: API_BASE_URL, onStatus: setVoiceStatus }), [])
  const [sinkId, setSinkId] = useState('')
  useEffect(() => { if (!sinkId && cable.output) setSinkId(cable.output.deviceId) }, [cable.output, sinkId])
  useEffect(() => { voice.setSink(devices.outputs.find((d) => d.deviceId === sinkId) || null) }, [sinkId, devices, voice])
  const [routeTest, setRouteTest] = useState(null)
  const testRoute = async () => {
    setRouteTest({ state: 'testing' })
    try {
      const r = await voice.testRoute(cable.input)
      setRouteTest({ state: 'done', ...r })
    } catch (err) {
      setRouteTest({ state: 'error', error: err.message })
    }
  }

  // ---- incoming
  const incoming = useIncomingSigns({ avatar: renderer, isEcho: (w) => voice.isEcho(w) })
  // phrases waiting: sign a little faster to catch up, never beyond MAX_SPEED (signs must stay readable)
  useEffect(() => { setSpeed(Math.min(MAX_SPEED, BASE_SPEED + 0.15 * incoming.backlog)) }, [incoming.backlog])
  const [listen, setListen] = useState({ state: 'off' })
  const [source, setSource] = useState(roomFromExtension ? 'extension' : surface ? 'local-room' : 'share')
  const [engine, setEngine] = useState('mudra')
  const [chromeStatus, setChromeStatus] = useState(null)
  useEffect(() => { chromeOnDeviceStatus().then(setChromeStatus) }, [])
  const captureRef = useRef(null) // { stop() } of the source and the captured stream

  const stopListening = useCallback((note) => {
    captureRef.current?.forEach((s) => s?.stop?.())
    captureRef.current = null
    incoming.stop()
    setListen({ state: 'off', note })
    if (embedded) window.parent.postMessage({ type: 'mudra:panel-stopped-listening' }, '*')
  }, [incoming, embedded])

  const startListening = async () => {
    setListen({ state: 'starting' })
    try {
      const onEvent = await incoming.start()
      const onStatus = (s) => setListen((l) => ({ ...l, ...s, state: s.state === 'stopped' ? 'off' : s.state }))
      if (source === 'extension' || source === 'local-room') {
        let room = roomFromExtension
        if (source === 'local-room') { // the newest capture running on this computer (the MUDRA extension)
          const active = await fetch(`${API_BASE_URL}/captions/active`).then((r) => r.json()).catch(() => ({ rooms: [] }))
          room = active.rooms?.[0]?.room
          if (!room) throw new Error('No MUDRA capture is running on this computer. Click the MUDRA toolbar button on the meeting tab first.')
          setRoomFromExtension(room)
        }
        captureRef.current = [listenToRoom(room, { apiBase: API_BASE_URL, onEvent, onStatus })]
        setListen({ state: 'listening', engine: 'sherpa-onnx', what: 'the meeting tab (captured by the MUDRA extension)' })
        return
      }
      let stream
      if (source === 'mic-test') {
        stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false } })
      } else {
        // the person picks the meeting tab ("Also share tab audio") or the screen with system audio (Zoom app)
        stream = await navigator.mediaDevices.getDisplayMedia({ video: true, audio: { suppressLocalAudioPlayback: false }, systemAudio: 'include', selfBrowserSurface: 'exclude' })
        stream.getVideoTracks().forEach((t) => t.stop()) // only the sound is used
      }
      const track = stream.getAudioTracks()[0]
      if (!track) {
        stream.getTracks().forEach((t) => t.stop())
        throw new Error('No audio was shared. Choose the meeting tab and tick "Also share tab audio" (or "Share system audio").')
      }
      track.onended = () => stopListening('The shared audio ended.')
      const src = engine === 'chrome'
        ? chromeOnDevice(track, { onEvent, onStatus })
        : await streamToMudraASR(track, { apiBase: API_BASE_URL, room: randomRoom(), onEvent, onStatus })
      captureRef.current = [src, { stop: () => stream.getTracks().forEach((t) => t.stop()) }]
      setListen({ state: 'listening', engine: engine === 'chrome' ? 'Chrome on-device' : 'sherpa-onnx', what: source === 'mic-test' ? 'the microphone (testing only)' : track.label || 'shared audio' })
    } catch (err) {
      incoming.stop()
      setListen({ state: 'error', error: err?.name === 'NotAllowedError' ? 'Sharing was cancelled.' : err.message })
    }
  }

  // the extension starts capture from its toolbar button; the panel follows its state
  useEffect(() => {
    if (!embedded) return undefined
    const onMessage = (e) => {
      if (!isExtensionOrigin(e.origin) || !e.data || typeof e.data !== 'object') return
      if (e.data.type === 'mudra:capture-stopped') stopListening('The extension stopped capturing the tab.')
    }
    window.addEventListener('message', onMessage)
    window.parent.postMessage({ type: 'mudra:panel-ready' }, '*')
    return () => window.removeEventListener('message', onMessage)
  }, [embedded, stopListening])
  useEffect(() => { if (roomFromExtension && avatarReady && listen.state === 'off' && !listen.note) startListening() }, [avatarReady, roomFromExtension]) // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => { if (roomFromExtension && source === 'local-room') setSource('extension') }, [roomFromExtension]) // eslint-disable-line react-hooks/exhaustive-deps

  // ---- in-client surfaces (Meet add-on, Zoom App)
  const [client, setClient] = useState(null)
  useEffect(() => {
    if (!surface) return
    const init = surface === 'zoom' ? initZoomApp() : initMeetAddon(surface)
    init.then((r) => {
      setClient(r)
      if (r.startingRoom && !roomFromExtension) setRoomFromExtension(r.startingRoom)
    })
  }, [surface]) // eslint-disable-line react-hooks/exhaustive-deps
  const surfaceNote = !surface ? null : client?.error ? client.error
    : surface === 'zoom' ? (client ? `Zoom App · ${client.context}${client.meeting?.meetingTopic ? ` · ${client.meeting.meetingTopic}` : ''}` : 'Connecting to Zoom…')
      : client ? (mainStage ? 'Google Meet · main stage' : 'Google Meet · side panel') : 'Connecting to Google Meet…'

  // ---- outgoing
  const outgoing = useOutgoingSpeech({ voice })
  useEffect(() => () => { voice.stop(); captureRef.current?.forEach((s) => s?.stop?.()) }, [voice])

  const sink = devices.outputs.find((d) => d.deviceId === sinkId)
  const routedToCable = Boolean(sink && CABLE_INPUT.test(sink.label))
  const popOut = () => window.open(window.location.href, 'mudra-meeting-panel', 'width=400,height=760')
  const stopAll = () => { stopListening(); outgoing.stop(); voice.stop(); if (embedded) window.parent.postMessage({ type: 'mudra:stop-capture' }, '*') }

  return (
    <div className="flex min-h-screen flex-col gap-2 bg-white p-2 text-sl-ink" style={{ fontSize: 13 }}>
      <header className="flex flex-wrap items-center gap-1.5">
        {!embedded && <strong className="mr-1 font-display text-sm">MUDRA</strong>}
        <span className={chip(server.state === 'online' ? TONES.ok : server.state === 'offline' ? TONES.bad : TONES.off)} title="Local MUDRA server">
          {server.state === 'online' ? 'Server on' : server.state === 'offline' ? 'Server off' : '...'}
        </span>
        <span className={chip(listen.state === 'listening' ? TONES.live : listen.state === 'error' ? TONES.bad : TONES.off)} role="status">
          {listen.state === 'listening' ? '● Listening to meeting' : listen.state === 'starting' ? 'Starting…' : 'Not listening'}
        </span>
        <span className={chip(outgoing.active ? TONES.live : TONES.off)}>{outgoing.active ? '● Camera on' : 'Camera off'}</span>
        <span className={chip(routedToCable ? TONES.ok : TONES.warn)} title="Where MUDRA's voice goes">
          {routedToCable ? 'Voice → meeting mic' : 'Voice → speakers only'}
        </span>
        {(listen.state === 'listening' || outgoing.active) && <button type="button" className={`${btn} ml-auto bg-rose-600 text-white`} onClick={stopAll}>Stop all</button>}
      </header>

      {/* incoming: avatar + caption */}
      <section aria-label="Incoming speech in ISL" className="rounded-xl border-2 border-sl-ink">
        <div ref={hostRef} className="relative h-[230px] w-full overflow-hidden rounded-t-[10px] bg-sl-cream-100" data-testid="avatar">
          {!avatarReady && <p className="absolute inset-0 grid place-items-center text-xs">Loading the avatar…</p>}
        </div>
        <div className="space-y-1 p-2">
          <p className="min-h-[2.5rem] text-sm leading-snug" aria-live="polite" data-testid="caption">
            <span>{incoming.caption.committed}</span>{' '}
            <span className="italic text-slate-500">{incoming.caption.provisional}</span>
            {!incoming.caption.committed && !incoming.caption.provisional && <span className="text-slate-500">Captions of what others say appear here.</span>}
          </p>
          <p className="text-[11px] text-slate-600" data-testid="sign-status">
            {incoming.signing ? <>Signing: <strong>{incoming.signing}</strong></> : 'Avatar idle'}
            {incoming.backlog > 0 && <> · {incoming.backlog} waiting</>}
            {incoming.lastLatency && <> · last sign {incoming.lastLatency.partialToAvatar} ms after it was heard</>}
          </p>
          {incoming.unsigned.length > 0 && <p className="text-[11px] text-amber-800">No ISL sign in MUDRA's vocabulary yet (caption only): {[...new Set(incoming.unsigned)].join(', ')}</p>}
          {incoming.skipped.length > 0 && <p className="text-[11px] text-amber-800">Skipped to keep up with the speaker: {incoming.skipped.join(', ')}</p>}
          <div className="flex flex-wrap items-center gap-1.5 pt-1">
            {listen.state !== 'listening' ? (
              <>
                <select aria-label="Meeting audio source" className="rounded border px-1 py-0.5 text-xs" value={source} onChange={(e) => setSource(e.target.value)}>
                  {roomFromExtension && <option value="extension">This tab (MUDRA extension)</option>}
                  {!roomFromExtension && <option value="local-room">MUDRA extension on this computer</option>}
                  <option value="share">Share a meeting tab or screen audio</option>
                  {testMode && <option value="mic-test">Microphone (testing only)</option>}
                </select>
                {source !== 'extension' && (
                  <select aria-label="Speech recognition engine" className="rounded border px-1 py-0.5 text-xs" value={engine} onChange={(e) => setEngine(e.target.value)}>
                    <option value="mudra">MUDRA local (sherpa-onnx)</option>
                    <option value="chrome" disabled={chromeStatus === 'unsupported' || chromeStatus === 'unavailable'}>Chrome on-device {chromeStatus && chromeStatus !== 'available' ? `(${chromeStatus})` : ''}</option>
                  </select>
                )}
                {engine === 'chrome' && source !== 'extension' && chromeStatus === 'downloadable' && (
                  <button type="button" className={btn} onClick={async () => { await installChromeOnDevice(); setChromeStatus(await chromeOnDeviceStatus()) }}>Get Chrome's English pack</button>
                )}
                <button type="button" className={`${btn} bg-sl-green`} onClick={startListening} disabled={!avatarReady || listen.state === 'starting' || (engine === 'mudra' && !server.asr?.available && source !== 'extension')}>Start listening</button>
              </>
            ) : (
              <button type="button" className={`${btn} bg-rose-600 text-white`} onClick={() => { stopListening(); if (embedded) window.parent.postMessage({ type: 'mudra:stop-capture' }, '*') }}>Stop listening</button>
            )}
          </div>
          {listen.state === 'listening' && <p className="text-[11px] text-slate-600">Listening to {listen.what} · {listen.engine} · on this computer, nothing is recorded.</p>}
          {listen.error && <p className="text-[11px] text-rose-700">{listen.error}</p>}
          {listen.note && <p className="text-[11px] text-slate-600">{listen.note}</p>}
          {server.state === 'online' && !server.asr?.available && <p className="text-[11px] text-amber-800">MUDRA speech recognition is not set up on the server: {server.asr?.error}</p>}
        </div>
      </section>

      {surfaceNote && (
        <p className="text-[11px] text-slate-600" data-testid="surface">
          {surfaceNote}
          {surface === 'meet-sidepanel' && client?.client && roomFromExtension && (
            <button type="button" className={`${btn} ml-2`} onClick={() => shareToMainStage(client.client, roomFromExtension).catch((e) => setClient((c) => ({ ...c, error: e.message })))}>Show on main stage</button>
          )}
        </p>
      )}

      {/* outgoing: signs -> voice */}
      {!mainStage && <section aria-label="Your signs spoken into the meeting" className="space-y-1.5 rounded-xl border-2 border-sl-ink p-2">
        <div className="flex items-center gap-2">
          <video ref={videoRef} className={`h-[72px] w-[96px] -scale-x-100 rounded bg-slate-200 object-cover ${outgoing.active ? '' : 'opacity-40'}`} muted playsInline />
          <div className="min-w-0 flex-1 text-xs">
            {outgoing.lastSign ? <p>Last sign: <strong>{outgoing.lastSign.label}</strong> {outgoing.lastSign.spoken ? '' : '(unsure: not spoken)'}</p> : <p className="text-slate-500">Your signs are spoken here.</p>}
            {outgoing.pendingWords.length > 0 && <p>Will say: “{outgoing.pendingWords.join(' ')}”</p>}
            <p data-testid="voice-status" className={voiceStatus.state === 'error' ? 'text-rose-700' : voiceStatus.state === 'local-only' ? 'text-amber-800' : 'text-slate-600'}>
              {voiceStatus.state === 'synthesizing' && `Preparing: “${voiceStatus.text}”`}
              {voiceStatus.state === 'playing' && `Speaking: “${voiceStatus.text}”`}
              {voiceStatus.state === 'sent' && `Sent to the meeting microphone: “${voiceStatus.text}”`}
              {voiceStatus.state === 'local-only' && `Played on ${voiceStatus.sink} only - the meeting did not hear it`}
              {voiceStatus.state === 'error' && voiceStatus.error}
            </p>
          </div>
        </div>
        <div className="flex flex-wrap items-center gap-1.5">
          {!outgoing.active
            ? <button type="button" className={`${btn} bg-sl-green`} onClick={() => outgoing.start(videoRef.current)} disabled={outgoing.state.status === 'starting'}>Start signing</button>
            : <button type="button" className={`${btn} bg-rose-600 text-white`} onClick={outgoing.stop}>Stop camera</button>}
          {outgoing.pendingWords.length > 0 && <button type="button" className={btn} onClick={outgoing.speakNow}>Say now</button>}
          {outgoing.state.error && <span className="text-[11px] text-rose-700">{outgoing.state.error}</span>}
        </div>
        <details className="text-xs" open={!routedToCable || Boolean(routeTest)}>
          <summary className="cursor-pointer font-bold">Voice route</summary>
          <div className="mt-1 space-y-1">
            <label className="flex items-center gap-1">Speak into
              <select aria-label="Voice output device" className="min-w-0 flex-1 rounded border px-1 py-0.5" value={sinkId} onChange={(e) => setSinkId(e.target.value)}>
                <option value="">Default speakers (meeting will not hear)</option>
                {devices.outputs.filter((d) => d.deviceId !== 'default' && d.deviceId !== 'communications').map((d) => <option key={d.deviceId} value={d.deviceId}>{d.label || 'Output device'}</option>)}
              </select>
            </label>
            {!devices.labelled && <p className="text-amber-800">Device names appear after you allow the camera (Start signing) or a test.</p>}
            {!cable.output && devices.labelled && <p className="text-amber-800">No virtual cable found. Install VB-Audio Virtual Cable, then pick its playback end here (“CABLE Input” or “Speakers (VB-Audio Virtual Cable)”) and “CABLE Output” as the microphone in Meet / Zoom.</p>}
            {cable.output && <p>In Meet / Zoom, choose <strong>{cable.input?.label || 'CABLE Output'}</strong> as the microphone.</p>}
            <div className="flex items-center gap-1.5">
              <button type="button" className={btn} onClick={testRoute} disabled={routeTest?.state === 'testing'}>Test route</button>
              <label className="flex items-center gap-1"><input type="checkbox" onChange={(e) => voice.setMonitor(e.target.checked)} /> Also play on my speakers</label>
            </div>
            {routeTest?.state === 'testing' && <p>Playing a test phrase…</p>}
            {routeTest?.state === 'done' && routeTest.heard === true && <p className="text-emerald-800">Verified: the test phrase reached {cable.input?.label} (level {routeTest.peak}). Select it as the meeting microphone.</p>}
            {routeTest?.state === 'done' && routeTest.heard === false && <p className="text-rose-700">Nothing arrived on {cable.input?.label}. Is “Speak into” set to the cable ({cable.output?.label || 'CABLE Input'})?</p>}
            {routeTest?.state === 'done' && routeTest.heard === null && <p className="text-amber-800">{routeTest.note}</p>}
            {routeTest?.state === 'error' && <p className="text-rose-700">{routeTest.error}</p>}
          </div>
        </details>
      </section>}

      <footer className="flex flex-wrap items-center gap-2 text-[11px] text-slate-600">
        {embedded && <button type="button" className={btn} onClick={popOut}>Pop out</button>}
        <span>Local only: speech is recognised and spoken on this computer; nothing is recorded. Signs come from MUDRA's recorded vocabulary ({incoming.vocabSize || '…'} words and phrases); a word without a sign is captioned, not invented.</span>
      </footer>
    </div>
  )
}
