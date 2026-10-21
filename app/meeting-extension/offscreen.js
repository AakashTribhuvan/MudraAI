/**
 * Offscreen page: the meeting tab's audio -> (a) back to the speakers, so the user still hears the meeting
 * (tab capture otherwise mutes the tab), and (b) 16 kHz 16-bit PCM in 100 ms chunks over one WebSocket to the local
 * MUDRA server's /asr?room=R. Only the newest chunk is sent each time; nothing is stored.
 * MUDRA's own voice is played by the panel to the virtual cable, not through this tab capture's playback.
 */
let capture = null // { stream, playback, asr, ws }

async function startCapture({ streamId, room, api }) {
  stopCapture()
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: { mandatory: { chromeMediaSource: 'tab', chromeMediaSourceId: streamId } },
    video: false,
  })
  // (a) keep the meeting audible
  const playback = new AudioContext()
  playback.createMediaStreamSource(stream).connect(playback.destination)
  // (b) speech recognition
  const ws = new WebSocket(`${api.replace(/^http/, 'ws')}/asr?room=${encodeURIComponent(room)}`)
  ws.binaryType = 'arraybuffer'
  await new Promise((resolve, reject) => {
    ws.onopen = resolve
    ws.onerror = () => reject(new Error(`The MUDRA server is not reachable at ${api}. Start it with start_all.bat.`))
  })
  const asr = new AudioContext({ sampleRate: 16000 })
  await asr.audioWorklet.addModule(chrome.runtime.getURL('pcm-worklet.js'))
  const node = new AudioWorkletNode(asr, 'mudra-pcm16')
  node.port.onmessage = (e) => { if (ws.readyState === WebSocket.OPEN) ws.send(e.data) }
  asr.createMediaStreamSource(stream).connect(node)
  const mute = asr.createGain()
  mute.gain.value = 0
  node.connect(mute).connect(asr.destination)
  ws.onclose = () => { if (capture?.ws === ws) { stopCapture(); notify('capture-ended', 'the MUDRA server closed the connection') } }
  stream.getAudioTracks()[0].onended = () => { stopCapture(); notify('capture-ended', 'the tab audio ended') }
  capture = { stream, playback, asr, ws }
}

function stopCapture() {
  if (!capture) return
  const c = capture
  capture = null
  try { if (c.ws.readyState === WebSocket.OPEN) c.ws.send(JSON.stringify({ type: 'stop' })) } catch { /* closed */ }
  setTimeout(() => { try { c.ws.close() } catch { /* closed */ } }, 1000)
  c.stream.getTracks().forEach((t) => t.stop())
  c.playback.close()
  c.asr.close()
}

function notify(type, reason) {
  chrome.runtime.sendMessage({ target: 'background', type, reason }).catch(() => {})
}

chrome.runtime.onMessage.addListener((msg, _sender, reply) => {
  if (msg?.target !== 'offscreen') return undefined
  if (msg.type === 'start-capture') {
    startCapture(msg).then(() => reply({ ok: true }), (err) => { stopCapture(); reply({ ok: false, error: String(err?.message || err) }) })
    return true
  }
  if (msg.type === 'stop-capture') { stopCapture(); reply({ ok: true }) }
  return undefined
})
