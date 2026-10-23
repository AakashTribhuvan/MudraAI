// The extension's panel frame inside the meeting page: hosts the MUDRA web app's meeting panel (/meet-panel) and
// relays its Stop to the extension. Messages are accepted only from the MUDRA web app's origin.
const params = new URLSearchParams(location.search)
const room = params.get('room')
const web = params.get('web') || 'http://localhost:3000'
let webOrigin = null
try { webOrigin = new URL(web).origin } catch { webOrigin = null }

if (!webOrigin || !room || room === 'none') {
  document.body.innerHTML = '<p class="err">The MUDRA panel could not start. Check the MUDRA app address in the extension options.</p>'
} else {
  const frame = document.createElement('iframe')
  frame.title = 'MUDRA meeting panel'
  frame.allow = 'camera; microphone; autoplay; speaker-selection; display-capture; clipboard-write'
  frame.src = `${webOrigin}/meet-panel?room=${encodeURIComponent(room)}&embed=extension`
  document.body.appendChild(frame)

  window.addEventListener('message', (e) => {
    if (e.origin !== webOrigin || e.source !== frame.contentWindow || !e.data || typeof e.data !== 'object') return
    if (e.data.type === 'mudra:stop-capture') chrome.runtime.sendMessage({ target: 'background', type: 'stop-capture' })
  })
  chrome.runtime.onMessage.addListener((msg) => {
    if (msg?.target === 'panel' && msg.type === 'capture-stopped') frame.contentWindow?.postMessage({ type: 'mudra:capture-stopped', reason: msg.reason }, webOrigin)
  })
}
