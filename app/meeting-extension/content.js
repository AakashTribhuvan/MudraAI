/**
 * MUDRA overlay in the meeting page (Google Meet, Zoom Web). Uses no meeting internals and no page selectors:
 * one fixed, draggable, minimisable box in a closed Shadow DOM (the page's styles cannot reach it), holding the
 * extension's panel page. Shown when the toolbar button starts capture; the meeting's own controls stay usable.
 */
(() => {
  if (window.__mudraOverlay) return
  window.__mudraOverlay = true
  let host = null
  let ui = null

  const CSS = `
    :host { all: initial; }
    .box { position: fixed; right: 16px; bottom: 96px; width: 380px; height: min(700px, calc(100vh - 140px)); z-index: 2147483000;
      display: flex; flex-direction: column; background: #fff; border: 2px solid #1d1b16; border-radius: 14px;
      box-shadow: 0 12px 32px rgba(0,0,0,.35); overflow: hidden; font: 600 12px/1.3 system-ui, sans-serif; color: #1d1b16; }
    .bar { display: flex; align-items: center; gap: 6px; padding: 6px 8px; background: #fdf6e3; border-bottom: 2px solid #1d1b16; cursor: move; user-select: none; }
    .title { font-weight: 800; font-size: 13px; }
    .live { display: inline-flex; align-items: center; gap: 4px; padding: 1px 7px; border-radius: 999px; background: #dc2626; color: #fff; font-size: 11px; }
    .live.off { background: #e2e8f0; color: #475569; }
    .grow { flex: 1; }
    button { font: inherit; border: 2px solid #1d1b16; border-radius: 8px; background: #fff; padding: 2px 8px; cursor: pointer; }
    button.stop { background: #dc2626; color: #fff; }
    iframe { flex: 1; border: 0; width: 100%; background: #fff; }
    .note { padding: 6px 8px; background: #fef3c7; color: #78350f; font-weight: 600; }
    .box.min { height: auto; width: auto; }
    .box.min iframe, .box.min .note { display: none; }
  `

  function build(room, web) {
    host = document.createElement('div')
    host.setAttribute('data-mudra-overlay', '')
    const root = host.attachShadow({ mode: 'closed' })
    root.innerHTML = `<style>${CSS}</style>
      <div class="box" role="complementary" aria-label="MUDRA ISL panel">
        <div class="bar">
          <span class="title">MUDRA</span>
          <span class="live" aria-live="polite">● Listening to this tab</span>
          <span class="grow"></span>
          <button class="min-btn" title="Minimise">–</button>
          <button class="stop" title="Stop listening to this tab">Stop</button>
          <button class="close" title="Close the panel" aria-label="Close">✕</button>
        </div>
        <div class="note" hidden></div>
        <iframe title="MUDRA panel" allow="camera; microphone; autoplay; speaker-selection; display-capture; clipboard-write"></iframe>
      </div>`
    const q = (s) => root.querySelector(s)
    ui = { box: q('.box'), live: q('.live'), note: q('.note'), frame: q('iframe'), stop: q('.stop') }
    ui.frame.src = chrome.runtime.getURL(`panel.html?room=${encodeURIComponent(room)}&web=${encodeURIComponent(web)}`)
    q('.min-btn').onclick = () => ui.box.classList.toggle('min')
    ui.stop.onclick = () => chrome.runtime.sendMessage({ target: 'background', type: 'stop-capture' })
    q('.close').onclick = () => { chrome.runtime.sendMessage({ target: 'background', type: 'stop-capture' }); remove() }
    drag(q('.bar'), ui.box)
    document.documentElement.appendChild(host)
  }

  function drag(handle, box) {
    handle.addEventListener('pointerdown', (e) => {
      if (e.target.closest('button')) return
      const r = box.getBoundingClientRect()
      const dx = e.clientX - r.left
      const dy = e.clientY - r.top
      handle.setPointerCapture(e.pointerId)
      const move = (ev) => {
        box.style.left = `${Math.max(0, Math.min(window.innerWidth - 120, ev.clientX - dx))}px`
        box.style.top = `${Math.max(0, Math.min(window.innerHeight - 40, ev.clientY - dy))}px`
        box.style.right = 'auto'
        box.style.bottom = 'auto'
      }
      const up = () => { handle.removeEventListener('pointermove', move); handle.removeEventListener('pointerup', up) }
      handle.addEventListener('pointermove', move)
      handle.addEventListener('pointerup', up)
    })
  }

  function remove() { host?.remove(); host = null; ui = null }

  function setLive(on, note) {
    if (!ui) return
    ui.live.textContent = on ? '● Listening to this tab' : 'Not listening'
    ui.live.classList.toggle('off', !on)
    ui.stop.hidden = !on
    ui.note.hidden = !note
    ui.note.textContent = note || ''
  }

  chrome.runtime.onMessage.addListener((msg) => {
    if (msg.type === 'show-panel') {
      if (!host) build(msg.room, msg.web)
      else ui.frame.src = chrome.runtime.getURL(`panel.html?room=${encodeURIComponent(msg.room)}&web=${encodeURIComponent(msg.web)}`)
      setLive(true)
    } else if (msg.type === 'capture-stopped') {
      setLive(false, `Stopped: ${msg.reason}. Click the MUDRA toolbar button to listen again.`)
    } else if (msg.type === 'capture-error') {
      if (!host) build('none', 'about:blank')
      setLive(false, `MUDRA could not start: ${msg.error}`)
    }
  })
})()
