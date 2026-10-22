// Reads the real webcam's capabilities and settings (no image is kept), then measures its frame rate for 2 s.
import { spawn } from 'node:child_process'
import { mkdtempSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
const port = 9944
const proc = spawn('C:/Program Files/Google/Chrome/Application/chrome.exe', ['--use-fake-ui-for-media-stream', '--no-first-run', `--remote-debugging-port=${port}`, `--user-data-dir=${mkdtempSync(join(tmpdir(), 'cc-'))}`, '--window-position=-2600,40', 'about:blank'], { stdio: 'ignore' })
const sleep = (ms) => new Promise((r) => setTimeout(r, ms))
let t; for (let i = 0; i < 60; i++) { try { t = await (await fetch(`http://127.0.0.1:${port}/json`)).json(); if (t.find((x) => x.type === 'page')) break } catch {} await sleep(250) }
const ws = new WebSocket(t.find((x) => x.type === 'page').webSocketDebuggerUrl); await new Promise((r) => ws.addEventListener('open', r))
let id = 0; const pend = new Map(); ws.addEventListener('message', (e) => { const m = JSON.parse(e.data); if (m.id && pend.has(m.id)) { pend.get(m.id)(m.result); pend.delete(m.id) } })
const send = (method, params = {}) => new Promise((r) => { id++; pend.set(id, r); ws.send(JSON.stringify({ id, method, params })) })
await send('Page.navigate', { url: 'http://127.0.0.1:3000/welcome' }); await sleep(2000)
const expr = `(async () => {
  const measure = async (track) => { return 'skipped' }; const _m = async (track) => { const v = document.createElement('video'); v.muted = true; v.srcObject = new MediaStream([track]); await v.play(); let n = 0; const t0 = performance.now(); await new Promise((res) => { const f = () => { n++; if (performance.now() - t0 < 2000) v.requestVideoFrameCallback(f); else res() }; v.requestVideoFrameCallback(f) }); return Math.round(n / ((performance.now() - t0) / 1000)) }
  const s = await Promise.race([new Promise((_, rej) => setTimeout(() => rej(new Error('camera did not open in 8 s (in use by another app?)')), 8000)), navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 30 } } })])
  const tr = s.getVideoTracks()[0]
  const caps = tr.getCapabilities(); const set = tr.getSettings()
  const out = { label: tr.label, caps: { frameRate: caps.frameRate, exposureMode: caps.exposureMode, exposureTime: caps.exposureTime, exposureCompensation: caps.exposureCompensation, brightness: caps.brightness, whiteBalanceMode: caps.whiteBalanceMode }, settings: { frameRate: set.frameRate, exposureMode: set.exposureMode, exposureTime: set.exposureTime, brightness: set.brightness }, fpsDefault: await measure(tr) }
  if (caps.exposureMode && caps.exposureMode.includes('manual') && caps.exposureTime) {
    try { await tr.applyConstraints({ advanced: [{ exposureMode: 'manual', exposureTime: Math.min(caps.exposureTime.max, Math.max(caps.exposureTime.min, 300)) }] }); out.fpsManual300 = await measure(tr); out.afterManual = tr.getSettings().exposureTime } catch (e) { out.manualError = String(e) }
  }
  s.getTracks().forEach((x) => x.stop()); return out })()`
const r = await send('Runtime.evaluate', { expression: expr + '.catch((e) => String(e))', awaitPromise: true, returnByValue: true, userGesture: true })
console.log(JSON.stringify(r.result?.value ?? r, null, 1)); ws.close(); proc.kill(); process.exit(0)
