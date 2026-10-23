// Live-recognition benchmark on this PC's real GPU: a normal Chrome window (off-screen, not throttled) with the fake
// webcam (signs.y4m), the real Communicate page, the real server. Prints the page's tracking numbers each second and
// writes the window into which the server's sign events fall.
// Usage: node bench_cam.mjs <label> [GPU|CPU] [seconds=36] [extra JS run before the camera starts]
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { mkdirSync } from 'node:fs'
const HERE = dirname(fileURLToPath(import.meta.url))
const ROOT = join(HERE, '..', '..')
const WORK = join(ROOT, 'experiments', 'tools', 'live_benchmark') // outputs (git-ignored)
mkdirSync(WORK, { recursive: true })
import { spawn } from 'node:child_process'
import { mkdtempSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
const [label, delegate = 'GPU', secs = '36', pre = ''] = process.argv.slice(2)
const y4m = join(WORK, process.env.CAMDIR || 'fakecam', 'signs.y4m')
const chrome = 'C:/Program Files/Google/Chrome/Application/chrome.exe'
const port = 9600 + Math.floor(Math.random() * 300)
const proc = spawn(chrome, ['--use-fake-ui-for-media-stream', '--use-fake-device-for-media-stream', `--use-file-for-fake-video-capture=${y4m}`,
  '--disable-backgrounding-occluded-windows', '--disable-renderer-backgrounding', '--disable-background-timer-throttling',
  ...(process.env.FLAGS ? process.env.FLAGS.split(' ') : []), '--no-first-run', '--no-default-browser-check', `--remote-debugging-port=${port}`, `--user-data-dir=${mkdtempSync(join(tmpdir(), 'bench-'))}`,
  '--window-position=-2600,40', '--window-size=1366,860', 'about:blank'], { stdio: 'ignore' })
const sleep = (ms) => new Promise((r) => setTimeout(r, ms))
let t; for (let i = 0; i < 60; i++) { try { t = await (await fetch(`http://127.0.0.1:${port}/json`)).json(); if (t.find((x) => x.type === 'page')) break } catch { /* starting */ } await sleep(250) }
const ws = new WebSocket(t.find((x) => x.type === 'page').webSocketDebuggerUrl)
await new Promise((r) => ws.addEventListener('open', r))
let id = 0; const pend = new Map(); const logs = []
ws.addEventListener('message', (e) => { const m = JSON.parse(e.data); if (m.id && pend.has(m.id)) { pend.get(m.id)(m.result); pend.delete(m.id) }
  if (m.method === 'Runtime.consoleAPICalled') { const s = m.params.args.map((a) => a.value ?? a.description).join(' '); if (/delegate|BENCH|error/i.test(s)) logs.push(s.slice(0, 200) + ' ' + JSON.stringify(m.params.args.map((a) => a.preview?.properties?.map((x) => x.name + '=' + x.value)).filter(Boolean))) } })
const send = (method, params = {}) => new Promise((r) => { id++; pend.set(id, r); ws.send(JSON.stringify({ id, method, params })) })
const ev = async (expression) => { const r = await send('Runtime.evaluate', { expression, awaitPromise: true, returnByValue: true, userGesture: true }); return r.exceptionDetails ? { error: r.exceptionDetails.exception?.description } : r.result?.value }
await send('Page.enable'); await send('Runtime.enable')
await send('Page.navigate', { url: 'http://127.0.0.1:3000/welcome' }); await sleep(1500)
await ev(`localStorage.setItem('signline_demo_session', JSON.stringify({ role: 'user', name: 'Bench', demo: true })); localStorage.setItem('signline_tracker_delegate', JSON.stringify({ delegate: '${delegate}', timings: {}, at: Date.now() }))`)
await send('Page.navigate', { url: 'http://127.0.0.1:3000/app/communication' }); await sleep(Number(process.env.PRELOAD_MS || 14000))
const gpu = await ev(`(() => { const c = document.createElement('canvas').getContext('webgl2'); const d = c && c.getExtension('WEBGL_debug_renderer_info'); return d ? c.getParameter(d.UNMASKED_RENDERER_WEBGL) : 'no webgl' })()`)
if (pre) console.log('pre:', JSON.stringify(await ev(pre)))
// latency probes: when the camera opens (the fake video starts at its first frame) and when each sign result arrives
await ev(`(() => { const g = navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices); navigator.mediaDevices.getUserMedia = async (c) => { const s = await g(c); if (c && c.video) window.__camAt = performance.now(); return s }
  window.__signs = []; const W = window.WebSocket; window.WebSocket = class extends W { constructor(...a) { super(...a); this.addEventListener('message', (m) => { try { const e = JSON.parse(m.data); if (e.event === 'sign_ended' || e.event === 'sign_started') window.__signs.push({ ev: e.event, t: performance.now(), status: e.result?.status, sign: e.result?.sign }) } catch {} }) } } })()`)
const startedAt = new Date()
await ev(`[...document.querySelectorAll('button')].find((b) => /Start Camera/i.test(b.textContent))?.click()`)
// sample the tracker's own numbers through its module (same instance as the page)
const rows = []
for (let i = 0; i < Number(secs); i++) {
  await sleep(1000)
  rows.push(await ev(`(() => { const h = window.__mudraTracker; return h ? { fps: Math.round(h.fps || 0), cam: Math.round(h.cameraFps || 0), ms: Math.round(h.inferenceMs || 0), d: h.delegate, warm: h.warmUpMs } : null })()`))
}
await ev(`[...document.querySelectorAll('button')].find((b) => /Stop Camera/i.test(b.textContent))?.click()`)
const endedAt = new Date()
const lat = await ev(`({ camAt: window.__camAt, signs: window.__signs })`)
const ok = rows.filter((r) => r && r.fps)
const med = (k) => { const v = ok.slice(5).map((r) => r[k]).sort((a, b) => a - b); return v[Math.floor(v.length / 2)] }
console.log(JSON.stringify({ label, gpu: gpu.slice(7, 40), delegate: ok.at(-1)?.d, trackingFps: med('fps'), cameraFps: med('cam'), inferenceMs: med('ms'), warmUpMs: ok.at(-1)?.warm, probe: logs.filter((l) => /Hand tracking delegate/.test(l)) }))
writeFileSync(join(WORK, `bench_${label}.json`), JSON.stringify({ label, startedAt, endedAt, rows, lat }))
ws.close(); proc.kill(); process.exit(0)
