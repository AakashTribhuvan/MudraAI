// Outgoing through the real VB-Cable: the panel speaks signed words into the cable's playback end; a "meeting"
// listener records CABLE Output (what Meet / Zoom would use as the microphone) and transcribes it with /asr.
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { mkdirSync } from 'node:fs'
const HERE = dirname(fileURLToPath(import.meta.url))
const ROOT = join(HERE, '..', '..')
const WORK = join(ROOT, 'experiments', 'tools', 'meeting_tests') // outputs (git-ignored)
mkdirSync(WORK, { recursive: true })
import { spawn } from 'node:child_process'
import { mkdtempSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
const chrome = 'C:/Program Files/Google/Chrome/Application/chrome.exe'
const port = 9870
const proc = spawn(chrome, ['--use-fake-ui-for-media-stream', '--autoplay-policy=no-user-gesture-required', '--no-first-run', '--no-default-browser-check',
  `--remote-debugging-port=${port}`, `--user-data-dir=${mkdtempSync(join(tmpdir(), 'cable-e2e-'))}`, '--window-position=80,60', '--window-size=460,860', 'about:blank'], { stdio: 'ignore' })
const sleep = (ms) => new Promise((r) => setTimeout(r, ms))
let t; for (let i = 0; i < 60; i++) { try { t = await (await fetch(`http://127.0.0.1:${port}/json`)).json(); if (t.find((x) => x.type === 'page')) break } catch { /* starting */ } await sleep(250) }
const ws = new WebSocket(t.find((x) => x.type === 'page').webSocketDebuggerUrl)
await new Promise((r) => ws.addEventListener('open', r))
let id = 0; const pend = new Map(); const logs = []
ws.addEventListener('message', (e) => { const m = JSON.parse(e.data); if (m.id && pend.has(m.id)) { pend.get(m.id)(m.result); pend.delete(m.id) }
  if (m.method === 'Runtime.consoleAPICalled' && ['error', 'warning'].includes(m.params.type)) logs.push(m.params.args.map((a) => a.value ?? a.description).join(' ').slice(0, 200)) })
const send = (method, params = {}) => new Promise((r) => { id++; pend.set(id, r); ws.send(JSON.stringify({ id, method, params })) })
const ev = async (expression) => { const r = await send('Runtime.evaluate', { expression, awaitPromise: true, returnByValue: true, userGesture: true }); return r.exceptionDetails ? { error: r.exceptionDetails.exception?.description } : r.result?.value }
await send('Page.enable'); await send('Runtime.enable')
await send('Page.navigate', { url: 'http://127.0.0.1:3000/meet-panel?test=1' })
await sleep(3000)
// microphone permission -> device names; then reload so the panel lists them
await ev(`navigator.mediaDevices.getUserMedia({ audio: true }).then((s) => s.getTracks().forEach((x) => x.stop()))`)
await send('Page.reload'); await sleep(5000)
const devices = await ev(`navigator.mediaDevices.enumerateDevices().then((d) => d.filter((x) => x.kind !== 'videoinput').map((x) => x.kind + ' | ' + x.deviceId.slice(0, 12) + ' | ' + x.label))`)
console.log('devices:\n  ' + devices.join('\n  '))
console.log('panel chips:', await ev(`document.querySelector('header').innerText.replace(/\\s+/g, ' ')`))
console.log('speak into:', await ev(`(() => { const s = document.querySelector('select[aria-label="Voice output device"]'); return s.options[s.selectedIndex].text })()`))
await ev(`[...document.querySelectorAll('button')].find((b) => b.textContent === 'Test route').click()`)
for (let i = 0; i < 20; i++) { await sleep(500); if (await ev(`/Verified|Nothing arrived|Played;/.test(document.querySelector('details').innerText)`)) break }
console.log('test route:', await ev(`[...document.querySelectorAll('details p')].map((p) => p.innerText).filter((x) => /Verified|Nothing|Played|test phrase|microphone/.test(x)).join(' / ')`))
// the "meeting": record CABLE Output and transcribe it locally
await ev(`(async () => {
  const d = (await navigator.mediaDevices.enumerateDevices()).find((x) => x.kind === 'audioinput' && /CABLE Output/i.test(x.label) && x.deviceId !== 'default' && x.deviceId !== 'communications')
  const mic = await navigator.mediaDevices.getUserMedia({ audio: { deviceId: { exact: d.deviceId }, echoCancellation: false, noiseSuppression: false, autoGainControl: false } })
  const m = await import('/src/features/meeting/speechSources.js')
  window.__heard = []
  window.__meeting = await m.streamToMudraASR(mic.getAudioTracks()[0], { apiBase: 'http://localhost:8010', onEvent: (e) => { if (e.event === 'final') window.__heard.push(e.text) } })
  return d.label
})()`)
await sleep(1500)
// the user's signs (injected into the real recogniser hook: no signing video exists for a camera test)
await ev(`import('/src/services/signRecognizer.js').then((m) => { const r = m.default; r.onPrediction = r.onPrediction || null })`)
// no real webcam in this test: video requests get a blank canvas stream (audio requests are untouched)
await ev(`(() => { const g = navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices); navigator.mediaDevices.getUserMedia = (c) => { if (c && c.video) { const cv = document.createElement('canvas'); cv.width = 640; cv.height = 480; const x = cv.getContext('2d'); setInterval(() => { x.fillStyle = '#888'; x.fillRect(0, 0, 640, 480) }, 100); return Promise.resolve(cv.captureStream(15)) } return g(c) } })()`)
await ev(`[...document.querySelectorAll('button')].find((b) => b.textContent === 'Start signing').click()`)
for (let i = 0; i < 40; i++) { await sleep(500); if (await ev(`document.body.innerText.includes('Camera on') || document.body.innerText.includes('Camera:') || document.body.innerText.includes('permission')`)) break }
console.log('camera:', await ev(`document.querySelector('header').innerText.replace(/\\s+/g, ' ')`))
const t0 = Date.now()
await ev(`import('/src/services/signRecognizer.js').then((m) => { const r = m.default; for (const s of ['HELLO', 'HOW_ARE_YOU']) r.onPrediction({ status: 'ok', sign: s, label: s, confidence: 0.92 }) })`)
for (let i = 0; i < 20; i++) { await sleep(500); const v = await ev(`document.querySelector('[data-testid=voice-status]').innerText`); if (/Sent to the meeting|only|error/i.test(v)) { console.log(`voice after ${Date.now() - t0} ms:`, v); break } }
await sleep(2500)
await ev(`import('/src/services/signRecognizer.js').then((m) => { const r = m.default; for (const s of ['I', 'NEED', 'WATER']) r.onPrediction({ status: 'ok', sign: s, label: s, confidence: 0.92 }) })`)
await sleep(1600)
for (let i = 0; i < 20; i++) { await sleep(500); const v = await ev(`document.querySelector('[data-testid=voice-status]').innerText`); if (/Sent to the meeting: “I/.test(v) || /error/i.test(v)) { console.log('voice 2:', v); break } }
await sleep(3500)
await ev(`window.__meeting.stop()`)
await sleep(2500)
console.log('what the meeting microphone carried (local ASR of CABLE Output):', JSON.stringify(await ev(`window.__heard`)))
console.log('said by MUDRA:', JSON.stringify(await ev(`[...document.querySelectorAll('section p')].map((p) => p.innerText).filter((x) => /Sent to/.test(x))`)))
await ev(`[...document.querySelectorAll('button')].find((b) => b.textContent === 'Stop camera')?.click()`)
writeFileSync(join(WORK, 'cable_e2e.png'), Buffer.from((await send('Page.captureScreenshot', { format: 'png' })).data, 'base64'))
console.log('console warnings/errors:', [...new Set(logs)].filter((l) => !/React Router/.test(l)).slice(0, 8).join(' || ') || 'none')
ws.close(); proc.kill(); process.exit(0)
