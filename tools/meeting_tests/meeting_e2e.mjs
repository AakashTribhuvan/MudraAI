// End-to-end incoming test: a WAV with known word timings is the (fake) microphone of headless Chrome; the
// meeting panel (test mode) listens, the avatar signs. Measures word onset -> first partial -> commit -> avatar.
// Usage: node meeting_e2e.mjs <wav> <words.json> <out.png> [engine=mudra]
import { spawn } from 'node:child_process'
import { mkdtempSync, readFileSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
const [wav, wordsJson, out, engine = 'mudra'] = process.argv.slice(2)
const words = JSON.parse(readFileSync(wordsJson, 'utf8').replace(/^\uFEFF/, ''))
const chrome = 'C:/Program Files/Google/Chrome/Application/chrome.exe'
const port = 9400 + Math.floor(Math.random() * 400)
const proc = spawn(chrome, ['--headless=new', '--enable-gpu', '--ignore-gpu-blocklist', '--use-fake-device-for-media-stream', '--use-fake-ui-for-media-stream',
  `--use-file-for-fake-audio-capture=${wav}`, '--autoplay-policy=no-user-gesture-required', `--remote-debugging-port=${port}`,
  `--user-data-dir=${mkdtempSync(join(tmpdir(), 'meet-e2e-'))}`, '--window-size=420,820', 'about:blank'], { stdio: 'ignore' })
const sleep = (ms) => new Promise((r) => setTimeout(r, ms))
let targets
for (let i = 0; i < 50; i++) { try { targets = await (await fetch(`http://127.0.0.1:${port}/json`)).json(); break } catch { await sleep(200) } }
const ws = new WebSocket(targets.find((t) => t.type === 'page').webSocketDebuggerUrl)
await new Promise((r) => ws.addEventListener('open', r))
let id = 0; const pending = new Map(); const logs = []
ws.addEventListener('message', (e) => {
  const m = JSON.parse(e.data)
  if (m.id && pending.has(m.id)) { pending.get(m.id)(m.result); pending.delete(m.id) }
  if (m.method === 'Runtime.consoleAPICalled' && ['error', 'warning'].includes(m.params.type)) logs.push(m.params.type + ': ' + m.params.args.map((a) => a.value ?? a.description ?? '').join(' ').slice(0, 200))
  if (m.method === 'Runtime.exceptionThrown') logs.push('exception: ' + (m.params.exceptionDetails.exception?.description || m.params.exceptionDetails.text).slice(0, 300))
})
const send = (method, params = {}) => new Promise((r) => { id += 1; pending.set(id, r); ws.send(JSON.stringify({ id, method, params })) })
const evaluate = async (expression) => (await send('Runtime.evaluate', { expression, awaitPromise: true, returnByValue: true })).result?.value
await send('Page.enable'); await send('Runtime.enable')
await send('Page.navigate', { url: 'http://127.0.0.1:3000/meet-panel?test=1' })
for (let i = 0; i < 60; i++) { await sleep(500); if (await evaluate(`!document.body.innerText.includes('Loading the avatar')`)) break }
await sleep(800)
// record when the fake microphone opens (Chrome starts the file from the beginning when the device opens)
await evaluate(`(() => { const g = navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices); navigator.mediaDevices.getUserMedia = async (c) => { const s = await g(c); window.__gumAt = performance.now(); return s } })()`)
await evaluate(`(() => { const set = (sel, v) => { const el = document.querySelector(sel); const d = Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype, 'value'); d.set.call(el, v); el.dispatchEvent(new Event('change', { bubbles: true })) }; set('select[aria-label="Meeting audio source"]', 'mic-test'); })()`)
await sleep(300)
if (engine !== 'mudra') await evaluate(`(() => { const el = document.querySelector('select[aria-label="Speech recognition engine"]'); Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype, 'value').set.call(el, '${engine}'); el.dispatchEvent(new Event('change', { bubbles: true })) })()`)
await sleep(300)
await evaluate(`[...document.querySelectorAll('button')].find((b) => b.textContent === 'Start listening').click()`)
const totalMs = words.at(-1).start_ms + 2500
const snaps = []
const t0 = Date.now()
while (Date.now() - t0 < totalMs) {
  await sleep(1000)
  snaps.push(await evaluate(`({ t: performance.now() - (window.__gumAt || 0), caption: document.querySelector('[data-testid=caption]').innerText, status: document.querySelector('[data-testid=sign-status]').innerText })`))
}
const shot = await send('Page.captureScreenshot', { format: 'png' })
writeFileSync(out, Buffer.from(shot.data, 'base64'))
const result = await evaluate(`(() => { const m = window.__mudraMeetingMetrics || { words: [], phrases: [] }; return { gumAt: window.__gumAt, words: m.words.map((w) => ({ word: w.word, firstPartialAt: w.firstPartialAt, commitAt: w.commitAt, reason: w.reason })), phrases: m.phrases.map((p) => ({ text: p.text, words: p.phrase.words, commitAt: p.commitAt, startAt: p.startAt })), echoes: m.echoes, unsigned: document.body.innerText.match(/caption only\\): (.*)/)?.[1] || '', skipped: document.body.innerText.match(/keep up with the speaker: (.*)/)?.[1] || '' } })()`)
await evaluate(`[...document.querySelectorAll('button')].find((b) => /Stop listening|Stop all/.test(b.textContent))?.click()`)
await sleep(500)
const after = await evaluate(`({ status: document.querySelector('[data-testid=sign-status]').innerText, listening: document.body.innerText.includes('● Listening') })`)
ws.close(); proc.kill()
// latency per heard word: from the word's onset in the audio (fake mic opened at gumAt)
const norm = (w) => w.toLowerCase().replace(/[^a-z0-9']/g, '')
const onset = new Map()
for (const w of words) { const k = norm(w.text); if (!onset.has(k)) onset.set(k, w.start_ms) }
const rows = []
for (const w of result.words) {
  const on = onset.get(w.word)
  if (on == null) { rows.push({ word: w.word, note: 'not in the script' }); continue }
  const ph = result.phrases.find((p) => p.words.includes(w.word) && p.startAt)
  rows.push({ word: w.word, onset_ms: Math.round(on), partial_after_onset: Math.round(w.firstPartialAt - result.gumAt - on), commit_after_onset: Math.round(w.commitAt - result.gumAt - on), commit_reason: w.reason,
    avatar_after_onset: ph ? Math.round(ph.startAt - result.gumAt - on) : null, phrase: ph?.text || null })
}
console.log(JSON.stringify({ engine, rows, phrases: result.phrases.map((p) => ({ text: p.text, signedAt: p.startAt ? Math.round(p.startAt - result.gumAt) : null })), unsigned: result.unsigned, skipped: result.skipped, echoes: result.echoes, snaps: snaps.map((s) => `${Math.round(s.t)}ms | ${s.caption.replace(/\n/g, ' ')} | ${s.status}`), afterStop: after, logs: [...new Set(logs)].slice(0, 10) }, null, 1))
process.exit(0)
