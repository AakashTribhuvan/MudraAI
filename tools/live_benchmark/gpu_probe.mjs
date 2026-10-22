import { spawn } from 'node:child_process'
import { mkdtempSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
const port = 9933
const proc = spawn('C:/Program Files/Google/Chrome/Application/chrome.exe', ['--no-first-run', `--remote-debugging-port=${port}`, `--user-data-dir=${mkdtempSync(join(tmpdir(), 'gp-'))}`, '--window-position=-2600,40', 'about:blank'], { stdio: 'ignore' })
const sleep = (ms) => new Promise((r) => setTimeout(r, ms))
let t; for (let i = 0; i < 60; i++) { try { t = await (await fetch(`http://127.0.0.1:${port}/json`)).json(); if (t.find((x) => x.type === 'page')) break } catch {} await sleep(250) }
const ws = new WebSocket(t.find((x) => x.type === 'page').webSocketDebuggerUrl); await new Promise((r) => ws.addEventListener('open', r))
let id = 0; const pend = new Map(); ws.addEventListener('message', (e) => { const m = JSON.parse(e.data); if (m.id && pend.has(m.id)) { pend.get(m.id)(m.result); pend.delete(m.id) } })
const send = (method, params = {}) => new Promise((r) => { id++; pend.set(id, r); ws.send(JSON.stringify({ id, method, params })) })
await send('Page.navigate', { url: 'http://127.0.0.1:3000/welcome' }); await sleep(2000)
const r = await send('Runtime.evaluate', { returnByValue: true, awaitPromise: true, expression: `(async () => { const out = {}; for (const pp of ['default', 'low-power', 'high-performance']) { const c = document.createElement('canvas').getContext('webgl2', { powerPreference: pp }); const d = c.getExtension('WEBGL_debug_renderer_info'); out[pp] = c.getParameter(d.UNMASKED_RENDERER_WEBGL).slice(0, 60) } const a = navigator.gpu && await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' }); out.webgpu_hp = a ? (a.info ? a.info.vendor + ' ' + a.info.architecture + ' ' + a.info.description : 'adapter') : 'none'; return out })()` })
console.log(JSON.stringify(r.result.value, null, 1)); ws.close(); proc.kill(); process.exit(0)
