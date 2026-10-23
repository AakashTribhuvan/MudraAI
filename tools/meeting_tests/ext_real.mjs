// Real-gesture extension test: a visible Chrome window, the extension's keyboard shortcut (Alt+Shift+M) pressed with
// real Windows keystrokes (an invocation, like a toolbar click), real tab capture of the mock meeting's audio.
// Usage: node ext_real.mjs <extDir> <extId> <out.png>
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { mkdirSync } from 'node:fs'
const HERE = dirname(fileURLToPath(import.meta.url))
const ROOT = join(HERE, '..', '..')
const WORK = join(ROOT, 'experiments', 'tools', 'meeting_tests') // outputs (git-ignored)
mkdirSync(WORK, { recursive: true })
import { spawn, execFileSync } from 'node:child_process'
import { mkdtempSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
const [extDir, extId, out] = process.argv.slice(2)
const chrome = 'C:/Program Files/Google/Chrome/Application/chrome.exe'
const proc = spawn(chrome, ['--remote-debugging-pipe', '--enable-unsafe-extension-debugging', '--autoplay-policy=no-user-gesture-required',
  '--no-first-run', '--no-default-browser-check', `--user-data-dir=${mkdtempSync(join(tmpdir(), 'ext-real-'))}`, '--window-position=80,60', '--window-size=1280,800', 'about:blank'],
  { stdio: ['ignore', 'ignore', 'pipe', 'pipe', 'pipe'] })
const write = proc.stdio[3]
const read = proc.stdio[4]
let buf = ''
let id = 0
const pending = new Map()
const events = []
read.on('data', (chunk) => {
  buf += chunk.toString('utf8')
  let i
  while ((i = buf.indexOf('\0')) >= 0) {
    const m = JSON.parse(buf.slice(0, i))
    buf = buf.slice(i + 1)
    if (m.id && pending.has(m.id)) { pending.get(m.id)(m); pending.delete(m.id) } else events.push(m)
  }
})
const send = (method, params = {}, sessionId) => new Promise((r) => { id += 1; pending.set(id, r); write.write(JSON.stringify({ id, method, params, ...(sessionId ? { sessionId } : {}) }) + '\0') })
const sleep = (ms) => new Promise((r) => setTimeout(r, ms))
const log = (...a) => console.log(...a)
const evalIn = async (sessionId, expression, userGesture = false) => {
  const r = await send('Runtime.evaluate', { expression, awaitPromise: true, returnByValue: true, userGesture }, sessionId)
  return r.result?.exceptionDetails ? { error: r.result.exceptionDetails.exception?.description || r.result.exceptionDetails.text } : r.result?.result?.value
}
const pressShortcut = () => execFileSync('powershell.exe', ['-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass', '-File', join(HERE, 'focus_keys.ps1'), '-ProcId', String(proc.pid), '-Keys', '%+m']).toString().trim()
const _oldPress = () => execFileSync('powershell.exe', ['-NoProfile', '-NonInteractive', '-Command',
  `$w = New-Object -ComObject WScript.Shell; $ok = $w.AppActivate(${proc.pid}); Start-Sleep -Milliseconds 700; $w.SendKeys('%+m'); "activated=$ok"`]).toString().trim()

await sleep(2000)
log('loadUnpacked', JSON.stringify((await send('Extensions.loadUnpacked', { path: extDir })).result))
const { targetId } = (await send('Target.createTarget', { url: 'http://127.0.0.1:5599/' })).result
const page = (await send('Target.attachToTarget', { targetId, flatten: true })).result.sessionId
await send('Runtime.enable', {}, page)
await sleep(2500)
// close the initial about:blank tab so the meeting tab is the active one
for (const t of (await send('Target.getTargets')).result.targetInfos) if (t.type === 'page' && t.url === 'about:blank') await send('Target.closeTarget', { targetId: t.targetId })
await send('Target.activateTarget', { targetId })
await evalIn(page, `document.getElementById('remote').play()`, true)
await sleep(1000)
log('meeting audio before', await evalIn(page, `(() => { const a = document.getElementById('remote'); return { paused: a.paused, t: Math.round(a.currentTime * 10) / 10 } })()`))
let sw
for (let i = 0; i < 20 && !sw; i++) {
  sw = (await send('Target.getTargets')).result.targetInfos.find((x) => x.type === 'service_worker' && x.url.includes(extId))
  if (!sw) await sleep(500)
}
const swS = (await send('Target.attachToTarget', { targetId: sw.targetId, flatten: true })).result.sessionId
await send('Runtime.enable', {}, swS)
log('shortcut', JSON.stringify(await evalIn(swS, `chrome.commands.getAll().then((c) => c.map((x) => x.name + '=' + x.shortcut))`)))
const tabId = await evalIn(swS, `chrome.tabs.query({ url: 'http://127.0.0.1:5599/*' }).then((t) => t[0]?.id)`)
// Windows may refuse focus to a new window: press again until the extension reports it started (badge ON)
for (let attempt = 1; attempt <= 4; attempt++) {
  log(`press Alt+Shift+M (attempt ${attempt}):`, pressShortcut())
  await sleep(3500)
  if (await evalIn(swS, `chrome.action.getBadgeText({ tabId: ${tabId} })`) === 'ON') break
}
log('badge after shortcut', JSON.stringify(await evalIn(swS, `chrome.action.getBadgeText({ tabId: ${tabId} })`)), 'session', JSON.stringify(await evalIn(swS, `chrome.storage.session.get('session')`)))
log('active capture rooms on the server', await (await fetch('http://127.0.0.1:8010/captions/active')).text())
log('overlay', await evalIn(page, `!!document.querySelector('[data-mudra-overlay]')`))
log('meeting audio during capture (element keeps playing; offscreen replays the captured sound)', await evalIn(page, `(() => { const a = document.getElementById('remote'); return { paused: a.paused, t: Math.round(a.currentTime * 10) / 10 } })()`))
let panel
for (let i = 0; i < 30 && !panel; i++) {
  const f = (await send('Target.getTargets')).result.targetInfos.find((x) => x.url.startsWith('http://localhost:3000/meet-panel'))
  if (f) panel = (await send('Target.attachToTarget', { targetId: f.targetId, flatten: true })).result.sessionId
  else await sleep(500)
}
log('panel frame', Boolean(panel))
for (let i = 0; i < 18; i++) {
  await sleep(1000)
  if (panel && i % 2 === 1) log(`t+${i + 1}s`, JSON.stringify(await evalIn(panel, `({ caption: document.querySelector('[data-testid=caption]')?.innerText, status: document.querySelector('[data-testid=sign-status]')?.innerText })`)))
}
const offscreen = (await send('Target.getTargets')).result.targetInfos.find((x) => x.url.includes('offscreen.html'))
log('offscreen page', offscreen ? 'running' : 'none')
const shot = await send('Page.captureScreenshot', { format: 'png' }, page)
writeFileSync(out, Buffer.from(shot.result.data, 'base64'))
for (let attempt = 1; attempt <= 4; attempt++) {
  log(`press Alt+Shift+M again to stop (attempt ${attempt}):`, pressShortcut())
  await sleep(2500)
  if (await evalIn(swS, `chrome.action.getBadgeText({ tabId: ${tabId} })`) !== 'ON') break
}
log('badge after stop', JSON.stringify(await evalIn(swS, `chrome.action.getBadgeText({ tabId: ${tabId} })`)), '| server rooms', await (await fetch('http://127.0.0.1:8010/captions/active')).text())
log('panel after stop', panel ? JSON.stringify(await evalIn(panel, `document.querySelector('header')?.innerText`)) : '-')
const errs = events.filter((e) => e.method === 'Runtime.exceptionThrown' || (e.method === 'Runtime.consoleAPICalled' && ['error', 'warning'].includes(e.params.type)))
log('errors/warnings', errs.map((e) => JSON.stringify(e.params.args?.map((a) => a.value) || e.params.exceptionDetails?.text).slice(0, 200)).join('\n') || 'none')
proc.kill()
process.exit(0)
