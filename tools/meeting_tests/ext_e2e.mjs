// Extension end-to-end on the mock meeting: load the unpacked test extension over the DevTools pipe, open the
// mock meeting (plays speech.wav), start capture through the test hook, then check: overlay shown, tab audio still
// playing, captions and signs in the panel, meeting buttons still clickable, Stop ends capture.
// Usage: node ext_e2e.mjs <extDir> <extId> <out.png>
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
const [extDir, extId, out] = process.argv.slice(2)
const chrome = 'C:/Program Files/Google/Chrome/Application/chrome.exe'
const proc = spawn(chrome, ['--headless=new', '--remote-debugging-pipe', '--enable-unsafe-extension-debugging', `--allowlisted-extension-id=${extId}`,
  '--autoplay-policy=no-user-gesture-required', '--enable-gpu', '--ignore-gpu-blocklist', '--use-fake-ui-for-media-stream',
  `--user-data-dir=${mkdtempSync(join(tmpdir(), 'ext-e2e-'))}`, '--window-size=1280,800', 'about:blank'], { stdio: ['ignore', 'ignore', 'pipe', 'pipe', 'pipe'] })
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
const evalIn = async (sessionId, expression) => {
  const r = await send('Runtime.evaluate', { expression, awaitPromise: true, returnByValue: true }, sessionId)
  return r.result?.exceptionDetails ? { error: r.result.exceptionDetails.exception?.description || r.result.exceptionDetails.text } : r.result?.result?.value
}
await sleep(1500)
const loaded = await send('Extensions.loadUnpacked', { path: extDir })
log('loadUnpacked', JSON.stringify(loaded.result || loaded.error))
const { targetId } = (await send('Target.createTarget', { url: 'http://127.0.0.1:5599/' })).result
const page = (await send('Target.attachToTarget', { targetId, flatten: true })).result.sessionId
await send('Runtime.enable', {}, page)
await sleep(3000)
log('page audio playing', await evalIn(page, `(() => { const a = document.getElementById('remote'); return { paused: a.paused, t: Math.round(a.currentTime * 10) / 10 } })()`))
// the extension's service worker
let sw
for (let i = 0; i < 20 && !sw; i++) {
  const t = (await send('Target.getTargets')).result.targetInfos
  sw = t.find((x) => x.type === 'service_worker' && x.url.includes(extId))
  if (!sw) await sleep(500)
}
const swSession = (await send('Target.attachToTarget', { targetId: sw.targetId, flatten: true })).result.sessionId
await send('Runtime.enable', {}, swSession)
const tabs = await evalIn(swSession, `chrome.tabs.query({ url: 'http://127.0.0.1:5599/*' }).then((t) => t.map((x) => x.id))`)
log('tabs', JSON.stringify(tabs))
const ROOM = 'e2eroom' + Date.now()
const started = await evalIn(swSession, `__mudraTestShowPanel(${tabs[0]}, '${ROOM}').then((s) => JSON.stringify(s), (e) => 'ERR ' + e.message)`)
log('show panel', started)
const producer = spawn(process.execPath, [join(HERE, 'room_producer.mjs'), join(WORK, 'mock_meeting', 'speech.wav'), ROOM, '6000'], { stdio: ['ignore', 'pipe', 'inherit'] })
producer.stdout.on('data', (d) => log(String(d).trim()))
await sleep(2500)
log('overlay in page', await evalIn(page, `!!document.querySelector('[data-mudra-overlay]')`))
log('page audio still playing', await evalIn(page, `(() => { const a = document.getElementById('remote'); return { paused: a.paused, t: Math.round(a.currentTime * 10) / 10 } })()`))
// find the panel frame (localhost:3000/meet-panel) and read it
let panel
for (let i = 0; i < 30 && !panel; i++) {
  const t = (await send('Target.getTargets')).result.targetInfos
  const f = t.find((x) => x.url.startsWith('http://localhost:3000/meet-panel'))
  if (f) panel = (await send('Target.attachToTarget', { targetId: f.targetId, flatten: true })).result.sessionId
  else await sleep(500)
}
log('panel frame attached', Boolean(panel))
const samples = []
for (let i = 0; i < 16; i++) {
  await sleep(1000)
  if (panel) samples.push(await evalIn(panel, `({ caption: document.querySelector('[data-testid=caption]')?.innerText, status: document.querySelector('[data-testid=sign-status]')?.innerText, chips: document.querySelector('header')?.innerText })`))
}
samples.forEach((s, i) => log(`t+${i + 1}s`, JSON.stringify(s)))
log('meeting button still clickable', await evalIn(page, `(() => { const b = document.getElementById('mic'); const r = b.getBoundingClientRect(); const el = document.elementFromPoint(r.x + r.width / 2, r.y + r.height / 2); el.click(); return { topElementIsButton: el === b, clicks: window.__micClicks } })()`))
const shot = await send('Page.captureScreenshot', { format: 'png' }, page)
writeFileSync(out, Buffer.from(shot.result.data, 'base64'))
// Stop from the panel, as the user would
if (panel) log('click stop', await evalIn(panel, `(() => { const b = [...document.querySelectorAll('button')].find((x) => /Stop listening/.test(x.textContent)); b?.click(); return Boolean(b) })()`))
await sleep(2000)
log('after stop: panel', panel ? JSON.stringify(await evalIn(panel, `document.querySelector('header')?.innerText`)) : '-')
log('after stop: badge', await evalIn(swSession, `chrome.action.getBadgeText({ tabId: ${tabs[0]} })`))
log('after stop: page audio', await evalIn(page, `(() => { const a = document.getElementById('remote'); return { paused: a.paused } })()`))
const errs = events.filter((e) => e.method === 'Runtime.exceptionThrown' || (e.method === 'Runtime.consoleAPICalled' && ['error', 'warning'].includes(e.params.type)))
log('errors', errs.map((e) => JSON.stringify(e.params).slice(0, 300)).join('\n') || 'none')
proc.kill()
process.exit(0)
