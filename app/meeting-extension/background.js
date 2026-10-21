/**
 * MUDRA meeting extension - service worker.
 *
 * Clicking the toolbar button on a Google Meet or Zoom Web tab (the user gesture tab capture needs):
 *   1. chrome.tabCapture.getMediaStreamId for that tab,
 *   2. the offscreen page captures the tab's audio, keeps it playing (the meeting stays audible) and streams it as
 *      16 kHz PCM to the local MUDRA server (/asr?room=R) - nothing is recorded or sent elsewhere,
 *   3. the content script shows the MUDRA panel in the page, listening to caption room R.
 * Clicking again, the panel's Stop, closing or leaving the tab stops the capture. One tab at a time.
 */
import { isMeetingUrl, loadConfig, randomRoom } from './config.js'

// { tabId, room }; kept in storage.session because the service worker may be stopped while the capture runs
let session = null
const loaded = chrome.storage.session.get('session').then((v) => { session = v.session || null })
const save = () => chrome.storage.session.set({ session })

async function ensureOffscreen() {
  if (await chrome.offscreen.hasDocument()) return
  await chrome.offscreen.createDocument({
    url: 'offscreen.html',
    reasons: ['USER_MEDIA', 'AUDIO_PLAYBACK'],
    justification: 'Captures the meeting tab audio for local speech recognition and keeps playing it so the meeting stays audible.',
  })
}

async function setBadge(tabId, on, error = false) {
  await chrome.action.setBadgeText({ tabId, text: on ? 'ON' : error ? '!' : '' })
  await chrome.action.setBadgeBackgroundColor({ tabId, color: error ? '#b45309' : '#dc2626' })
  await chrome.action.setTitle({ tabId, title: on ? 'MUDRA is listening to this tab - click to stop' : 'MUDRA: start ISL captions for this meeting tab' })
}

async function stop(reason = 'stopped') {
  if (!session) return
  const { tabId } = session
  session = null
  await save()
  try { await chrome.runtime.sendMessage({ target: 'offscreen', type: 'stop-capture' }) } catch { /* offscreen already gone */ }
  try { await chrome.tabs.sendMessage(tabId, { type: 'capture-stopped', reason }) } catch { /* tab closed */ }
  chrome.runtime.sendMessage({ target: 'panel', type: 'capture-stopped', reason }).catch(() => {}) // the panel frame
  await setBadge(tabId, false).catch(() => {})
}

async function start(tab) {
  const config = await loadConfig()
  if (!isMeetingUrl(tab.url)) {
    // not a supported meeting page: open the panel on its own (share a tab or the screen's audio from there)
    await chrome.windows.create({ url: `${config.web}/meet-panel`, type: 'popup', width: 400, height: 780 })
    return
  }
  if (session) await stop('another tab started')
  await ensureOffscreen() // first, as in Chrome's tabCapture sample: the stream id is used right after it is made
  const streamId = await chrome.tabCapture.getMediaStreamId({ targetTabId: tab.id })
  const room = randomRoom()
  session = { tabId: tab.id, room }
  await save()
  const reply = await chrome.runtime.sendMessage({ target: 'offscreen', type: 'start-capture', streamId, room, api: config.api })
  if (!reply?.ok) {
    console.warn('MUDRA: tab audio capture failed:', reply?.error)
    session = null
    await save()
    await setBadge(tab.id, false, true)
    await chrome.tabs.sendMessage(tab.id, { type: 'capture-error', error: reply?.error || 'Tab audio could not be captured.' }).catch(() => {})
    return
  }
  await setBadge(tab.id, true)
  await chrome.tabs.sendMessage(tab.id, { type: 'show-panel', room, web: config.web })
}

chrome.action.onClicked.addListener(async (tab) => {
  await loaded
  try {
    if (session?.tabId === tab.id) await stop('stopped from the toolbar')
    else await start(tab)
  } catch (err) {
    console.error('MUDRA:', err)
    await setBadge(tab.id, false, true).catch(() => {})
    await chrome.tabs.sendMessage(tab.id, { type: 'capture-error', error: String(err?.message || err) }).catch(() => {})
  }
})

chrome.runtime.onMessage.addListener((msg, sender, reply) => {
  if (msg?.target !== 'background') return undefined
  if (msg.type === 'status') { loaded.then(() => reply({ active: Boolean(session && sender.tab?.id === session.tabId), room: session?.room })); return true }
  if (msg.type === 'stop-capture') { loaded.then(() => stop('stopped from the panel')).then(() => reply({ ok: true })); return true }
  if (msg.type === 'capture-ended') { loaded.then(() => stop(msg.reason || 'the tab audio ended')); return undefined }
  return undefined
})

chrome.tabs.onRemoved.addListener((tabId) => loaded.then(() => { if (session?.tabId === tabId) stop('the tab was closed') }))
chrome.tabs.onUpdated.addListener((tabId, info) => loaded.then(() => {
  if (session?.tabId === tabId && info.url && !isMeetingUrl(info.url)) stop('the tab left the meeting')
}))
