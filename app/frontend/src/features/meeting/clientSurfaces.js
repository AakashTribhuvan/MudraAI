/**
 * In-client surfaces for the meeting panel: a Google Meet add-on (side panel + main stage) and a Zoom App.
 * The official SDKs are loaded from their vendors' CDNs when the panel runs inside that client; nothing here
 * accesses meeting media (Meet's Media API is a closed preview; Zoom RTMS needs an approved app), so the audio still
 * comes from the MUDRA extension's tab capture or a shared tab / screen, joined through a local caption room.
 *
 * Both need registration under the owner's accounts and an HTTPS address before they run (docs/MEETING_INTEGRATION.md):
 *   Meet: a Google Cloud project with the Google Workspace Marketplace SDK, add-on side panel and main stage URLs,
 *         VITE_MEET_CLOUD_PROJECT_NUMBER.
 *   Zoom: a Zoom App (Marketplace, "General app"), home URL, domain allow list, the capabilities below.
 */

const MEET_SDK = import.meta.env?.VITE_MEET_ADDONS_SDK_URL || 'https://www.gstatic.com/meetjs/addons/1.2.0/meet.addons.js'
const ZOOM_SDK = import.meta.env?.VITE_ZOOM_APPS_SDK_URL || 'https://appssdk.zoom.us/sdk.min.js'
const MEET_PROJECT = import.meta.env?.VITE_MEET_CLOUD_PROJECT_NUMBER || ''

const loaded = new Map()
function loadScript(src) {
  if (!loaded.has(src)) {
    loaded.set(src, new Promise((resolve, reject) => {
      const s = document.createElement('script')
      s.src = src
      s.async = true
      s.onload = resolve
      s.onerror = () => { loaded.delete(src); reject(new Error(`Could not load ${src}`)) }
      document.head.appendChild(s)
    }))
  }
  return loaded.get(src)
}

/** surface: 'meet-sidepanel' | 'meet-mainstage' | 'zoom' | null (the ?surface= parameter of /meet-panel). */
export function surfaceFromUrl(search = window.location.search) {
  const s = new URLSearchParams(search).get('surface')
  return ['meet-sidepanel', 'meet-mainstage', 'zoom'].includes(s) ? s : null
}

/** Google Meet add-on: session + side panel or main stage client. Returns { client, startingRoom, error }. */
export async function initMeetAddon(surface) {
  if (!MEET_PROJECT) return { error: 'VITE_MEET_CLOUD_PROJECT_NUMBER is not set (the add-on is not registered yet).' }
  try {
    await loadScript(MEET_SDK)
    const session = await window.meet.addon.createAddonSession({ cloudProjectNumber: MEET_PROJECT })
    if (surface === 'meet-mainstage') {
      const client = await session.createMainStageClient()
      const state = await client.getActivityStartingState?.()
      let room = null
      try { room = JSON.parse(state?.additionalData || '{}').room || null } catch { room = null }
      return { client, startingRoom: room }
    }
    return { client: await session.createSidePanelClient() }
  } catch (err) {
    return { error: `Meet add-on: ${err?.message || err}` }
  }
}

/** Side panel: show the avatar to everyone in the call on the main stage (an add-on activity). */
export async function shareToMainStage(client, room) {
  const url = new URL(window.location.href)
  url.search = `?surface=meet-mainstage${room ? `&room=${encodeURIComponent(room)}` : ''}`
  await client.startActivity({ mainStageUrl: url.toString(), additionalData: JSON.stringify({ room }) })
}

/** Zoom App inside the Zoom client: configure the SDK and read where it runs. Returns { context, meeting, error }. */
export async function initZoomApp() {
  try {
    await loadScript(ZOOM_SDK)
    const sdk = window.zoomSdk
    const cfg = await sdk.config({ capabilities: ['getRunningContext', 'getMeetingContext', 'getUserContext'], version: '0.16' })
    let meeting = null
    if (cfg.runningContext === 'inMeeting') {
      try { meeting = await sdk.getMeetingContext() } catch { meeting = null }
    }
    return { context: cfg.runningContext, meeting }
  } catch (err) {
    return { error: `Zoom App: ${err?.message || err}` }
  }
}
