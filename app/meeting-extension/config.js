// Where the local MUDRA app runs (start_all.bat): the web app serves the panel, the API does speech recognition.
export const DEFAULTS = { web: 'http://localhost:3000', api: 'http://127.0.0.1:8010' }

export async function loadConfig() {
  const stored = await chrome.storage.local.get(['web', 'api'])
  return { web: stored.web || DEFAULTS.web, api: stored.api || DEFAULTS.api }
}

export const SUPPORTED = [/^https:\/\/meet\.google\.com\/[a-z]{3}-[a-z]{4}-[a-z]{3}/, /^https:\/\/([a-z0-9-]+\.)?zoom\.us\/wc\//]
export const isMeetingUrl = (url) => SUPPORTED.some((re) => re.test(url || ''))

export function randomRoom() {
  return Array.from(crypto.getRandomValues(new Uint8Array(12)), (b) => b.toString(16).padStart(2, '0')).join('')
}
