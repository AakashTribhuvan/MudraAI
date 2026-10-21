import { DEFAULTS, loadConfig } from './config.js'

const web = document.getElementById('web')
const api = document.getElementById('api')
loadConfig().then((c) => { web.value = c.web; api.value = c.api })
document.getElementById('save').onclick = async () => {
  const clean = (v, d) => { try { return new URL(v || d).origin } catch { return d } }
  await chrome.storage.local.set({ web: clean(web.value, DEFAULTS.web), api: clean(api.value, DEFAULTS.api) })
  document.getElementById('saved').textContent = 'Saved.'
}
