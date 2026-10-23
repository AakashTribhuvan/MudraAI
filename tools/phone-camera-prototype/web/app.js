const $ = (id) => document.getElementById(id)

const controls = {
  start_hold: { formatter: (value) => `${Number(value).toFixed(2)} s` },
  stop_hold: { formatter: (value) => `${Number(value).toFixed(2)} s` },
  mode_hold: { formatter: (value) => `${Number(value).toFixed(2)} s` },
  movement_threshold: { formatter: (value) => Number(value).toFixed(2) },
  raised_below_shoulder: { formatter: (value) => Number(value).toFixed(2) }
}

const statusNames = { idle: "IDLE", starting: "STARTING", signing: "SIGNING", ending: "ENDING" }
const modeNames = { idle: "—", static: "STATIC", dynamic: "DYNAMIC", setup: "SETUP" }
const settingTimers = new Map()
const cameraSelect = $("camera-select")
const setupMessage = $("setup-message")
const setupPreview = $("setup-camera-feed")
const cameraFeed = $("camera-feed")
const dynamicCameraFeed = $("dynamic-camera-feed")
const setupPlaceholder = $("setup-placeholder")
const videoPlaceholder = $("video-placeholder")
const dynamicVideoPlaceholder = $("dynamic-video-placeholder")
const cameraSetup = $("camera-setup")
const modelWorkspaces = $("model-workspaces")
const staticSection = $("static-sign-section")
const dynamicSection = $("dynamic-sign-section")
const flipCameraButtons = document.querySelectorAll("[data-flip-camera]")

let statusSnapshot = null
let settingsInitialized = false
let currentSection = "static"
let modelInitialized = false
let switchingCamera = false
let renderedCameraOptions = null
let renderedCatalog = null
let renderedSetup = null
let renderedTracking = null
let renderedFeedSection = null

async function requestJson(path, payload) {
  const response = await fetch(path, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
    cache: "no-store"
  })
  const result = await response.json()
  if (!response.ok) throw new Error(result.error || `Request failed (${response.status})`)
  return result
}

function renderCameraOptions(status) {
  const options = status.cameras || []
  const selected = status.selected_camera || {}
  const value = `${selected.index}:${selected.backend}`
  cameraSelect.replaceChildren()
  for (const camera of options) {
    const option = document.createElement("option")
    option.value = `${camera.index}:${camera.backend_key}`
    option.textContent =
      `Camera ${camera.index} · ${camera.backend} · ${camera.width} × ${camera.height}`
    cameraSelect.append(option)
  }
  cameraSelect.disabled = options.length === 0 || switchingCamera
  $("start-tracking").disabled = options.length === 0 || switchingCamera
  if (options.some((camera) => `${camera.index}:${camera.backend_key}` === value)) {
    cameraSelect.value = value
  }
  const active = options.find((camera) => `${camera.index}:${camera.backend_key}` === value)
  $("setup-camera-name").textContent = active
    ? `Camera ${active.index} · ${active.width} × ${active.height}`
    : "No camera selected"
  if (!options.length) {
    setupMessage.textContent = "No live cameras found. Connect Camo, then restart the app."
  } else if (!switchingCamera) {
    setupMessage.textContent = "Preview the selected camera, then start tracking."
  }
  $("active-camera-name").textContent = active
    ? `Camera ${active.index} · ${active.backend} · ${active.width} × ${active.height}`
    : "Camera selected"
}

function renderModelCatalog(catalog, selectedMode) {
  const selectedModel = catalog?.[selectedMode] || {}
  const dynamicModel = catalog?.dynamic || {}
  const words = selectedModel.words || []
  $("static-trained-count").textContent = selectedModel.trained_words ?? 0
  $("static-live-count").textContent = selectedModel.live_words ?? 0
  $("static-held-count").textContent =
    `${Math.max(0, (selectedModel.trained_words || 0) - (selectedModel.live_words || 0))} labels withheld: no calibrated live threshold`
  $("dynamic-trained-count").textContent = dynamicModel.trained_words ?? 0
  $("reference-word-count").textContent = words.length

  const list = $("sign-reference-list")
  list.replaceChildren()
  if (!words.length) {
    const empty = document.createElement("span")
    empty.className = "reference-empty"
    empty.textContent = "No static model labels are loaded."
    list.append(empty)
    return
  }

  for (const word of words) {
    const card = document.createElement("button")
    card.type = "button"
    card.className = "sign-reference-item"
    card.title = `${word.label.replace(/_/g, " ")}${word.live ? " · live eligible" : " · withheld from live predictions"}`
    const image = document.createElement("img")
    image.alt = ""
    image.loading = "lazy"
    if (word.preview_url) image.src = word.preview_url
    const label = document.createElement("span")
    label.textContent = word.label.replace(/_/g, " ")
    const state = document.createElement("small")
    state.textContent = word.live ? "LIVE-ELIGIBLE" : "PREVIEW ONLY"
    card.append(image, label, state)
    card.addEventListener("click", () => {
      document.querySelectorAll(".sign-reference-item.is-selected").forEach((item) => {
        item.classList.remove("is-selected")
      })
      card.classList.add("is-selected")
      $("recognition-caption").textContent =
        `${word.live ? "Live-eligible reference" : "Preview only; live prediction withheld"} · folder label user-confirmed; sign form not independently validated.`
    })
    image.addEventListener("error", () => {
      image.hidden = true
    })
    list.append(card)
  }
}

function renderRecognition(recognition) {
  const label = $("recognition-label")
  const caption = $("recognition-caption")
  const states = {
    disabled: ["OFF", "Select Static Pose Model to enable its matcher."],
    waiting: ["READY", "Hold a tracked pose to try a match."],
    starting: ["STARTING", "Waiting for the sign boundary to settle."],
    ending: ["ENDING", "Return to a held pose to continue matching."],
    dynamic: ["STATIC ONLY", "The still-image model does not recognize dynamic signs."],
    confirming: ["CHECKING…", "Waiting for the same pose to persist."],
    tracking: ["CHECK TRACKING", recognition.reason?.replace(/_/g, " ") || "Keep the pose and hands clearly in frame."],
    unknown: ["UNKNOWN", recognition.reason?.replace(/_/g, " ") || "No reliable match; try a clearer held pose."],
    recognized: [recognition.label?.replace(/_/g, " ") || "MATCH", "Experimental match · source label not verified."]
  }
  const [title, message] = states[recognition.state] || states.waiting
  label.textContent = title
  caption.textContent = message
}

function renderRuntime(runtime) {
  const state = runtime.state || "idle"
  const mode = runtime.mode || "idle"
  const stateName = statusNames[state] || state.toUpperCase()
  $("activity-state").textContent = stateName
  $("activity-mode").textContent = modeNames[mode] || "—"
  $("video-state-chip").textContent = `${stateName}${mode !== "idle" ? ` · ${modeNames[mode] || mode}` : ""}`
  $("segment-time").textContent = Number(runtime.elapsed || 0).toFixed(1)
  $("wrist-speed").textContent = Number(runtime.wrist_speed || 0).toFixed(2)
  $("hands-count").textContent = Number(runtime.hands || 0)
  $("fps-value").textContent = Math.round(Number(runtime.fps || 0))
  $("raised-status").textContent = runtime.raised ? "Signing-height zone active" : "Hands below active zone"
  $("resolution-value").textContent =
    runtime.resolution || "Camera size pending"
  $("camera-status").textContent = runtime.event === "refreshed"
    ? "Static tracking refreshed · every 5 seconds"
    : Number.isFinite(runtime.static_refresh_in)
      ? `Static refresh in ${Number(runtime.static_refresh_in).toFixed(1)}s`
      : "Live preview · not recorded"
  $("state-caption").textContent = state === "signing" && mode === "static"
      ? "Still hands are tracked as a sign; the static segment refreshes every 5 seconds"
      : state === "signing" && mode === "dynamic"
        ? "Sustained hand movement detected"
        : state === "ending"
          ? "Waiting for rest to confirm the end"
          : state === "starting"
            ? "Checking that activity continues"
            : "Waiting for a clear signing movement"
  renderRecognition(runtime.recognition || {})
  $("dynamic-tracking-state").textContent =
    `Tracker: ${stateName} · ${runtime.resolution || "camera size pending"} · ${Math.round(Number(runtime.fps || 0))} FPS · no dynamic prediction model`
}

function renderWorkspace(status) {
  statusSnapshot = status
  for (const button of flipCameraButtons) {
    const mirrored = status.camera_mirrored !== false
    button.setAttribute("aria-pressed", String(mirrored))
    button.textContent = mirrored ? "Flip camera · mirrored" : "Flip camera · natural"
  }
  if (!modelInitialized) {
    currentSection = status.model_mode || "static"
    modelInitialized = true
  }
  const setupVisible = status.camera_setup && !status.tracking_enabled
  if (renderedSetup !== setupVisible || renderedTracking !== status.tracking_enabled) {
    cameraSetup.hidden = !setupVisible
    modelWorkspaces.hidden = !status.tracking_enabled
    if (setupVisible) {
      setupPreview.src = setupPreview.dataset.streamSrc
      cameraFeed.removeAttribute("src")
      dynamicCameraFeed.removeAttribute("src")
      renderedFeedSection = null
    } else {
      setupPreview.removeAttribute("src")
      renderActiveSection()
    }
    renderedSetup = setupVisible
    renderedTracking = status.tracking_enabled
  } else if (status.tracking_enabled) {
    renderActiveSection()
  }
  const cameraSignature = JSON.stringify([status.cameras, status.selected_camera])
  if (cameraSignature !== renderedCameraOptions) {
    renderCameraOptions(status)
    renderedCameraOptions = cameraSignature
  }
  const catalogSignature = JSON.stringify([status.model_mode, status.model_catalog])
  if (catalogSignature !== renderedCatalog) {
    renderModelCatalog(status.model_catalog, status.model_mode || "static")
    renderedCatalog = catalogSignature
  }
  if (!settingsInitialized) {
    for (const [key, formatter] of Object.entries({
      start_hold: (value) => `${Number(value).toFixed(2)} s`,
      stop_hold: (value) => `${Number(value).toFixed(2)} s`,
      mode_hold: (value) => `${Number(value).toFixed(2)} s`,
      movement_threshold: (value) => Number(value).toFixed(2),
      raised_below_shoulder: (value) => Number(value).toFixed(2)
    })) {
      if (status.settings && Object.hasOwn(status.settings, key)) {
        $(key).value = status.settings[key]
        $(`${key}_value`).textContent = formatter(status.settings[key])
      }
    }
    settingsInitialized = true
  }
  renderRuntime(status.runtime || {})
}

function renderActiveSection() {
  const staticActive = currentSection !== "dynamic"
  staticSection.hidden = !staticActive
  dynamicSection.hidden = staticActive
  for (const [id, selected] of [
    ["static-section-tab", currentSection === "static"],
    ["intro-section-tab", currentSection === "intro"],
    ["dynamic-section-tab", currentSection === "dynamic"]
  ]) {
    $(id).classList.toggle("is-active", selected)
    $(id).setAttribute("aria-selected", String(selected))
  }
  const section = staticActive ? "static" : "dynamic"
  if (renderedFeedSection === section) return
  const activeFeed = staticActive ? cameraFeed : dynamicCameraFeed
  const inactiveFeed = staticActive ? dynamicCameraFeed : cameraFeed
  inactiveFeed.removeAttribute("src")
  activeFeed.src = activeFeed.dataset.streamSrc
  renderedFeedSection = section
}

async function selectModel(section) {
  currentSection = section
  await requestJson("/api/model", { mode: section })
  renderActiveSection()
}

async function refreshStatus() {
  try {
    const response = await fetch("/api/status", { cache: "no-store" })
    if (!response.ok) throw new Error(`Status request failed (${response.status})`)
    const status = await response.json()
    renderWorkspace(status)
  } catch (error) {
    setupMessage.textContent = error.message || "Waiting for the local camera service."
    $("connection-state").textContent = "RECONNECTING"
  }
}

async function updateSettings(changes) {
  $("settings-status").textContent = "Applying changes…"
  try {
    await requestJson("/api/settings", changes)
    $("settings-status").textContent = "Live settings applied"
  } catch (error) {
    $("settings-status").textContent = error.message || "Could not apply settings"
  }
}

cameraSelect.addEventListener("change", async () => {
  const [index, backend] = cameraSelect.value.split(":")
  switchingCamera = true
  cameraSelect.disabled = true
  $("start-tracking").disabled = true
  setupMessage.textContent = "Switching camera preview…"
  try {
    await requestJson("/api/camera", { index: Number(index), backend })
  } catch (error) {
    setupMessage.textContent = error.message || "Could not select camera."
  } finally {
    switchingCamera = false
    if (statusSnapshot) {
      renderedCameraOptions = null
      renderCameraOptions(statusSnapshot)
    }
  }
})

$("start-tracking").addEventListener("click", async () => {
  $("start-tracking").disabled = true
  setupMessage.textContent = "Starting pose trackers…"
  try {
    await requestJson("/api/session", { enabled: true })
  } catch (error) {
    setupMessage.textContent = error.message || "Could not start tracking."
    $("start-tracking").disabled = false
  }
})

$("flip-camera").addEventListener("click", async () => {
  const mirrored = !(statusSnapshot?.camera_mirrored ?? true)
  try {
    await requestJson("/api/camera/flip", { mirrored })
    await refreshStatus()
  } catch (error) {
    $("setup-message").textContent = error.message || "Could not flip the camera image."
    $("camera-status").textContent = error.message || "Could not flip the camera image."
  }
})

$("stop-tracking").addEventListener("click", async () => {
  try {
    await requestJson("/api/session", { enabled: false })
  } catch (error) {
    $("camera-status").textContent = error.message || "Could not stop tracking."
  }
})

$("static-section-tab").addEventListener("click", () => {
  selectModel("static").catch((error) => {
    $("recognition-caption").textContent = error.message
  })
  $("intro-section-tab").addEventListener("click", () => {
    selectModel("intro").catch((error) => {
      $("recognition-caption").textContent = error.message
    })
  })
})
$("dynamic-section-tab").addEventListener("click", () => {
  selectModel("dynamic").catch((error) => {
    $("dynamic-tracking-state").textContent = error.message
  })
})

for (const [key, config] of Object.entries(controls)) {
  $(key).addEventListener("input", (event) => {
    const value = Number(event.target.value)
    $(`${key}_value`).textContent = config.formatter(value)
    window.clearTimeout(settingTimers.get(key))
    settingTimers.set(key, window.setTimeout(() => updateSettings({ [key]: value }), 140))
  })
}

$("reset-settings").addEventListener("click", async () => {
  for (const timer of settingTimers.values()) window.clearTimeout(timer)
  await updateSettings({
    start_hold: 0.15,
    stop_hold: 0.7,
    mode_hold: 0.45,
    movement_threshold: 1.0,
    raised_below_shoulder: 1.0
  })
})

setupPreview.addEventListener("load", () => { setupPlaceholder.hidden = true })
setupPreview.addEventListener("error", () => { setupPlaceholder.hidden = false })
cameraFeed.addEventListener("load", () => { videoPlaceholder.hidden = true })
cameraFeed.addEventListener("error", () => { videoPlaceholder.hidden = false })
dynamicCameraFeed.addEventListener("load", () => {
  dynamicCameraFeed.hidden = false
  dynamicVideoPlaceholder.hidden = true
})
dynamicCameraFeed.addEventListener("error", () => {
  dynamicCameraFeed.hidden = true
  dynamicVideoPlaceholder.hidden = false
})

refreshStatus()
window.setInterval(refreshStatus, 500)
