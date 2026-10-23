/**
 * MUDRA Hand Tracker Service
 * MediaPipe Tasks (HandLandmarker + PoseLandmarker) on the webcam, matching the
 * landmark pipeline used to train the recognition model. Models and WASM are served
 * locally from /models and /wasm, so no camera data or model download leaves the device.
 *
 * Results passed to onResults:
 *   multiHandLandmarks  Array<Array<{x,y,z}>>   up to 2 hands, 21 points each
 *   multiHandedness     Array<{label, score}>   MediaPipe labels (assume a mirrored image)
 *   poseLandmarks       Array<{x,y,z,visibility}> | null   33 body points
 *   width, height       video size in pixels
 *   diag                {cameraFps, trackingFps, inferenceMs, delegate}: what limits the frame rate
 */

import { FilesetResolver, HandLandmarker, PoseLandmarker } from '@mediapipe/tasks-vision'

const ASSET_BASE = import.meta.env.BASE_URL
const WASM_PATH = `${ASSET_BASE}wasm`
const HAND_MODEL = `${ASSET_BASE}models/hand_landmarker.task`
const POSE_MODEL = `${ASSET_BASE}models/pose_landmarker_lite.task`

// MediaPipe Hand Connection Pairs
export const HAND_CONNECTIONS = [
  // Thumb
  [0, 1], [1, 2], [2, 3], [3, 4],
  // Index
  [0, 5], [5, 6], [6, 7], [7, 8],
  // Middle
  [0, 9], [9, 10], [10, 11], [11, 12],
  // Ring
  [0, 13], [13, 14], [14, 15], [15, 16],
  // Pinky
  [0, 17], [17, 18], [18, 19], [19, 20],
  // Palm Base Knuckle Bridge
  [5, 9], [9, 13], [13, 17]
]

// Upper-body pose connections (shoulders, arms, face outline)
export const POSE_CONNECTIONS = [
  [11, 12], [11, 13], [13, 15], [12, 14], [14, 16], [11, 23], [12, 24], [23, 24]
]

async function createWithFallback(factory, vision, options, first = 'GPU') {
  const other = first === 'GPU' ? 'CPU' : 'GPU'
  try {
    const landmarker = await factory.createFromOptions(vision, { ...options, baseOptions: { ...options.baseOptions, delegate: first } })
    return { landmarker, delegate: first }
  } catch (firstError) {
    console.warn(`${first} delegate unavailable, falling back to ${other}:`, firstError)
    const landmarker = await factory.createFromOptions(vision, { ...options, baseOptions: { ...options.baseOptions, delegate: other } })
    return { landmarker, delegate: other }
  }
}

const HAND_OPTIONS = {
  baseOptions: { modelAssetPath: HAND_MODEL },
  runningMode: 'VIDEO',
  numHands: 2,
  minHandDetectionConfidence: 0.5,
  minHandPresenceConfidence: 0.5,
  minTrackingConfidence: 0.5
}
const POSE_OPTIONS = {
  baseOptions: { modelAssetPath: POSE_MODEL },
  runningMode: 'VIDEO',
  numPoses: 1,
  minPoseDetectionConfidence: 0.5
}

// Faster delegate: some laptops run MediaPipe slower on the (integrated) GPU than on the CPU. The first
// PROBE_FRAMES frames of a session are timed; above SLOW_MS the other delegate is loaded and timed too, the faster
// one is kept and remembered in this browser (DELEGATE_KEY), so the next session starts with it.
const PROBE_WARMUP = 5
const POSE_SLOW_MS = 50
// pose on every frame below POSE_FAST_MS per frame (hands + pose), back to every other frame above POSE_BACK_MS
// (two limits, so the rate does not flip on every frame around one value)
const POSE_FAST_MS = 22
const POSE_BACK_MS = 30
const PROBE_FRAMES = 20
const SLOW_MS = 35
const DELEGATE_KEY = 'signline_tracker_delegate'
const median = (xs) => [...xs].sort((a, b) => a - b)[Math.floor(xs.length / 2)]
const DECISION_DAYS = 30 // a measured choice is kept this long, then measured again (drivers change)
const readSaved = () => {
  try { return JSON.parse(localStorage.getItem(DELEGATE_KEY) || 'null') } catch { return null }
}
const readDelegate = () => readSaved()?.delegate || 'GPU'
/** True when this browser already measured both delegates recently: no more probing, no model swaps. */
const decided = () => {
  const s = readSaved()
  return Boolean(s?.timings && s.timings.GPU != null && s.timings.CPU != null && Date.now() - (s.at || 0) < DECISION_DAYS * 864e5)
}
const saveDelegate = (delegate, timings) => {
  try { localStorage.setItem(DELEGATE_KEY, JSON.stringify({ delegate, timings, at: Date.now() })) } catch { /* private window */ }
}

/**
 * The GPU the browser draws with (WebGL renderer name), e.g. "Intel(R) UHD Graphics" or "NVIDIA GeForce RTX 5060".
 * integrated: an on-chip GPU. On a laptop that also has an NVIDIA / AMD card, Windows gives Chrome the on-chip one
 * unless Chrome is set to "High performance" (measured on the development laptop: 46 ms -> 17 ms per frame).
 */
/** WebGL renderer name -> { name: short readable name, integrated: an on-chip GPU }. */
export function describeGpu(renderer = '') {
  // "ANGLE (Intel, Intel(R) UHD Graphics (0x0000A7A8) Direct3D11 vs_5_0 ps_5_0, D3D11)" -> "Intel(R) UHD Graphics"
  const name = renderer.replace(/^ANGLE \([^,]+,\s*/, '').replace(/\s*\(0x[0-9a-f]+\).*$/i, '').replace(/\s*Direct3D.*$/, '').trim()
  return { name, integrated: /intel|uhd|iris|radeon\(tm\) graphics|vega \d+ graphics|microsoft basic|swiftshader/i.test(renderer) }
}

let gpuCache = null
export function browserGpu() {
  if (gpuCache) return gpuCache
  let renderer = ''
  try {
    const gl = document.createElement('canvas').getContext('webgl2') || document.createElement('canvas').getContext('webgl')
    const ext = gl?.getExtension('WEBGL_debug_renderer_info')
    renderer = ext ? gl.getParameter(ext.UNMASKED_RENDERER_WEBGL) : ''
  } catch { renderer = '' }
  gpuCache = describeGpu(renderer)
  return gpuCache
}

const ema = (prev, value, k = 0.1) => (prev ? (1 - k) * prev + k * value : value)

// Resolutions offered in the UI (the browser may negotiate another size; the video reports the real one)
export const CAMERA_RESOLUTIONS = { 480: { width: 640, height: 480 }, 720: { width: 1280, height: 720 } }

/** Why the camera could not start, from the getUserMedia error name. */
export function cameraErrorKind(error) {
  switch (error?.name) {
    case 'NotAllowedError':
    case 'SecurityError':
      return 'denied'
    case 'NotFoundError':
    case 'OverconstrainedError':
      return 'not_found'
    case 'NotReadableError':
    case 'AbortError':
      return 'in_use'
    default:
      return 'unknown'
  }
}

const BLACK_LUMA = 8 // mean brightness (0-255) below which a frame counts as black
const FROZEN_S = 3 // no new video frame for this long = frozen feed

class HandTrackerService {
  constructor() {
    this.handLandmarker = null
    this.poseLandmarker = null
    this.videoElement = null
    this.stream = null
    this.isTracking = false
    this.animationFrameId = null
    this.onResultsCallback = null
    this.initializationPromise = null
    this.lastVideoTime = -1
    this.lastTimestamp = 0
    this.delegate = null // 'GPU' or 'CPU' (CPU if either landmarker fell back)
    this.cameraFps = 0 // frames the camera delivers
    this.inferenceMs = 0 // hand + pose time per processed frame
    this.healthTimer = null
    this.onIssueCallback = null // called with 'black' | 'frozen' | null when the feed's health changes
    this.issue = null
  }

  /** Video inputs ({deviceId, label}); labels are empty until camera permission has been granted. */
  async listCameras() {
    if (!navigator.mediaDevices?.enumerateDevices) return []
    const devices = await navigator.mediaDevices.enumerateDevices()
    return devices.filter((d) => d.kind === 'videoinput').map((d, i) => ({ deviceId: d.deviceId, label: d.label || `Camera ${i + 1}` }))
  }

  /** Label and size of the running camera, for diagnostics. */
  cameraInfo() {
    const track = this.stream?.getVideoTracks?.()[0]
    const s = track?.getSettings?.() || {}
    return { label: track?.label || '', width: s.width || 0, height: s.height || 0, frameRate: s.frameRate || 0 }
  }

  /**
   * Loads the hand and pose landmarkers once (shared by Practice and Communicate).
   */
  async initialize() {
    if (this.handLandmarker && this.poseLandmarker) return this
    if (this.initializationPromise) return this.initializationPromise

    this.initializationPromise = (async () => {
      const vision = await FilesetResolver.forVisionTasks(WASM_PATH)
      this.vision = vision
      const first = readDelegate()
      const hands = await createWithFallback(HandLandmarker, vision, HAND_OPTIONS, first)
      const pose = await createWithFallback(PoseLandmarker, vision, POSE_OPTIONS, first)
      this.handLandmarker = hands.landmarker
      this.poseLandmarker = pose.landmarker
      this.delegate = hands.delegate === 'GPU' && pose.delegate === 'GPU' ? 'GPU' : 'CPU'
      this._warmUp()
      return this
    })().catch((error) => {
      this.initializationPromise = null
      console.error('Failed to initialize MediaPipe landmarkers:', error)
      throw error
    })

    return this.initializationPromise
  }

  /**
   * The first detections compile the GPU programs: measured 3-5 s for the first frame (2026-10-21), which froze the
   * picture and lost the first sign after Start. Two detections on a blank image while the page loads do it before
   * the camera starts. Timestamps 1, 2 stay below the real ones (performance.now()).
   */
  _warmUp() {
    try {
      const blank = document.createElement('canvas')
      blank.width = 640
      blank.height = 480
      blank.getContext('2d').fillRect(0, 0, 640, 480)
      const t0 = performance.now()
      for (const ts of [1, 2]) {
        this.handLandmarker.detectForVideo(blank, ts)
        this.poseLandmarker.detectForVideo(blank, ts)
      }
      this.lastTimestamp = 2
      this.warmUpMs = Math.round(performance.now() - t0)
    } catch (err) {
      console.warn('Tracker warm-up failed (the first frames will be slow):', err)
    }
  }

  /** Loads and warms up the trackers in the background when a camera page opens (no camera access). */
  preload() {
    if (!this.preloading) this.preloading = this.initialize().catch(() => null)
    return this.preloading
  }

  /**
   * Starts user webcam and landmark tracking
   */
  async startCamera(videoElement, onResults, { deviceId = '', resolution = 480, onIssue = null } = {}) {
    if (!videoElement) throw new Error('Video element is required')

    this.videoElement = videoElement
    this.onResultsCallback = onResults
    this.onIssueCallback = onIssue

    await this.initialize()

    if (!navigator.mediaDevices?.getUserMedia) {
      const err = new Error('This browser cannot open a camera (needs a secure page: localhost or https).')
      err.name = 'NotFoundError'
      throw err
    }
    const size = CAMERA_RESOLUTIONS[resolution] || CAMERA_RESOLUTIONS[480]
    const portraitPhone = window.matchMedia?.('(max-width: 767px) and (orientation: portrait)').matches
    const video = portraitPhone
      ? { width: { ideal: 480 }, height: { ideal: 854 }, aspectRatio: { ideal: 9 / 16 }, frameRate: { ideal: 24 } }
      : { width: { ideal: size.width }, height: { ideal: size.height }, frameRate: { ideal: 30 } }
    let stream
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: deviceId ? { ...video, deviceId: { exact: deviceId } } : { ...video, facingMode: 'user' },
        audio: false
      })
    } catch (error) {
      if (!deviceId || error?.name !== 'OverconstrainedError') throw error
      // the remembered camera is gone (unplugged / renamed): fall back to the default camera
      stream = await navigator.mediaDevices.getUserMedia({ video: { ...video, facingMode: 'user' }, audio: false })
    }
    this.stream = stream
    this.videoElement.srcObject = stream
    await this.videoElement.play()

    this.isTracking = true
    this.lastVideoTime = -1
    this._countCameraFrames()
    this._watchHealth()
    this._processVideoLoop()
    return true
  }

  /**
   * Checks the feed every second: a camera can deliver frames that are black (lens cover, privacy
   * shutter, another app holding the sensor) or stop delivering frames at all (frozen driver).
   */
  _watchHealth() {
    clearInterval(this.healthTimer)
    const probe = document.createElement('canvas')
    probe.width = 32
    probe.height = 24
    const ctx = probe.getContext('2d', { willReadFrequently: true })
    let lastTime = -1
    let lastChange = performance.now()
    this.issue = null
    this.healthTimer = setInterval(() => {
      const video = this.videoElement
      if (!this.isTracking || !video) return
      const now = performance.now()
      if (video.currentTime !== lastTime) {
        lastTime = video.currentTime
        lastChange = now
      }
      let issue = null
      if (now - lastChange > FROZEN_S * 1000) {
        issue = 'frozen'
      } else if (ctx && video.readyState >= 2) {
        ctx.drawImage(video, 0, 0, probe.width, probe.height)
        const px = ctx.getImageData(0, 0, probe.width, probe.height).data
        let sum = 0
        for (let i = 0; i < px.length; i += 4) sum += 0.299 * px[i] + 0.587 * px[i + 1] + 0.114 * px[i + 2]
        if (sum / (px.length / 4) < BLACK_LUMA) issue = 'black'
      }
      if (issue !== this.issue) {
        this.issue = issue
        this.onIssueCallback?.(issue)
      }
    }, 1000)
  }

  /**
   * Counts the frames the camera delivers, independent of how many are tracked: a webcam in a dim
   * room lowers its own frame rate, which no amount of GPU fixes.
   */
  _countCameraFrames() {
    const video = this.videoElement
    if (!video?.requestVideoFrameCallback) {
      this.cameraFps = this.stream?.getVideoTracks?.()[0]?.getSettings?.().frameRate || 0
      return
    }
    let last = 0
    const onFrame = (now) => {
      if (!this.isTracking || this.videoElement !== video) return
      if (last) this.cameraFps = ema(this.cameraFps, 1000 / Math.max(now - last, 1))
      last = now
      video.requestVideoFrameCallback(onFrame)
    }
    video.requestVideoFrameCallback(onFrame)
  }

  /**
   * Times the current delegate on the live camera; if slow, loads and times the other one and keeps the faster (see
   * SLOW_MS). Measured on real frames (blank frames favour the CPU wrongly: no hands = a different workload). A swap
   * is applied only on a frame without hands, so it never interrupts a sign, and the decision is kept in this browser
   * for DECISION_DAYS, so later sessions do not probe at all.
   */
  _probeDelegate(ms, handsInView) {
    const p = (this.probe ||= { samples: [], timings: {}, done: decided(), loading: false })
    if (p.swap && !handsInView) { // apply a pending swap between signs
      const { hands, pose, delegate, closeOld } = p.swap
      p.swap = null
      if (closeOld) { this.handLandmarker.close?.(); this.poseLandmarker.close?.() } // only once they are replaced
      this.handLandmarker = hands
      this.poseLandmarker = pose
      this.lastPose = null
      this.delegate = delegate
      return
    }
    if (p.done || p.loading || p.swap) return
    p.samples.push(ms)
    if (p.samples.length < PROBE_WARMUP + PROBE_FRAMES) return
    const current = this.delegate
    p.timings[current] = Math.round(median(p.samples.slice(PROBE_WARMUP)))
    p.samples = []
    const other = current === 'GPU' ? 'CPU' : 'GPU'
    if (p.timings[other] == null && p.timings[current] > SLOW_MS && this.vision) {
      p.loading = true // keep tracking with the current delegate while the other one loads
      Promise.all([
        HandLandmarker.createFromOptions(this.vision, { ...HAND_OPTIONS, baseOptions: { ...HAND_OPTIONS.baseOptions, delegate: other } }),
        PoseLandmarker.createFromOptions(this.vision, { ...POSE_OPTIONS, baseOptions: { ...POSE_OPTIONS.baseOptions, delegate: other } })
      ]).then(([hands, pose]) => {
        p.previous = { delegate: current, hands: this.handLandmarker, pose: this.poseLandmarker }
        const blank = document.createElement('canvas') // compile the new delegate's GPU programs before using it
        blank.width = 64
        blank.height = 48
        hands.detectForVideo(blank, ++this.lastTimestamp)
        pose.detectForVideo(blank, this.lastTimestamp)
        p.swap = { hands, pose, delegate: other }
      }).catch((err) => {
        console.warn(`${other} delegate could not be loaded:`, err)
        p.done = true
      }).finally(() => { p.loading = false })
      return
    }
    // decision: the faster delegate stays, the other is closed (going back also waits for a frame without hands)
    if (p.previous && p.timings[p.previous.delegate] < p.timings[current]) {
      p.swap = { hands: p.previous.hands, pose: p.previous.pose, delegate: p.previous.delegate, closeOld: true }
    } else if (p.previous) {
      p.previous.hands.close?.()
      p.previous.pose.close?.()
    }
    p.previous = null
    p.done = true
    saveDelegate(p.swap ? p.swap.delegate : this.delegate, p.timings)
    console.info('Hand tracking delegate:', p.swap ? p.swap.delegate : this.delegate, p.timings)
  }

  /**
   * Processing loop: one hand + pose inference per NEW camera frame. requestVideoFrameCallback fires once per frame
   * the camera delivers, right when it arrives; the old requestAnimationFrame loop ran on screen refreshes and, with a
   * webcam stream, often processed the same camera frame two or three times (measured 2026-10-21: 49 "frames"/s sent
   * to the server from a 25 fps camera), wasting tracking time and repeating frames in the sign. While a frame is
   * being processed, newer ones replace older ones, so tracking always works on the latest frame.
   */
  _processVideoLoop() {
    if (!this.isTracking || !this.videoElement) return
    const video = this.videoElement
    if (typeof video.requestVideoFrameCallback === 'function') {
      const onFrame = (_now, meta) => {
        if (!this.isTracking || this.videoElement !== video) return
        this._processFrame(video, meta?.presentedFrames)
        this.videoFrameHandle = video.requestVideoFrameCallback(onFrame)
      }
      this.videoFrameHandle = video.requestVideoFrameCallback(onFrame)
      return
    }
    // older browsers: poll on animation frames, skipping frames whose video time did not change
    const poll = () => {
      if (!this.isTracking || this.videoElement !== video) return
      if (video.currentTime !== this.lastVideoTime) {
        this.lastVideoTime = video.currentTime
        this._processFrame(video)
      }
      this.animationFrameId = requestAnimationFrame(poll)
    }
    this.animationFrameId = requestAnimationFrame(poll)
  }

  _processFrame(video, presented) {
    if (video.readyState < 2 || video.paused) return
    if (presented != null) {
      if (presented === this.lastPresented) return // the same camera frame again
      this.lastPresented = presented
    }
    {
      // Landmarkers require strictly increasing timestamps
      const timestamp = Math.max(Math.round(performance.now()), this.lastTimestamp + 1)
      this.lastTimestamp = timestamp
      try {
        const started = performance.now()
        const hands = this.handLandmarker.detectForVideo(video, timestamp)
        // Pose on every frame when tracking is fast (a discrete GPU: ~20 ms for hands + pose), so the body points and
        // the hand-to-shoulder features are never a frame old; on slower machines every other frame (the body moves
        // slowly), every third above POSE_SLOW_MS
        this.frameCount = (this.frameCount || 0) + 1
        if (this.inferenceMs > POSE_SLOW_MS) this.poseEvery = 3
        else if (this.inferenceMs > POSE_BACK_MS || !this.poseEvery) this.poseEvery = 2
        else if (this.inferenceMs < POSE_FAST_MS) this.poseEvery = 1
        else if (this.poseEvery === 3) this.poseEvery = 2
        const poseEvery = this.poseEvery
        if (poseEvery === 1 || this.frameCount % poseEvery === 1 || !this.lastPose) {
          this.lastPose = this.poseLandmarker.detectForVideo(video, timestamp)
        }
        const pose = this.lastPose
        const spent = performance.now() - started
        this.inferenceMs = ema(this.inferenceMs, spent)
        this._probeDelegate(spent, (hands.landmarks || []).length > 0)
        const dt = this.lastFrameAt ? (timestamp - this.lastFrameAt) / 1000 : 0
        this.lastFrameAt = timestamp
        if (dt > 0) this.fps = ema(this.fps, 1 / dt)
        this.onResultsCallback?.({
          multiHandLandmarks: hands.landmarks || [],
          multiHandedness: (hands.handedness || hands.handednesses || []).map((categories) => ({
            label: categories?.[0]?.categoryName,
            score: categories?.[0]?.score ?? 0
          })),
          poseLandmarks: pose.landmarks?.[0] || null,
          width: video.videoWidth,
          height: video.videoHeight,
          timestamp,
          fps: this.fps || 0,
          diag: {
            cameraFps: Math.round(this.cameraFps || 0),
            trackingFps: Math.round(this.fps || 0),
            inferenceMs: Math.round(this.inferenceMs || 0),
            delegate: this.delegate,
            gpu: browserGpu().name,
            integratedGpu: browserGpu().integrated,
            camera: this.stream?.getVideoTracks?.()[0]?.label || ''
          }
        })
      } catch (err) {
        console.warn('Frame processing exception:', err)
      }
    }
  }

  /**
   * Stops camera stream and tracking loop
   */
  stopCamera() {
    this.isTracking = false
    clearInterval(this.healthTimer)
    this.healthTimer = null
    this.issue = null
    this.lastPose = null
    this.lastFrameAt = 0
    this.fps = 0
    this.cameraFps = 0
    this.inferenceMs = 0

    if (this.animationFrameId) {
      cancelAnimationFrame(this.animationFrameId)
      this.animationFrameId = null
    }
    if (this.videoFrameHandle != null && this.videoElement?.cancelVideoFrameCallback) {
      this.videoElement.cancelVideoFrameCallback(this.videoFrameHandle)
    }
    this.videoFrameHandle = null
    this.lastPresented = null
    this.poseEvery = 0

    if (this.stream) {
      this.stream.getTracks().forEach((track) => track.stop())
      this.stream = null
    }

    if (this.videoElement) {
      this.videoElement.srcObject = null
    }
  }

  /**
   * Renders the upper-body pose as a faint guide behind the hands.
   */
  drawPose(ctx, poseLandmarks, width, height, mirror = true) {
    if (!ctx || !poseLandmarks || poseLandmarks.length < 25) return

    ctx.save()
    if (mirror) {
      ctx.translate(width, 0)
      ctx.scale(-1, 1)
    }
    // Tracking can place finger points a few pixels beyond the image edge. Clip every
    // overlay to the captured frame so the visible sign box never escapes the preview.
    ctx.beginPath()
    ctx.rect(0, 0, width, height)
    ctx.clip()
    ctx.lineWidth = 2.5
    ctx.lineCap = 'round'
    ctx.strokeStyle = 'rgba(253, 186, 116, 0.55)'
    POSE_CONNECTIONS.forEach(([i, j]) => {
      const p1 = poseLandmarks[i]
      const p2 = poseLandmarks[j]
      if (!p1 || !p2 || (p1.visibility ?? 1) < 0.5 || (p2.visibility ?? 1) < 0.5) return
      ctx.beginPath()
      ctx.moveTo(p1.x * width, p1.y * height)
      ctx.lineTo(p2.x * width, p2.y * height)
      ctx.stroke()
    })
    ctx.restore()
  }

  /**
   * Renders high-fidelity MUDRA-styled hand skeleton on HTML5 canvas
   */
  drawSkeleton(ctx, landmarks, width, height, mirror = true) {
    if (!ctx || !landmarks || landmarks.length < 21) return

    ctx.save()
    if (mirror) {
      ctx.translate(width, 0)
      ctx.scale(-1, 1)
    }
    ctx.beginPath()
    ctx.rect(0, 0, width, height)
    ctx.clip()

    // 1. Draw Bone Connections with elegant gradient stroke
    ctx.lineWidth = 3.5
    ctx.lineCap = 'round'
    ctx.lineJoin = 'round'

    HAND_CONNECTIONS.forEach(([i, j]) => {
      const p1 = landmarks[i]
      const p2 = landmarks[j]
      if (!p1 || !p2) return

      const x1 = p1.x * width
      const y1 = p1.y * height
      const x2 = p2.x * width
      const y2 = p2.y * height

      // Gradient from Soft Lavender to Peach
      const gradient = ctx.createLinearGradient(x1, y1, x2, y2)
      gradient.addColorStop(0, 'rgba(139, 92, 246, 0.85)') // #8B5CF6
      gradient.addColorStop(1, 'rgba(253, 186, 116, 0.90)') // #FDBA74

      ctx.strokeStyle = gradient
      ctx.beginPath()
      ctx.moveTo(x1, y1)
      ctx.lineTo(x2, y2)
      ctx.stroke()
    })

    // 2. Draw Keypoint Nodes (Joint Dots)
    landmarks.forEach((p, idx) => {
      const x = p.x * width
      const y = p.y * height

      const isFingertip = [4, 8, 12, 16, 20].includes(idx)
      const isWrist = idx === 0

      // Outer soft glow aura
      ctx.beginPath()
      ctx.arc(x, y, isFingertip ? 7 : isWrist ? 6 : 4.5, 0, 2 * Math.PI)
      ctx.fillStyle = isFingertip
        ? 'rgba(249, 115, 22, 0.35)'
        : 'rgba(139, 92, 246, 0.25)'
      ctx.fill()

      // Inner solid node
      ctx.beginPath()
      ctx.arc(x, y, isFingertip ? 4.5 : isWrist ? 4 : 3, 0, 2 * Math.PI)
      ctx.fillStyle = isFingertip ? '#F97316' : '#8B5CF6'
      ctx.strokeStyle = '#FFFFFF'
      ctx.lineWidth = 1.5
      ctx.fill()
      ctx.stroke()
    })

    const xs = landmarks.map((point) => Math.max(0, Math.min(1, point.x)) * width)
    const ys = landmarks.map((point) => Math.max(0, Math.min(1, point.y)) * height)
    const pad = Math.min(24, Math.round(Math.min(width, height) * 0.045))
    const left = Math.max(0, Math.min(...xs) - pad)
    const top = Math.max(0, Math.min(...ys) - pad)
    const right = Math.min(width, Math.max(...xs) + pad)
    const bottom = Math.min(height, Math.max(...ys) + pad)
    if (right > left && bottom > top) {
      ctx.strokeStyle = 'rgba(168, 224, 99, 0.95)'
      ctx.lineWidth = Math.max(2, Math.min(width, height) * 0.004)
      ctx.setLineDash([Math.max(6, width * 0.012), Math.max(4, width * 0.008)])
      ctx.strokeRect(left, top, right - left, bottom - top)
      ctx.setLineDash([])
    }

    ctx.restore()
  }
}

export const handTracker = new HandTrackerService()
if (import.meta.env?.DEV && typeof window !== 'undefined') window.__mudraTracker = handTracker // dev diagnostics only
export default handTracker
