import { FaceLandmarker, FilesetResolver } from '@mediapipe/tasks-vision'

const ASSET_BASE = import.meta.env.BASE_URL
const WASM_PATH = `${ASSET_BASE}wasm`
const MODEL_PATH = `${ASSET_BASE}models/face_landmarker.task`
const TARGET_FPS = 5 // facial cues change slowly; keep the GPU for hand and body tracking
const SLOW_FPS = 1 // while hand tracking cannot keep up with the camera

class FaceExpressionTracker {
  constructor() {
    this.landmarker = null
    this.initializationPromise = null
    this.videoElement = null
    this.onResult = null
    this.onError = null
    this.frameId = null
    this.running = false
    this.lastVideoTime = -1
    this.lastInferenceAt = 0
    this.targetFps = TARGET_FPS
  }

  /** Hands come first: run face cues at 1 fps while hand tracking is the bottleneck. */
  setSlow(slow) {
    this.targetFps = slow ? SLOW_FPS : TARGET_FPS
  }

  async initialize() {
    if (this.landmarker) return this.landmarker
    if (this.initializationPromise) return this.initializationPromise

    this.initializationPromise = (async () => {
      const vision = await FilesetResolver.forVisionTasks(WASM_PATH)
      this.landmarker = await FaceLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: MODEL_PATH,
          delegate: 'GPU'
        },
        runningMode: 'VIDEO',
        numFaces: 1,
        outputFaceBlendshapes: true
      })
      return this.landmarker
    })().catch((error) => {
      this.initializationPromise = null
      this.landmarker = null
      throw error
    })

    return this.initializationPromise
  }

  async start(videoElement, onResult, onError) {
    this.stop()
    this.videoElement = videoElement
    this.onResult = onResult
    this.onError = onError
    this.running = true

    try {
      await this.initialize()
      this.scheduleFrame()
    } catch (error) {
      this.running = false
      this.onError?.(error)
      throw error
    }
  }

  scheduleFrame() {
    if (!this.running) return
    this.frameId = requestAnimationFrame((timestamp) => this.processFrame(timestamp))
  }

  processFrame(timestamp) {
    if (!this.running || !this.videoElement) return

    const video = this.videoElement
    const isNewVideoFrame = video.currentTime !== this.lastVideoTime
    const isDue = timestamp - this.lastInferenceAt >= 1000 / this.targetFps

    if (video.readyState >= 2 && !video.paused && isNewVideoFrame && isDue) {
      try {
        this.lastVideoTime = video.currentTime
        this.lastInferenceAt = timestamp
        const result = this.landmarker.detectForVideo(video, Math.round(timestamp))
        this.onResult?.(result)
      } catch (error) {
        this.onError?.(error)
      }
    }

    this.scheduleFrame()
  }

  stop() {
    this.running = false
    if (this.frameId) cancelAnimationFrame(this.frameId)
    this.frameId = null
    this.videoElement = null
    this.onResult = null
    this.onError = null
    this.lastVideoTime = -1
    this.lastInferenceAt = 0
  }

  close() {
    this.stop()
    this.landmarker?.close?.()
    this.landmarker = null
    this.initializationPromise = null
  }
}

export const faceExpressionTracker = new FaceExpressionTracker()
export default faceExpressionTracker
