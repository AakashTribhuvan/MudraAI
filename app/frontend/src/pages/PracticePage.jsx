import React, { useState, useRef, useEffect, useCallback } from 'react'
import { 
  Hand, 
  Sparkles, 
  Camera, 
  CameraOff, 
  CheckCircle2, 
  AlertTriangle, 
  RotateCcw, 
  TrendingUp, 
  Award, 
  Info, 
  CheckCircle, 
  ShieldCheck,
  ArrowRight,
  ExternalLink
} from 'lucide-react'
import handTracker from '../services/handTracker'
import CameraSourcePicker, { useCameraSources } from '../components/CameraSourcePicker'
import { evaluateSignAttempt, isCoachableSign } from '../services/islClassifier'
import { createSessionRecorder } from '../services/progressStore'
import { ISL_VOCABULARY_LEXICON, ISLRTC_DICTIONARY_URL } from '../services/islDictionary'
import MudraAvatarViewer from '../components/avatar/MudraAvatarViewer'

export default function PracticePage({ targetSignId = 'HELLO', onNavigate }) {
  const videoRef = useRef(null)
  const canvasRef = useRef(null)

  const [activeSignId, setActiveSignId] = useState(targetSignId || 'HELLO')
  const [cameraActive, setCameraActive] = useState(false)
  const [cameraLoading, setCameraLoading] = useState(false)
  const [cameraError, setCameraError] = useState(null)
  const { sources: cameraSources, deviceId: cameraDeviceId, setDeviceId: setCameraDeviceId } = useCameraSources(cameraActive)

  // Real-time evaluation state (0% on startup / no hand)
  const [evaluation, setEvaluation] = useState({
    handDetected: false,
    overallSimilarity: 0,
    fingerConfigScore: 0,
    positionScore: 0,
    orientationScore: 0,
    status: 'NO_HAND',
    feedbackMessage: 'Start camera and position your hand in the frame to begin practice.',
    detailedChecks: {
      detection: { pass: false, label: 'Hand in Camera Frame' },
      fingerShape: { pass: false, score: 0, label: 'Finger Flexion & Spread' },
      spatialPosition: { pass: false, score: 0, label: 'Hand Height & Centering' },
      palmOrientation: { pass: false, score: 0, label: 'Palm Facing Direction' }
    }
  })

  // Recent scores history for real progression tracking
  const [recentScores, setRecentScores] = useState([])
  const sessionRecorderRef = useRef(null)
  if (!sessionRecorderRef.current) sessionRecorderRef.current = createSessionRecorder()
  const coachableSigns = ISL_VOCABULARY_LEXICON.filter((s) => isCoachableSign(s.id))

  const activeSign = ISL_VOCABULARY_LEXICON.find((s) => s.id === activeSignId) || ISL_VOCABULARY_LEXICON[0]

  // Update active sign if passed from props
  useEffect(() => {
    if (targetSignId) {
      setActiveSignId(targetSignId)
    }
  }, [targetSignId])

  // Handle incoming MediaPipe results
  const handleMediaPipeResults = useCallback((results) => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    handTracker.fitCanvas(canvas, results)
    ctx.clearRect(0, 0, canvas.width, canvas.height)

    if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
      const primaryHand = results.multiHandLandmarks[0]

      // Draw glowing skeleton on canvas
      results.multiHandLandmarks.forEach((landmarks) => {
        handTracker.drawSkeleton(ctx, landmarks, canvas.width, canvas.height, true)
      })

      // Run genuine mathematical biometric evaluation
      const evalResult = evaluateSignAttempt(activeSignId, primaryHand, results.multiHandLandmarks)
      setEvaluation(evalResult)
      if (evalResult.supported) {
        sessionRecorderRef.current.add(activeSignId, evalResult.overallSimilarity)
      }

      if (evalResult.overallSimilarity > 50) {
        setRecentScores((prev) => {
          if (prev.length === 0 || Math.abs(prev[prev.length - 1] - evalResult.overallSimilarity) >= 4) {
            return [...prev.slice(-3), evalResult.overallSimilarity]
          }
          return prev
        })
      }
    } else {
      // Strictly NO hand detected: Zero confidence
      setEvaluation({
        handDetected: false,
        overallSimilarity: 0,
        fingerConfigScore: 0,
        positionScore: 0,
        orientationScore: 0,
        status: 'NO_HAND',
        feedbackMessage: 'Hand not detected. Position your hand clearly in front of the camera.',
        detailedChecks: {
          detection: { pass: false, label: 'Hand in Camera Frame' },
          fingerShape: { pass: false, score: 0, label: 'Finger Flexion & Spread' },
          spatialPosition: { pass: false, score: 0, label: 'Hand Height & Centering' },
          palmOrientation: { pass: false, score: 0, label: 'Palm Facing Direction' }
        }
      })
    }
  }, [activeSignId])

  // Start Camera
  const handleStartCamera = async () => {
    setCameraLoading(true)
    setCameraError(null)
    try {
      if (videoRef.current) {
        await handTracker.startCamera(videoRef.current, handleMediaPipeResults, { deviceId: cameraDeviceId })
        setCameraActive(true)
      }
    } catch (err) {
      console.error('Camera startup error:', err)
      setCameraError('Camera access was denied or is unavailable. Please check browser camera permissions.')
      setCameraActive(false)
    } finally {
      setCameraLoading(false)
    }
  }

  const handleCameraSourceChange = async (deviceId) => {
    setCameraDeviceId(deviceId)
    if (!cameraActive || !videoRef.current) return
    handTracker.stopCamera()
    try {
      await handTracker.startCamera(videoRef.current, handleMediaPipeResults, { deviceId })
    } catch (err) {
      setCameraError(err?.message || 'Could not switch camera.')
      setCameraActive(false)
    }
  }

  // Stop Camera
  const handleStopCamera = () => {
    handTracker.stopCamera()
    sessionRecorderRef.current.flush()
    setCameraActive(false)
    if (canvasRef.current) {
      const ctx = canvasRef.current.getContext('2d')
      if (ctx) ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height)
    }
    setEvaluation({
      handDetected: false,
      overallSimilarity: 0,
      fingerConfigScore: 0,
      positionScore: 0,
      orientationScore: 0,
      status: 'NO_HAND',
      feedbackMessage: 'Camera is standby. Start camera to begin gesture practice.',
      detailedChecks: {
        detection: { pass: false, label: 'Hand in Camera Frame' },
        fingerShape: { pass: false, score: 0, label: 'Finger Flexion & Spread' },
        spatialPosition: { pass: false, score: 0, label: 'Hand Height & Centering' },
        palmOrientation: { pass: false, score: 0, label: 'Palm Facing Direction' }
      }
    })
  }

  useEffect(() => {
    return () => {
      handTracker.stopCamera()
      sessionRecorderRef.current.flush()
    }
  }, [])

  return (
    <div className="practice-page pt-6 lg:pt-10 pb-24 lg:pb-16 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 space-y-10">
      <style>{`
        .practice-mobile-layout {
          position: relative;
        }
        @media (max-width: 767px) {
          .practice-page {
            padding-top: 5rem !important;
            padding-bottom: 2rem !important;
            gap: 1.25rem !important;
          }
          .practice-page > .practice-header {
            gap: 0.65rem;
            padding-bottom: 0.75rem;
          }
          .practice-page > .practice-header p {
            display: none;
          }
          .practice-page > .practice-header h1 {
            font-size: 1.65rem;
            line-height: 1.1;
          }
          .practice-page > .practice-header > div:first-child > div:first-child {
            display: none;
          }
          .practice-page > .practice-header > div:last-child {
            width: 100%;
            justify-content: flex-start;
            padding: 0.25rem;
            overflow-x: auto;
            flex-wrap: nowrap;
          }
          .practice-page > .practice-header > div:last-child button {
            flex: 0 0 auto;
            min-height: 2.25rem;
          }
          .practice-mobile-layout {
            display: block;
          }
          .practice-avatar-viewer {
            height: clamp(560px, 72dvh, 760px) !important;
          }
          .practice-avatar-stage {
            position: relative;
            height: min(65dvh, 440px);
            overflow: hidden;
            border: 1px solid rgba(213, 196, 253, 0.9);
            border-radius: 1.5rem;
            background: linear-gradient(180deg, rgba(250, 248, 255, 0.92), rgba(255, 248, 241, 0.9));
            box-shadow: 0 12px 30px rgba(47, 41, 79, 0.1);
          }
          .practice-avatar-stage .practice-avatar-viewer {
            height: 100% !important;
          }
          .practice-avatar-stage .practice-avatar-viewer canvas {
            transform: scale(0.8);
            transform-origin: center center;
          }
          .practice-camera-column {
            display: block;
          }
          .practice-camera-column > .practice-feedback-card {
            margin-top: 1rem;
            padding: 1rem;
            border-radius: 1.5rem;
          }
          .practice-camera-panel {
            position: absolute;
            top: clamp(17rem, 45dvh, 20rem);
            bottom: auto;
            right: 1rem;
            z-index: 20;
            width: clamp(96px, 25vw, 112px);
            aspect-ratio: 3 / 4;
            min-height: 0;
            padding: 0.55rem;
            border-radius: 1rem;
            box-shadow: 0 12px 28px rgba(25, 21, 44, 0.28);
          }
          .practice-camera-panel .practice-camera-status,
          .practice-camera-panel .practice-camera-controls {
            display: none;
          }
          .practice-camera-panel video,
          .practice-camera-panel canvas {
            object-fit: cover;
            border-radius: 0.7rem;
          }
          .practice-camera-panel > div:not(.practice-camera-status):not(.practice-camera-controls) {
            font-size: 0.7rem;
          }
          .practice-camera-panel .practice-camera-empty p {
            display: none;
          }
          .practice-camera-panel .practice-camera-empty button {
            padding: 0.45rem 0.6rem;
            font-size: 0.65rem;
          }
          .practice-camera-panel .practice-camera-empty svg {
            width: 1.25rem;
            height: 1.25rem;
            margin-bottom: 0.35rem;
          }
          .practice-feedback-card {
            background: transparent;
            border: 0;
            box-shadow: none;
            backdrop-filter: none;
            -webkit-backdrop-filter: none;
          }
          .practice-feedback-card .practice-camera-panel {
            background: #19152c;
          }
          .practice-feedback-column {
            margin-top: 1rem;
          }
          .practice-feedback-column .space-y-4 {
            gap: 0.75rem;
          }
        }
        @media (min-width: 768px) and (max-width: 1279px) {
          .practice-mobile-layout {
            gap: 1.25rem;
          }
        }
        @media (prefers-reduced-motion: reduce) {
          .practice-camera-panel,
          .practice-avatar-viewer {
            transition: none !important;
          }
        }
      `}</style>
      
      {/* Header */}
      <div className="practice-header flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 pb-6 border-b border-mudra-lavender-200/60">
        <div>
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-mudra-lavender-100 text-mudra-lavender-700 text-xs font-bold font-mono mb-1">
            <Sparkles className="w-3.5 h-3.5 text-mudra-lavender-600" />
            <span>SIGN COACH (RULE-BASED) & 3D AVATAR REFERENCE</span>
          </div>
          <h1 className="font-display font-extrabold text-3xl sm:text-4xl text-mudra-indigo-900">
            Interactive Practice Studio
          </h1>
          <p className="text-sm text-mudra-indigo-600 mt-1">
            Watch the 3D MUDRA instructor on the left, then hold the sign on the right: hand shape and position are compared with hand-written rules (single frame, movement is not checked).
          </p>
        </div>

        {/* Target Sign Selector */}
        <div className="flex items-center gap-1.5 bg-white/80 p-1.5 rounded-2xl border border-mudra-lavender-200 shadow-xs flex-wrap">
          {coachableSigns.map((s) => (
            <button
              key={s.id}
              type="button"
              onClick={() => setActiveSignId(s.id)}
              className={`px-3 py-1.5 rounded-xl text-xs font-bold transition-all flex items-center gap-1 cursor-pointer ${
                activeSignId === s.id
                  ? 'bg-mudra-lavender-600 text-white shadow-xs'
                  : 'text-mudra-indigo-700 hover:bg-mudra-lavender-50'
              }`}
            >
              <span>{s.label}</span>
            </button>
          ))}
        </div>
      </div>

      {/* Main Split Practice Studio */}
      <div className="practice-mobile-layout grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
        
        {/* Left Column: 3D MUDRA Avatar Target Reference */}
        <div className="lg:col-span-5 space-y-4">
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-xs font-bold uppercase tracking-wider text-mudra-indigo-500 font-display">
                1. 3D Reference Demonstration
              </span>
              <span className="text-xs font-mono font-bold text-mudra-lavender-700 bg-mudra-lavender-100 px-2.5 py-0.5 rounded-full">
                Target: {activeSign.label}
              </span>
            </div>

            {/* 3D Avatar Viewer */}
            <div className="practice-avatar-stage">
              <MudraAvatarViewer
                signText={activeSign.id}
                className="practice-avatar-viewer w-full h-[340px] sm:h-[380px]"
              />
            </div>

            {/* Anatomy Guidelines */}
            <div className="glass-card rounded-2xl p-4 border border-white space-y-2 text-xs text-mudra-indigo-800">
              <div className="flex items-center justify-between">
                <span className="font-bold text-mudra-indigo-900 uppercase tracking-wider text-[11px]">
                  Target Pose Mechanics (ISLRTC):
                </span>
                <a
                  href={activeSign.sourceUrl}
                  target="_blank"
                  rel="noreferrer"
                  className="text-[10px] text-mudra-lavender-700 font-semibold hover:underline flex items-center gap-0.5"
                >
                  <span>Lexicon standard</span>
                  <ExternalLink className="w-2.5 h-2.5" />
                </a>
              </div>
              <div className="space-y-1">
                <div><strong className="text-mudra-lavender-800">Shape:</strong> {activeSign.handShapeDesc}</div>
                <div><strong className="text-mudra-lavender-800">Position:</strong> {activeSign.positionDesc}</div>
                <div><strong className="text-mudra-lavender-800">Movement:</strong> {activeSign.movementDesc}</div>
              </div>
            </div>
          </div>
        </div>

        {/* Right Column: User's Turn with Live Camera & Evaluation Scorecard */}
        <div className="practice-camera-column lg:col-span-7 space-y-4">
          <div className="practice-feedback-card glass-card rounded-3xl p-6 sm:p-7 border border-white shadow-xl space-y-6">
            
            <div className="flex items-center justify-between pb-3 border-b border-mudra-lavender-200/60">
              <div className="flex items-center gap-2">
                <span className={`w-2.5 h-2.5 rounded-full ${cameraActive ? 'bg-emerald-500 animate-pulse' : 'bg-amber-400'}`}></span>
                <span className="text-xs font-bold uppercase tracking-wider text-mudra-lavender-700 font-display">
                  2. YOUR TURN — Live Camera Feedback
                </span>
              </div>
              
              {/* Real Gesture Similarity Badge */}
              <span className={`text-xs font-mono font-bold px-3 py-1 rounded-full ${
                !evaluation.handDetected || evaluation.supported === false
                  ? 'bg-gray-100 text-gray-600'
                  : evaluation.overallSimilarity >= 80
                  ? 'bg-emerald-100 text-emerald-800'
                  : evaluation.overallSimilarity >= 50
                  ? 'bg-amber-100 text-amber-800'
                  : 'bg-rose-100 text-rose-800'
              }`}>
                {!evaluation.handDetected
                  ? 'No Hand Detected'
                  : evaluation.supported === false
                  ? 'Live scoring not available'
                  : `Gesture similarity: ${evaluation.overallSimilarity}%`}
              </span>
            </div>

            {/* Camera Area */}
            <div className="practice-camera-panel relative aspect-video bg-mudra-indigo-950 rounded-2xl overflow-hidden shadow-md flex flex-col justify-between p-4 text-white border border-mudra-indigo-800">
              
              <video
                ref={videoRef}
                className={`absolute inset-0 w-full h-full object-cover transform -scale-x-100 ${
                  cameraActive ? 'opacity-100' : 'opacity-0 pointer-events-none'
                }`}
                playsInline
                muted
              />

              <canvas
                ref={canvasRef}
                width={640}
                height={480}
                className="absolute inset-0 w-full h-full object-cover z-10 pointer-events-none"
              />

              {!cameraActive && (
                <div className="practice-camera-empty absolute inset-0 flex flex-col items-center justify-center p-6 text-center z-0">
                  <Camera className="w-8 h-8 text-mudra-lavender-400 mb-2" />
                  <div className="font-display font-bold text-sm text-white mb-1">
                    Camera is Standby
                  </div>
                  <p className="text-[11px] text-mudra-indigo-300 max-w-xs mb-3">
                    Start camera to analyze your hand landmarks in real time against the {activeSign.label} model.
                  </p>
                  <button
                    type="button"
                    onClick={handleStartCamera}
                    disabled={cameraLoading}
                    className="btn-primary py-2 px-4 text-xs font-bold shadow-md cursor-pointer"
                  >
                    <span>{cameraLoading ? 'Starting MediaPipe...' : 'Start Camera'}</span>
                  </button>
                </div>
              )}

              {cameraSources.length > 1 && (
                <div className="absolute top-4 right-4 z-30 rounded-xl bg-black/60 px-3 py-2">
                  <CameraSourcePicker sources={cameraSources} deviceId={cameraDeviceId} onChange={handleCameraSourceChange} />
                </div>
              )}

              {/* Status Pill */}
              <div className="practice-camera-status flex items-center justify-between text-xs z-20">
                <span className="bg-black/60 px-2.5 py-1 rounded-full border border-white/10 font-mono text-[11px]">
                  {cameraActive ? (evaluation.handDetected ? '21 Landmarks Tracking' : 'Awaiting Hand') : 'Camera Offline'}
                </span>
                {evaluation.overallSimilarity >= 80 && (
                  <span className="bg-emerald-500/90 text-white px-2.5 py-1 rounded-full font-semibold text-[11px] flex items-center gap-1 shadow-xs">
                    <CheckCircle className="w-3 h-3" />
                    <span>High Match (≥ 80%)</span>
                  </span>
                )}
              </div>

              {/* Camera Controls */}
              {cameraActive && (
                <div className="practice-camera-controls flex items-center justify-between gap-2 pt-2 border-t border-white/10 z-20">
                  <button
                    type="button"
                    onClick={handleStopCamera}
                    className="px-3 py-1.5 rounded-xl bg-rose-500/90 hover:bg-rose-600 text-white text-xs font-bold transition-all flex items-center gap-1 cursor-pointer"
                  >
                    <CameraOff className="w-3.5 h-3.5" />
                    <span>Stop Camera</span>
                  </button>
                </div>
              )}
            </div>

            {/* Multi-Dimensional Feedback Scorecard */}
            <div className="space-y-4 pt-2">
              <div className="flex items-center justify-between">
                <h3 className="font-display font-bold text-base text-mudra-indigo-900">
                  Live Landmark Evaluation
                </h3>
                <span className="text-xs text-mudra-indigo-500 font-mono">
                  {evaluation.handDetected ? 'Measured from user camera' : 'Hand required'}
                </span>
              </div>

              {/* 4 Real Physical Metric Rows */}
              <div className="space-y-2.5">
                {/* Hand Detection */}
                <div className="p-3 rounded-xl bg-mudra-ivory-100/90 border border-mudra-lavender-200 flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <CheckCircle2 className={`w-4 h-4 ${evaluation.handDetected ? 'text-emerald-600' : 'text-gray-400'} shrink-0`} />
                    <span className="text-xs font-semibold text-mudra-indigo-900">Hand Detected in Frame</span>
                  </div>
                  <span className="text-xs font-mono font-bold text-mudra-indigo-950">
                    {evaluation.handDetected ? 'DETECTED' : 'NOT IN FRAME'}
                  </span>
                </div>

                {/* Finger Flexion & Spread */}
                <div className="p-3 rounded-xl bg-mudra-ivory-100/90 border border-mudra-lavender-200 flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <CheckCircle2 className={`w-4 h-4 ${evaluation.fingerConfigScore >= 75 ? 'text-emerald-600' : evaluation.fingerConfigScore > 0 ? 'text-amber-500' : 'text-gray-400'} shrink-0`} />
                    <span className="text-xs font-semibold text-mudra-indigo-900">Finger Flexion & Spread Configuration</span>
                  </div>
                  <span className="text-xs font-mono font-bold text-mudra-indigo-950">
                    {evaluation.handDetected ? `${evaluation.fingerConfigScore}%` : '--'}
                  </span>
                </div>

                {/* Spatial Height */}
                <div className="p-3 rounded-xl bg-mudra-ivory-100/90 border border-mudra-lavender-200 flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <CheckCircle2 className={`w-4 h-4 ${evaluation.positionScore >= 75 ? 'text-emerald-600' : evaluation.positionScore > 0 ? 'text-amber-500' : 'text-gray-400'} shrink-0`} />
                    <span className="text-xs font-semibold text-mudra-indigo-900">Hand Elevation & Spatial Centering</span>
                  </div>
                  <span className="text-xs font-mono font-bold text-mudra-indigo-950">
                    {evaluation.handDetected ? `${evaluation.positionScore}%` : '--'}
                  </span>
                </div>

                {/* Palm Orientation */}
                <div className="p-3 rounded-xl bg-mudra-ivory-100/90 border border-mudra-lavender-200 flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <CheckCircle2 className={`w-4 h-4 ${evaluation.orientationScore >= 75 ? 'text-emerald-600' : evaluation.orientationScore > 0 ? 'text-amber-500' : 'text-gray-400'} shrink-0`} />
                    <span className="text-xs font-semibold text-mudra-indigo-900">Palm Normal & Facing Direction</span>
                  </div>
                  <span className="text-xs font-mono font-bold text-mudra-indigo-950">
                    {evaluation.handDetected ? `${evaluation.orientationScore}%` : '--'}
                  </span>
                </div>
              </div>

              {/* Actionable Coaching Message */}
              <div className="p-4 rounded-2xl bg-gradient-to-r from-mudra-lavender-50 via-white to-mudra-peach-50 border border-mudra-lavender-300">
                <div className="text-[11px] font-bold uppercase tracking-wider text-mudra-lavender-800 font-display mb-1">
                  Coach Guidance
                </div>
                <p className="text-xs sm:text-sm text-mudra-indigo-900 font-medium leading-relaxed">
                  "{evaluation.feedbackMessage}"
                </p>
              </div>

              {/* Recent Progress */}
              {recentScores.length > 0 && (
                <div className="flex items-center justify-between pt-2 border-t border-mudra-lavender-200/60 text-xs">
                  <div className="flex items-center gap-2">
                    <TrendingUp className="w-4 h-4 text-emerald-600" />
                    <span className="font-semibold text-mudra-indigo-900">Recent Similarity Scores:</span>
                    <span className="font-mono text-mudra-indigo-600">
                      {recentScores.map((s) => `${s}%`).join(' → ')}
                    </span>
                  </div>
                </div>
              )}

            </div>

          </div>
        </div>

      </div>

    </div>
  )
}
