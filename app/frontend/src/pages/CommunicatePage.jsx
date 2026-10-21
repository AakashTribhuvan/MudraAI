import React, { useState, useRef, useEffect, useCallback } from 'react'
import { InputModeSwitch, PhrasePanel, SpellPanel } from '../components/communicate/LibraryModes'
import { fetchLibrary, fetchModels, letterSpeller, phraseRecorder } from '../services/libraryClient'
import { 
  Camera, 
  CameraOff, 
  Play, 
  Pause, 
  RotateCcw, 
  Volume2, 
  Sparkles, 
  CheckCircle2, 
  Copy, 
  Radio, 
  Info,
  Layers,
  AlertCircle,
  Mic,
  MicOff,
  Send,
  ArrowRightLeft
} from 'lucide-react'
import handTracker, { cameraErrorKind } from '../services/handTracker'
import signRecognizer, { API_BASE_URL } from '../services/signRecognizer'
import faceExpressionTracker from '../services/faceExpressionTracker'
import { classifyISLSign } from '../services/islClassifier'
import { ISL_VOCABULARY_LEXICON, buildMultilingualSentence, findISLLexiconItem, readableSign } from '../services/islDictionary'
import { analyzeExpression, FACE_UNAVAILABLE_EXPRESSION, INITIAL_EXPRESSION } from '../services/expressionAnalyzer'
import MudraAvatarViewer from '../components/avatar/MudraAvatarViewer'
import SignReference from '../components/SignReference'
import { useAccessibility } from '../context/AccessibilityContext'

const AUTO_SPEAK_PAUSE_MS = 2500
// An unsure sign still adds its best guess at this confidence or more (below it the sign is dropped)
const AUTO_ADD_MIN = 0.5
const LOW_FPS = 15
// Per-browser camera preferences (localStorage can be unavailable: then nothing is remembered)
function readPref(key, fallback) {
  try { return localStorage.getItem(key) ?? fallback } catch { return fallback }
}
function writePref(key, value) {
  try { localStorage.setItem(key, String(value)) } catch { /* not remembered */ }
}
const CAMERA_ERRORS = {
  denied: 'Camera permission was denied. Allow the camera for this site (the camera icon in the address bar), then try again.',
  not_found: 'No camera was found. Connect or enable a webcam, then try again.',
  in_use: 'The camera is in use by another app (Zoom, Teams, Camera…) or blocked by the system. Close that app, then try again.',
  unknown: 'The camera could not be started. Check that it works in another app, then try again.'
}
// Signs islClassifier.js can detect (offline fallback)
const RULE_SIGNS = ['HELLO', 'WATER', 'NAMASTE', 'THANK_YOU', 'YES', 'NO', 'PEACE', 'HELP']
// Close-up framing: shoulders wider than this share of the frame, or lower than this height, leave no room for the hands
const FRAMING = { maxShoulderWidth: 0.6, maxShoulderY: 0.72 }

export default function CommunicatePage() {
  // Mode toggle: 'sign-to-speech' (ISL -> Voice) vs 'speech-to-sign' (Voice/Text -> 3D ISL Avatar)
  const [activeMode, setActiveMode] = useState('sign-to-speech')

  // --- Mode A (Sign to Speech) State ---
  const videoRef = useRef(null)
  const canvasRef = useRef(null)
  const cameraPanelRef = useRef(null)

  const [cameraActive, setCameraActive] = useState(false)
  const [cameraLoading, setCameraLoading] = useState(false)
  const [cameraError, setCameraError] = useState(null)
  const [cameraIssue, setCameraIssue] = useState(null) // 'black' | 'frozen' while the camera runs
  const [cameras, setCameras] = useState([])
  const [cameraId, setCameraId] = useState(() => readPref('mudra_camera_id', ''))
  const [cameraRes, setCameraRes] = useState(() => Number(readPref('mudra_camera_res', '480')) || 480)
  const [isPaused, setIsPaused] = useState(false)
  const [isSpeaking, setIsSpeaking] = useState(false)
  const [copied, setCopied] = useState(false)

  const [currentDetection, setCurrentDetection] = useState({
    sign: 'UNKNOWN',
    confidence: 0,
    isUnknown: true,
    metadata: null
  })

  const [recognizedTokens, setRecognizedTokens] = useState([])
  const [generatedSentence, setGeneratedSentence] = useState('')
  const [selectedLang, setSelectedLang] = useState('en')
  const [faceStatus, setFaceStatus] = useState('initializing')
  const [expression, setExpression] = useState(INITIAL_EXPRESSION)

  const stableSignRef = useRef({ sign: null, count: 0 })
  const lastCommittedSignRef = useRef(null)
  const expressionHistoryRef = useRef([])
  const expressionPendingRef = useRef({ tone: null, count: 0 })
  const smoothedConfidenceRef = useRef(0)
  const [suggestions, setSuggestions] = useState([])
  // Framing check: the model was trained on signers seen from the waist up
  const [framing, setFraming] = useState(null) // null | 'no_shoulders' | 'too_close'
  const [trackerFps, setTrackerFps] = useState(0)
  const [trackerDiag, setTrackerDiag] = useState(null) // {cameraFps, trackingFps, inferenceMs, delegate}
  const framingRef = useRef({ frames: 0, noShoulders: 0, tooClose: 0 })
  // Speak the sentence automatically after a pause in signing
  const [autoSpeak, setAutoSpeak] = useState(() => {
    try {
      return window.localStorage.getItem('mudra.autoSpeak') !== 'off'
    } catch {
      return true
    }
  })
  const autoSpeakTimerRef = useRef(null)
  const lastSpokenRef = useRef('')
  const sentenceRef = useRef('')
  const [modelState, setModelState] = useState({ status: signRecognizer.status, info: null, error: null })
  // Words the camera model knows, so people can check a word before signing it
  const [vocab, setVocab] = useState([])
  const { speechRate } = useAccessibility()
  const [vocabQuery, setVocabQuery] = useState('')
  // Quick-add chips: online, the app words the model does not know; offline, the signs the rule engine detects
  const chipSigns = modelState.status === 'online'
    ? ISL_VOCABULARY_LEXICON.filter((v) => !vocab.length || !vocab.some((w) => w.sign === v.id))
    : ISL_VOCABULARY_LEXICON.filter((v) => RULE_SIGNS.includes(v.id))
  const [vocabOpen, setVocabOpen] = useState(false)
  const [referenceSign, setReferenceSign] = useState(null) // word whose training example is shown
  // Words (trained word model, automatic) | Spell (fingerspelling letters) | Phrase (choose from top guesses)
  // "Next word" cue: a word was just added, so the next sign can start (a clear pause between words)
  const [nextCue, setNextCue] = useState(null)
  useEffect(() => {
    if (!nextCue) return undefined
    const t = setTimeout(() => setNextCue(null), 1400)
    return () => clearTimeout(t)
  }, [nextCue])
  const [inputMode, setInputMode] = useState('words')
  const inputModeRef = useRef('words')
  const [libraryModels, setLibraryModels] = useState({})
  // Word model for Words mode (GET /models): the deployed model by default, experimental ones opt-in
  const [wordModels, setWordModels] = useState([])
  const [wordModel, setWordModel] = useState(() => readPref('signline_word_model', 'sequence'))
  useEffect(() => { fetchModels().then(setWordModels) }, [])
  useEffect(() => {
    const known = !wordModels.length || wordModels.some((m) => m.id === wordModel)
    const id = known ? wordModel : 'sequence'
    try { signRecognizer.setModel(id) } catch { signRecognizer.setModel('sequence') }
    writePref('signline_word_model', id)
  }, [wordModel, wordModels])
  useEffect(() => { fetchLibrary().then((l) => setLibraryModels(l.models || {})) }, [])
  const changeInputMode = (mode) => {
    inputModeRef.current = mode
    setInputMode(mode)
    signRecognizer.reset() // drop a half-finished word segment
    setSuggestions([])
  }
  useEffect(() => {
    if (modelState.status !== 'online') return
    fetch(`${API_BASE_URL}/labels`)
      .then((r) => r.json())
      .then((body) => setVocab(body.labels || []))
      .catch(() => setVocab([]))
  }, [modelState.status])
  const vocabMatches = vocabQuery.trim()
    ? vocab.filter((v) => v.label.toLowerCase().includes(vocabQuery.trim().toLowerCase()))
    : vocab
  const modelOnlineRef = useRef(false)
  const lastModelSignRef = useRef(null)
  const lastHandSeenRef = useRef(0)
  const isPausedRef = useRef(isPaused)
  useEffect(() => {
    isPausedRef.current = isPaused
  }, [isPaused])

  // Automatically keep constructed sentence synchronized with tokens and chosen language
  useEffect(() => {
    setGeneratedSentence(buildMultilingualSentence(recognizedTokens, selectedLang))
  }, [recognizedTokens, selectedLang])

  useEffect(() => {
    sentenceRef.current = generatedSentence
  }, [generatedSentence])

  useEffect(() => {
    try {
      window.localStorage.setItem('mudra.autoSpeak', autoSpeak ? 'on' : 'off')
    } catch {
      /* storage unavailable: the choice is just not remembered */
    }
  }, [autoSpeak])

  // After each new word, wait for a pause (no new sign starting) and then speak the whole sentence once
  useEffect(() => {
    clearTimeout(autoSpeakTimerRef.current)
    if (!autoSpeak || recognizedTokens.length === 0) return undefined
    autoSpeakTimerRef.current = setTimeout(() => {
      const text = sentenceRef.current
      if (text && text !== lastSpokenRef.current) speakText(text)
    }, AUTO_SPEAK_PAUSE_MS)
    return () => clearTimeout(autoSpeakTimerRef.current)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [recognizedTokens, autoSpeak])

  const replaceLastToken = useCallback((sign) => {
    setRecognizedTokens((prev) => (prev.length ? [...prev.slice(0, -1), sign] : [sign]))
  }, [])

  // Auto mode: letters within SPELL_JOIN_MS of each other build one spelled word; a recognised word ends it
  const spellingRef = useRef(0)
  const SPELL_JOIN_MS = 3000
  const addLetter = useCallback((letter) => {
    const now = Date.now()
    const joining = now - spellingRef.current < SPELL_JOIN_MS
    spellingRef.current = now
    setRecognizedTokens((prev) => (joining && prev.length ? [...prev.slice(0, -1), prev[prev.length - 1] + letter] : [...prev, letter]))
  }, [])

  const commitToken = useCallback((sign) => {
    setRecognizedTokens((prev) => (prev.length > 0 && prev[prev.length - 1] === sign ? prev : [...prev, sign]))
  }, [])

  // Trained recognition model (backend /predict). Falls back to the rule engine when offline.
  useEffect(() => {
    signRecognizer.onStatusChange = (state) => {
      modelOnlineRef.current = state.status === 'online'
      setModelState(state)
    }
    signRecognizer.onPrediction = (p) => {
      if (isPausedRef.current) return
      const top = p.top_k?.[0]
      // Fully automatic: a confident result is added; an unsure one is added too (its best guess) when the
      // model gives it at least AUTO_ADD_MIN, with the runner-ups offered as optional one-tap replacements.
      const confident = p.status === 'ok'
      const unsure = !confident && (p.status === 'uncertain' || p.status === 'no_sign') &&
        (p.stats?.hand_rate ?? 1) >= 0.5 && top && top.confidence >= AUTO_ADD_MIN
      const added = confident ? { sign: p.sign, label: p.label } : unsure ? { sign: top.sign, label: top.label } : null
      setCurrentDetection({
        sign: added ? added.sign : 'UNKNOWN',
        label: added ? added.label : null,
        candidate: top ? top.label : null,
        confidence: Math.round((confident ? p.confidence : top?.confidence ?? 0) * 100),
        isUnknown: !added,
        status: confident ? 'ok' : unsure ? 'unsure' : (p.stats?.hand_rate ?? 0) >= 0.5 ? 'missed' : p.status,
        source: p.custom ? 'custom' : 'model',
        metadata: added ? findISLLexiconItem(added.sign) : null
      })
      // The server classified one complete sign (start -> end), so each sign adds at most one word
      if (added) {
        spellingRef.current = 0
        commitToken(added.sign)
        setNextCue({ label: added.label || readableSign(added.sign), at: Date.now() })
      }
      // Unsure: the other likely words can replace the one just added (optional, nothing to click otherwise)
      setSuggestions(unsure ? (p.top_k || []).slice(1, 3) : [])
    }
    signRecognizer.onLetter = (ev) => {
      if (isPausedRef.current) return
      addLetter(ev.letter)
      setCurrentDetection({
        sign: ev.sign, label: `Letter ${ev.letter}`, candidate: null, confidence: Math.round(ev.confidence * 100),
        isUnknown: false, status: 'ok', source: 'letters', metadata: null
      })
    }
    signRecognizer.onProvisional = (guess) => {
      if (isPausedRef.current) return
      setCurrentDetection((prev) => (prev.status === 'signing' ? { ...prev, candidate: guess.label } : prev))
    }
    signRecognizer.onActivity = (state) => {
      if (isPausedRef.current) return
      if (state === 'signing') {
        clearTimeout(autoSpeakTimerRef.current) // still signing: the sentence is not finished
        setCurrentDetection({ sign: 'UNKNOWN', confidence: 0, isUnknown: true, status: 'signing', source: 'model', metadata: null })
      } else if (state === 'idle') {
        // too short to be a sign: go back to waiting instead of staying on "Signing…"
        setCurrentDetection((prev) => (prev.status === 'signing'
          ? { sign: 'UNKNOWN', confidence: 0, isUnknown: true, status: 'too_short', source: 'model', metadata: null }
          : prev))
      }
    }
    signRecognizer.connect()
    return () => {
      signRecognizer.onPrediction = null
      signRecognizer.onLetter = null
      signRecognizer.onActivity = null
      signRecognizer.onProvisional = null
      signRecognizer.onStatusChange = null
      signRecognizer.reset()
    }
  }, [commitToken])

  useEffect(() => {
    let active = true
    faceExpressionTracker.initialize()
      .then(() => {
        if (active) setFaceStatus('ready')
      })
      .catch((error) => {
        console.error('Face Landmarker initialization failed:', error)
        if (active) {
          setFaceStatus('error')
          setExpression({ ...FACE_UNAVAILABLE_EXPRESSION, reason: 'Facial analysis unavailable' })
        }
      })

    return () => {
      active = false
      faceExpressionTracker.close()
    }
  }, [])

  const resetExpression = (status = 'ready') => {
    expressionHistoryRef.current = []
    expressionPendingRef.current = { tone: null, count: 0 }
    smoothedConfidenceRef.current = 0
    setFaceStatus(status)
    setExpression(status === 'error' ? FACE_UNAVAILABLE_EXPRESSION : INITIAL_EXPRESSION)
  }

  const handleFaceResults = (result) => {
    const faceDetected = Boolean(result.faceLandmarks?.length)
    const categories = result.faceBlendshapes?.[0]?.categories || []

    if (!faceDetected) {
      expressionHistoryRef.current = []
      expressionPendingRef.current = { tone: null, count: 0 }
      setFaceStatus('not-detected')
      setExpression(FACE_UNAVAILABLE_EXPRESSION)
      return
    }

    const prediction = analyzeExpression(categories)
    if (!prediction.available) {
      setFaceStatus('insufficient')
      setExpression(prediction)
      return
    }

    expressionHistoryRef.current = [...expressionHistoryRef.current, prediction.tone].slice(-15)
    const counts = expressionHistoryRef.current.reduce((map, tone) => {
      map[tone] = (map[tone] || 0) + 1
      return map
    }, {})
    const dominantTone = Object.entries(counts).sort((a, b) => b[1] - a[1])[0]?.[0]
    const pending = expressionPendingRef.current

    if (dominantTone === pending.tone) {
      pending.count += 1
    } else {
      pending.tone = dominantTone
      pending.count = 1
    }

    if (pending.count >= 4 || expression.tone === dominantTone) {
      smoothedConfidenceRef.current = smoothedConfidenceRef.current
        ? smoothedConfidenceRef.current * 0.7 + prediction.confidence * 0.3
        : prediction.confidence
      setFaceStatus('detected')
      setExpression({
        ...prediction,
        tone: dominantTone,
        confidence: smoothedConfidenceRef.current
      })
    }
  }

  const handleFaceError = (error) => {
    console.error('Face Landmarker runtime error:', error)
    faceExpressionTracker.stop()
    setFaceStatus('error')
    setExpression({ ...FACE_UNAVAILABLE_EXPRESSION, reason: error?.message || 'Facial analysis unavailable' })
  }

  // --- Mode B (Speech / Text to 3D Avatar) State ---
  const [textInput, setTextInput] = useState('HELLO NAMASTE')
  const [activeAvatarText, setActiveAvatarText] = useState('HELLO NAMASTE')
  const [isListening, setIsListening] = useState(false)
  const recognitionRef = useRef(null)

  const fpsShownAtRef = useRef(0)
  function setTrackerFpsThrottled(fps, diag) {
    const now = performance.now()
    if (now - fpsShownAtRef.current < 1000) return
    fpsShownAtRef.current = now
    setTrackerFps(Math.round(fps))
    if (diag) {
      setTrackerDiag(diag)
      // Tracking (not the camera) is the bottleneck: give the hands the GPU time the face tracker uses
      faceExpressionTracker.setSlow(diag.cameraFps > 0 && diag.trackingFps < Math.min(diag.cameraFps - 3, LOW_FPS))
    }
  }

  // Why the frame rate is low: the camera itself (dim light) or the tracking (CPU / busy GPU)
  const slowReason = !trackerDiag || trackerFps <= 0 || trackerFps >= LOW_FPS ? null
    : trackerDiag.cameraFps > 0 && trackerDiag.cameraFps < LOW_FPS && trackerDiag.trackingFps >= trackerDiag.cameraFps - 3 ? 'camera'
    : 'tracking'


  // Framing check over ~2 s windows: shoulders visible, and room below them for the hands
  function updateFraming(results) {
    const f = framingRef.current
    const pose = results.poseLandmarks
    const ls = pose?.[11]
    const rs = pose?.[12]
    const visible = (pt) => pt && (pt.visibility ?? 1) >= 0.5 && pt.x >= 0 && pt.x <= 1 && pt.y >= 0 && pt.y <= 1
    f.frames += 1
    if (!visible(ls) || !visible(rs)) {
      f.noShoulders += 1
    } else if (Math.abs(ls.x - rs.x) > FRAMING.maxShoulderWidth || (ls.y + rs.y) / 2 > FRAMING.maxShoulderY) {
      f.tooClose += 1
    }
    if (f.frames >= 60) {
      setFraming(f.noShoulders > 0.5 * f.frames ? 'no_shoulders' : f.tooClose > 0.5 * f.frames ? 'too_close' : null)
      framingRef.current = { frames: 0, noShoulders: 0, tooClose: 0 }
    }
  }

  // Handle MediaPipe tracking results
  const handleMediaPipeResults = useCallback((results) => {
    if (isPausedRef.current) return

    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    ctx.clearRect(0, 0, canvas.width, canvas.height)

    const hasHands = results.multiHandLandmarks && results.multiHandLandmarks.length > 0
    updateFraming(results)
    if (results.fps) setTrackerFpsThrottled(results.fps, results.diag)
    handTracker.drawPose(ctx, results.poseLandmarks, canvas.width, canvas.height, true)
    if (hasHands) {
      results.multiHandLandmarks.forEach((landmarks) => {
        handTracker.drawSkeleton(ctx, landmarks, canvas.width, canvas.height, true)
      })
    }

    // Spell / Phrase modes: frames go to the letter or phrase recogniser instead of the word stream
    if (inputModeRef.current !== 'words') {
      if (inputModeRef.current === 'spell') letterSpeller.addFrame(results)
      else phraseRecorder.addFrame(results)
      return
    }

    // Trained model path: stream landmarks; predictions arrive via signRecognizer.onPrediction
    if (modelOnlineRef.current) {
      const now = performance.now()
      if (hasHands) {
        lastHandSeenRef.current = now
      } else if (now - lastHandSeenRef.current > 800) {
        // Hands lowered for a while: show the resting state (segmentation itself runs on the server)
        setCurrentDetection((prev) => (prev.status === 'no_hands' || prev.status === 'ok' ? prev : {
          sign: 'UNKNOWN', confidence: 0, isUnknown: true, status: 'no_hands', source: 'model', metadata: null
        }))
      }
      signRecognizer.addFrame(results)
      return
    }

    // Offline fallback: rule-based hand-shape matching for 8 static signs
    if (hasHands) {
      const primaryHand = results.multiHandLandmarks[0]
      const classification = { ...classifyISLSign(primaryHand, results.multiHandLandmarks), source: 'rules' }
      setCurrentDetection(classification)

      // Buffer commitment on stable hold (14 frames = ~450ms)
      if (!classification.isUnknown && classification.sign !== 'UNKNOWN') {
        if (stableSignRef.current.sign === classification.sign) {
          stableSignRef.current.count += 1
          if (stableSignRef.current.count >= 14) {
            if (lastCommittedSignRef.current !== classification.sign) {
              setRecognizedTokens(prev => {
                if (prev.length > 0 && prev[prev.length - 1] === classification.sign) {
                  return prev; // Block consecutive duplicates from hand holds/jitters
                }
                return [...prev, classification.sign];
              })
              lastCommittedSignRef.current = classification.sign
            }
          }
        } else {
          stableSignRef.current = { sign: classification.sign, count: 1 }
        }
      } else {
        stableSignRef.current = { sign: null, count: 0 }
        lastCommittedSignRef.current = null
      }
    } else {
      setCurrentDetection({
        sign: 'UNKNOWN',
        confidence: 0,
        isUnknown: true,
        source: 'rules',
        metadata: null
      })
      stableSignRef.current = { sign: null, count: 0 }
      lastCommittedSignRef.current = null
    }
  }, [])

  useEffect(() => {
    handTracker.listCameras().then(setCameras).catch(() => setCameras([]))
    handTracker.preload() // load + warm up tracking now, so Start Camera is fast
  }, [])

  const handleStartCamera = async () => {
    setCameraLoading(true)
    setCameraError(null)
    setCameraIssue(null)
    try {
      if (videoRef.current) {
        await handTracker.startCamera(videoRef.current, handleMediaPipeResults, {
          deviceId: cameraId, resolution: cameraRes, onIssue: setCameraIssue
        })
        setCameraActive(true)
        if (window.matchMedia?.('(max-width: 1023px)').matches) {
          cameraPanelRef.current?.scrollIntoView({ block: 'start', behavior: 'smooth' }) // camera + results on one screen
        }
        handTracker.listCameras().then(setCameras).catch(() => {}) // labels appear once permission is granted
        try {
          await faceExpressionTracker.start(videoRef.current, handleFaceResults, handleFaceError)
        } catch (error) {
          handleFaceError(error)
        }
      }
    } catch (err) {
      console.error('Camera startup failed:', err)
      setCameraError(CAMERA_ERRORS[cameraErrorKind(err)])
      setCameraActive(false)
    } finally {
      setCameraLoading(false)
    }
  }

  const handleStopCamera = () => {
    handTracker.stopCamera()
    setCameraIssue(null)
    signRecognizer.reset()
    lastModelSignRef.current = null
    faceExpressionTracker.stop()
    setCameraActive(false)
    if (canvasRef.current) {
      const ctx = canvasRef.current.getContext('2d')
      if (ctx) ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height)
    }
    setCurrentDetection({
      sign: 'UNKNOWN',
      confidence: 0,
      isUnknown: true,
      metadata: null
    })
    resetExpression('ready')
  }

  const handleSpeakSentence = () => speakText(generatedSentence)

  function speakText(text) {
    if (!text) return
    lastSpokenRef.current = text

    if ('speechSynthesis' in window) {
      window.speechSynthesis.cancel()
      const utterance = new SpeechSynthesisUtterance(text)
      
      // Use localized Indian voice locales for rich audio synthesis
      if (selectedLang === 'hi') {
        utterance.lang = 'hi-IN'
      } else if (selectedLang === 'mr') {
        utterance.lang = 'mr-IN'
      } else {
        utterance.lang = 'en-IN'
      }

      utterance.rate = 0.95 * speechRate
      utterance.pitch = 1.0
      setIsSpeaking(true)
      utterance.onend = () => setIsSpeaking(false)
      utterance.onerror = () => setIsSpeaking(false)
      window.speechSynthesis.speak(utterance)
    } else {
      setIsSpeaking(false) // no speech synthesis in this browser: do not pretend to speak
    }
  }

  const handleClearBuffer = () => {
    lastSpokenRef.current = ''
    setRecognizedTokens([])
    setGeneratedSentence('')
    lastCommittedSignRef.current = null
    stableSignRef.current = { sign: null, count: 0 }
  }

  const handleCopyText = () => {
    if (!generatedSentence) return
    navigator.clipboard.writeText(generatedSentence)
    setCopied(true)
    setTimeout(() => setCopied(false), 1500)
  }

  const handleSimulateSign = (v) => {
    // Manual entry: no recognition happened, so no confidence is shown
    setCurrentDetection({
      sign: v.id,
      label: v.label,
      confidence: null,
      isUnknown: false,
      source: 'manual',
      metadata: v
    })
    setRecognizedTokens(prev => {
      if (prev.length > 0 && prev[prev.length - 1] === v.id) {
        return prev;
      }
      return [...prev, v.id];
    })
  }

  // Voice Input for Speech to 3D Sign Mode
  const handleToggleVoiceInput = () => {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition
    if (!SpeechRecognition) {
      alert('Speech recognition is not supported in this browser. You can type in the box directly.')
      return
    }

    if (isListening) {
      recognitionRef.current?.stop()
      setIsListening(false)
    } else {
      const recognition = new SpeechRecognition()
      recognition.continuous = false
      recognition.interimResults = false
      recognition.lang = 'en-US'

      recognition.onresult = (event) => {
        const transcript = event.results[0][0].transcript
        setTextInput(transcript)
        setActiveAvatarText(transcript.toUpperCase())
        setIsListening(false)
      }

      recognition.onerror = () => setIsListening(false)
      recognition.onend = () => setIsListening(false)

      recognitionRef.current = recognition
      recognition.start()
      setIsListening(true)
    }
  }

  const handleTranslateToSign = (e) => {
    e?.preventDefault()
    if (textInput.trim()) {
      setActiveAvatarText(textInput.trim().toUpperCase())
    }
  }

  useEffect(() => {
    return () => {
      handTracker.stopCamera()
      faceExpressionTracker.stop()
      recognitionRef.current?.stop()
    }
  }, [])

  return (
    <div className="communicate-page pt-6 lg:pt-10 pb-24 lg:pb-16 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 space-y-8">
      <style>{`
        /* Below the desktop layout (lg) the two columns stack: the results card goes right under the camera,
           the manual-word chips and the word list after it (the left column's children join the grid). */
        @media (max-width: 1023px) {
          .communicate-page { padding-bottom: 10rem; }
          .communicate-sign-grid { gap: 1rem; }
          .communicate-left { display: contents; }
          .communicate-left > * { margin-top: 0 !important; }
          .communicate-camera-panel { order: 1; scroll-margin-top: 4.5rem; }
          .communicate-right { order: 2; }
          .communicate-chips { order: 3; }
          .communicate-vocab { order: 4; }
        }
        /* Phones: the camera follows the screen height, so the detected sign fits on the same screen */
        @media (max-width: 767px) {
          .communicate-page { padding-top: 1rem; }
          .communicate-page > :not([hidden]) ~ :not([hidden]) { margin-top: 1rem !important; }
          .communicate-header { padding-bottom: 0.75rem; gap: 0.75rem; }
          .communicate-camera-panel {
            width: 100%;
            max-width: 100%;
            aspect-ratio: auto;
            height: clamp(220px, 42dvh, 380px);
            min-height: 0;
          }
          .communicate-camera-panel video,
          .communicate-camera-panel canvas {
            object-fit: cover;
          }
          .communicate-camera-panel .communicate-hint {
            bottom: 3.75rem;
            padding: 0.5rem 0.75rem;
            font-size: 0.75rem;
            line-height: 1rem;
          }
          /* results first; the model picker (a setting) moves to the end of the card */
          .communicate-results { padding: 1rem; display: flex; flex-direction: column; }
          .communicate-results > :not([hidden]) ~ :not([hidden]) { margin-top: 1rem !important; }
          .communicate-words { display: contents; }
          .communicate-model-picker { order: 1; margin: 1rem 0 0 !important; }
          .communicate-camera-panel .communicate-cue { top: 3.25rem; }
        }
        /* Phones from 360 px: the camera buttons stay on one row, so the tips above them have room */
        @media (min-width: 360px) and (max-width: 767px) {
          .communicate-controls { flex-wrap: nowrap; gap: 0.375rem; }
          .communicate-controls > div { gap: 0.375rem; }
          .communicate-controls button { padding: 0.375rem 0.625rem; white-space: nowrap; }
        }
        @media (min-width: 360px) and (max-width: 389px) {
          .communicate-controls, .communicate-controls > div { gap: 0.25rem; }
          .communicate-controls button { padding: 0.375rem 0.4375rem; }
        }
        @media (max-width: 767px) {
          .communicate-camera-panel .communicate-cue { white-space: nowrap; font-size: 0.75rem; padding: 0.25rem 0.75rem; }
        }
        /* Short screens (phones in landscape): the camera keeps room for the controls around it */
        @media (max-width: 1023px) and (max-height: 560px) {
          .communicate-camera-panel { aspect-ratio: auto; height: max(200px, calc(100dvh - 9rem)); min-height: 0; }
        }
        @media (min-width: 768px) and (max-width: 1279px) {
          .communicate-camera-panel {
            aspect-ratio: 4 / 3;
          }
        }
      `}</style>
      
      {/* Header & 2-Way Mode Switcher */}
      <div className="communicate-header flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 pb-6 border-b border-mudra-lavender-200/60">
        <div>
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-mudra-lavender-100 text-mudra-lavender-700 text-xs font-bold font-mono mb-1">
            <Radio className="w-3.5 h-3.5 text-mudra-lavender-600 animate-pulse" />
            <span>2-WAY REAL-TIME COMMUNICATION BRIDGE</span>
          </div>
          <h1 className="font-display font-extrabold text-3xl sm:text-4xl text-mudra-indigo-900">
            Communicate Hub
          </h1>
          <p className="text-sm text-mudra-indigo-600 mt-1">
            Seamless two-way bridge between Indian Sign Language gestures and spoken language.
          </p>
        </div>

        {/* Mode Switcher Tabs */}
        <div className="flex items-center gap-1.5 p-1.5 rounded-2xl bg-white/90 border border-mudra-lavender-200 shadow-xs">
          <button
            type="button"
            onClick={() => {
              setActiveMode('sign-to-speech')
            }}
            className={`px-3.5 py-2 rounded-xl text-xs font-bold transition-all flex items-center gap-1.5 ${
              activeMode === 'sign-to-speech'
                ? 'bg-mudra-lavender-600 text-white shadow-xs'
                : 'text-mudra-indigo-700 hover:bg-mudra-lavender-50'
            }`}
          >
            <Camera className="w-3.5 h-3.5" />
            <span>Sign &rarr; Voice</span>
          </button>

          <button
            type="button"
            onClick={() => {
              handleStopCamera()
              setActiveMode('speech-to-sign')
            }}
            className={`px-3.5 py-2 rounded-xl text-xs font-bold transition-all flex items-center gap-1.5 ${
              activeMode === 'speech-to-sign'
                ? 'bg-mudra-lavender-600 text-white shadow-xs'
                : 'text-mudra-indigo-700 hover:bg-mudra-lavender-50'
            }`}
          >
            <Sparkles className="w-3.5 h-3.5" />
            <span>Voice &rarr; 3D Avatar</span>
          </button>
        </div>
      </div>

      {/* MODE 1: SIGN TO SPEECH (Signer to Listener) */}
      {activeMode === 'sign-to-speech' && (
        <div className="communicate-sign-grid grid grid-cols-1 lg:grid-cols-12 gap-8 items-start animate-fadeIn">
          
          {/* Left: Live Webcam + Canvas Landmark Overlay */}
          <div className="communicate-left lg:col-span-7 space-y-5">
            <div ref={cameraPanelRef} className="communicate-camera-panel relative aspect-video sm:aspect-[4/3] min-h-[320px] sm:min-h-0 bg-mudra-indigo-900 rounded-3xl overflow-hidden shadow-xl border-2 border-mudra-indigo-700/90 flex flex-col justify-between p-4 sm:p-5 text-white">
              
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

              {cameraActive && nextCue && inputMode === 'words' && (
                <div key={nextCue.at} role="status"
                     className="communicate-cue sl-motion absolute left-1/2 top-4 z-30 -translate-x-1/2 animate-sl-fade-up rounded-full border-2 border-sl-ink bg-sl-green px-4 py-1.5 text-sm font-extrabold text-sl-ink shadow-sl-sm">
                  ✓ Added “{nextCue.label}”: next word
                </div>
              )}

              {!cameraActive && (
                <div className="absolute inset-0 flex flex-col items-center justify-center p-6 text-center z-0">
                  <div className="w-16 h-16 rounded-2xl bg-mudra-indigo-800 border border-mudra-indigo-500/80 flex items-center justify-center text-mudra-lavender-200 mb-3 shadow-lg">
                    <Camera className="w-8 h-8 text-mudra-lavender-400" />
                  </div>
                  <h3 className="font-display font-bold text-base text-white mb-1">
                    Camera is Offline
                  </h3>
                  <p className="text-xs leading-relaxed text-mudra-indigo-100 max-w-sm mb-5">
                    Turn on camera to track hand landmarks and recognize ISL signs in real time.
                  </p>
                  {cameraError && (
                    <p role="alert" className="mb-4 max-w-sm rounded-xl bg-rose-500/90 px-3 py-2 text-xs font-semibold text-white">
                      {cameraError}
                    </p>
                  )}
                  <div className="mb-4 flex flex-wrap items-center justify-center gap-2 text-xs">
                    <label className="sr-only" htmlFor="camera-select">Camera</label>
                    <select
                      id="camera-select"
                      value={cameraId}
                      onChange={(e) => { setCameraId(e.target.value); writePref('mudra_camera_id', e.target.value) }}
                      className="max-w-[14rem] rounded-xl bg-mudra-indigo-800 px-2 py-1.5 text-white border border-mudra-indigo-500"
                    >
                      <option value="">Default camera</option>
                      {cameras.filter((c) => c.deviceId).map((c) => (
                        <option key={c.deviceId} value={c.deviceId}>{c.label}</option>
                      ))}
                    </select>
                    <label className="sr-only" htmlFor="camera-res">Resolution</label>
                    <select
                      id="camera-res"
                      value={cameraRes}
                      onChange={(e) => { setCameraRes(Number(e.target.value)); writePref('mudra_camera_res', e.target.value) }}
                      className="rounded-xl bg-mudra-indigo-800 px-2 py-1.5 text-white border border-mudra-indigo-500"
                    >
                      <option value={480}>640×480 (faster)</option>
                      <option value={720}>1280×720</option>
                    </select>
                  </div>
                  <button
                    type="button"
                    onClick={handleStartCamera}
                    disabled={cameraLoading}
                    className="btn-primary min-h-11 rounded-2xl py-2.5 px-5 text-xs font-bold shadow-lg"
                  >
                    <Camera className="w-4 h-4" />
                    <span>{cameraLoading ? 'Starting MediaPipe...' : 'Start Camera Stream'}</span>
                  </button>
                </div>
              )}

              {cameraActive && cameraIssue && (
                <div role="alert" className="absolute top-14 left-1/2 -translate-x-1/2 z-40 w-[90%] max-w-md rounded-2xl bg-rose-500/95 px-4 py-2.5 text-center text-sm font-semibold text-white shadow-lg">
                  {cameraIssue === 'black'
                    ? 'The camera picture is black. Check the lens cover or privacy shutter, or pick another camera.'
                    : 'The camera stopped sending pictures. Stop and restart the camera, or close other apps using it.'}
                </div>
              )}

              {cameraActive && !cameraIssue && !framing && trackerFps > 0 && trackerFps < LOW_FPS && (
                <div className="communicate-hint absolute bottom-20 left-1/2 -translate-x-1/2 z-30 w-[90%] max-w-md rounded-2xl bg-amber-400/95 px-4 py-2.5 text-center text-sm font-semibold text-amber-950 shadow-lg">
                  {slowReason === 'camera'
                    ? `Your camera sends only ${trackerDiag.cameraFps} frames per second. Webcams slow down in dim light: add light in front of you.`
                    : trackerDiag?.integratedGpu
                    ? `Hand tracking is slow (${trackerFps} fps): the browser uses the on-chip graphics (${trackerDiag.gpu}). If this computer also has an NVIDIA or AMD card, set Chrome to "High performance" in Windows Settings > System > Display > Graphics, then restart Chrome.`
                    : trackerDiag?.delegate === 'CPU'
                    ? `Hand tracking runs on the CPU (${trackerFps} fps). Turn on hardware acceleration in the browser settings, or close other tabs.`
                    : `Hand tracking is slow (${trackerFps} fps, ${trackerDiag?.inferenceMs ?? '?'} ms per frame). Close other tabs or apps using the GPU.`}
                </div>
              )}

              {cameraActive && framing && (
                <div className="communicate-hint absolute bottom-20 left-1/2 -translate-x-1/2 z-30 w-[90%] max-w-md rounded-2xl bg-amber-400/95 px-4 py-2.5 text-center text-sm font-semibold text-amber-950 shadow-lg">
                  {framing === 'no_shoulders'
                    ? 'Move back so both shoulders are in view: recognition needs your upper body.'
                    : 'Move back a little: your chest and both hands should stay in view while you sign.'}
                </div>
              )}

              {/* Status Header */}
              <div className="flex items-center justify-between z-20">
                <div className="flex items-center gap-2 bg-black/60 backdrop-blur-md px-3 py-1.5 rounded-full border border-white/10 text-xs">
                  <span className={`w-2 h-2 rounded-full ${cameraActive ? 'bg-emerald-400 animate-pulse' : 'bg-amber-400'}`}></span>
                  <span className="font-mono">{cameraActive ? 'LIVE TRACKING' : 'OFFLINE'}</span>
                </div>

                {cameraActive && (
                  <div className="flex items-center gap-2 bg-black/60 backdrop-blur-md px-3 py-1.5 rounded-full border border-white/10 text-xs font-mono text-mudra-peach-300">
                    <span>
                      {currentDetection.source === 'model'
                        ? `Model · ${trackerFps} fps${trackerDiag?.delegate ? ` · ${trackerDiag.delegate}` : ''}`
                        : currentDetection.source === 'manual'
                        ? 'Manual entry'
                        : `Rules · ${currentDetection.confidence ?? 0}%`}
                    </span>
                  </div>
                )}
              </div>

              {/* Bottom Controls */}
              {cameraActive && (
                <div className="communicate-controls flex flex-wrap items-center justify-between gap-3 z-20 pt-2 border-t border-white/10">
                  <div className="flex items-center gap-2">
                    <button
                      type="button"
                      onClick={handleStopCamera}
                      className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-rose-500/90 hover:bg-rose-600 text-white text-xs font-bold transition-all"
                    >
                      <CameraOff className="w-3.5 h-3.5" />
                      <span>Stop Camera</span>
                    </button>

                    <button
                      type="button"
                      onClick={() => setIsPaused(!isPaused)}
                      className="px-3 py-1.5 rounded-xl bg-white/15 hover:bg-white/25 text-white text-xs font-semibold backdrop-blur-md border border-white/10 flex items-center gap-1.5"
                    >
                      {isPaused ? <Play className="w-3.5 h-3.5 text-emerald-400" /> : <Pause className="w-3.5 h-3.5" />}
                      <span>{isPaused ? 'Resume' : 'Pause'}</span>
                    </button>
                  </div>

                  <button
                    type="button"
                    onClick={handleClearBuffer}
                    className="px-3 py-1.5 rounded-xl bg-white/15 hover:bg-white/25 text-white text-xs font-semibold backdrop-blur-md border border-white/10 flex items-center gap-1.5"
                  >
                    <RotateCcw className="w-3.5 h-3.5" />
                    <span>Clear Buffer</span>
                  </button>
                </div>
              )}
            </div>

            {/* Quick Sign Simulator Chips */}
            <div className="communicate-chips glass-card rounded-3xl p-4 border border-white space-y-2">
              <div className="flex items-center justify-between text-xs">
                <span className="font-bold text-mudra-indigo-900 font-display">
                  {modelState.status === 'online' ? 'Add a word manually:' : 'Rule-based signs (hold in camera or click to add):'}
                </span>
                {modelState.status === 'online' && (
                  <span className="text-[10px] text-mudra-indigo-500">Not recognised from the camera yet: teach them on the Record page</span>
                )}
                <span className="text-[10px] text-mudra-indigo-500 font-mono">{chipSigns.length} Signs</span>
              </div>
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
                {chipSigns.map((v) => (
                  <button
                    key={v.id}
                    type="button"
                    onClick={() => handleSimulateSign(v)}
                    className="p-2 rounded-xl text-left border bg-white border-mudra-lavender-200 hover:bg-mudra-lavender-50 transition-all flex items-center justify-between text-xs"
                  >
                    <span className="font-bold text-mudra-indigo-900">{v.label}</span>
                    <span className="text-[10px] font-mono text-emerald-700 bg-emerald-50 px-1.5 py-0.5 rounded">ISL</span>
                  </button>
                ))}
              </div>
            </div>

            {modelState.status === 'online' && vocab.length > 0 && (
              <div className="communicate-vocab glass-card rounded-2xl p-4 border border-mudra-lavender-200/80 space-y-3">
                <button
                  type="button"
                  onClick={() => setVocabOpen((o) => !o)}
                  className="w-full flex items-center justify-between text-xs font-bold text-mudra-indigo-900 font-display"
                >
                  <span>What can I sign? {vocab.length} words the camera recognises</span>
                  <span className="font-mono text-mudra-indigo-500">{vocabOpen ? 'hide' : 'show'}</span>
                </button>
                {vocabOpen && (
                  <>
                    <input
                      value={vocabQuery}
                      onChange={(e) => setVocabQuery(e.target.value)}
                      placeholder="Search a word, e.g. water"
                      className="w-full rounded-xl border border-mudra-lavender-300 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-mudra-indigo-400"
                    />
                    <div className="max-h-48 overflow-y-auto flex flex-wrap gap-1.5">
                      {vocabMatches.slice(0, 400).map((v) => (
                        <button
                          key={v.sign}
                          type="button"
                          disabled={!v.has_reference}
                          onClick={() => setReferenceSign(v.sign)}
                          title={v.test_clips ? `Tested: ${v.test_correct}/${v.test_clips} correct` : undefined}
                          className={`px-2 py-0.5 rounded-full border text-xs ${referenceSign === v.sign
                            ? 'bg-mudra-indigo-700 border-mudra-indigo-700 text-white'
                            : 'bg-white border-mudra-lavender-200 text-mudra-indigo-900 hover:bg-mudra-lavender-50'} disabled:cursor-default`}
                        >
                          {v.label}
                        </button>
                      ))}
                      {vocabMatches.length === 0 && (
                        <span className="text-xs text-mudra-indigo-600">
                          Not in the model yet. Teach it on the Record page (5 recordings) and it works here straight away.
                        </span>
                      )}
                    </div>
                    {referenceSign && (
                      <SignReference
                        sign={referenceSign}
                        testCorrect={vocab.find((v) => v.sign === referenceSign)?.test_correct}
                        testClips={vocab.find((v) => v.sign === referenceSign)?.test_clips}
                        onClose={() => setReferenceSign(null)}
                      />
                    )}
                    <p className="text-[11px] text-mudra-indigo-600">
                      Tap a word to see how the model expects it to be signed. Sign one word from start to finish, then
                      lower your hands or hold them still for a moment. Stand so your shoulders and both hands are in view.
                    </p>
                  </>
                )}
              </div>
            )}
          </div>

          {/* Right: Interpretation Panel & Speech Out */}
          <div className="communicate-right lg:col-span-5 space-y-5 lg:pt-0">
            <div className="communicate-results glass-card rounded-3xl p-5 sm:p-6 border border-mudra-lavender-200/80 shadow-lg space-y-7">
              <InputModeSwitch mode={inputMode} onChange={changeInputMode} />

              {inputMode === 'spell' ? (
                <SpellPanel cameraActive={cameraActive} accuracy={libraryModels.letters}
                            onCommitWord={(word) => setRecognizedTokens((prev) => [...prev, word])} />
              ) : inputMode === 'phrase' ? (
                <PhrasePanel cameraActive={cameraActive} accuracy={libraryModels.phrases}
                             onCommitPhrase={(phrase) => setRecognizedTokens((prev) => [...prev, phrase])} />
              ) : (
              <div className="communicate-words">
                {wordModels.length > 1 && (
                  <label className="communicate-model-picker mb-3 block text-xs font-semibold text-mudra-indigo-700">
                    Recognition model
                    <select value={wordModel} onChange={(e) => setWordModel(e.target.value)}
                            className="mt-1 block w-full rounded-xl border border-mudra-lavender-300 bg-white px-2 py-1.5 text-sm">
                      {wordModels.map((m) => (
                        <option key={m.id} value={m.id}>
                          {m.experimental ? 'Experimental: ' : ''}{m.name} · {m.live_words} words
                          {m.test_accuracy != null ? ` · ${Math.round(m.test_accuracy * 100)}% (${m.accuracy_split})` : ''}
                        </option>
                      ))}
                    </select>
                    {wordModels.find((m) => m.id === wordModel)?.our_test && (() => {
                      const t = wordModels.find((m) => m.id === wordModel).our_test
                      const right = t.model_correct ?? t.static_correct
                      const alarms = t.false_alarm_rate ?? t.static_false_alarm_rate
                      if (wordModel === 'merged' || wordModel === 'auto') {
                        const sel = wordModels.find((m) => m.id === wordModel)
                        return (
                          <span className="mt-1 block font-normal text-amber-800">
                            SignLine words answer every word they know (most accurate on our tests). Only when they are
                            unsure may a teammate model add a word they lack: {sel.words.join(', ').toLowerCase()} (70%
                            confidence or more; on SignLine's 1,501 test clips this added no wrong answers).
                            {wordModel === 'auto' ? ' Hold a hand shape still in front of you to spell a letter; letters close together make one word.' : ''}
                            {' '}Live accuracy is not measured yet.
                          </span>
                        )
                      }
                      return (
                        <span className="mt-1 block font-normal text-amber-800">
                          On SignLine's own test clips: {right} of {t.shared_clips} shared-word signs right (the
                          default model: {t.word_model_correct} of {t.shared_clips}); answers on {Math.round(alarms * 1000) / 10}% of
                          other signs.{wordModel === 'intro_static' ? ' Holds a still pose for about half a second.' : ''}
                        </span>
                      )
                    })()}
                  </label>
                )}
                <div className="flex items-center justify-between mb-2">
                  <span className="text-xs font-bold uppercase tracking-[0.12em] text-mudra-lavender-800 font-display">
                    Detected Sign
                  </span>
                  <span className={`shrink-0 text-[11px] font-mono font-bold px-2.5 py-1 rounded-full border ${
                    currentDetection.status === 'uncertain'
                      ? 'bg-amber-100 text-amber-800 border-amber-200'
                      : currentDetection.isUnknown
                      ? 'bg-gray-100 text-gray-600 border-gray-200'
                      : 'bg-emerald-100 text-emerald-800 border-emerald-200'
                  }`}>
                    {currentDetection.source === 'manual'
                      ? 'Added manually'
                      : currentDetection.source === 'custom' && !currentDetection.isUnknown
                      ? 'Your recorded word'
                      : currentDetection.status === 'unsure'
                      ? `Added, unsure · ${currentDetection.confidence}%`
                      : currentDetection.status === 'uncertain'
                      ? `Uncertain · ${currentDetection.confidence}%`
                      : currentDetection.status === 'signing'
                      ? 'Signing…'
                      : currentDetection.status === 'no_hands'
                      ? 'No hands in view'
                      : currentDetection.isUnknown
                      ? (currentDetection.source === 'model' ? 'Waiting for a sign' : 'Below 68% match')
                      : `${currentDetection.confidence}% confidence`}
                  </span>
                </div>

                <div className="flex items-center gap-3 min-h-12">
                  <span className={`font-display font-extrabold text-3xl sm:text-4xl tracking-tight leading-none break-words ${
                    currentDetection.isUnknown ? 'text-mudra-indigo-600 italic' : 'text-mudra-indigo-950'
                  }`}>
                    {currentDetection.isUnknown
                      ? (currentDetection.status === 'uncertain' ? 'Not sure yet'
                        : currentDetection.status === 'missed' ? "Didn't catch that"
                        : currentDetection.status === 'signing' ? 'Signing…'
                        : currentDetection.status === 'too_short' ? 'Too short, sign again'
                        : 'Waiting for a sign')
                      : (currentDetection.label || readableSign(currentDetection.sign))}
                  </span>
                </div>
                {currentDetection.status === 'signing' && currentDetection.candidate && (
                  <p className="text-xs text-mudra-indigo-600 mt-1">
                    Looks like <strong>{currentDetection.candidate}</strong>… (finish the sign, then hold your hands still)
                  </p>
                )}
                {!currentDetection.isUnknown && currentDetection.sign && currentDetection.sign !== 'UNKNOWN' && (
                  <button
                    type="button"
                    onClick={() => { setVocabOpen(true); setReferenceSign(currentDetection.sign) }}
                    className="mt-1 text-xs font-semibold text-mudra-indigo-700 underline hover:text-mudra-indigo-900"
                  >
                    See how "{currentDetection.label || readableSign(currentDetection.sign)}" is signed
                  </button>
                )}
                {suggestions.length > 0 && (
                  <div className="mt-2 flex flex-wrap items-center gap-2 text-xs">
                    <span className="text-mudra-indigo-600">Added "{currentDetection.label}". Wrong? Replace with:</span>
                    {suggestions.map((s) => (
                      <button
                        key={s.sign}
                        type="button"
                        onClick={() => { replaceLastToken(s.sign); setSuggestions([]) }}
                        className="px-2.5 py-1 rounded-full border border-mudra-lavender-300 bg-white font-semibold text-mudra-indigo-900 hover:bg-mudra-lavender-50"
                      >
                        {s.label} <span className="font-mono text-mudra-indigo-400">{Math.round(s.confidence * 100)}%</span>
                      </button>
                    ))}
                  </div>
                )}
                <div className="mt-3 flex flex-wrap items-center gap-2 text-[11px]">
                  <span className={`inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full font-mono font-semibold ${
                    modelState.status === 'online' ? 'bg-emerald-50 text-emerald-800' : 'bg-amber-50 text-amber-800'
                  }`}>
                    <span className={`w-1.5 h-1.5 rounded-full ${modelState.status === 'online' ? 'bg-emerald-500' : 'bg-amber-500'}`}></span>
                    {modelState.status === 'online'
                      ? `Trained model · ${modelState.info?.num_signs ?? ''} words`
                      : modelState.status === 'checking'
                      ? 'Connecting to recognition model...'
                      : 'Offline: rule-based fallback, 8 signs'}
                  </span>
                  {modelState.status !== 'online' && modelState.status !== 'checking' && (
                    <button
                      type="button"
                      onClick={() => signRecognizer.connect()}
                      className="px-2 py-0.5 rounded-full border border-mudra-lavender-300 text-mudra-lavender-800 font-semibold hover:bg-mudra-lavender-50"
                    >
                      Retry connection
                    </button>
                  )}
                </div>
                {modelState.status === 'offline' && modelState.error && (
                  <p className="text-[11px] text-mudra-indigo-500 mt-1">{modelState.error}</p>
                )}
              </div>
              )}

              {/* Committed Sequence Buffer */}
              <div className="space-y-2 pt-2 border-t border-mudra-lavender-200/60">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <span className="text-xs font-bold uppercase tracking-wider text-mudra-indigo-500 font-display">
                    Recognized
                  </span>
                  <span className="text-[10px] text-mudra-indigo-400 font-mono">
                    {recognizedTokens.length} Tokens
                  </span>
                </div>

                <div className="flex items-center gap-2 flex-wrap min-h-[42px] p-2.5 rounded-xl bg-mudra-ivory-100/90 border border-mudra-lavender-200">
                  {recognizedTokens.length > 0 ? (
                    recognizedTokens.map((token, i) => (
                      <React.Fragment key={i}>
                        <span className="px-2.5 py-1 rounded-lg bg-mudra-indigo-900 text-white font-mono text-xs font-semibold shadow-xs">
                          {readableSign(token)}
                        </span>
                        {i < recognizedTokens.length - 1 && (
                          <span className="text-mudra-indigo-400 text-xs font-bold">&rarr;</span>
                        )}
                      </React.Fragment>
                    ))
                  ) : (
                    <span className="text-xs text-mudra-indigo-400 italic">
                      {modelState.status === 'online'
                        ? 'Sign a word, then lower your hands or hold them still for a moment; it is added when the sign ends'
                        : 'Hold a sign steady in camera to commit token'}
                    </span>
                  )}
                </div>
              </div>

              {/* Natural Sentence Output */}
              <div className="space-y-2 pt-2 border-t border-mudra-lavender-200/60">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-bold uppercase tracking-wider text-mudra-indigo-500 font-display">
                    Meaning
                  </span>
                  
                  {/* Language Selector Tabs */}
                  <div className="flex items-center gap-1 p-1 rounded-xl bg-mudra-ivory-100 border border-mudra-lavender-200 shadow-xs shrink-0">
                    <button
                      type="button"
                      onClick={() => setSelectedLang('en')}
                      className={`min-w-9 px-2.5 py-1.5 rounded-lg text-[10px] font-bold tracking-wider uppercase transition-all ${
                        selectedLang === 'en'
                          ? 'bg-mudra-lavender-600 text-white shadow-xs'
                          : 'text-mudra-indigo-700 hover:bg-mudra-lavender-50'
                      }`}
                    >
                      EN
                    </button>
                    <button
                      type="button"
                      onClick={() => setSelectedLang('hi')}
                      className={`min-w-9 px-2.5 py-1.5 rounded-lg text-[10px] font-bold tracking-wider uppercase transition-all ${
                        selectedLang === 'hi'
                          ? 'bg-mudra-lavender-600 text-white shadow-xs'
                          : 'text-mudra-indigo-700 hover:bg-mudra-lavender-50'
                      }`}
                    >
                      हिन्दी
                    </button>
                    <button
                      type="button"
                      onClick={() => setSelectedLang('mr')}
                      className={`min-w-9 px-2.5 py-1.5 rounded-lg text-[10px] font-bold tracking-wider uppercase transition-all ${
                        selectedLang === 'mr'
                          ? 'bg-mudra-lavender-600 text-white shadow-xs'
                          : 'text-mudra-indigo-700 hover:bg-mudra-lavender-50'
                      }`}
                    >
                      मराठी
                    </button>
                  </div>
                </div>

                <div className="p-4 rounded-2xl bg-gradient-to-br from-mudra-lavender-50 via-white to-mudra-peach-50 border border-mudra-lavender-300 min-h-[64px] flex items-center">
                  <div className="font-display font-semibold text-lg sm:text-xl text-mudra-indigo-950">
                    {generatedSentence ? `"${generatedSentence}"` : <span className="text-mudra-indigo-400 italic">No sentence generated yet</span>}
                  </div>
                </div>
              </div>

              {/* Non-manual (facial) cue layer — experimental, describes movement, not emotion */}
              <div className="space-y-3 pt-2 border-t border-mudra-lavender-200/60">
                <div className="flex items-start gap-3">
                  <span
                    key={`${expression.tone}-${faceStatus}`}
                    className="text-3xl leading-none animate-fadeIn"
                    aria-hidden="true"
                  >
                    {faceStatus === 'not-detected' || faceStatus === 'error' ? '🤔' : expression.emoji}
                  </span>
                  <div className="min-w-0 flex-1 space-y-1">
                    <div className="flex flex-wrap items-center justify-between gap-2">
                      <span className="text-[10px] font-bold uppercase tracking-[0.14em] text-mudra-lavender-700 font-display">
                        FACIAL CUES (EXPERIMENTAL)
                      </span>
                      {faceStatus === 'detected' && expression.available && (
                        <span className="text-[10px] font-mono font-bold text-mudra-indigo-600">
                          cue score {Math.round(expression.confidence * 100)} (uncalibrated)
                        </span>
                      )}
                    </div>
                    <div className="font-display font-bold text-lg text-mudra-indigo-950">
                      {faceStatus === 'not-detected'
                        ? 'Face not detected'
                        : faceStatus === 'error'
                        ? 'Facial cues unavailable'
                        : faceStatus === 'insufficient'
                        ? 'No strong facial cue'
                        : expression.label}
                    </div>
                    <p className="text-xs leading-relaxed text-mudra-indigo-600">
                      {faceStatus === 'initializing'
                        ? 'Loading facial analysis...'
                        : !cameraActive
                        ? 'Start the camera to observe facial cues.'
                        : faceStatus === 'not-detected'
                        ? 'Face not detected.'
                        : faceStatus === 'error'
                        ? expression.reason
                        : faceStatus === 'insufficient'
                        ? 'Insufficient facial cues.'
                        : 'Describes facial movement only. It is not a reading of emotion.'}
                    </p>
                    <p className="text-[10px] font-medium text-mudra-indigo-500">
                      Facial analysis runs locally on your device. No images are stored or uploaded.
                    </p>
                  </div>
                </div>
              </div>

              {/* Action Buttons: Speak & Copy */}
              <label className="flex items-center gap-2 text-xs font-semibold text-mudra-indigo-800 cursor-pointer select-none">
                <input type="checkbox" checked={autoSpeak} onChange={(e) => setAutoSpeak(e.target.checked)} className="accent-mudra-indigo-700 w-4 h-4" />
                Speak automatically when I pause ({AUTO_SPEAK_PAUSE_MS / 1000} s)
              </label>
              <div className="flex flex-wrap items-center gap-3 pt-2">
                <button
                  type="button"
                  onClick={handleSpeakSentence}
                  disabled={!generatedSentence}
                  className={`btn-primary min-h-12 min-w-0 flex-[1_1_12rem] rounded-2xl py-3.5 text-sm shadow-md ${
                    isSpeaking ? 'bg-mudra-lavender-700 animate-pulse' : ''
                  }`}
                >
                  <Volume2 className={`w-4 h-4 ${isSpeaking ? 'animate-bounce' : ''}`} />
                  <span>{isSpeaking ? 'Speaking Out Loud...' : 'Speak Sentence'}</span>
                </button>

                <button
                  type="button"
                  onClick={handleCopyText}
                  disabled={!generatedSentence}
                  className="btn-secondary shrink-0 py-3.5 px-4 text-xs font-semibold"
                  title="Copy sentence"
                >
                  {copied ? <CheckCircle2 className="w-4 h-4 text-emerald-600" /> : <Copy className="w-4 h-4" />}
                  <span>{copied ? 'Copied' : 'Copy'}</span>
                </button>

                <button
                  type="button"
                  onClick={handleClearBuffer}
                  className="btn-secondary shrink-0 py-3.5 px-3 text-xs"
                  title="Clear tokens"
                >
                  <RotateCcw className="w-4 h-4" />
                </button>
              </div>

            </div>
          </div>

        </div>
      )}

      {/* MODE 2: SPEECH / TEXT TO 3D AVATAR (Listener to Signer) */}
      {activeMode === 'speech-to-sign' && (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start animate-fadeIn">
          
          {/* Left: Input & Speech Dictation */}
          <div className="lg:col-span-6 space-y-5">
            <div className="glass-card rounded-3xl p-6 sm:p-7 border border-white shadow-xl space-y-5">
              <div>
                <div className="inline-flex items-center gap-1.5 text-xs font-bold text-mudra-peach-warm uppercase tracking-wider font-mono mb-1">
                  <Mic className="w-3.5 h-3.5" />
                  <span>Hearing Speaker Input</span>
                </div>
                <h2 className="font-display font-bold text-xl sm:text-2xl text-mudra-indigo-900">
                  Speak or Type to Translate to ISL
                </h2>
                <p className="text-xs text-mudra-indigo-600 mt-1">
                  The MUDRA 3D Avatar will perform the sign language translation in real time.
                </p>
              </div>

              <form onSubmit={handleTranslateToSign} className="space-y-3">
                <div className="relative">
                  <textarea
                    rows={3}
                    value={textInput}
                    onChange={(e) => setTextInput(e.target.value)}
                    placeholder="Type words or letters (e.g. 'HELLO', 'NAMASTE', 'THANK YOU', 'YES', 'NO', 'TIME')..."
                    className="w-full p-4 rounded-2xl bg-white border border-mudra-lavender-200 text-sm text-mudra-indigo-900 focus:outline-none focus:border-mudra-lavender-500 shadow-inner resize-none font-medium"
                  />

                  <button
                    type="button"
                    onClick={handleToggleVoiceInput}
                    className={`absolute bottom-3 right-3 p-2.5 rounded-xl border transition-all ${
                      isListening
                        ? 'bg-rose-500 text-white animate-pulse border-rose-600 shadow-md'
                        : 'bg-mudra-lavender-50 text-mudra-lavender-700 border-mudra-lavender-200 hover:bg-mudra-lavender-100'
                    }`}
                    title={isListening ? 'Listening...' : 'Click to Speak via Microphone'}
                  >
                    {isListening ? <MicOff className="w-4 h-4" /> : <Mic className="w-4 h-4" />}
                  </button>
                </div>

                <div className="flex items-center justify-between gap-3">
                  <div className="flex items-center gap-1 text-[11px] text-mudra-indigo-500">
                    <Sparkles className="w-3 h-3 text-mudra-lavender-600" />
                    <span>Avatar: 314 recorded word signs, 5 built-in signs, A-Z from real letter photos</span>
                  </div>

                  <button
                    type="submit"
                    className="btn-primary py-2.5 px-5 text-xs font-bold flex items-center gap-2"
                  >
                    <span>Sign with 3D Avatar</span>
                    <Send className="w-3.5 h-3.5" />
                  </button>
                </div>
              </form>

              {/* Sample phrases */}
              <div className="pt-2 border-t border-mudra-lavender-200/60 space-y-2">
                <span className="text-[11px] font-bold uppercase text-mudra-indigo-500 font-display">
                  Quick Phrases:
                </span>
                <div className="flex flex-wrap gap-2">
                  {['HELLO', 'NAMASTE', 'THANK YOU', 'YES', 'NO', 'TIME', 'HOME', 'GOODBYE'].map((phrase) => (
                    <button
                      key={phrase}
                      type="button"
                      onClick={() => {
                        setTextInput(phrase)
                        setActiveAvatarText(phrase)
                      }}
                      className="px-3 py-1 rounded-xl bg-white hover:bg-mudra-lavender-50 border border-mudra-lavender-200 text-xs font-semibold text-mudra-indigo-800 transition-colors shadow-xs"
                    >
                      {phrase}
                    </button>
                  ))}
                </div>
              </div>
            </div>
          </div>

          {/* Right: 3D Avatar Screen */}
          <div className="lg:col-span-6 space-y-4">
            <MudraAvatarViewer
              signText={activeAvatarText}
              className="w-full h-[400px] sm:h-[450px]"
            />
          </div>

        </div>
      )}

    </div>
  )
}
