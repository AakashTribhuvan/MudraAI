import React, { useState, useRef, useEffect, useCallback } from 'react'
import { Camera, CameraOff, Circle, CheckCircle2, AlertCircle, Trash2, Info, Users, Sparkles, Database } from 'lucide-react'
import { readableSign } from '../services/islDictionary'
import handTracker from '../services/handTracker'
import signRecognizer, { toFrame } from '../services/signRecognizer'
import { listRecordings, saveRecording, deleteRecording } from '../services/recordingClient'
import CameraSourcePicker, { useCameraSources } from '../components/CameraSourcePicker'

// App words the trained model does not know yet (no training data so far)
const SUGGESTED_WORDS = ['Namaste', 'Water', 'Help', 'Yes', 'No', 'Goodbye', 'Home', 'Person']
const DURATIONS = [2, 3, 4, 5]
const COUNTDOWN_S = 3
const TARGET_PER_WORD = 5
const SIGNER_KEY = 'mudra.recorder.signer'

const readSigner = () => {
  try {
    return window.localStorage.getItem(SIGNER_KEY) || ''
  } catch {
    return ''
  }
}

const QUALITY_MESSAGES = {
  too_short: 'The recording was too short. Try again.',
  low_hand_detection: 'Your hands were out of view most of the time. Step back so both hands stay in the frame.',
  no_pose: 'Your shoulders were not visible. Step back so your upper body is in the frame.'
}

/**
 * Barrier checks for one recorded word (GET /recordings "quality"): takes that agree, takes left out,
 * the nearest other taught word, and which known word the trained model reads the takes as.
 */
function WordQuality({ q, readable }) {
  if (!q) return null
  const notes = []
  if (q.looks_like) notes.push(`The model reads ${q.looks_like.takes} of these takes as "${readable(q.looks_like.sign)}". Check the sign, or record it more distinctly.`)
  if (q.too_close && q.nearest) notes.push(`Very close to "${readable(q.nearest)}" (${Math.round(q.nearest_similarity * 100)}% alike), so the two can be mixed up. Record clearer, more varied takes of both.`)
  if (q.consistency != null && q.consistency < 0.7) notes.push(`The takes differ a lot (${Math.round(q.consistency * 100)}% alike). Record a few more, signed the same way.`)
  if (q.left_out?.length) notes.push(`${q.left_out.length} take(s) did not match the others and are left out.`)
  return (
    <div className="space-y-1 text-xs">
      {q.consistency != null && <p className="text-mudra-indigo-600">Takes agree: {Math.round(q.consistency * 100)}%</p>}
      {notes.map((n) => (
        <p key={n} className="flex items-start gap-1 rounded-md bg-amber-50 px-2 py-1 text-amber-900">
          <AlertCircle className="mt-0.5 h-3 w-3 shrink-0" aria-hidden="true" /> {n}
        </p>
      ))}
    </div>
  )
}

export default function RecordPage() {
  const videoRef = useRef(null)
  const canvasRef = useRef(null)
  const framesRef = useRef(null) // frames being captured, or null when not recording
  const sizeRef = useRef({ width: 640, height: 480 })
  const handsSeenRef = useRef(0)

  const [cameraActive, setCameraActive] = useState(false)
  const [cameraError, setCameraError] = useState(null)
  const { sources: cameraSources, deviceId: cameraDeviceId, setDeviceId: setCameraDeviceId } = useCameraSources(cameraActive)
  const [backend, setBackend] = useState({ status: 'checking', error: null })
  const [signer, setSigner] = useState(readSigner)
  const [word, setWord] = useState(SUGGESTED_WORDS[0])
  const [duration, setDuration] = useState(3)
  const [phase, setPhase] = useState('idle') // idle | countdown | recording | saving
  const [countdown, setCountdown] = useState(0)
  const [elapsed, setElapsed] = useState(0)
  const [lastResult, setLastResult] = useState(null)
  const [summary, setSummary] = useState({ total: 0, signs: [], custom_words: [] })
  const [handsVisible, setHandsVisible] = useState(false)

  const refresh = useCallback(async () => {
    try {
      setSummary(await listRecordings())
    } catch (err) {
      setBackend({ status: 'offline', error: err.message })
    }
  }, [])

  useEffect(() => {
    signRecognizer.connect().then((status) => {
      setBackend({ status, error: signRecognizer.error })
      if (status === 'online') refresh()
    })
  }, [refresh])

  useEffect(() => {
    try {
      window.localStorage.setItem(SIGNER_KEY, signer)
    } catch {
      /* storage unavailable: the name is just not remembered */
    }
  }, [signer])

  useEffect(() => { handTracker.preload(); return () => handTracker.stopCamera() }, [])

  const handleResults = useCallback((results) => {
    const canvas = canvasRef.current
    const ctx = canvas?.getContext('2d')
    if (ctx) {
      ctx.clearRect(0, 0, canvas.width, canvas.height)
      handTracker.drawPose(ctx, results.poseLandmarks, canvas.width, canvas.height, true)
      ;(results.multiHandLandmarks || []).forEach((lm) => handTracker.drawSkeleton(ctx, lm, canvas.width, canvas.height, true))
    }
    sizeRef.current = { width: results.width || 640, height: results.height || 480 }
    const hasHands = (results.multiHandLandmarks || []).length > 0
    handsSeenRef.current = hasHands ? performance.now() : handsSeenRef.current
    setHandsVisible(performance.now() - handsSeenRef.current < 500)
    if (framesRef.current) framesRef.current.push(toFrame(results, performance.now() / 1000))
  }, [])

  const startCamera = async () => {
    setCameraError(null)
    try {
      await handTracker.startCamera(videoRef.current, handleResults, { deviceId: cameraDeviceId })
      setCameraActive(true)
    } catch (err) {
      setCameraError(err?.message || 'Could not start the camera.')
    }
  }

  const handleCameraSourceChange = async (deviceId) => {
    setCameraDeviceId(deviceId)
    if (!cameraActive || !videoRef.current) return
    handTracker.stopCamera()
    try {
      await handTracker.startCamera(videoRef.current, handleResults, { deviceId })
    } catch (err) {
      setCameraError(err?.message || 'Could not switch camera.')
      setCameraActive(false)
    }
  }

  const stopCamera = () => {
    handTracker.stopCamera()
    framesRef.current = null
    setPhase('idle')
    setCameraActive(false)
  }

  const finishRecording = useCallback(async () => {
    const frames = framesRef.current || []
    framesRef.current = null
    setPhase('saving')
    try {
      const saved = await saveRecording({ label: word, signer: signer.trim() || 'anonymous', ...sizeRef.current, frames })
      setLastResult({ ok: !saved.quality_flag, saved })
      await refresh()
    } catch (err) {
      setLastResult({ ok: false, error: err.message })
    }
    setPhase('idle')
  }, [word, signer, refresh])

  const record = useCallback(() => {
    if (!cameraActive || phase !== 'idle' || backend.status !== 'online' || !word.trim()) return
    setLastResult(null)
    setPhase('countdown')
    let left = COUNTDOWN_S
    setCountdown(left)
    const tick = setInterval(() => {
      left -= 1
      if (left > 0) {
        setCountdown(left)
        return
      }
      clearInterval(tick)
      framesRef.current = []
      setPhase('recording')
      const started = performance.now()
      const timer = setInterval(() => {
        const s = (performance.now() - started) / 1000
        setElapsed(s)
        if (s >= duration) {
          clearInterval(timer)
          finishRecording()
        }
      }, 100)
    }, 1000)
  }, [cameraActive, phase, backend.status, word, duration, finishRecording])

  useEffect(() => {
    const onKey = (e) => {
      if (e.code === 'Space' && e.target.tagName !== 'INPUT') {
        e.preventDefault()
        record()
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [record])

  const handleDelete = async (sampleId) => {
    await deleteRecording(sampleId)
    if (lastResult?.saved?.sample_id === sampleId) setLastResult(null)
    refresh()
  }

  const current = summary.signs.find((s) => s.sign.toLowerCase() === word.trim().toLowerCase().replace(/[\s-]+/g, '_'))
  const canRecord = cameraActive && phase === 'idle' && backend.status === 'online' && word.trim()

  return (
    <div className="pt-6 lg:pt-10 pb-24 lg:pb-16 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 space-y-8">
      <header className="space-y-2">
        <h1 className="font-display text-3xl sm:text-4xl font-bold text-mudra-indigo-900">Record signs</h1>
        <p className="text-mudra-indigo-700 max-w-3xl">
          Teach MUDRA new words and help it learn more signers. Record each word {TARGET_PER_WORD} times.
          New words work in Communicate straight away and are added to the next training run.
        </p>
        <p className="text-sm text-mudra-indigo-600 flex items-start gap-2 max-w-3xl">
          <Info className="w-4 h-4 mt-0.5 shrink-0" />
          Only body and hand landmark points are saved, on this computer, never video.
        </p>
      </header>

      {backend.status === 'offline' && (
        <div className="rounded-2xl border border-amber-300 bg-amber-50 p-4 text-amber-900 text-sm flex gap-2">
          <AlertCircle className="w-5 h-5 shrink-0" />
          <span>{backend.error || 'The recognition server is not running.'} Recording needs it to save signs.</span>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
        <div className="lg:col-span-7 space-y-4">
          <div className="relative aspect-[4/3] bg-mudra-indigo-900 rounded-3xl overflow-hidden shadow-xl border-2 border-mudra-indigo-700/90">
            <video ref={videoRef} className={`absolute inset-0 w-full h-full object-cover -scale-x-100 ${cameraActive ? 'opacity-100' : 'opacity-0'}`} playsInline muted />
            <canvas ref={canvasRef} width={640} height={480} className="absolute inset-0 w-full h-full object-cover z-10 pointer-events-none" />
            {cameraSources.length > 1 && (
              <div className="absolute top-4 right-4 z-30 rounded-xl bg-black/60 px-3 py-2 text-white">
                <CameraSourcePicker sources={cameraSources} deviceId={cameraDeviceId} onChange={handleCameraSourceChange} />
              </div>
            )}

            {!cameraActive && (
              <div className="absolute inset-0 flex flex-col items-center justify-center text-center text-white p-6 gap-3">
                <Camera className="w-10 h-10 text-mudra-lavender-300" />
                <p className="font-semibold">Camera is off</p>
                {cameraError && <p className="text-sm text-red-200">{cameraError}</p>}
              </div>
            )}

            {phase === 'countdown' && (
              <div className="absolute inset-0 z-20 flex flex-col items-center justify-center bg-black/30 text-white">
                <span className="text-8xl font-display font-bold">{countdown}</span>
                <span className="mt-2 text-lg">Hands down, get ready</span>
              </div>
            )}

            {phase === 'recording' && (
              <div className="absolute top-4 left-4 z-20 flex items-center gap-2 rounded-full bg-red-600 px-4 py-1.5 text-white font-semibold shadow">
                <Circle className="w-3 h-3 fill-white animate-pulse" />
                Sign "{word}" now, then lower your hands · {Math.max(0, duration - elapsed).toFixed(1)} s
              </div>
            )}

            {cameraActive && phase === 'idle' && (
              <div className={`absolute top-4 left-4 z-20 rounded-full px-3 py-1 text-xs font-semibold ${handsVisible ? 'bg-emerald-500 text-white' : 'bg-white/80 text-mudra-indigo-800'}`}>
                {handsVisible ? 'Hands in view' : 'Show your hands to check the framing'}
              </div>
            )}
          </div>

          <div className="flex flex-wrap gap-3">
            {!cameraActive ? (
              <button onClick={startCamera} className="flex items-center gap-2 rounded-xl bg-mudra-indigo-700 px-5 py-3 font-semibold text-white hover:bg-mudra-indigo-800">
                <Camera className="w-5 h-5" /> Start camera
              </button>
            ) : (
              <button onClick={stopCamera} className="flex items-center gap-2 rounded-xl border border-mudra-indigo-300 px-5 py-3 font-semibold text-mudra-indigo-800 hover:bg-mudra-lavender-50">
                <CameraOff className="w-5 h-5" /> Stop camera
              </button>
            )}
            <button
              onClick={record}
              disabled={!canRecord}
              className="flex items-center gap-2 rounded-xl bg-red-600 px-5 py-3 font-semibold text-white hover:bg-red-700 disabled:cursor-not-allowed disabled:opacity-40"
            >
              <Circle className="w-4 h-4 fill-white" />
              {phase === 'saving' ? 'Saving…' : phase === 'idle' ? `Record "${word || '…'}"` : 'Recording…'}
            </button>
            <span className="self-center text-sm text-mudra-indigo-600">or press Space</span>
          </div>

          {lastResult && (
            <div className={`rounded-2xl border p-4 text-sm flex gap-2 ${lastResult.ok ? 'border-emerald-300 bg-emerald-50 text-emerald-900' : 'border-amber-300 bg-amber-50 text-amber-900'}`}>
              {lastResult.ok ? <CheckCircle2 className="w-5 h-5 shrink-0" /> : <AlertCircle className="w-5 h-5 shrink-0" />}
              <div className="space-y-1">
                {lastResult.error && <p>Not saved: {lastResult.error}</p>}
                {lastResult.saved && lastResult.ok && (
                  <>
                    <p className="font-semibold">Saved "{lastResult.saved.label_readable}" ({lastResult.saved.label_count} of {TARGET_PER_WORD}).</p>
                    <p>
                      {lastResult.saved.in_model
                        ? lastResult.saved.custom_active
                          ? 'Your way of signing this word is now active: try it on the Communicate page.'
                          : 'MUDRA knows this word. After 3 good recordings, your own way of signing it is used too.'
                        : lastResult.saved.custom_active
                          ? 'New word active: try it on the Communicate page.'
                          : 'Saved for training.'}
                    </p>
                    <WordQuality q={lastResult.saved.quality} readable={readableSign} />
                  </>
                )}
                {lastResult.saved && !lastResult.ok && (
                  <p>{QUALITY_MESSAGES[lastResult.saved.quality_flag] || 'The recording could not be used.'} It was not counted.</p>
                )}
              </div>
            </div>
          )}

          <ol className="text-sm text-mudra-indigo-700 list-decimal pl-5 space-y-1">
            <li>Stand so your head, shoulders and both hands are in view.</li>
            <li>Press Record and keep your hands down during the countdown.</li>
            <li>Sign the word once, at normal speed, then lower your hands.</li>
          </ol>
          <p className="text-xs text-mudra-indigo-600">
            {TARGET_PER_WORD} takes make a word work for you straight away. For the trained model, aim for 10-15 takes from at
            least 3 different people, on different days.
          </p>
        </div>

        <div className="lg:col-span-5 space-y-6">
          <section className="rounded-3xl border border-mudra-lavender-200 bg-white p-5 shadow-sm space-y-4">
            <label className="block space-y-1">
              <span className="text-sm font-semibold text-mudra-indigo-800">Your name</span>
              <input
                value={signer}
                onChange={(e) => setSigner(e.target.value)}
                maxLength={64}
                placeholder="So recordings can be grouped by person"
                className="w-full rounded-xl border border-mudra-lavender-300 px-3 py-2 focus:outline-none focus:ring-2 focus:ring-mudra-indigo-400"
              />
            </label>

            <div className="space-y-2">
              <span className="text-sm font-semibold text-mudra-indigo-800">Word to record</span>
              <div className="flex flex-wrap gap-2">
                {SUGGESTED_WORDS.map((w) => (
                  <button
                    key={w}
                    onClick={() => setWord(w)}
                    className={`rounded-full px-3 py-1 text-sm font-medium border ${word === w ? 'bg-mudra-indigo-700 text-white border-mudra-indigo-700' : 'border-mudra-lavender-300 text-mudra-indigo-800 hover:bg-mudra-lavender-50'}`}
                  >
                    {w}
                  </button>
                ))}
              </div>
              <input
                value={word}
                onChange={(e) => setWord(e.target.value)}
                maxLength={64}
                placeholder="Or type any word"
                className="w-full rounded-xl border border-mudra-lavender-300 px-3 py-2 focus:outline-none focus:ring-2 focus:ring-mudra-indigo-400"
              />
              {current && (
                <p className="text-xs text-mudra-indigo-600">
                  {current.recordings} recorded so far{current.in_model ? ' · already in the model' : current.custom_active ? ' · active as a new word' : ''}
                </p>
              )}
            </div>

            <div className="space-y-2">
              <span className="text-sm font-semibold text-mudra-indigo-800">Recording length</span>
              <div className="flex gap-2">
                {DURATIONS.map((d) => (
                  <button
                    key={d}
                    onClick={() => setDuration(d)}
                    className={`rounded-lg px-3 py-1 text-sm font-medium border ${duration === d ? 'bg-mudra-indigo-700 text-white border-mudra-indigo-700' : 'border-mudra-lavender-300 text-mudra-indigo-800'}`}
                  >
                    {d} s
                  </button>
                ))}
              </div>
            </div>
          </section>

          <section className="rounded-3xl border border-mudra-lavender-200 bg-white p-5 shadow-sm space-y-3">
            <div className="flex items-center justify-between">
              <h2 className="font-display font-bold text-lg text-mudra-indigo-900 flex items-center gap-2">
                <Database className="w-5 h-5" /> Recorded so far
              </h2>
              <span className="text-sm text-mudra-indigo-600">{summary.total} recordings</span>
            </div>
            {summary.signs.length === 0 && <p className="text-sm text-mudra-indigo-600">Nothing recorded yet.</p>}
            <ul className="divide-y divide-mudra-lavender-100">
              {summary.signs.map((s) => (
                <li key={s.sign} className="py-3 space-y-2">
                  <div className="flex items-center justify-between gap-2">
                    <button onClick={() => setWord(s.label)} className="font-semibold text-mudra-indigo-900 hover:underline text-left">
                      {s.label}
                    </button>
                    <div className="flex items-center gap-2 text-xs">
                      {s.custom_active && (
                        <span className="rounded-full bg-emerald-100 px-2 py-0.5 font-semibold text-emerald-800 flex items-center gap-1">
                          <Sparkles className="w-3 h-3" /> {s.in_model ? 'your examples active' : 'new word active'}
                        </span>
                      )}
                      {s.in_model && <span className="rounded-full bg-mudra-lavender-100 px-2 py-0.5 font-semibold text-mudra-indigo-700">in model</span>}
                      <span className="text-mudra-indigo-700">{s.recordings}/{TARGET_PER_WORD}</span>
                    </div>
                  </div>
                  <div className="h-1.5 rounded-full bg-mudra-lavender-100 overflow-hidden">
                    <div className="h-full bg-mudra-indigo-600" style={{ width: `${Math.min(100, (s.recordings / TARGET_PER_WORD) * 100)}%` }} />
                  </div>
                  <p className="text-xs text-mudra-indigo-600 flex items-center gap-1">
                    <Users className="w-3 h-3" /> {s.signers.length ? s.signers.join(', ') : 'no usable recordings'}
                    {s.rejected > 0 && ` · ${s.rejected} not usable`}
                  </p>
                  <WordQuality q={s.quality} readable={readableSign} />
                  <div className="flex flex-wrap gap-1">
                    {s.items.map((it, i) => (
                      <span
                        key={it.sample_id}
                        className={`inline-flex items-center gap-1 rounded-md px-2 py-0.5 text-xs ${it.quality_flag || s.quality?.left_out?.includes(it.sample_id) ? 'bg-amber-50 text-amber-800' : 'bg-mudra-lavender-50 text-mudra-indigo-800'}`}
                        title={it.quality_flag ? QUALITY_MESSAGES[it.quality_flag] : s.quality?.left_out?.includes(it.sample_id) ? 'Left out: does not match the other takes' : `${it.signer}, ${it.duration_s.toFixed(1)} s`}
                      >
                        #{i + 1}
                        <button onClick={() => handleDelete(it.sample_id)} aria-label={`Delete recording ${i + 1} of ${s.label}`} className="hover:text-red-600">
                          <Trash2 className="w-3 h-3" />
                        </button>
                      </span>
                    ))}
                  </div>
                </li>
              ))}
            </ul>
          </section>
        </div>
      </div>
    </div>
  )
}
