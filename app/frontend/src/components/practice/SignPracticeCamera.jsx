import { useCallback, useEffect, useRef, useState } from 'react'
import { Camera, CameraOff, Check, Hand, ScanLine, X } from 'lucide-react'
import handTracker, { cameraErrorKind } from '../../services/handTracker'
import signRecognizer, { API_BASE_URL } from '../../services/signRecognizer'
import { isLetter, letterSpeller, phraseRecorder } from '../../services/libraryClient'
import { atRest, bestMatch, describe, frameFeatures, referenceFeatures, signingFrames } from '../../features/practice/handFeatures'
import StatusPill from '../ui/StatusPill'

const CAMERA_ERRORS = {
  denied: 'Camera permission was denied. Allow the camera for this site, then try again.',
  not_found: 'No camera was found.',
  in_use: 'The camera is in use by another app or tab. Close it, then try again.',
  unknown: 'The camera could not be started.',
}

function Meter({ label, score, hint }) {
  const d = describe(score)
  const colour = d.level === 'good' ? 'bg-sl-green-600' : d.level === 'mid' ? 'bg-sl-sun' : d.level === 'low' ? 'bg-rose-400' : 'bg-sl-cream-300'
  const mark = d.level === 'good' ? '✓' : d.level === 'mid' ? '~' : d.level === 'low' ? '✗' : '–'
  return (
    <div>
      <div className="flex items-baseline justify-between gap-2 text-sm">
        <span className="font-bold">{label}</span>
        <span className="font-semibold text-sl-ink-600"><span aria-hidden="true">{mark} </span>{score == null ? '--' : `${Math.round(score * 100)}%`}</span>
      </div>
      <div className="mt-1 h-2.5 overflow-hidden rounded-full bg-sl-cream-200" role="meter" aria-label={label}
           aria-valuemin={0} aria-valuemax={100} aria-valuenow={score == null ? undefined : Math.round(score * 100)}>
        <div className={`h-full rounded-full ${colour} transition-all duration-200`} style={{ width: `${Math.round((score ?? 0) * 100)}%` }} />
      </div>
      <p className="mt-0.5 text-xs text-sl-ink-600">{d.text}{hint ? ` · ${hint}` : ''}</p>
    </div>
  )
}

/**
 * Camera practice for one word, letter or phrase: live comparison with its recorded example (GET /reference)
 * and an optional check by the matching model (words: WebSocket /stream; letters: /letters/predict;
 * phrases: /phrases/predict on a recorded clip). Guidance only; nothing is stored.
 */
export default function SignPracticeCamera({ sign, label, kind }) {
  const checkKind = kind || (isLetter(sign) ? 'letter' : 'word') // word | letter | phrase
  const videoRef = useRef(null)
  const canvasRef = useRef(null)
  const refFeatures = useRef([])
  const smooth = useRef({ shape: null, position: null, palm: null })
  const checking = useRef(false)
  const checkKindRef = useRef(checkKind)
  checkKindRef.current = checkKind
  const lastPanel = useRef(0)
  const [on, setOn] = useState(false)
  const [starting, setStarting] = useState(false)
  const [error, setError] = useState('')
  const [refState, setRefState] = useState('loading')
  const [panel, setPanel] = useState({ hands: 0, shape: null, position: null, palm: null, handsMatch: true, body: false, resting: false })
  const [check, setCheck] = useState({ active: false, state: 'idle', result: null })

  // the example of the current word
  useEffect(() => {
    let alive = true
    setRefState('loading')
    smooth.current = { shape: null, position: null, palm: null }
    fetch(`${API_BASE_URL}/reference/${encodeURIComponent(sign)}`)
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error('no example'))))
      .then((clip) => {
        if (!alive) return
        refFeatures.current = signingFrames(referenceFeatures(clip)) // compare with the signing part only
        setRefState(refFeatures.current.length ? 'ready' : 'none')
      })
      .catch(() => alive && setRefState('none'))
    setCheck((c) => ({ ...c, result: null, state: c.active ? 'waiting' : 'idle' }))
    return () => { alive = false }
  }, [sign])

  const onResults = useCallback((results) => {
    const canvas = canvasRef.current
    const ctx = canvas?.getContext('2d')
    if (ctx) {
      handTracker.fitCanvas(canvas, results)
      ctx.clearRect(0, 0, canvas.width, canvas.height)
      handTracker.drawPose(ctx, results.poseLandmarks, canvas.width, canvas.height, true)
      for (const lm of results.multiHandLandmarks || []) handTracker.drawSkeleton(ctx, lm, canvas.width, canvas.height, true)
    }
    if (checking.current) {
      if (checkKindRef.current === 'letter') letterSpeller.addFrame(results)
      else if (checkKindRef.current === 'phrase') phraseRecorder.addFrame(results)
      else signRecognizer.addFrame(results)
    }

    const aspect = results.width && results.height ? results.width / results.height : 4 / 3
    const live = frameFeatures(results.multiHandLandmarks || [], results.poseLandmarks, aspect)
    const n = live.hands.filter(Boolean).length
    const resting = Boolean(n) && live.hasBody && atRest(live)
    const m = n && !resting && refFeatures.current.length ? bestMatch(live, refFeatures.current) : null
    const s = smooth.current
    const ema = (prev, v) => (v == null ? null : prev == null ? v : prev * 0.7 + v * 0.3)
    if (m) {
      s.shape = ema(s.shape, m.shape)
      s.position = ema(s.position, m.position)
      s.palm = ema(s.palm, m.palm)
    } else if (!n || resting) {
      s.shape = s.position = s.palm = null
    }
    const now = performance.now()
    if (now - lastPanel.current > 150) {
      lastPanel.current = now
      setPanel({ hands: n, ...s, handsMatch: m ? m.handsMatch : true, body: live.hasBody, resting })
    }
  }, [])

  const stopCheck = () => {
    checking.current = false
    signRecognizer.onPrediction = null
    signRecognizer.onActivity = null
    signRecognizer.reset()
    letterSpeller.stop()
    letterSpeller.onGuess = null
    phraseRecorder.cancel()
  }

  const stop = useCallback(() => {
    stopCheck()
    handTracker.stopCamera()
    setOn(false)
    setCheck({ active: false, state: 'idle', result: null })
  }, [])

  useEffect(() => () => { // leaving the page: release the camera
    if (handTracker.isTracking) handTracker.stopCamera()
    checking.current = false
    letterSpeller.stop()
    phraseRecorder.cancel()
    signRecognizer.onPrediction = null
    signRecognizer.onActivity = null
    signRecognizer.reset()
  }, [])

  const start = async () => {
    setStarting(true)
    setError('')
    try {
      await handTracker.startCamera(videoRef.current, onResults, { resolution: 480 })
      setOn(true)
    } catch (e) {
      setError(CAMERA_ERRORS[cameraErrorKind(e)])
    } finally {
      setStarting(false)
    }
  }

  const startCheck = async () => {
    if (checkKind === 'letter') {
      letterSpeller.onGuess = (r) => r.status !== 'no_hands' && setCheck((c) => ({ ...c, state: 'waiting', result: r }))
      letterSpeller.start()
      checking.current = true
      setCheck({ active: true, state: 'waiting', result: null })
      return
    }
    if (checkKind === 'phrase') {
      phraseRecorder.start()
      checking.current = true
      setCheck({ active: true, state: 'signing', result: null })
      return
    }
    if (signRecognizer.status !== 'online') await signRecognizer.connect()
    if (signRecognizer.status !== 'online') {
      setCheck({ active: false, state: 'offline', result: null })
      return
    }
    signRecognizer.onActivity = (state) => setCheck((c) => ({ ...c, state: state === 'signing' ? 'signing' : c.state }))
    signRecognizer.onPrediction = (p) => setCheck((c) => ({ ...c, state: 'waiting', result: p }))
    checking.current = true
    setCheck({ active: true, state: 'waiting', result: null })
  }

  const finishPhrase = async () => {
    checking.current = false
    setCheck((c) => ({ ...c, state: 'checking' }))
    try {
      const res = await phraseRecorder.stop()
      setCheck({ active: true, state: 'done', result: res.status === 'too_short' ? { status: 'no_sign', top_k: [] } : res })
    } catch {
      setCheck({ active: false, state: 'offline', result: null })
    }
  }

  const r = check.result
  const top = r?.top_k || []
  const matched = r && (r.sign === sign || top.slice(0, 3).some((t) => t.sign === sign))
  const recognised = r?.sign || top[0]?.sign

  return (
    <section aria-label={`Practise ${label} with your camera`} className="space-y-3 rounded-sl-lg border-2 border-sl-ink bg-white p-4">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h3 className="flex items-center gap-2 font-display text-lg font-bold"><Camera className="h-5 w-5" aria-hidden="true" /> Practise “{label}”</h3>
        {on ? (
          <button type="button" onClick={stop} className="inline-flex min-h-9 items-center gap-2 rounded-full border-2 border-sl-ink px-3 text-sm font-bold hover:bg-sl-cream-200">
            <CameraOff className="h-4 w-4" aria-hidden="true" /> Stop camera
          </button>
        ) : (
          <button type="button" onClick={start} disabled={starting} className="btn-primary min-h-9 px-4 py-1.5 text-sm">
            <Camera className="h-4 w-4" aria-hidden="true" /> {starting ? 'Starting…' : 'Start camera'}
          </button>
        )}
      </div>
      {error && <p role="alert" className="rounded-sl bg-rose-50 p-3 text-sm font-semibold text-rose-700">{error}</p>}

      <div className="grid gap-4 md:grid-cols-[1.2fr_1fr]">
        <div className="relative aspect-[4/3] overflow-hidden rounded-sl bg-sl-ink">
          <video ref={videoRef} playsInline muted className="absolute inset-0 h-full w-full -scale-x-100 object-cover" />
          <canvas ref={canvasRef} width={640} height={480} className="absolute inset-0 h-full w-full object-cover" />
          {!on && (
            <div className="absolute inset-0 flex flex-col items-center justify-center p-4 text-center text-white">
              <Hand className="h-8 w-8" aria-hidden="true" />
              <p className="mt-2 text-sm">Start the camera, then copy the example. Video stays on your device.</p>
            </div>
          )}
        </div>

        <div className="space-y-3">
          <div className="flex items-center justify-between gap-2">
            <h4 className="text-sm font-bold uppercase tracking-wider text-sl-ink-600">Live landmark evaluation</h4>
            <StatusPill status={refState === 'ready' ? 'live' : refState === 'loading' ? 'unavailable' : 'error'}>
              {refState === 'ready' ? 'Example loaded' : refState === 'loading' ? 'Loading example' : 'No example'}
            </StatusPill>
          </div>
          <div className="flex items-center justify-between rounded-sl border border-sl-cream-300 px-3 py-2 text-sm">
            <span className="font-bold">Hand detected in frame</span>
            <span className={`font-extrabold ${panel.hands ? 'text-sl-green-700' : 'text-rose-700'}`}>
              {!on ? '--' : panel.hands ? `✓ ${panel.hands === 2 ? 'Both hands' : 'One hand'}` : '✗ Not in frame'}
            </span>
          </div>
          {on && panel.resting && (
            <p role="status" className="rounded-sl bg-sl-sun-100 px-3 py-2 text-sm font-semibold">Hands at rest: raise them to start the sign.</p>
          )}
          <Meter label="Finger shape & spread" score={on ? panel.shape : null} hint={!panel.handsMatch && on && panel.hands ? 'the example uses both hands' : ''} />
          <Meter label="Hand height & position" score={on ? panel.position : null} hint={on && panel.hands && !panel.body ? 'shoulders not in view' : ''} />
          <Meter label="Palm facing direction" score={on ? panel.palm : null} />
          <p className="text-xs text-sl-ink-600">Compared with the recorded example’s closest moment: a guide, not a grade.</p>
        </div>
      </div>

      {on && (
        <div className="space-y-2 rounded-sl border-2 border-dashed border-sl-ink-200 p-3">
          {!check.active ? (
            <div className="flex flex-wrap items-center justify-between gap-2">
              <p className="text-sm">Ready? Let SignLine’s model try to recognise your sign.</p>
              <button type="button" onClick={startCheck} className="btn-secondary min-h-9 px-4 py-1.5 text-sm">
                <ScanLine className="h-4 w-4" aria-hidden="true" /> Check my sign
              </button>
            </div>
          ) : (
            <div className="space-y-2" aria-live="polite">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <p className="text-sm font-semibold">
                  {checkKind === 'letter' ? 'Hold the letter shape still in view.'
                    : checkKind === 'phrase' ? (check.state === 'signing' ? 'Sign the whole phrase, then press Done.' : check.state === 'checking' ? 'Checking…' : 'Checked.')
                    : check.state === 'signing' ? 'Signing… hold still at the end.' : 'Sign the word now, then hold your hands still or lower them.'}
                </p>
                {checkKind === 'phrase' && check.state === 'signing' && (
                  <button type="button" onClick={finishPhrase} className="btn-primary min-h-9 px-4 py-1.5 text-sm">Done</button>
                )}
                {checkKind === 'phrase' && check.state === 'done' && (
                  <button type="button" onClick={startCheck} className="btn-secondary min-h-9 px-4 py-1.5 text-sm">Try again</button>
                )}
                <button type="button" onClick={() => { stopCheck(); setCheck({ active: false, state: 'idle', result: null }) }}
                        className="inline-flex items-center gap-1 text-sm font-bold underline underline-offset-4">
                  <X className="h-4 w-4" aria-hidden="true" /> Stop checking
                </button>
              </div>
              {r && (
                <div className={`flex items-start gap-3 rounded-sl border-2 p-3 ${matched ? 'border-sl-green-700 bg-sl-green-50' : 'border-sl-ink-200 bg-sl-cream-50'}`}>
                  {matched ? <Check className="mt-0.5 h-5 w-5 shrink-0 text-sl-green-700" aria-hidden="true" /> : <ScanLine className="mt-0.5 h-5 w-5 shrink-0" aria-hidden="true" />}
                  <p className="text-sm">
                    {matched
                      ? <><strong>SignLine recognised “{label}”.</strong> {r.status === 'ok' ? 'It would be added to a sentence.' : 'It was one of its top guesses.'}</>
                      : r.status === 'no_sign' || !recognised
                        ? <><strong>No sign recognised.</strong> Try the whole movement from start to finish.</>
                        : <><strong>Recognised as “{top[0]?.label || recognised}”</strong> ({Math.round((top[0]?.confidence || 0) * 100)}%), not “{label}”. Watch the example again.</>}
                    <span className="block text-xs text-sl-ink-600">The model has mostly learned from other signers, so it can be wrong about your signing.</span>
                  </p>
                </div>
              )}
            </div>
          )}
          {check.state === 'offline' && <p className="text-sm text-rose-700">The recognition server is not reachable, so signs cannot be checked.</p>}
        </div>
      )}
    </section>
  )
}
