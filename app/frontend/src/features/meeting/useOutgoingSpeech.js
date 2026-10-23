import { useCallback, useEffect, useRef, useState } from 'react'
import handTracker from '../../services/handTracker'
import signRecognizer from '../../services/signRecognizer'
import { canonicalSentence, readableSign } from '../../services/islDictionary'

const SPEAK_PAUSE_MS = 1400 // no new sign for this long: say what was signed since the last time

/**
 * What is said for the signs: a curated sentence for that exact sequence of signs (HELLO, WATER -> "Hello, I need
 * water."), otherwise the signed words themselves. Single-sign expansions of the Communicate page (WATER -> "I need
 * water.") are not used here: other people hear what was signed, and a repeated word is said once.
 */
export function meetingSentence(tokens) {
  const curated = canonicalSentence(tokens, 'en')
  if (curated) return curated
  const words = tokens.map((t) => readableSign(t).toLowerCase().replace(/^i$/, 'I')).filter((w, i, all) => i === 0 || w !== all[i - 1])
  const text = words.join(' ')
  return text ? text.charAt(0).toUpperCase() + text.slice(1) + '.' : ''
}

/**
 * Outgoing: the user's signs (webcam -> MediaPipe -> MUDRA word model, as on the Communicate page) -> speech.
 * Only confident results are spoken to other people (an unsure guess is shown, not said). Words collected since
 * the last utterance are spoken once the user pauses; the voice goes where voiceOutput's sink points.
 */
export function useOutgoingSpeech({ voice }) {
  const [active, setActive] = useState(false)
  const [state, setState] = useState({ status: 'off' }) // off | starting | ready | signing | error
  const [said, setSaid] = useState([]) // [{ text, at }]
  const [pendingWords, setPendingWords] = useState([])
  const [lastSign, setLastSign] = useState(null)
  const pendingRef = useRef([])
  const timerRef = useRef(null)
  const voiceRef = useRef(voice)
  voiceRef.current = voice

  const flushSpeech = useCallback(() => {
    const tokens = pendingRef.current
    if (!tokens.length) return
    pendingRef.current = []
    setPendingWords([])
    const text = meetingSentence(tokens)
    setSaid((s) => [...s, { text, at: Date.now() }].slice(-8))
    voiceRef.current.speak(text)
  }, [])

  const schedule = useCallback(() => {
    clearTimeout(timerRef.current)
    timerRef.current = setTimeout(flushSpeech, SPEAK_PAUSE_MS)
  }, [flushSpeech])

  const start = useCallback(async (videoEl) => {
    setState({ status: 'starting' })
    const status = await signRecognizer.connect()
    if (status !== 'online') {
      setState({ status: 'error', error: signRecognizer.error || 'The recognition server is not reachable.' })
      return
    }
    signRecognizer.onPrediction = (p) => {
      if (p.status === 'ok') {
        pendingRef.current = [...pendingRef.current, p.sign]
        setPendingWords(pendingRef.current.map(readableSign))
        setLastSign({ sign: p.sign, label: p.label || readableSign(p.sign), confidence: p.confidence, spoken: true })
        schedule()
      } else {
        const top = p.top_k?.[0]
        setLastSign(top ? { sign: top.sign, label: top.label, confidence: top.confidence, spoken: false } : null)
      }
      setState({ status: 'ready' })
    }
    signRecognizer.onActivity = (a) => {
      if (a === 'signing') { clearTimeout(timerRef.current); setState({ status: 'signing' }) } // wait for the sentence
      else if (a === 'idle') { setState({ status: 'ready' }); if (pendingRef.current.length) schedule() }
    }
    try {
      await handTracker.startCamera(videoEl, (results) => signRecognizer.addFrame(results), { resolution: 480 })
      setActive(true)
      setState({ status: 'ready' })
    } catch (err) {
      setState({ status: 'error', error: err?.name === 'NotAllowedError' ? 'Camera permission was refused.' : `Camera: ${err?.message || err}` })
    }
  }, [schedule])

  const stop = useCallback(() => {
    clearTimeout(timerRef.current)
    pendingRef.current = []
    setPendingWords([])
    handTracker.stopCamera()
    signRecognizer.onPrediction = null
    signRecognizer.onActivity = null
    signRecognizer.reset()
    setActive(false)
    setState({ status: 'off' })
  }, [])

  useEffect(() => () => { clearTimeout(timerRef.current); if (handTracker.videoElement) handTracker.stopCamera(); signRecognizer.reset() }, [])

  return { active, state, said, pendingWords, lastSign, start, stop, speakNow: flushSpeech }
}
