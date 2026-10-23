import { useCallback, useEffect, useRef, useState } from 'react'
import { fetchVocabulary } from '../vocabulary'
import { buildSpokenLexicon, createSignPlanner } from './signPlanner'
import { createSignScheduler } from './signScheduler'
import { createTranscriptCommitter } from './transcriptCommitter'

/**
 * Incoming meeting speech -> live caption + ISL avatar phrases.
 * transcript events -> createTranscriptCommitter (stable words) -> createSignPlanner (vocabulary signs)
 * -> createSignScheduler (ordered, bounded avatar queue). Also measures, per word: first partial, commit, and the
 * first avatar frame of the phrase that signs it (window.__mudraMeetingMetrics in development).
 */
export function useIncomingSigns({ avatar, isEcho, spellUnknown = false }) {
  const [caption, setCaption] = useState({ committed: '', provisional: '' })
  const [unsigned, setUnsigned] = useState([])
  const [skipped, setSkipped] = useState([])
  const [signing, setSigning] = useState(null)
  const [backlog, setBacklog] = useState(0)
  const [lastLatency, setLastLatency] = useState(null)
  const [vocabSize, setVocabSize] = useState(0)
  const pipeRef = useRef(null)
  const metricsRef = useRef({ words: [], phrases: [], echoes: 0 })
  const avatarRef = useRef(avatar)
  avatarRef.current = avatar
  const echoRef = useRef(isEcho)
  echoRef.current = isEcho

  const build = useCallback(async () => {
    const vocab = await fetchVocabulary()
    const signs = Object.values(vocab.labels || {}).filter((l) => l.has_reference).map((l) => l.sign)
    const lexicon = buildSpokenLexicon(signs)
    setVocabSize(lexicon.size)
    const m = metricsRef.current
    const firstSeen = new Map() // word position key -> first partial time
    const scheduler = createSignScheduler({
      play: (text, opts) => avatarRef.current.executeSignSequence(text, opts),
      cancel: (id) => avatarRef.current.cancelPhrase(id),
      clear: () => avatarRef.current.clearQueue(),
      onEvent: (ev) => {
        if (ev.type === 'start') {
          setSigning(ev.phrase.text)
          const rec = m.phrases.find((p) => p.phrase === ev.phrase)
          if (rec) {
            rec.startAt = ev.at
            const lat = { partialToAvatar: Math.round(ev.at - rec.firstPartialAt), commitToAvatar: Math.round(ev.at - rec.commitAt), words: ev.phrase.words.join(' ') }
            setLastLatency(lat)
          }
        } else if (ev.type === 'end') setSigning(null)
        else if (ev.type === 'skipped') setSkipped((s) => [...s, ...ev.phrase.words].slice(-12))
        setBacklog(scheduler.pending.length)
      },
    })
    const planner = createSignPlanner({
      lexicon, spellUnknown,
      onPhrase: (phrase) => {
        if (phrase.unsigned.length) setUnsigned((u) => [...u, ...phrase.unsigned].slice(-12))
        if (!phrase.text) return
        const firsts = phrase.words.map((w) => m.words.find((x) => x.word === w && !x.phrase)).filter(Boolean)
        firsts.forEach((x) => { x.phrase = phrase.text })
        m.phrases.push({ phrase, text: phrase.text, commitAt: phrase.at, firstPartialAt: Math.min(...firsts.map((x) => x.firstPartialAt).filter(Number.isFinite), phrase.at) })
        scheduler.enqueue(phrase).then(() => setBacklog(scheduler.pending.length))
      },
    })
    const committer = createTranscriptCommitter({
      onCaption: setCaption,
      onCommit: (words, info) => {
        if (echoRef.current?.(words)) { m.echoes++; return } // MUDRA's own voice coming back from the meeting
        words.forEach((word, k) => {
          const key = `${info.segment}:${word}`
          m.words.push({ word, segment: info.segment, commitAt: info.at, firstPartialAt: firstSeen.get(key) ?? info.at, reason: info.reason, k })
        })
        planner.push(words, info.at, info.final)
      },
    })
    const timer = setInterval(() => { const now = performance.now(); committer.tick(now); planner.tick(now) }, 100)
    pipeRef.current = {
      committer, planner, scheduler, timer,
      onTranscript(ev) {
        const now = ev.t_client ?? performance.now()
        for (const w of (ev.text || '').toLowerCase().split(/\s+/).filter(Boolean)) {
          const key = `${ev.segment}:${w.replace(/[^a-z0-9']/g, '')}`
          if (!firstSeen.has(key)) firstSeen.set(key, now)
        }
        if (ev.event === 'partial') committer.partial(ev.segment, ev.text, now)
        else if (ev.event === 'final') { committer.final(ev.segment, ev.text, now); planner.tick(now) }
      },
    }
    if (import.meta.env?.DEV) window.__mudraMeetingMetrics = m
    return pipeRef.current
  }, [spellUnknown])

  const start = useCallback(async () => {
    if (!pipeRef.current) await build()
    return pipeRef.current.onTranscript
  }, [build])

  const stop = useCallback(() => {
    const p = pipeRef.current
    if (!p) return
    clearInterval(p.timer)
    p.planner.reset() // Stop cancels pending work: nothing more is signed
    p.scheduler.stop()
    p.committer.reset()
    pipeRef.current = null
    setSigning(null)
    setBacklog(0)
  }, [])

  useEffect(() => () => { if (pipeRef.current) clearInterval(pipeRef.current.timer) }, [])

  return { start, stop, caption, unsigned, skipped, signing, backlog, lastLatency, vocabSize, metrics: metricsRef.current }
}
