import { describe, expect, it, vi } from 'vitest'
import { createTranscriptCommitter } from '../transcriptCommitter'
import { buildSpokenLexicon, createSignPlanner } from '../signPlanner'
import { createSignScheduler } from '../signScheduler'
import { createVoiceOutput, findCable } from '../voiceOutput'
import { meetingSentence } from '../useOutgoingSpeech'

const SIGNS = ['TOMORROW', 'TODAY', 'TIME', 'THANK_YOU', 'WATER', 'HELP', 'I', 'I_AM_FINE', 'HOW_ARE_YOU', 'GOOD', 'GOOD_MORNING', 'HELLO', 'NEED', 'FRIEND', 'BIG_LARGE', 'HOUSE', 'ME', 'YOU']

describe('transcript committer', () => {
  it('commits stable words while the phrase is still being spoken, each word once', () => {
    const commits = []
    const c = createTranscriptCommitter({ stableMs: 400, onCommit: (w) => commits.push(...w) })
    c.partial(0, 'thank', 0)
    c.partial(0, 'thank you i', 300)
    c.partial(0, 'thank you i', 350) // the same guess again
    c.tick(500) // "thank" stable for 500 ms -> committed; "you" only 200 ms
    expect(commits).toEqual(['thank'])
    c.partial(0, 'thank you i need water plea', 900)
    expect(commits).toEqual(['thank', 'you', 'i']) // "plea" is the growing last word: not yet
    c.final(0, 'thank you i need water please', 1500)
    c.final(0, 'thank you i need water please', 1600) // a repeated final
    expect(commits).toEqual(['thank', 'you', 'i', 'need', 'water', 'please'])
    expect(c.stats.duplicates).toBe(1)
  })

  it('never re-commits a word the recogniser rewrites later', () => {
    const commits = []
    const c = createTranscriptCommitter({ stableMs: 100, onCommit: (w) => commits.push(...w) })
    c.partial(1, 'i need wonder', 0)
    c.tick(200)
    c.partial(1, 'i kneed water now', 300) // rewrites "need" (already committed)
    c.final(1, 'i kneed water now', 400)
    expect(commits).toEqual(['i', 'need', 'water', 'now'])
    expect(c.stats.revisions).toBeGreaterThan(0)
  })

  it('a new phrase without the old final keeps what was heard; captions show provisional words', () => {
    const commits = []
    let caption = null
    const c = createTranscriptCommitter({ stableMs: 1000, onCommit: (w) => commits.push(...w), onCaption: (x) => { caption = x } })
    c.partial(0, 'hello friend', 0)
    expect(caption).toEqual({ committed: '', provisional: 'hello friend', segment: 0 })
    c.partial(1, 'water', 100)
    expect(commits).toEqual(['hello', 'friend'])
    expect(caption.committed).toBe('hello friend')
  })
})

describe('sign planner', () => {
  const lex = buildSpokenLexicon(SIGNS)

  it('maps words to the longest vocabulary sign and skips ISL-omitted words', () => {
    const phrases = []
    const p = createSignPlanner({ lexicon: lex, onPhrase: (x) => phrases.push(x) })
    p.push(['hello', 'how', 'are', 'you', 'i', 'need', 'the', 'water'], 0, true)
    expect(phrases.map((x) => x.signs)).toEqual([['HELLO', 'HOW_ARE_YOU', 'I'], ['NEED', 'WATER']])
    expect(phrases[0].text).toBe('HELLO HOW ARE YOU I')
    expect(p.stats.skipped).toBe(1) // "the"
  })

  it('waits for the next word when it may complete a longer sign, then signs alone after holdMs', () => {
    const phrases = []
    const p = createSignPlanner({ lexicon: lex, holdMs: 700, onPhrase: (x) => phrases.push(x) })
    p.push(['thank'], 0)
    expect(phrases).toEqual([])
    p.push(['you'], 200)
    expect(phrases[0].signs).toEqual(['THANK_YOU'])
    p.push(['good'], 300)
    p.tick(600)
    expect(phrases.length).toBe(1) // still waiting: "good morning"?
    p.tick(1100)
    expect(phrases[1].signs).toEqual(['GOOD'])
    p.push(['i', 'am'], 1200)
    p.push(['fine'], 1300)
    expect(phrases[2].signs).toEqual(['I_AM_FINE'])
  })

  it('rejoins words the recogniser splits in two ("to morrow"), even when they are committed apart', () => {
    const phrases = []
    const p = createSignPlanner({ lexicon: lex, holdMs: 700, onPhrase: (x) => phrases.push(x) })
    p.push(['see', 'you', 'to', 'morrow'], 0, true)
    expect(phrases[0].signs).toEqual(['YOU', 'TOMORROW'])
    p.push(['to'], 100)
    expect(phrases.length).toBe(1) // "to" waits: it may start "to day"
    p.push(['day'], 300)
    expect(phrases[1].signs).toEqual(['TODAY'])
    p.push(['time', 'table'], 400, true) // TIME is a sign itself: not glued into "timetable"
    expect(phrases[2].signs).toEqual(['TIME'])
  })

  it('never invents a sign: unknown words are reported, or spelled when asked; plurals and aliases map', () => {
    const phrases = []
    const p = createSignPlanner({ lexicon: lex, onPhrase: (x) => phrases.push(x) })
    p.push(['friends', 'home', 'large', 'quantum'], 0, true)
    expect(phrases[0].signs).toEqual(['FRIEND', 'HOUSE', 'BIG_LARGE'])
    expect(phrases[0].unsigned).toEqual(['quantum'])
    const spelled = []
    const q = createSignPlanner({ lexicon: lex, spellUnknown: true, maxSpell: 5, onPhrase: (x) => spelled.push(x) })
    q.push(['ravi', 'quantum'], 0, true)
    expect(spelled[0].text).toBe('RAVI')
    expect(spelled[0].unsigned).toEqual(['quantum'])
  })
})

describe('sign scheduler', () => {
  const fakeAvatar = () => {
    const queue = []
    let n = 0
    return {
      queue,
      play: vi.fn(async (text, opts) => { const id = `p${++n}`; queue.push({ id, text, opts }); return id }),
      cancel: vi.fn((id) => { const i = queue.findIndex((q) => q.id === id); if (i < 1) return false; queue.splice(i, 1); return true }),
      clear: vi.fn(() => queue.splice(0)),
      startNext: () => queue[0]?.opts.onStart(1),
      finish: () => { const q = queue.shift(); q?.opts.onEnd(2) },
    }
  }

  it('plays phrases in order and keeps the backlog bounded (drops the oldest waiting phrase)', async () => {
    const a = fakeAvatar()
    const events = []
    const s = createSignScheduler({ play: a.play, cancel: a.cancel, clear: a.clear, maxPending: 2, onEvent: (e) => events.push(e.type) })
    await s.enqueue({ text: 'HELLO', words: ['hello'] })
    a.startNext() // HELLO is being signed
    for (const w of ['WATER', 'HELP', 'FRIEND']) await s.enqueue({ text: w, words: [w.toLowerCase()] })
    expect(a.queue.map((q) => q.text)).toEqual(['HELLO', 'HELP', 'FRIEND']) // WATER skipped to stay live
    expect(s.stats.skipped).toBe(1)
    expect(s.stats.skippedWords).toEqual(['water'])
    expect(s.pending.map((p) => p.text)).toEqual(['HELP', 'FRIEND'])
    s.stop()
    expect(a.clear).toHaveBeenCalled()
    expect(events).toContain('stopped')
  })

  it('drops a waiting phrase once it would be signed too late after it was heard', async () => {
    const a = fakeAvatar()
    let t = 0
    const s = createSignScheduler({ play: a.play, cancel: a.cancel, clear: a.clear, maxLagMs: 6000, now: () => t })
    await s.enqueue({ text: 'HELLO', words: ['hello'], at: 0 })
    a.startNext()
    await s.enqueue({ text: 'WATER', words: ['water'], at: 500 })
    await s.enqueue({ text: 'HELP', words: ['help'], at: 4000 })
    expect(s.stats.skipped).toBe(0) // both still timely
    t = 7000 // HELLO took long; WATER was heard 6.5 s ago
    await s.enqueue({ text: 'FRIEND', words: ['friend'], at: 6900 })
    expect(a.queue.map((q) => q.text)).toEqual(['HELLO', 'HELP', 'FRIEND'])
    expect(s.stats.skippedWords).toEqual(['water'])
  })

  it('ignores a phrase whose queueing finished after Stop', async () => {
    let release
    const play = vi.fn(() => new Promise((r) => { release = r }))
    const s = createSignScheduler({ play, cancel: () => false, clear: () => {} })
    const pending = s.enqueue({ text: 'HELLO', words: ['hello'] })
    s.stop()
    release('p1')
    expect(await pending).toBe(null)
    expect(s.stats.queued).toBe(0)
  })
})

describe('voice output', () => {
  const okFetch = vi.fn(async () => ({ ok: true, blob: async () => new Blob(['RIFF']) }))
  class FakeAudio {
    constructor() { FakeAudio.last = this; this.sinkId = '' }
    async setSinkId(id) { this.sinkId = id }
    async play() { setTimeout(() => this.onended?.(), 0) }
    pause() {}
  }
  globalThis.URL.createObjectURL ??= () => 'blob:x'
  globalThis.URL.revokeObjectURL ??= () => {}

  it('routes speech to the chosen output and says whether the meeting can hear it', async () => {
    const statuses = []
    const v = createVoiceOutput({ apiBase: 'http://x', fetchImpl: okFetch, AudioImpl: FakeAudio, onStatus: (s) => statuses.push(s) })
    await v.speak('thank you')
    expect(statuses.at(-1).state).toBe('local-only') // default speakers: not sent to the meeting
    v.setSink({ deviceId: 'cable-1', label: 'CABLE Input (VB-Audio Virtual Cable)' })
    await v.speak('hello')
    expect(FakeAudio.last.sinkId).toBe('cable-1')
    expect(statuses.at(-1).state).toBe('sent')
    expect(okFetch).toHaveBeenLastCalledWith('http://x/tts', expect.objectContaining({ method: 'POST' }))
  })

  it('reports server errors instead of pretending to speak', async () => {
    const statuses = []
    const bad = async () => ({ ok: false, status: 503, json: async () => ({ detail: 'not on Windows' }) })
    const v = createVoiceOutput({ apiBase: 'http://x', fetchImpl: bad, AudioImpl: FakeAudio, onStatus: (s) => statuses.push(s) })
    await v.speak('hello')
    expect(statuses.at(-1)).toMatchObject({ state: 'error', error: 'Speech server: not on Windows' })
  })

  it('recognises its own speech coming back from the meeting (echo guard) and finds the cable', async () => {
    const v = createVoiceOutput({ apiBase: 'http://x', fetchImpl: okFetch, AudioImpl: FakeAudio })
    await v.speak('I need water')
    expect(v.isEcho(['need', 'water'])).toBe(true)
    expect(v.isEcho(['hello', 'friend'])).toBe(false)
    const cable = findCable({ outputs: [{ label: 'Speakers (Realtek)' }, { label: 'CABLE Input (VB-Audio Virtual Cable)' }], inputs: [{ label: 'CABLE Output (VB-Audio Virtual Cable)' }] })
    expect(cable.output.label).toMatch(/CABLE Input/)
    expect(cable.input.label).toMatch(/CABLE Output/)
    // VB-Cable 4.5 names (Windows 11): stereo playback end preferred over the 16-channel one, aliases skipped
    const v45 = findCable({
      outputs: [{ deviceId: 'default', label: 'Default - Speakers (VB-Audio Virtual Cable)' }, { deviceId: 'a', label: 'CABLE In 16 Ch (VB-Audio Virtual Cable)' }, { deviceId: 'b', label: 'Speakers (VB-Audio Virtual Cable)' }],
      inputs: [{ deviceId: 'communications', label: 'Communications - CABLE Output (VB-Audio Virtual Cable)' }, { deviceId: 'c', label: 'CABLE Output (VB-Audio Virtual Cable)' }],
    })
    expect([v45.output.deviceId, v45.input.deviceId]).toEqual(['b', 'c'])
    // as Chrome lists them on Windows 11 with VB-Cable 4.5 (measured 2026-10-21)
    const chromeNames = findCable({ outputs: [{ deviceId: 'x', label: 'CABLE In 16ch (VB-Audio Virtual Cable)' }, { deviceId: 'y', label: 'CABLE Input (VB-Audio Virtual Cable)' }], inputs: [] })
    expect(chromeNames.output.deviceId).toBe('y')
  })

  it('says what was signed: curated sentences for whole sequences, no single-sign expansion, no repeats', () => {
    expect(meetingSentence(['HELLO', 'WATER'])).toBe('Hello, I need water.')
    expect(meetingSentence(['WATER'])).toBe('Water.')
    expect(meetingSentence(['I', 'NEED', 'WATER'])).toBe('I need water.')
    expect(meetingSentence(['HOW_ARE_YOU', 'HOW_ARE_YOU'])).toBe('How are you.')
  })
})
