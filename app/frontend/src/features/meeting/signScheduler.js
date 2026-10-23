/**
 * Planned sign phrases -> the avatar, in order, with a bounded backlog.
 *
 * Each phrase becomes one avatar sequence (useAvatarRenderer's executeSignSequence with append). Signing is
 * slower than speech, so the backlog is bounded two ways: a waiting phrase heard more than `maxLagMs` ago is
 * dropped (reported as skipped) because signing it that late no longer matches the conversation, and at most
 * `maxPending` phrases wait. The caller can also speed the avatar up a little while phrases wait (onEvent's
 * `backlog`). Stop clears everything.
 *
 *   const s = createSignScheduler({ play, cancel, clear, onEvent })
 *   s.enqueue(phrase); s.stop()
 * play(text, { append: true, onStart, onEnd }) -> Promise<id | null>; cancel(id) -> boolean; clear()
 */

export function createSignScheduler({ play, cancel, clear, maxPending = 4, maxLagMs = 6000, now = () => performance.now(), onEvent } = {}) {
  let pending = [] // [{ id, phrase }] queued and not started, oldest first
  let current = null
  let generation = 0
  const stats = { queued: 0, started: 0, finished: 0, skipped: 0, skippedWords: [] }

  const drop = (entry) => {
    if (!cancel(entry.id)) return
    stats.skipped++
    stats.skippedWords.push(...entry.phrase.words)
    onEvent?.({ type: 'skipped', phrase: entry.phrase, at: now(), backlog: pending.length })
  }
  // too late to sign (heard more than maxLagMs ago) or too many waiting: drop the oldest waiting phrases
  const trim = () => {
    const t = now()
    while (pending.length && (pending.length > maxPending || t - (pending[0].phrase.at ?? t) > maxLagMs)) drop(pending.shift())
  }

  return {
    stats,
    get pending() { return pending.map((p) => p.phrase) },
    get current() { return current },
    async enqueue(phrase) {
      if (!phrase?.text) return null
      const gen = generation
      const entry = { id: null, phrase, started: false }
      const id = await play(phrase.text, {
        append: true,
        onStart: (t) => {
          entry.started = true
          pending = pending.filter((p) => p !== entry)
          current = phrase
          stats.started++
          onEvent?.({ type: 'start', phrase, at: t, backlog: pending.length })
          trim()
        },
        onEnd: (t) => {
          if (current === phrase) current = null
          stats.finished++
          onEvent?.({ type: 'end', phrase, at: t })
        },
      })
      if (gen !== generation || id == null) return null // stopped meanwhile, or the avatar is not ready
      entry.id = id
      if (!entry.started) pending.push(entry)
      stats.queued++
      onEvent?.({ type: 'queued', phrase, at: now(), backlog: pending.length })
      trim()
      return id
    },
    stop() {
      generation++
      pending = []
      current = null
      clear?.()
      onEvent?.({ type: 'stopped', at: performance.now() })
    },
  }
}
