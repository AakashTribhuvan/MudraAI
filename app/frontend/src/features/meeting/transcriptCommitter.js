/**
 * Live transcript -> stable words, without waiting for the end of the sentence.
 *
 * A streaming recogniser sends growing guesses for the phrase being spoken ("thank", "thank you i", "thank you i
 * need") and a final text at each pause. A word is committed once it has stayed the same for `stableMs` and is not
 * the last word of the guess (the last word may still grow: "plea" -> "please"); the final text commits the rest.
 * Each word is committed exactly once: repeated guesses, a repeated final, or a guess that rewrites a word already
 * committed never produce duplicates (rewrites are counted in stats.revisions).
 *
 *   const c = createTranscriptCommitter({ onCommit: (words, info) => ..., onCaption: (caption) => ... })
 *   c.partial(segment, text, now); c.final(segment, text, now); c.tick(now); c.reset()
 */

const normalise = (text) => (text || '').toLowerCase().replace(/[^a-z0-9' ]+/g, ' ').split(/\s+/).filter(Boolean)

export function createTranscriptCommitter({ stableMs = 450, maxCaptionWords = 24, onCommit, onCaption } = {}) {
  let seg = null // { id, words, since[], committed, final }
  let history = [] // committed words of earlier phrases (for the caption)
  const finished = new Set() // segment ids that are complete
  const stats = { partials: 0, finals: 0, committed: 0, revisions: 0, duplicates: 0 }

  const caption = () => {
    const done = seg ? seg.words.slice(0, seg.committed) : []
    const provisional = seg && !seg.final ? seg.words.slice(seg.committed) : []
    const committed = [...history, ...done].slice(-maxCaptionWords)
    onCaption?.({ committed: committed.join(' '), provisional: provisional.join(' '), segment: seg?.id ?? null })
  }

  const commit = (count, now, reason) => {
    if (!seg || count <= seg.committed) return
    const words = seg.words.slice(seg.committed, count)
    seg.committed = count
    stats.committed += words.length
    onCommit?.(words, { segment: seg.id, at: now, reason, final: seg.final })
  }

  const close = (now) => { // the open phrase ends (a new one started without its final): keep what was heard
    if (!seg) return
    if (!seg.final) commit(seg.words.length, now, 'next-phrase')
    finished.add(seg.id)
    history = [...history, ...seg.words.slice(0, seg.committed)].slice(-maxCaptionWords)
    seg = null
  }

  const open = (id, now) => {
    if (seg && seg.id !== id) close(now)
    if (!seg) seg = { id, words: [], since: [], committed: 0, final: false }
  }

  /** Words held unchanged for stableMs, except the last one, are committed. */
  const settle = (now) => {
    if (!seg || seg.final) return
    let k = seg.committed
    while (k < seg.words.length - 1 && now - seg.since[k] >= stableMs) k++
    commit(k, now, 'stable')
  }

  return {
    stats,
    partial(id, text, now = performance.now()) {
      if (finished.has(id)) { stats.duplicates++; return }
      stats.partials++
      open(id, now)
      const words = normalise(text)
      // positions whose word changed restart their stability clock; committed words cannot change any more
      const since = words.map((w, i) => (seg.words[i] === w ? seg.since[i] : now))
      for (let i = 0; i < seg.committed; i++) {
        if (words[i] !== seg.words[i]) { stats.revisions++; words[i] = seg.words[i]; since[i] = seg.since[i] }
      }
      if (words.length < seg.committed) { // the guess got shorter than what is committed: keep the committed words
        for (let i = words.length; i < seg.committed; i++) { words[i] = seg.words[i]; since[i] = seg.since[i] }
      }
      seg.words = words
      seg.since = since
      settle(now)
      caption()
    },
    final(id, text, now = performance.now()) {
      if (finished.has(id)) { stats.duplicates++; return }
      stats.finals++
      open(id, now)
      const words = normalise(text)
      const kept = seg.words.slice(0, seg.committed)
      for (let i = 0; i < kept.length; i++) if (words[i] !== kept[i]) stats.revisions++
      seg.words = [...kept, ...words.slice(kept.length)]
      seg.final = true
      commit(seg.words.length, now, 'final')
      close(now)
      caption()
    },
    /** Call regularly (e.g. every 100 ms): commits words that became stable without a new guess. */
    tick(now = performance.now()) {
      const before = seg?.committed
      settle(now)
      if (seg && seg.committed !== before) caption()
    },
    reset() {
      seg = null
      history = []
      finished.clear()
      caption()
    },
  }
}
