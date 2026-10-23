/**
 * Committed spoken words -> ISL sign phrases for the avatar, using only MUDRA's vocabulary (GET /labels: every
 * word with a recorded sign). Longest match first ("thank you" -> THANK_YOU, "how are you" -> HOW_ARE_YOU); a word
 * that may start a longer sign waits up to holdMs for the next word. Words with no sign are never given an invented
 * one: by default they are listed as "not signed" (shown in the caption); with spellUnknown, short ones (names, up
 * to maxSpell letters) are fingerspelled with the letter shapes, as on the Communicate page.
 *
 *   const lex = buildSpokenLexicon(signs)
 *   const p = createSignPlanner({ lexicon: lex, onPhrase: (phrase) => ... })
 *   p.push(words, now); p.tick(now); p.flush(now)
 */

// Words ISL leaves out (articles, copulas, auxiliaries, fillers): skipped without a "not signed" note
export const SKIP_WORDS = new Set(('a an the is am are was were be been being to of and or but so do does did ' +
  'will would shall should can could may might must has have had um uh er ah oh okay ok just very really ' +
  'that\'s it\'s i\'m you\'re we\'re they\'re there their there\'s here at in on for with by as if then than').split(' '))

// Sign labels that name two English words, and everyday spoken forms of signs in the vocabulary
const SPLIT_LABELS = { BIG_LARGE: ['big', 'large'], SMALL_LITTLE: ['small', 'little'], STORE_OR_SHOP: ['store', 'shop'],
  STREET_OR_ROAD: ['street', 'road'], RACE_ETHNICITY: ['race', 'ethnicity'], EX_MONSOON: ['monsoon'], YOU_PLURAL: [] }
const SPOKEN_ALIASES = {
  thanks: 'THANK_YOU', 'thank u': 'THANK_YOU', hi: 'HELLO', home: 'HOUSE', phone: 'TELEPHONE', mobile: 'CELL_PHONE',
  tv: 'TELEVISION', mom: 'MOTHER', mum: 'MOTHER', mummy: 'MOTHER', dad: 'FATHER', papa: 'FATHER', kid: 'CHILD',
  kids: 'CHILD', children: 'CHILD', men: 'MAN', women: 'WOMAN', people: 'CROWD', 'i am fine': 'I_AM_FINE',
  "i'm fine": 'I_AM_FINE', 'love you': 'I_LOVE_YOU', 'what time': 'WHAT_TIME', 'how much': 'HOW_MUCH',
  'what is your name': 'WHAT_IS_YOUR_NAME', "what's your name": 'WHAT_IS_YOUR_NAME', tshirt: 'T_SHIRT',
  't shirt': 'T_SHIRT', railway: 'TRAIN', station: 'TRAIN_STATION', ticket: 'TRAIN_TICKET', tired: 'FEDUP',
}

const singular = (w) => (w.endsWith('ies') && w.length > 4 ? w.slice(0, -3) + 'y'
  : w.endsWith('ses') || w.endsWith('xes') || w.endsWith('ches') || w.endsWith('shes') ? w.replace(/es$/, '')
    : w.endsWith('s') && !w.endsWith('ss') && w.length > 3 ? w.slice(0, -1) : w)

/** signs: sign labels with a recorded clip (e.g. ['THANK_YOU', 'WATER', ...]). */
export function buildSpokenLexicon(signs) {
  const phrases = new Map() // 'thank you' -> 'THANK_YOU'
  const known = new Set(signs)
  for (const sign of signs) {
    if (sign.startsWith('LETTER_') || sign.startsWith('_')) continue
    const forms = SPLIT_LABELS[sign] || [sign.toLowerCase().replace(/_/g, ' ')]
    for (const f of forms) if (!phrases.has(f)) phrases.set(f, sign)
  }
  for (const [spoken, sign] of Object.entries(SPOKEN_ALIASES)) if (known.has(sign)) phrases.set(spoken, sign)
  const prefixes = new Set()
  const splitStarts = new Set() // "to" of "to morrow": streaming ASR splits some words in two
  let maxWords = 1
  for (const p of phrases.keys()) {
    const w = p.split(' ')
    maxWords = Math.max(maxWords, w.length)
    for (let n = 1; n < w.length; n++) prefixes.add(w.slice(0, n).join(' '))
    if (w.length === 1) for (let k = 2; k <= p.length - 2; k++) splitStarts.add(p.slice(0, k))
  }
  const lookup = (words) => {
    const key = words.join(' ')
    if (phrases.has(key)) return phrases.get(key)
    if (words.length === 1 && phrases.has(singular(words[0]))) return phrases.get(singular(words[0]))
    return null
  }
  // "to morrow" -> TOMORROW, "to day" -> TODAY (measured with both sherpa-onnx models), only when the first part is
  // not a sign itself
  const joined = (a, b) => (a && b && splitStarts.has(a) && !lookup([a]) ? lookup([a + b]) : null)
  const mayJoin = (a) => splitStarts.has(a) && !lookup([a])
  return { phrases, prefixes, maxWords, lookup, joined, mayJoin, size: phrases.size }
}

/** The text the avatar renderer matches back to the same signs: the label's words ("THANK YOU", "BIG LARGE"). */
export const signText = (sign) => sign.replace(/_/g, ' ')

export function createSignPlanner({ lexicon, holdMs = 700, joinHoldMs = 1500, spellUnknown = false, maxSpell = 6, maxSigns = 3, onPhrase } = {}) {
  let buffer = [] // [{ word, at }]
  const stats = { signed: 0, unsigned: 0, skipped: 0, spelled: 0, phrases: 0 }

  // Resolve the buffer from the start; stops at a word that may still become part of a longer sign
  const resolve = (now, final) => {
    const items = []
    const unsigned = []
    let i = 0
    while (i < buffer.length) {
      const rest = buffer.slice(i).map((b) => b.word)
      const glued = lexicon.joined?.(rest[0], rest[1])
      if (glued) { items.push({ sign: glued, words: [rest[0] + rest[1]] }); i += 2; continue }
      let match = null
      for (let n = Math.min(lexicon.maxWords, rest.length); n >= 1 && !match; n--) {
        const sign = lexicon.lookup(rest.slice(0, n))
        if (sign) match = { sign, n }
      }
      // could the words up to the end of the buffer still grow into a longer sign ("i am" -> I_AM_FINE)? wait
      // ("to" of "to morrow" waits longer: the second half is often still growing in the recogniser's guess)
      const age = now - buffer[buffer.length - 1].at
      const waiting = !final && ((lexicon.prefixes.has(rest.join(' ')) && age < holdMs) ||
        (rest.length === 1 && lexicon.mayJoin?.(rest[0]) && age < joinHoldMs))
      if (waiting) break
      if (match) {
        items.push({ sign: match.sign, words: rest.slice(0, match.n) })
        i += match.n
        continue
      }
      const w = rest[0]
      if (SKIP_WORDS.has(w)) stats.skipped++
      else if (spellUnknown && /^[a-z]+$/.test(w) && w.length <= maxSpell) items.push({ spell: w, words: [w] })
      else { unsigned.push(w); stats.unsigned++ }
      i++
    }
    buffer = buffer.slice(i)
    return { items, unsigned }
  }

  const emit = (now, final) => {
    const { items, unsigned } = resolve(now, final)
    if (!items.length && !unsigned.length) return
    for (let k = 0; k < Math.max(items.length, 1); k += maxSigns) {
      const group = items.slice(k, k + maxSigns)
      const phrase = {
        signs: group.filter((g) => g.sign).map((g) => g.sign),
        text: group.map((g) => (g.sign ? signText(g.sign) : g.spell.toUpperCase())).join(' '),
        words: group.flatMap((g) => g.words),
        spelled: group.filter((g) => g.spell).map((g) => g.spell),
        unsigned: k === 0 ? unsigned : [],
        at: now,
      }
      stats.signed += phrase.signs.length
      stats.spelled += phrase.spelled.length
      if (phrase.text) stats.phrases++
      onPhrase?.(phrase)
    }
  }

  return {
    stats,
    push(words, now = performance.now(), final = false) {
      buffer.push(...words.map((word) => ({ word, at: now })))
      emit(now, final)
    },
    /** A held word whose longer sign did not come within holdMs is signed on its own. */
    tick(now = performance.now()) { if (buffer.length) emit(now, false) },
    flush(now = performance.now()) { if (buffer.length) emit(now, true) },
    reset() { buffer = [] },
    get waiting() { return buffer.map((b) => b.word) },
  }
}
