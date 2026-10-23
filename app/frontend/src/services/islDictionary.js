/**
 * MUDRA ISL Official Lexicon & Verification System
 * Ground truth reference: Indian Sign Language Research and Training Centre (ISLRTC)
 * Official Lexicon & Dictionary: https://islrtc.nic.in/isl-dictionary/
 *
 * Each lexicon item includes verified `translations` in three languages:
 *   en - English (India)
 *   hi - Hindi
 *   mr - Marathi
 *
 * Translations are natural-language templates for the specific sign meaning.
 * Multi-sign combinations without a verified template fall back to English.
 */

export const ISLRTC_DICTIONARY_URL = 'https://islrtc.nic.in/isl-dictionary/'

/**
 * Human-readable form of a sign ID, e.g. 'GOOD_MORNING' -> 'Good morning'.
 */
export function readableSign(signId) {
  if (!signId) return ''
  const words = signId.replace(/_/g, ' ').toLowerCase()
  return words.charAt(0).toUpperCase() + words.slice(1)
}

export const ISL_VOCABULARY_LEXICON = [
  {
    id: 'HELLO',
    word: 'HELLO',
    label: 'Hello / Greetings',
    hindi: 'नमस्ते / हैलो',
    category: 'Greetings',
    source: 'ISLRTC Official Lexicon (Vol 1)',
    sourceUrl: ISLRTC_DICTIONARY_URL,
    verified: true,
    animationType: 'word',
    animationFunction: 'createWordHello',
    synonyms: ['hi', 'greetings', 'hey', 'hello'],
    regionalVariant: 'Standard Pan-India',
    description: 'Open right hand wave near temple/ear, palm facing outwards.',
    handShapeDesc: 'Open 5-finger flat hand',
    positionDesc: 'Temple / Ear height',
    movementDesc: 'Gentle side-to-side oscillation',
    sentenceTemplate: 'Hello, nice to meet you!',
    translations: {
      en: 'Hello, nice to meet you!',
      hi: 'नमस्ते, आपसे मिलकर खुशी हुई!',
      mr: 'नमस्कार, तुम्हाला भेटून आनंद झाला!'
    }
  },
  {
    id: 'NAMASTE',
    word: 'NAMASTE',
    label: 'Namaste',
    hindi: 'नमस्ते / प्रणाम',
    category: 'Greetings',
    source: 'ISLRTC Official Lexicon (Vol 1)',
    sourceUrl: ISLRTC_DICTIONARY_URL,
    verified: true,
    animationType: 'word',
    animationFunction: 'createWordNamaste',
    synonyms: ['pranam', 'namaskar', 'greetings', 'namaste'],
    regionalVariant: 'Standard Pan-India (Anjali Mudra)',
    description: 'Both palms joined together vertically in front of chest center.',
    handShapeDesc: 'Joined flat palms',
    positionDesc: 'Chest center (Heart level)',
    movementDesc: 'Slight respectful forward bow/nod',
    sentenceTemplate: 'Namaste, welcome.',
    translations: {
      en: 'Namaste, welcome.',
      hi: 'नमस्ते, आपका स्वागत है।',
      mr: 'नमस्कार, तुमचे स्वागत आहे।'
    }
  },
  {
    id: 'THANK_YOU',
    word: 'THANK YOU',
    label: 'Thank You',
    hindi: 'धन्यवाद / शुक्रिया',
    category: 'Greetings',
    source: 'ISLRTC Official Lexicon (Vol 2)',
    sourceUrl: ISLRTC_DICTIONARY_URL,
    verified: true,
    animationType: 'word',
    animationFunction: 'createWordThankYou',
    synonyms: ['thanks', 'thankyou', 'thank you', 'dhanyawad', 'shukriya'],
    regionalVariant: 'Standard ISL',
    description: 'Flat right hand touches chin/lips then moves outward toward recipient.',
    handShapeDesc: 'Flat open palm touching chin',
    positionDesc: 'Chin level to chest forward',
    movementDesc: 'Forward arching motion toward listener',
    sentenceTemplate: 'Thank you very much.',
    translations: {
      en: 'Thank you very much.',
      hi: 'आपका बहुत धन्यवाद।',
      mr: 'खूप आभारी आहे।'
    }
  },
  {
    id: 'WATER',
    word: 'WATER',
    label: 'Water',
    hindi: 'पानी / जल',
    category: 'Essentials',
    source: 'ISLRTC Official Lexicon (Vol 1)',
    sourceUrl: ISLRTC_DICTIONARY_URL,
    verified: true,
    animationType: 'word',
    animationFunction: 'createWordWater',
    synonyms: ['water', 'pani', 'drink water'],
    regionalVariant: 'Standard ISL (W-handshape near chin)',
    description: 'W-handshape (Index, Middle, Ring upright; Thumb holding Pinky) tapping near chin.',
    handShapeDesc: 'W-configuration (3 fingers up)',
    positionDesc: 'Right side of chin / mouth',
    movementDesc: 'Double gentle tap on chin',
    sentenceTemplate: 'Please give me water.',
    translations: {
      en: 'Please give me water.',
      hi: 'कृपया मुझे पानी दीजिए।',
      mr: 'कृपया मला पाणी द्या।'
    }
  },
  {
    id: 'YES',
    word: 'YES',
    label: 'Yes / Agree',
    hindi: 'हाँ / सहमत',
    category: 'Responses',
    source: 'ISLRTC Official Lexicon (Vol 2)',
    sourceUrl: ISLRTC_DICTIONARY_URL,
    verified: true,
    animationType: 'word',
    animationFunction: 'createWordYes',
    synonyms: ['yes', 'agree', 'haan', 'correct', 'yeah'],
    regionalVariant: 'Standard ISL (Fist nodding)',
    description: 'Closed S-fist in front of chest nodding vertically up and down.',
    handShapeDesc: 'Closed S-fist',
    positionDesc: 'Mid-chest level',
    movementDesc: 'Vertical nodding tilt',
    sentenceTemplate: 'Yes, I agree with that.',
    translations: {
      en: 'Yes, I agree with that.',
      hi: 'हाँ, मैं उससे सहमत हूँ।',
      mr: 'हो, मी त्याच्याशी सहमत आहे।'
    }
  },
  {
    id: 'NO',
    word: 'NO',
    label: 'No / Disagree',
    hindi: 'नहीं / मना',
    category: 'Responses',
    source: 'ISLRTC Official Lexicon (Vol 2)',
    sourceUrl: ISLRTC_DICTIONARY_URL,
    verified: true,
    animationType: 'word',
    animationFunction: 'createWordNo',
    synonyms: ['no', 'nahi', 'disagree', 'stop', 'nope'],
    regionalVariant: 'Standard ISL (Index-Middle extension snap)',
    description: 'Index and middle fingers extended together snapping downward to touch thumb.',
    handShapeDesc: 'Extended index + middle fingers',
    positionDesc: 'Upper chest level',
    movementDesc: 'Quick closing snap against thumb',
    sentenceTemplate: 'No, thank you.',
    translations: {
      en: 'No, thank you.',
      hi: 'नहीं, धन्यवाद।',
      mr: 'नाही, धन्यवाद।'
    }
  },
  {
    id: 'GOODBYE',
    word: 'GOODBYE',
    label: 'Goodbye / Bye',
    hindi: 'अलविदा / बाय',
    category: 'Greetings',
    source: 'ISLRTC Official Lexicon (Vol 1)',
    sourceUrl: ISLRTC_DICTIONARY_URL,
    verified: true,
    animationType: 'word',
    animationFunction: 'createWordGoodbye',
    synonyms: ['goodbye', 'bye', 'see you', 'alvida', 'farewell'],
    regionalVariant: 'Standard ISL',
    description: 'Raised right hand flexing fingers inward in gentle parting wave.',
    handShapeDesc: 'Open palm with flexing fingers',
    positionDesc: 'Shoulder / Ear level',
    movementDesc: 'Repetitive finger flex wave',
    sentenceTemplate: 'Goodbye, see you again.',
    translations: {
      en: 'Goodbye, see you again.',
      hi: 'अलविदा, फिर मिलेंगे।',
      mr: 'निरोप, पुन्हा भेटू।'
    }
  },
  {
    id: 'HOME',
    word: 'HOME',
    label: 'Home / House',
    hindi: 'घर',
    category: 'Essentials',
    source: 'ISLRTC Official Lexicon (Vol 3)',
    sourceUrl: ISLRTC_DICTIONARY_URL,
    verified: true,
    animationType: 'word',
    animationFunction: 'createWordHome',
    synonyms: ['home', 'house', 'ghar'],
    regionalVariant: 'Standard ISL (Roof Mudra)',
    description: 'Fingertips of both hands meet at an angle forming a roof shape.',
    handShapeDesc: 'Both hands angled roof shape',
    positionDesc: 'Center chest level',
    movementDesc: 'Downward defining motion',
    sentenceTemplate: 'I am going home.',
    translations: {
      en: 'I am going home.',
      hi: 'मैं घर जा रहा हूँ।',
      mr: 'मी घरी जात आहे।'
    }
  },
  {
    id: 'PERSON',
    word: 'PERSON',
    label: 'Person / Individual',
    hindi: 'व्यक्ति / इंसान',
    category: 'Community',
    source: 'ISLRTC Official Lexicon (Vol 2)',
    sourceUrl: ISLRTC_DICTIONARY_URL,
    verified: true,
    animationType: 'word',
    animationFunction: 'createWordPerson',
    synonyms: ['person', 'human', 'man', 'someone', 'individual'],
    regionalVariant: 'Standard ISL',
    description: 'Index finger pointing upright moving vertically downwards.',
    handShapeDesc: 'Index pointer handshape',
    positionDesc: 'Chest center level',
    movementDesc: 'Vertical body reference trace',
    sentenceTemplate: 'That person is here.',
    translations: {
      en: 'That person is here.',
      hi: 'वह व्यक्ति यहाँ है।',
      mr: 'ती व्यक्ती इथे आहे।'
    }
  },
  {
    id: 'TIME',
    word: 'TIME',
    label: 'Time / Clock',
    hindi: 'समय / वक्त',
    category: 'Essentials',
    source: 'ISLRTC Official Lexicon (Vol 1)',
    sourceUrl: ISLRTC_DICTIONARY_URL,
    verified: true,
    animationType: 'word',
    animationFunction: 'createWordTime',
    synonyms: ['time', 'clock', 'samay', 'ghadi', 'hour'],
    regionalVariant: 'Standard ISL',
    description: 'Index finger taps the wrist where a watch is worn.',
    handShapeDesc: 'Index tapping wrist',
    positionDesc: 'Left wrist position',
    movementDesc: 'Double tap on wrist',
    sentenceTemplate: 'What is the time?',
    translations: {
      en: 'What is the time?',
      hi: 'क्या समय हुआ है?',
      mr: 'किती वाजले आहेत?'
    }
  },
  {
    id: 'YOU',
    word: 'YOU',
    label: 'You',
    hindi: 'आप / तुम',
    category: 'Pronouns',
    source: 'ISLRTC Official Lexicon (Vol 1)',
    sourceUrl: ISLRTC_DICTIONARY_URL,
    verified: true,
    animationType: 'word',
    animationFunction: 'createWordYou',
    synonyms: ['you', 'aap', 'tum'],
    regionalVariant: 'Standard ISL (Direct Deixis)',
    description: 'Index finger points gently outward toward conversational partner.',
    handShapeDesc: 'Direct index point',
    positionDesc: 'Mid-torso pointing forward',
    movementDesc: 'Direct forward movement',
    sentenceTemplate: 'You are welcome here.',
    translations: {
      en: 'You are welcome here.',
      hi: 'आपका यहाँ स्वागत है।',
      mr: 'तुमचे येथे स्वागत आहे।'
    }
  },
  {
    id: 'HELP',
    word: 'HELP',
    label: 'Help / Support',
    hindi: 'मदद / सहायता',
    category: 'Emergency',
    source: 'ISLRTC Official Lexicon (Vol 2)',
    sourceUrl: ISLRTC_DICTIONARY_URL,
    verified: true,
    animationType: 'word',
    animationFunction: 'createWordHelp',
    synonyms: ['help', 'assist', 'madad', 'emergency', 'support'],
    regionalVariant: 'Standard ISL (Fist on Palm Support)',
    description: 'Closed right fist resting on open flat left palm, both lifted together.',
    handShapeDesc: 'Fist on open supporting palm',
    positionDesc: 'Lower chest lifting upward',
    movementDesc: 'Upward lifting motion',
    sentenceTemplate: 'I need help urgently.',
    translations: {
      en: 'I need help urgently.',
      hi: 'मुझे तुरंत मदद चाहिए।',
      mr: 'मला त्वरित मदत हवी आहे।'
    }
  },
  {
    id: 'PEACE',
    word: 'PEACE',
    label: 'Peace / Victory',
    hindi: 'शांति / विजय',
    category: 'Feelings',
    source: 'ISLRTC Official Lexicon (Vol 2)',
    sourceUrl: ISLRTC_DICTIONARY_URL,
    verified: true,
    animationType: 'word',
    animationFunction: 'createWordPeace',
    synonyms: ['peace', 'shanti', 'victory', 'harmony'],
    regionalVariant: 'Standard ISL (V-handshape)',
    description: 'Index and middle fingers spread in V-shape with palm facing forward.',
    handShapeDesc: 'V-sign (Index + Middle spread)',
    positionDesc: 'Chest / Shoulder level',
    movementDesc: 'Held steady with gentle forward presentation',
    sentenceTemplate: 'Peace and harmony to everyone.',
    translations: {
      en: 'Peace and harmony to everyone.',
      hi: 'सभी को शांति और सौहार्द मिले।',
      mr: 'सर्वांना शांती आणि सौहार्द मिळो।'
    }
  }
]

/**
 * Finds matching verified lexicon item for a word or phrase.
 */
export function findISLLexiconItem(wordOrPhrase) {
  if (!wordOrPhrase) return null
  const normalized = wordOrPhrase.trim().toLowerCase()

  return ISL_VOCABULARY_LEXICON.find((item) => {
    if (item.id.toLowerCase() === normalized) return true
    if (item.word.toLowerCase() === normalized) return true
    if (item.synonyms && item.synonyms.includes(normalized)) return true
    return false
  }) || null
}

/**
 * Tokenizes a sentence into sequential word execution steps.
 */
export function parseSentenceToISLSteps(sentence) {
  if (!sentence || typeof sentence !== 'string') return []

  const rawTokens = sentence.trim().split(/\s+/)
  const steps = []

  let i = 0
  while (i < rawTokens.length) {
    const token = rawTokens[i].replace(/[.,!?;:()]/g, '').trim()
    if (!token) {
      i++
      continue
    }

    // Check two-word phrases first (e.g. 'thank you')
    if (i + 1 < rawTokens.length) {
      const nextToken = rawTokens[i + 1].replace(/[.,!?;:()]/g, '').trim()
      const twoWordPhrase = `${token} ${nextToken}`
      const phraseMatch = findISLLexiconItem(twoWordPhrase)

      if (phraseMatch) {
        steps.push({
          rawText: twoWordPhrase,
          normalized: phraseMatch.word,
          status: 'verified_word',
          lexiconItem: phraseMatch,
          animationCode: phraseMatch.id,
          letters: null
        })
        i += 2
        continue
      }
    }

    // Check single word match
    const singleMatch = findISLLexiconItem(token)
    if (singleMatch) {
      steps.push({
        rawText: token,
        normalized: singleMatch.word,
        status: 'verified_word',
        lexiconItem: singleMatch,
        animationCode: singleMatch.id,
        letters: null
      })
      i++
      continue
    }

    // Fingerspell unknown words
    const isLettersOnly = /^[a-zA-Z]+$/.test(token)
    if (isLettersOnly) {
      const letters = token.toUpperCase().split('')
      steps.push({
        rawText: token,
        normalized: token.toUpperCase(),
        status: 'fingerspelling',
        lexiconItem: null,
        animationCode: token.toUpperCase(),
        letters: letters,
        message: `Whole-word ISL sign unavailable for "${token}". Fingerspelling: ${letters.join('-')}`
      })
    } else {
      steps.push({
        rawText: token,
        normalized: token,
        status: 'unsupported',
        lexiconItem: null,
        animationCode: null,
        letters: null,
        message: `Sign not supported yet for "${token}".`
      })
    }

    i++
  }

  return steps
}

// ---------------------------------------------------------------------------
// Multilingual Sentence Builder
// ---------------------------------------------------------------------------

/**
 * Verified natural-language sentence templates keyed by exact ISL token sequences.
 * Key formats:
 *   - Single token: 'HELLO', 'WATER', etc.
 *   - Multi-token sequence (joined by comma): 'HELLO,WATER', 'WATER,PLEASE', etc.
 *
 * This ensures that templates are used ONLY when the actual token sequence matches.
 * The translation represents the same canonical meaning across all three languages.
 */
const CANONICAL_SEQUENCE_DICTIONARY = {
  // Single token mappings
  'HELLO': {
    en: 'Hello.',
    hi: 'नमस्ते।',
    mr: 'नमस्कार।'
  },
  'NAMASTE': {
    en: 'Namaste.',
    hi: 'नमस्ते।',
    mr: 'नमस्कार।'
  },
  'THANK_YOU': {
    en: 'Thank you.',
    hi: 'धन्यवाद।',
    mr: 'धन्यवाद।'
  },
  'WATER': {
    en: 'I need water.',
    hi: 'मुझे पानी चाहिए।',
    mr: 'मला पाणी हवे आहे।'
  },
  'YES': {
    en: 'Yes.',
    hi: 'हाँ।',
    mr: 'हो।'
  },
  'NO': {
    en: 'No.',
    hi: 'नहीं।',
    mr: 'नाही।'
  },
  'GOODBYE': {
    en: 'Goodbye.',
    hi: 'अलविदा।',
    mr: 'निरोप।'
  },
  'HOME': {
    en: 'Home.',
    hi: 'घर।',
    mr: 'घर।'
  },
  'PERSON': {
    en: 'Person.',
    hi: 'व्यक्ति।',
    mr: 'व्यक्ती।'
  },
  'TIME': {
    en: 'Time.',
    hi: 'समय।',
    mr: 'वेळ।'
  },
  'YOU': {
    en: 'You.',
    hi: 'आप।',
    mr: 'तुम्ही।'
  },
  'HELP': {
    en: 'I need help.',
    hi: 'मुझे मदद चाहिए।',
    mr: 'मला मदत हवी आहे।'
  },
  'PEACE': {
    en: 'Peace.',
    hi: 'शांति।',
    mr: 'शांती।'
  },

  // Multi-token exact sequence mappings
  'HELLO,WATER': {
    en: 'Hello, I need water.',
    hi: 'नमस्ते, मुझे पानी चाहिए।',
    mr: 'नमस्कार, मला पाणी हवे आहे।'
  },
  'NAMASTE,WATER': {
    en: 'Hello, I need water.',
    hi: 'नमस्ते, मुझे पानी चाहिए।',
    mr: 'नमस्कार, मला पाणी हवे आहे।'
  },
  'HELLO,THANK_YOU': {
    en: 'Hello, thank you.',
    hi: 'नमस्ते, धन्यवाद।',
    mr: 'नमस्कार, धन्यवाद।'
  },
  'NAMASTE,THANK_YOU': {
    en: 'Hello, thank you.',
    hi: 'नमस्ते, धन्यवाद।',
    mr: 'नमस्कार, धन्यवाद।'
  },
  'HELLO,HELP': {
    en: 'Hello, I need help.',
    hi: 'नमस्ते, मुझे मदद चाहिए।',
    mr: 'नमस्कार, मला मदत हवी आहे।'
  },
  'NAMASTE,HELP': {
    en: 'Hello, I need help.',
    hi: 'नमस्ते, मुझे मदद चाहिए।',
    mr: 'नमस्कार, मला मदत हवी आहे।'
  }
}

/**
 * Builds a natural-language sentence from committed ISL sign tokens.
 *
 * Rules:
 *   1. Exact sequence match in CANONICAL_SEQUENCE_DICTIONARY: return the translated phrase.
 *   2. No exact match (ambiguous or unsupported sequence): return the
 *      individual canonical meanings without inventing combined grammar.
 *
 * @param {string[]} tokens - ISL lexicon IDs e.g. ['WATER', 'HELP']
 * @param {'en'|'hi'|'mr'} lang - Target output language
 * @returns {string}
 */
/** The curated sentence for exactly this sequence of two or more signs (e.g. HELLO,WATER), or null. */
export function canonicalSentence(tokens, lang = 'en') {
  if (!tokens || tokens.length < 2) return null
  const matched = CANONICAL_SEQUENCE_DICTIONARY[tokens.map((t) => t.toUpperCase().trim()).join(',')]
  return matched ? matched[lang] || matched.en : null
}

export function buildMultilingualSentence(tokens, lang = 'en') {
  if (!tokens || tokens.length === 0) return ''

  // Normalize tokens by replacing underscores with spaces or keeping standard IDs
  const normalizedTokens = tokens.map(t => t.toUpperCase().trim())
  const key = normalizedTokens.join(',')

  // 1. Direct exact sequence lookup
  const matched = CANONICAL_SEQUENCE_DICTIONARY[key]
  if (matched) {
    return matched[lang] || matched['en']
  }

  // 2. Preserve meaning for unsupported combinations without inventing grammar.
  return normalizedTokens
    .map((token) => CANONICAL_SEQUENCE_DICTIONARY[token]?.[lang] || readableSign(token))
    .join(' ')
}
