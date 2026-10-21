import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import {
  AlertTriangle, CheckCircle2, ChevronLeft, Loader2, MapPin, Pause, Play, Plus, RotateCcw, Send, SkipForward, Undo2, X,
} from 'lucide-react'
import Button from '../../components/ui/Button'
import { emergencyApi, forgetLastSession } from '../../features/emergency/api'
import {
  DemoTag, ErrorNote, Hint, Notice, ProvenanceTag, Section, StatusTimeline, confidenceText,
} from '../../features/emergency/components'

const INPUT_TABS = [['BUTTONS', 'Buttons'], ['SIGN', 'Signs'], ['GESTURE', 'My gestures']]
const CONFIDENCE_CHOICES = [[0.9, 'Sure'], [0.55, 'Unsure'], [0.3, 'Not clear']]
const POLL_STATES = new Set(['SENT_TO_CONFIGURED_DESTINATION', 'ACKNOWLEDGED'])
const signLabel = (s) => s.replace(/^CHOICE:/, '').replace(/^INTENT:/, '').replaceAll('_', ' ').toLowerCase()

export default function EmergencySessionPage() {
  const { sessionId } = useParams()
  const navigate = useNavigate()
  const [view, setView] = useState(null)
  const [config, setConfig] = useState(null)
  const [error, setError] = useState(null)
  const [busy, setBusy] = useState(false)
  const [tab, setTab] = useState(null)
  const locationRef = useRef(null)

  const run = useCallback(async (fn) => {
    setBusy(true)
    setError(null)
    try {
      const next = await fn()
      if (next?.id) setView(next)
      return next
    } catch (e) {
      setError(e.message)
      return null
    } finally {
      setBusy(false)
    }
  }, [])

  useEffect(() => {
    if (!emergencyApi.hasSession(sessionId)) {
      setError('This report belongs to another browser tab or has expired. Start a new one.')
      return
    }
    emergencyApi.config().then(setConfig).catch(() => {})
    emergencyApi.session(sessionId).then((v) => { setView(v); setTab(v.mode) }).catch((e) => setError(e.message))
  }, [sessionId])

  // after sharing, keep the status current so the user sees a responder's acknowledgement
  useEffect(() => {
    if (!view || !POLL_STATES.has(view.status)) return undefined
    const t = setInterval(() => emergencyApi.session(sessionId).then(setView).catch(() => {}), 4000)
    return () => clearInterval(t)
  }, [view?.status, sessionId]) // eslint-disable-line react-hooks/exhaustive-deps

  const act = useCallback((path, body) => run(() => emergencyApi.act(sessionId, path, body)), [run, sessionId])
  const categories = useMemo(() => config?.categories || [], [config])

  if (!view) {
    return (
      <div className="space-y-4">
        <ErrorNote error={error} />
        {error ? <Link to="/emergency" className="font-bold underline">Start a new emergency report</Link>
          : <p className="flex items-center gap-2 text-sl-ink-600"><Loader2 className="h-5 w-5 animate-spin" aria-hidden="true" /> Loading your report…</p>}
      </div>
    )
  }

  const shareBrowserLocation = () => {
    if (!navigator.geolocation) { setError('This browser cannot share a location. Type it instead.'); return }
    setBusy(true)
    navigator.geolocation.getCurrentPosition(
      (pos) => run(() => emergencyApi.setLocation(sessionId, {
        source: 'BROWSER', lat: pos.coords.latitude, lon: pos.coords.longitude, accuracy_m: pos.coords.accuracy, permission: 'GRANTED',
      })),
      () => { setBusy(false); setError('Location was not shared (permission refused or unavailable). You can type it instead.') },
      { enableHighAccuracy: true, timeout: 15000, maximumAge: 60000 },
    )
  }

  const answer = async (q, value) => {
    const next = await act(`/questions/${q.id}/answer`, { value })
    if (next && (q.key === 'LOCATION' || q.key === 'LOCATION_SAFE')) {
      if (value === 'SHARE') shareBrowserLocation()
      if (value === 'ENTER') locationRef.current?.focus()
    }
  }

  const ctx = view.context
  // once confirmed the report is locked: unanswered questions no longer apply
  const pending = view.editable ? view.questions.filter((q) => q.status === 'PENDING') : []
  const answered = view.questions.filter((q) => q.status === 'ANSWERED')
  const editable = view.editable

  return (
    <div className="space-y-4 pb-20 lg:pb-0">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <Link to="/emergency" className="inline-flex items-center gap-1 text-sm font-bold text-sl-ink-600 hover:text-sl-ink">
            <ChevronLeft className="h-4 w-4" aria-hidden="true" /> Emergency
          </Link>
          <h1 className="font-display text-2xl font-extrabold sm:text-3xl">Your emergency report</h1>
        </div>
        <div className="flex flex-wrap items-center gap-2" aria-live="polite">
          <span className="rounded-full border-2 border-sl-ink bg-white px-3 py-1 text-sm font-bold">{view.status_label}</span>
          {view.demo && <DemoTag>Demo mode</DemoTag>}
        </div>
      </div>
      <Notice compact />
      <ErrorNote error={error} onClose={() => setError(null)} />

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-[minmax(0,1fr)_22rem]">
        <div className="min-w-0 space-y-4">
          {editable && (
            <Section title="Tell SignLine" aside={
              <div role="tablist" aria-label="How to communicate" className="flex rounded-full border-2 border-sl-ink bg-sl-cream-50 p-0.5">
                {INPUT_TABS.map(([id, label]) => (
                  <button key={id} role="tab" type="button" aria-selected={tab === id} onClick={() => setTab(id)}
                          className={`min-h-9 rounded-full px-3 text-xs font-bold ${tab === id ? 'bg-sl-ink text-white' : 'text-sl-ink'}`}>
                    {label}
                  </button>
                ))}
              </div>
            }>
              {tab === 'BUTTONS' && <ButtonsInput config={config} busy={busy} act={act} />}
              {tab === 'SIGN' && <SignInput view={view} config={config} busy={busy} act={act} />}
              {tab === 'GESTURE' && <GestureInput busy={busy} act={act} />}
            </Section>
          )}

          <Section title="What you said" aside={editable && (
            <div className="flex gap-2">
              <Button size="sm" variant="outline" disabled={busy || !view.can_undo} onClick={() => act('/sequence/undo')}>
                <Undo2 className="h-4 w-4" aria-hidden="true" /> Undo
              </Button>
              <Button size="sm" variant="ghost" disabled={busy || !view.sequence.length} onClick={() => act('/sequence/reset')}>
                <RotateCcw className="h-4 w-4" aria-hidden="true" /> Clear
              </Button>
            </div>
          )}>
            <SequenceList view={view} config={config} busy={busy} act={act} />
          </Section>

          <Section title="What SignLine understood">
            <Understanding ctx={ctx} generalMessage={view.general_message} confirmed={Boolean(view.incident)} />
          </Section>

          {(pending.length > 0 || answered.length > 0) && (
            <Section title="A few questions" aside={<span className="text-xs text-sl-ink-600">All optional</span>}>
              <ul className="space-y-3">
                {pending.map((q) => (
                  <li key={q.id} className="rounded-sl border-2 border-sl-ink bg-sl-cream-50 p-3">
                    <p className="font-bold">{q.text}</p>
                    <p className="text-xs text-sl-ink-600">{q.reason}</p>
                    <div className="mt-2 flex flex-wrap gap-2">
                      {q.options.map((o) => (
                        <Button key={o.value} size="sm" variant="outline" disabled={busy || !editable} onClick={() => answer(q, o.value)}>{o.label}</Button>
                      ))}
                      <Button size="sm" variant="ghost" disabled={busy || !editable} onClick={() => act(`/questions/${q.id}/skip`)}>Skip</Button>
                    </div>
                  </li>
                ))}
                {answered.map((q) => (
                  <li key={q.id} className="flex items-start justify-between gap-3 text-sm">
                    <span>{q.text} <strong>{q.options.find((o) => o.value === q.answer)?.label || String(q.answer)}</strong></span>
                    <ProvenanceTag value="USER_REPORTED" />
                  </li>
                ))}
              </ul>
            </Section>
          )}

          <Section title="Location" aside={<span className="text-xs text-sl-ink-600">Only if you choose to share it</span>}>
            <LocationPanel view={view} busy={busy} editable={editable} inputRef={locationRef} onBrowser={shareBrowserLocation}
                           onSet={(body) => run(() => emergencyApi.setLocation(sessionId, body))}
                           onClear={() => run(() => emergencyApi.clearLocation(sessionId))} />
          </Section>
        </div>

        <aside id="em-report" className="min-w-0 scroll-mt-20 space-y-4 lg:sticky lg:top-20 lg:self-start">
          <ReportPanel view={view} categories={categories} busy={busy} act={act}
                       onDone={() => { forgetLastSession(); navigate('/emergency') }} />
          <Section title="Status">
            <StatusTimeline history={view.history} />
          </Section>
        </aside>
      </div>

      {/* Mobile: the main action stays reachable without scrolling to the report panel */}
      <div className="fixed inset-x-0 bottom-0 z-40 border-t-2 border-sl-ink bg-sl-cream-50 px-4 py-2 lg:hidden"
           style={{ paddingBottom: 'calc(0.5rem + env(safe-area-inset-bottom))' }}>
        {['SESSION_CREATED', 'COMMUNICATING', 'POSSIBLE_EMERGENCY_DETECTED'].includes(view.status) ? (
          <Button className="w-full bg-rose-600 text-white hover:bg-rose-700" disabled={busy || (!view.sequence.length && !ctx.category)}
                  onClick={async () => { if (await act('/review')) document.getElementById('em-report')?.scrollIntoView({ behavior: 'smooth' }) }}>
            Review my report
          </Button>
        ) : (
          <a href="#em-report" className="flex min-h-11 items-center justify-center rounded-full border-2 border-sl-ink bg-white text-sm font-bold">
            {view.status_label} · see my report
          </a>
        )}
      </div>
    </div>
  )
}

/* ------------------------------------------------------------------ inputs */
function ButtonsInput({ config, busy, act }) {
  const [need, setNeed] = useState(null)
  if (!config) return <p className="text-sm text-sl-ink-600">Loading options…</p>
  const details = need ? config.buttons.details[need.id] : null
  if (details) {
    return (
      <div>
        <button type="button" onClick={() => setNeed(null)} className="mb-2 inline-flex items-center gap-1 text-sm font-bold text-sl-ink-600">
          <ChevronLeft className="h-4 w-4" aria-hidden="true" /> {need.label}: what is happening?
        </button>
        <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
          {details.map((d) => (
            <button key={d.id} type="button" disabled={busy} onClick={async () => { await act('/selections', { need: need.id, detail: d.id }); setNeed(null) }}
                    className="min-h-14 rounded-sl border-2 border-sl-ink bg-white px-4 text-left font-bold hover:bg-rose-50 disabled:opacity-60">
              {d.label}
            </button>
          ))}
        </div>
      </div>
    )
  }
  return (
    <div className="grid grid-cols-2 gap-2 sm:grid-cols-3">
      {config.buttons.needs.map((n) => (
        <button key={n.id} type="button" disabled={busy}
                onClick={() => (config.buttons.details[n.id] ? setNeed(n) : act('/selections', { need: n.id }))}
                className="min-h-16 rounded-sl border-2 border-sl-ink bg-white px-3 text-center font-display text-base font-extrabold shadow-sl-sm hover:bg-rose-50 disabled:opacity-60">
          {n.label}
        </button>
      ))}
    </div>
  )
}

function SignInput({ view, config, busy, act }) {
  const [query, setQuery] = useState('')
  const [confidence, setConfidence] = useState(0.9)
  const sc = view.scenario
  const vocab = (config?.vocabulary || []).filter((w) => w.toLowerCase().includes(query.toLowerCase().trim()))
  return (
    <div className="space-y-4">
      {sc && !sc.id.startsWith('E_') && (
        <div className="rounded-sl border-2 border-sl-ink bg-sl-sky-100 p-3">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <p className="text-sm font-bold">Scenario {sc.id[0]}: {sc.title} <span className="font-normal text-sl-ink-600">({sc.cursor}/{sc.total} signs)</span></p>
            <DemoTag>Simulated recognition</DemoTag>
          </div>
          {sc.hint && <p className="mt-1 text-xs">{sc.hint}</p>}
          <div className="mt-2 flex flex-wrap gap-2">
            <Button size="sm" disabled={busy || sc.paused || sc.cursor >= sc.total} onClick={() => act('/scenario', { action: 'next' })}>
              <SkipForward className="h-4 w-4" aria-hidden="true" /> Next sign
            </Button>
            <Button size="sm" variant="outline" disabled={busy || sc.paused || sc.cursor >= sc.total} onClick={() => act('/scenario', { action: 'all' })}>
              <Play className="h-4 w-4" aria-hidden="true" /> Play all
            </Button>
            <Button size="sm" variant="outline" disabled={busy} onClick={() => act('/scenario', { action: sc.paused ? 'resume' : 'pause' })}>
              {sc.paused ? <Play className="h-4 w-4" aria-hidden="true" /> : <Pause className="h-4 w-4" aria-hidden="true" />} {sc.paused ? 'Resume' : 'Pause'}
            </Button>
            <Button size="sm" variant="ghost" disabled={busy} onClick={() => act('/scenario', { action: 'restart' })}>
              <RotateCcw className="h-4 w-4" aria-hidden="true" /> Restart
            </Button>
          </div>
        </div>
      )}
      {sc?.id.startsWith('E_') && (
        <div className="rounded-sl border-2 border-sl-ink bg-sl-sky-100 p-3">
          <p className="text-sm font-bold">Scenario E: {sc.description}</p>
          <Button size="sm" className="mt-2" disabled={busy || sc.cursor > 0} onClick={() => act('/scenario', { action: 'next' })}>
            <Play className="h-4 w-4" aria-hidden="true" /> {sc.cursor > 0 ? 'Gesture used' : 'Use the gesture'}
          </Button>
        </div>
      )}
      <div>
        <div className="flex flex-wrap items-end justify-between gap-2">
          <label className="text-sm font-bold" htmlFor="sign-search">Add a sign <span className="font-normal text-sl-ink-600">(demo: stands in for camera recognition)</span></label>
          <fieldset className="flex items-center gap-1 text-xs">
            <legend className="sr-only">Recognition confidence to simulate</legend>
            {CONFIDENCE_CHOICES.map(([v, label]) => (
              <label key={v} className={`cursor-pointer rounded-full border px-2 py-1 font-bold ${confidence === v ? 'border-sl-ink bg-sl-ink text-white' : 'border-sl-cream-300'}`}>
                <input type="radio" name="confidence" className="sr-only" checked={confidence === v} onChange={() => setConfidence(v)} />
                {label}
              </label>
            ))}
          </fieldset>
        </div>
        <input id="sign-search" value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Search signs"
               className="mt-2 min-h-11 w-full rounded-sl border-2 border-sl-ink bg-white px-3 text-sm" />
        <div className="mt-2 flex max-h-40 flex-wrap gap-1.5 overflow-y-auto">
          {vocab.map((w) => (
            <button key={w} type="button" disabled={busy} onClick={() => act('/mock-sign', { sign: w, confidence })}
                    className="min-h-9 rounded-full border border-sl-cream-300 bg-sl-cream-50 px-3 text-xs font-bold hover:border-sl-ink disabled:opacity-60">
              {signLabel(w)}
            </button>
          ))}
          {query.trim() && !vocab.length && (
            <button type="button" disabled={busy} onClick={() => act('/mock-sign', { sign: query.trim(), confidence })}
                    className="min-h-9 rounded-full border border-dashed border-sl-ink px-3 text-xs font-bold">
              Add “{query.trim()}” (not a known sign: shown as not recognised)
            </button>
          )}
        </div>
      </div>
    </div>
  )
}

function GestureInput({ busy, act }) {
  const [gestures, setGestures] = useState(null)
  const [asking, setAsking] = useState(null)
  useEffect(() => { emergencyApi.gestures().then(setGestures).catch(() => setGestures([])) }, [])
  if (!gestures) return <p className="text-sm text-sl-ink-600">Loading your gestures…</p>
  if (!gestures.length) {
    return <p className="text-sm">No gestures yet. <Link to="/emergency/gestures" className="font-bold underline">Set one up</Link> (it takes a minute).</p>
  }
  const use = async (g) => { setAsking(null); await act(`/gestures/${g.id}/trigger`) }
  return (
    <div className="space-y-2">
      <Hint>Gesture detection by camera is not connected yet: tap the gesture you made. Its prepared message is added to your report.</Hint>
      <ul className="grid gap-2 sm:grid-cols-2">
        {gestures.filter((g) => g.enabled).map((g) => (
          <li key={g.id} className="rounded-sl border-2 border-sl-ink bg-white p-3">
            <p className="font-bold">{g.name}</p>
            <p className="text-xs text-sl-ink-600">{g.category_label}: “{g.message}”</p>
            {asking === g.id ? (
              <div className="mt-2 flex gap-2">
                <Button size="sm" disabled={busy} onClick={() => use(g)}>Yes, use it</Button>
                <Button size="sm" variant="ghost" onClick={() => setAsking(null)}>No</Button>
              </div>
            ) : (
              <Button size="sm" className="mt-2" disabled={busy} onClick={() => (g.require_confirmation ? setAsking(g.id) : use(g))}>Use this gesture</Button>
            )}
          </li>
        ))}
      </ul>
    </div>
  )
}

/* ------------------------------------------------------------------ sequence and understanding */
function SequenceList({ view, config, busy, act }) {
  const [editing, setEditing] = useState(null)
  const [value, setValue] = useState('')
  if (!view.sequence.length) return <p className="text-sm text-sl-ink-600">Nothing yet. Use the buttons, signs or a gesture above.</p>
  return (
    <div>
      <ol className="flex flex-wrap gap-2" aria-label="Your signs and selections, in order">
        {view.sequence.map((e) => {
          const unknown = e.event_type === 'UNKNOWN_SIGN'
          const low = e.confidence != null && e.confidence < 0.6
          return (
            <li key={e.event_id}>
              <button type="button" disabled={!view.editable || e.event_type === 'BUTTON_SELECTED' || e.event_type === 'GESTURE_DETECTED'}
                      onClick={() => { setEditing(e); setValue(e.sign) }}
                      className={`flex min-h-11 flex-col items-start rounded-sl border-2 px-3 py-1 text-left ${unknown || low ? 'border-dashed border-rose-700 bg-rose-50' : 'border-sl-ink bg-white'} ${editing?.event_id === e.event_id ? 'ring-4 ring-sl-green' : ''}`}>
                <span className="text-sm font-extrabold">{signLabel(e.sign)}{unknown ? ' ?' : ''}</span>
                <span className="text-[11px] text-sl-ink-600">
                  {e.event_type === 'BUTTON_SELECTED' ? 'button' : e.event_type === 'GESTURE_DETECTED' ? 'your gesture' : confidenceText(e.confidence)}
                  {unknown ? ' · not recognised' : low ? ' · unsure' : ''}
                  {e.corrected ? ` · corrected from ${signLabel(e.original_sign)}` : ''}
                  {e.repeated ? ' · repeated' : ''}
                </span>
              </button>
            </li>
          )
        })}
      </ol>
      {editing && view.editable && (
        <form className="mt-3 flex flex-wrap items-end gap-2 rounded-sl border-2 border-sl-ink bg-sl-cream-50 p-3"
              onSubmit={async (ev) => { ev.preventDefault(); await act('/sequence/correct', { event_id: editing.event_id, sign: value }); setEditing(null) }}>
          <label className="flex-1 text-sm font-bold">
            Correct “{signLabel(editing.sign)}” to
            <input list="em-vocab" value={value} onChange={(ev) => setValue(ev.target.value)}
                   className="mt-1 min-h-11 w-full rounded-sl border-2 border-sl-ink bg-white px-3 font-normal" />
            <datalist id="em-vocab">{(config?.vocabulary || []).map((w) => <option key={w} value={w} />)}</datalist>
          </label>
          <Button type="submit" size="sm" disabled={busy || !value.trim()}>Save</Button>
          <Button size="sm" variant="outline" disabled={busy} onClick={async () => { await act('/sequence/remove', { event_id: editing.event_id }); setEditing(null) }}>
            <X className="h-4 w-4" aria-hidden="true" /> Remove
          </Button>
          <Button size="sm" variant="ghost" onClick={() => setEditing(null)}>Cancel</Button>
        </form>
      )}
      {view.editable && <Hint>Tap a sign to correct or remove it. The original recognition is kept in the history.</Hint>}
    </div>
  )
}

function Understanding({ ctx, generalMessage, confirmed }) {
  if (!ctx.category) {
    return generalMessage
      ? <p className="rounded-sl bg-sl-green-50 p-3 text-sm">{generalMessage} <span className="text-sl-ink-600">If you need help, choose what kind with the buttons above.</span></p>
      : <p className="text-sm text-sl-ink-600">Not clear yet. Add more signs or tap what you need.</p>
  }
  return (
    <div className="space-y-3">
      <div className="rounded-sl border-2 border-sl-ink bg-rose-50 p-3">
        <p className="text-xs font-bold uppercase tracking-wide text-rose-800">{confirmed ? 'Confirmed by you' : 'Possible'}: {ctx.category_label}</p>
        <p className="mt-1 text-lg font-bold">{ctx.summary}</p>
        {ctx.user_corrected && <p className="mt-1 text-xs font-bold text-sl-green-700">Corrected by you</p>}
      </div>
      {ctx.uncertain.length > 0 && (
        <div className="flex items-start gap-2 rounded-sl border border-dashed border-rose-700 p-3 text-sm">
          <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-rose-700" aria-hidden="true" />
          <p>Please check: {ctx.uncertain.map((u) => `${signLabel(u.sign)} (${u.reason.toLowerCase()})`).join(', ')}.</p>
        </div>
      )}
      <details className="text-sm">
        <summary className="cursor-pointer font-bold">Why SignLine thinks this</summary>
        <ul className="mt-2 space-y-2">
          {ctx.interpretations.map((i) => (
            <li key={i.rule_id + i.category} className="flex items-start justify-between gap-3 rounded-sl bg-sl-cream-50 p-2">
              <span>
                <strong>{i.category}</strong>{i.uncertain ? ' (uncertain)' : ''}: {i.explanation}
                {i.alternatives.length > 0 && <span className="text-sl-ink-600"> Could also be: {i.alternatives.join(', ').toLowerCase().replaceAll('_', ' ')}.</span>}
              </span>
              <ProvenanceTag value={i.provenance} />
            </li>
          ))}
        </ul>
        {ctx.missing.length > 0 && <p className="mt-2 text-sl-ink-600">Still unknown: {ctx.missing.join(', ').toLowerCase()}.</p>}
      </details>
    </div>
  )
}

/* ------------------------------------------------------------------ location */
function LocationPanel({ view, busy, editable, inputRef, onBrowser, onSet, onClear }) {
  const [text, setText] = useState('')
  const loc = view.location
  return (
    <div className="space-y-3">
      {loc ? (
        <div className="flex flex-wrap items-center justify-between gap-2 rounded-sl border-2 border-sl-ink bg-sl-green-50 p-3 text-sm">
          <span className="flex items-start gap-2">
            <MapPin className="mt-0.5 h-4 w-4 shrink-0" aria-hidden="true" />
            <span>
              {loc.text || `${loc.lat.toFixed(5)}, ${loc.lon.toFixed(5)}`}
              {loc.accuracy_m != null && <span className="text-sl-ink-600"> (±{loc.accuracy_m} m)</span>}
              {loc.simulated && <> <DemoTag /></>}
            </span>
          </span>
          {editable && <Button size="sm" variant="ghost" disabled={busy} onClick={onClear}>Remove</Button>}
        </div>
      ) : <p className="text-sm text-sl-ink-600">No location shared. Responders will only see a location if you add one.</p>}
      {editable && (
        <>
          <div className="flex flex-wrap gap-2">
            <Button size="sm" variant="outline" disabled={busy} onClick={onBrowser}><MapPin className="h-4 w-4" aria-hidden="true" /> Share my current location</Button>
            {view.demo && <Button size="sm" variant="ghost" disabled={busy} onClick={() => onSet({ source: 'SIMULATED' })}>Use a simulated location (demo)</Button>}
          </div>
          <form className="flex gap-2" onSubmit={(e) => { e.preventDefault(); if (text.trim()) onSet({ source: 'MANUAL', text: text.trim() }); setText('') }}>
            <label htmlFor="em-location" className="sr-only">Type a location</label>
            <input id="em-location" ref={inputRef} value={text} onChange={(e) => setText(e.target.value)} maxLength={300}
                   placeholder="Or type where you are (landmark, street, floor)"
                   className="min-h-11 flex-1 rounded-sl border-2 border-sl-ink bg-white px-3 text-sm" />
            <Button type="submit" size="sm" disabled={busy || !text.trim()}><Plus className="h-4 w-4" aria-hidden="true" /> Add</Button>
          </form>
          <Hint>Your browser asks before sharing your location. It is stored only with this report.</Hint>
        </>
      )}
    </div>
  )
}

/* ------------------------------------------------------------------ review, confirm, share */
function ReportPanel({ view, categories, busy, act, onDone }) {
  const ctx = view.context
  const [editing, setEditing] = useState(false)
  const [category, setCategory] = useState('')
  const [summary, setSummary] = useState('')
  const status = view.status
  const incident = view.incident
  const lastDispatch = incident?.dispatch_attempts.at(-1)

  const startEdit = () => { setCategory(ctx.category || ''); setSummary(ctx.summary || ''); setEditing(true) }
  const saveEdit = async () => {
    await act('/correction', { category: category || null, summary: summary !== ctx.summary ? summary : null })
    setEditing(false)
  }

  return (
    <Section title="Your report" className="border-2 border-sl-ink shadow-sl">
      {['SESSION_CREATED', 'COMMUNICATING', 'POSSIBLE_EMERGENCY_DETECTED'].includes(status) && (
        <div className="space-y-3">
          <p className="text-sm">{ctx.summary || 'Say or choose what is happening first.'}</p>
          <Button className="w-full bg-rose-600 text-white hover:bg-rose-700" disabled={busy || (!view.sequence.length && !ctx.category)} onClick={() => act('/review')}>
            Review my report
          </Button>
        </div>
      )}

      {status === 'AWAITING_USER_CONFIRMATION' && (
        <div className="space-y-3">
          {editing ? (
            <div className="space-y-2">
              <label className="block text-sm font-bold">Kind of help
                <select value={category} onChange={(e) => setCategory(e.target.value)} className="mt-1 min-h-11 w-full rounded-sl border-2 border-sl-ink bg-white px-2 font-normal">
                  <option value="">Choose…</option>
                  {categories.map((c) => <option key={c.id} value={c.id}>{c.label}</option>)}
                </select>
              </label>
              <label className="block text-sm font-bold">Message for responders
                <textarea value={summary} onChange={(e) => setSummary(e.target.value)} rows={3} maxLength={1000}
                          className="mt-1 w-full rounded-sl border-2 border-sl-ink bg-white p-2 font-normal" />
              </label>
              <div className="flex gap-2">
                <Button size="sm" disabled={busy} onClick={saveEdit}>Save</Button>
                <Button size="sm" variant="ghost" onClick={() => setEditing(false)}>Cancel</Button>
              </div>
            </div>
          ) : (
            <>
              <p className="text-xs font-bold uppercase tracking-wide text-sl-ink-600">Is this right?</p>
              <p className="font-bold">{ctx.category_label || 'Kind of help not chosen'}</p>
              <p className="text-sm">{ctx.summary || 'No message yet.'}</p>
              {view.location ? <p className="text-xs text-sl-ink-600">Location will be included.</p> : <p className="text-xs text-sl-ink-600">No location included.</p>}
              <div className="flex flex-col gap-2">
                <Button className="w-full" disabled={busy || !ctx.category} onClick={() => act('/confirm')}>
                  <CheckCircle2 className="h-4 w-4" aria-hidden="true" /> Yes, this is right
                </Button>
                <Button className="w-full" variant="outline" disabled={busy} onClick={startEdit}>No, let me correct it</Button>
              </div>
            </>
          )}
        </div>
      )}

      {incident && (
        <div className="space-y-3">
          <p className="font-bold">{incident.category_label}</p>
          <p className="text-sm">{incident.summary}</p>
          {incident.emergency_contact && (
            <p className="text-xs text-sl-ink-600">Prepared from your gesture. Emergency number chosen: {incident.emergency_contact.number} {incident.emergency_contact.service} (call it yourself from the gesture page; sharing here does not call anyone).</p>
          )}
          {status === 'READY_TO_SHARE' && <p className="text-sm font-bold">Report prepared. Not shared yet.</p>}
          {incident.video && (
            <p className="rounded-sl border border-sl-cream-300 p-2 text-xs">
              Your gesture video is attached with your consent:{' '}
              <strong>{{ ATTACHED: 'not shared yet', DELIVERED: 'delivered with the report', FAILED: 'not delivered' }[incident.video_sharing_status] || incident.video_sharing_status}</strong>.
            </p>
          )}
          {status === 'READY_TO_SHARE' && (
            <Button className="w-full bg-rose-600 text-white hover:bg-rose-700" disabled={busy} onClick={() => act('/share')}>
              <Send className="h-4 w-4" aria-hidden="true" /> Share with the demo responder desk
            </Button>
          )}
          {status === 'DELIVERY_FAILED' && (
            <div className="space-y-2 rounded-sl border-2 border-rose-700 bg-rose-50 p-3 text-sm text-rose-900">
              <p className="font-bold">Not delivered. Nobody has received this report.</p>
              {lastDispatch?.error && <p>{lastDispatch.error}</p>}
              <Button className="w-full" disabled={busy} onClick={() => act('/share')}>Try again</Button>
            </div>
          )}
          {status === 'SENT_TO_CONFIGURED_DESTINATION' && (
            <p className="rounded-sl bg-sl-sky-100 p-3 text-sm">
              Delivered to the SignLine demo responder desk (receipt {lastDispatch?.receipt_id}, simulated). Waiting for a responder to acknowledge it…
            </p>
          )}
          {status === 'ACKNOWLEDGED' && <p className="rounded-sl bg-sl-green-100 p-3 text-sm font-bold">A responder at the demo desk has seen your report.</p>}
          {status === 'RESOLVED' && <p className="rounded-sl bg-sl-green-100 p-3 text-sm font-bold">The responder marked this report as resolved.</p>}
        </div>
      )}

      {status === 'CANCELLED' && <p className="text-sm font-bold">This report was cancelled. Nothing was shared.</p>}

      <div className="mt-4 flex flex-col gap-2 border-t border-dashed border-sl-cream-300 pt-3">
        {view.allowed_next.includes('CANCELLED') && (
          <Button size="sm" variant="ghost" disabled={busy} onClick={() => act('/cancel')}>Cancel this report</Button>
        )}
        {['CANCELLED', 'RESOLVED'].includes(status) && <Button size="sm" variant="outline" onClick={onDone}>Start a new report</Button>}
      </div>
    </Section>
  )
}
