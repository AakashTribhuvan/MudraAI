import { useCallback, useEffect, useState } from 'react'
import { Link, Navigate, useNavigate, useParams } from 'react-router-dom'
import { ChevronLeft, LogOut, MapPin, RefreshCw } from 'lucide-react'
import Button from '../../components/ui/Button'
import { emergencyApi, responderAuth, setResponderAuth } from '../../features/emergency/api'
import { DemoTag, ErrorNote, Hint, ProvenanceTag, Section, StatusTimeline, confidenceText } from '../../features/emergency/components'

const FILTERS = [['', 'All'], ['SENT_TO_CONFIGURED_DESTINATION', 'New'], ['ACKNOWLEDGED', 'Acknowledged'], ['RESOLVED', 'Resolved']]
const FACT_GROUPS = [
  ['What the user confirmed', ['CATEGORY', 'SUMMARY']],
  ['Signs and selections', ['SIGN', 'SELECTION', 'MESSAGE']],
  ['Answers', ['ANSWER']],
  ['System interpretation', ['INTERPRETATION', 'UNCERTAINTY']],
  ['Other', ['LOCATION', 'NEED', 'SYSTEM']],
]

function useResponderGuard() {
  const navigate = useNavigate()
  const auth = responderAuth()
  const valid = auth && auth.expiresAt > Date.now()
  const onAuthError = useCallback((e) => {
    if (e.status === 401 || e.status === 403) { setResponderAuth(null); navigate('/responder/login', { replace: true }); return null }
    return e.message
  }, [navigate])
  return { auth: valid ? auth : null, onAuthError }
}

export function ResponderLoginPage() {
  const navigate = useNavigate()
  const [username, setUsername] = useState('demo.responder')
  const [password, setPassword] = useState('')
  const [error, setError] = useState(null)
  const [busy, setBusy] = useState(false)
  const submit = async (e) => {
    e.preventDefault()
    setBusy(true)
    setError(null)
    try { await emergencyApi.login(username, password); navigate('/responder', { replace: true }) } catch (err) { setError(err.message); setBusy(false) }
  }
  return (
    <div className="mx-auto max-w-md space-y-4 pt-6">
      <h1 className="font-display text-3xl font-extrabold">Responder sign in</h1>
      <p className="text-sm text-sl-ink-600">For the SignLine demo responder desk. Emergency users never need an account.</p>
      <ErrorNote error={error} />
      <form onSubmit={submit} className="space-y-3 rounded-sl-lg border-2 border-sl-ink bg-white p-5 shadow-sl">
        <label className="block text-sm font-bold">Username
          <input value={username} onChange={(e) => setUsername(e.target.value)} autoComplete="username" required
                 className="mt-1 min-h-11 w-full rounded-sl border-2 border-sl-ink px-3 font-normal" />
        </label>
        <label className="block text-sm font-bold">Password
          <input type="password" value={password} onChange={(e) => setPassword(e.target.value)} autoComplete="current-password" required
                 className="mt-1 min-h-11 w-full rounded-sl border-2 border-sl-ink px-3 font-normal" />
        </label>
        <Button type="submit" variant="purple" className="w-full" disabled={busy}>Sign in</Button>
      </form>
      <Hint>
        Demo password: set SIGNLINE_EM_DEMO_RESPONDER_PASSWORD on the server, or read it from
        app/backend/var/demo_responder_credentials.txt (created on first start, never committed).
      </Hint>
    </div>
  )
}

export function ResponderDashboardPage() {
  const { auth, onAuthError } = useResponderGuard()
  const navigate = useNavigate()
  const [filter, setFilter] = useState('')
  const [rows, setRows] = useState(null)
  const [stats, setStats] = useState(null)
  const [error, setError] = useState(null)

  const load = useCallback(() => {
    emergencyApi.incidents(filter).then(setRows).catch((e) => setError(onAuthError(e)))
    emergencyApi.analytics().then(setStats).catch(() => {})
  }, [filter, onAuthError])
  useEffect(() => {
    if (!auth) return undefined
    load()
    const t = setInterval(load, 5000)
    return () => clearInterval(t)
  }, [load, auth?.token]) // eslint-disable-line react-hooks/exhaustive-deps

  if (!auth) return <Navigate to="/responder/login" replace />
  const signOut = () => { setResponderAuth(null); navigate('/responder/login') }

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="font-display text-2xl font-extrabold sm:text-3xl">Incoming reports</h1>
          <p className="text-sm text-sl-ink-600">Signed in as {auth.user.display_name} ({auth.user.role.toLowerCase()}). Refreshes every 5 s.</p>
        </div>
        <div className="flex gap-2">
          <Button size="sm" variant="outline" onClick={load}><RefreshCw className="h-4 w-4" aria-hidden="true" /> Refresh</Button>
          <Button size="sm" variant="ghost" onClick={signOut}><LogOut className="h-4 w-4" aria-hidden="true" /> Sign out</Button>
        </div>
      </div>
      <ErrorNote error={error} onClose={() => setError(null)} />
      <div role="tablist" aria-label="Filter by status" className="flex flex-wrap gap-2">
        {FILTERS.map(([id, label]) => (
          <button key={id} role="tab" type="button" aria-selected={filter === id} onClick={() => setFilter(id)}
                  className={`min-h-9 rounded-full border-2 px-4 text-sm font-bold ${filter === id ? 'border-sl-ink bg-sl-purple text-white' : 'border-sl-cream-300 bg-white'}`}>
            {label}
          </button>
        ))}
      </div>
      {!rows ? <p className="text-sm text-sl-ink-600">Loading…</p> : !rows.length ? (
        <p className="rounded-sl border border-dashed border-sl-ink-200 p-6 text-center text-sm text-sl-ink-600">No reports. Shared reports appear here.</p>
      ) : (
        <ul className="space-y-2">
          {rows.map((r) => (
            <li key={r.id}>
              <Link to={`/responder/incidents/${r.id}`} className="flex flex-col gap-1 rounded-sl border-2 border-sl-ink bg-white p-4 hover:bg-sl-purple-50 sm:flex-row sm:items-center sm:justify-between">
                <span>
                  <span className="flex flex-wrap items-center gap-2">
                    <strong>{r.category_label}</strong>
                    {r.priority === 'HIGH' && <span className="rounded-full border border-rose-700 bg-rose-50 px-2 text-xs font-bold text-rose-800">High (user’s answers)</span>}
                    {r.demo && <DemoTag>Demo</DemoTag>}
                    {r.has_location && <MapPin className="h-4 w-4" aria-label="Location shared" />}
                  </span>
                  <span className="block text-sm">{r.summary}</span>
                  {r.communication_needs.length > 0 && <span className="block text-xs text-sl-ink-600">{r.communication_needs.join(' · ')}</span>}
                </span>
                <span className="shrink-0 text-right text-xs">
                  <span className="block font-bold">{r.status_label}</span>
                  <span className="text-sl-ink-600">{new Date(r.created_at).toLocaleString()}</span>
                </span>
              </Link>
            </li>
          ))}
        </ul>
      )}
      {stats && (
        <Section title="Usage (aggregated)">
          <p className="mb-2 text-xs text-sl-ink-600">{stats.note}</p>
          <dl className="grid grid-cols-2 gap-2 text-sm sm:grid-cols-4">
            {[['Sessions', 'SESSION_STARTED'], ['Confirmed', 'USER_CONFIRMED'], ['Delivered', 'SHARE_SUCCEEDED'], ['Failed deliveries', 'SHARE_FAILED']].map(([label, key]) => (
              <div key={key} className="rounded-sl bg-sl-cream-50 p-3"><dt className="text-xs text-sl-ink-600">{label}</dt><dd className="text-xl font-extrabold">{stats.events[key] || 0}</dd></div>
            ))}
          </dl>
        </Section>
      )}
    </div>
  )
}

export function ResponderIncidentPage() {
  const { incidentId } = useParams()
  const { auth, onAuthError } = useResponderGuard()
  const [inc, setInc] = useState(null)
  const [note, setNote] = useState('')
  const [error, setError] = useState(null)
  const [busy, setBusy] = useState(false)
  const [videoUrl, setVideoUrl] = useState(null)

  useEffect(() => {
    if (auth) emergencyApi.incident(incidentId).then(setInc).catch((e) => setError(onAuthError(e)))
  }, [incidentId, auth?.token]) // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => () => { if (videoUrl) URL.revokeObjectURL(videoUrl) }, [videoUrl])
  const showVideo = () => emergencyApi.incidentVideo(incidentId).then((b) => setVideoUrl(URL.createObjectURL(b))).catch((e) => setError(onAuthError(e)))
  if (!auth) return <Navigate to="/responder/login" replace />

  const run = async (fn) => {
    setBusy(true)
    setError(null)
    try { setInc(await fn()) } catch (e) { setError(onAuthError(e)) } finally { setBusy(false) }
  }

  return (
    <div className="space-y-4">
      <Link to="/responder" className="inline-flex items-center gap-1 text-sm font-bold text-sl-ink-600 hover:text-sl-ink">
        <ChevronLeft className="h-4 w-4" aria-hidden="true" /> All reports
      </Link>
      <ErrorNote error={error} onClose={() => setError(null)} />
      {inc && (
        <>
          <div className="flex flex-wrap items-start justify-between gap-3">
            <div>
              <h1 className="font-display text-2xl font-extrabold sm:text-3xl">{inc.category_label}</h1>
              <p className="text-lg">{inc.summary}</p>
              <p className="mt-1 flex flex-wrap items-center gap-2 text-xs text-sl-ink-600">
                {inc.id} · {new Date(inc.created_at).toLocaleString()} {inc.demo && <DemoTag>Demo report</DemoTag>}
              </p>
            </div>
            <div className="flex flex-col items-end gap-2">
              <span className="rounded-full border-2 border-sl-ink bg-white px-3 py-1 text-sm font-bold">{inc.status_label}</span>
              {inc.status === 'SENT_TO_CONFIGURED_DESTINATION' && <Button variant="purple" disabled={busy} onClick={() => run(() => emergencyApi.acknowledge(inc.id))}>Acknowledge</Button>}
              {inc.status === 'ACKNOWLEDGED' && <Button disabled={busy} onClick={() => run(() => emergencyApi.resolve(inc.id))}>Mark resolved</Button>}
            </div>
          </div>
          {inc.communication_needs.length > 0 && (
            <div className="rounded-sl border-2 border-sl-ink bg-sl-sun-100 p-3 text-sm">
              <strong>How to communicate:</strong> {inc.communication_needs.join(' · ')} <ProvenanceTag value="USER_REPORTED" />
            </div>
          )}
          <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_20rem]">
            <div className="space-y-4">
              {FACT_GROUPS.map(([title, kinds]) => {
                const facts = inc.facts.filter((f) => kinds.includes(f.kind))
                if (!facts.length) return null
                return (
                  <Section key={title} title={title}>
                    <ul className="space-y-1.5">
                      {facts.map((f, i) => (
                        <li key={`${f.key}-${i}`} className="flex items-start justify-between gap-3 text-sm">
                          <span>{f.value}</span><ProvenanceTag value={f.provenance} />
                        </li>
                      ))}
                    </ul>
                  </Section>
                )
              })}
              <Section title="Sign sequence">
                <ol className="flex flex-wrap gap-2">
                  {inc.sequence.map((e) => (
                    <li key={e.event_id} className="rounded-sl border border-sl-cream-300 px-2 py-1 text-xs">
                      <strong>{e.sign}</strong> · {confidenceText(e.confidence)}{e.corrected ? ` · corrected from ${e.original_sign}` : ''}
                    </li>
                  ))}
                </ol>
              </Section>
            </div>
            <aside className="space-y-4">
              {inc.emergency_message && (
                <Section title="Gesture report">
                  <pre className="whitespace-pre-wrap font-sans text-sm">{inc.emergency_message}</pre>
                  {inc.emergency_contact && <p className="mt-2 text-xs text-sl-ink-600">Number the user chose to call: {inc.emergency_contact.number} {inc.emergency_contact.service}</p>}
                </Section>
              )}
              {inc.video && (
                <Section title="Gesture video (shared with consent)">
                  {videoUrl
                    ? <video src={videoUrl} controls playsInline className="aspect-[4/3] w-full rounded-sl border-2 border-sl-ink bg-black object-contain" aria-label="Gesture video shared by the user" />
                    : <Button size="sm" variant="outline" onClick={showVideo}>Show video ({inc.video.duration_seconds} s)</Button>}
                  <p className="mt-1 text-xs text-sl-ink-600">Shared {inc.video.sharing_consent_at ? new Date(inc.video.sharing_consent_at).toLocaleString() : ''}. Viewing is recorded in the audit log. Do not copy it elsewhere.</p>
                </Section>
              )}
              <Section title="Location">
                {inc.location ? (
                  <p className="text-sm">
                    {inc.location.text || `${inc.location.lat}, ${inc.location.lon} (±${inc.location.accuracy_m} m)`}{' '}
                    {inc.location.simulated && <DemoTag />}
                  </p>
                ) : <p className="text-sm text-sl-ink-600">Not shared by the user.</p>}
              </Section>
              <Section title="Notes (responders only)">
                <ul className="mb-3 space-y-2 text-sm">
                  {inc.notes.map((n, i) => <li key={i}><p>{n.text}</p><p className="text-xs text-sl-ink-600">{n.by} · {new Date(n.at).toLocaleTimeString()}</p></li>)}
                </ul>
                <form onSubmit={(e) => { e.preventDefault(); if (note.trim()) run(() => emergencyApi.addNote(inc.id, note.trim())).then(() => setNote('')) }} className="space-y-2">
                  <label htmlFor="note" className="sr-only">Add a note</label>
                  <textarea id="note" value={note} onChange={(e) => setNote(e.target.value)} rows={2} maxLength={2000}
                            className="w-full rounded-sl border-2 border-sl-ink p-2 text-sm" placeholder="Add a note" />
                  <Button type="submit" size="sm" variant="outline" disabled={busy || !note.trim()}>Add note</Button>
                </form>
              </Section>
              <Section title="Status"><StatusTimeline history={inc.history} viewer="responder" /></Section>
              <Section title="Delivery">
                <ul className="space-y-1 text-xs">
                  {inc.dispatch_attempts.map((d) => <li key={d.id}>{d.status} · {d.destination}{d.receipt_id ? ` · ${d.receipt_id}` : ''}{d.simulated ? ' (simulated)' : ''}</li>)}
                </ul>
              </Section>
            </aside>
          </div>
        </>
      )}
    </div>
  )
}
