import { useEffect, useMemo, useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import { ChevronLeft, FlaskConical, Pencil, Phone, RotateCcw, Share2, Siren, Trash2 } from 'lucide-react'
import Button from '../../components/ui/Button'
import { emergencyApi } from '../../features/emergency/api'
import { ErrorNote, Hint, ProvenanceTag, Section } from '../../features/emergency/components'
import { SHARE_STATUS, shareReport, shareSupport, telLink } from '../../features/emergency/dial'
import GestureRecorder from './GestureRecorder'
import { GestureFields, toBody } from './GestureWizard'

/** One saved gesture: everything about it, its video, and the emergency assistance flow built from it. */
export default function EmergencyGestureDetailPage() {
  const { gestureId } = useParams()
  const navigate = useNavigate()
  const [config, setConfig] = useState(null)
  const [contacts, setContacts] = useState([])
  const [g, setG] = useState(null)
  const [videoBlob, setVideoBlob] = useState(null)
  const [mode, setMode] = useState('view') // view | edit | retake | assist
  const [form, setForm] = useState(null)
  const [retakeConsent, setRetakeConsent] = useState(false)
  const [pending, setPending] = useState(null) // retaken video waiting for consent
  const [tested, setTested] = useState(null)
  const [error, setError] = useState(null)
  const [notFound, setNotFound] = useState(false)
  const videoUrl = useMemo(() => (videoBlob ? URL.createObjectURL(videoBlob) : null), [videoBlob])
  useEffect(() => () => { if (videoUrl) URL.revokeObjectURL(videoUrl) }, [videoUrl])

  const loadVideo = (gesture) => {
    setVideoBlob(null)
    if (gesture?.has_video) emergencyApi.gestureVideo(gesture.id).then(setVideoBlob).catch((e) => setError(e.message))
  }
  useEffect(() => {
    emergencyApi.config().then(setConfig).catch(() => {})
    emergencyApi.contacts().then((c) => setContacts(c.contacts)).catch(() => {})
    emergencyApi.gestures().then((list) => {
      const found = list.find((x) => x.id === gestureId)
      if (!found) return setNotFound(true)
      setG(found)
      loadVideo(found)
    }).catch((e) => setError(e.message))
  }, [gestureId])

  if (notFound) {
    return <div className="space-y-3"><p>This gesture does not exist on this device.</p><Link className="font-bold underline" to="/emergency/gestures">Back to my gestures</Link></div>
  }
  if (!g) return <p role="status">Loading…</p>

  const saveEdit = async (e) => {
    e.preventDefault()
    try { setG(await emergencyApi.updateGesture(g.id, toBody(form))); setMode('view') } catch (err) { setError(err.message) }
  }
  const saveRetake = async () => {
    try {
      const updated = await emergencyApi.uploadGestureVideo(g.id, pending.blob, pending.seconds)
      setG(updated); setVideoBlob(pending.blob); setPending(null); setRetakeConsent(false); setMode('view')
    } catch (err) { setError(err.message) }
  }
  const removeVideo = async () => {
    if (!window.confirm('Delete the recorded video of this gesture?')) return
    try { setG(await emergencyApi.deleteGestureVideo(g.id)); setVideoBlob(null) } catch (err) { setError(err.message) }
  }
  const removeGesture = async () => {
    if (!window.confirm(`Delete the gesture “${g.name}” and its video?`)) return
    try { await emergencyApi.deleteGesture(g.id); navigate('/emergency/gestures') } catch (err) { setError(err.message) }
  }

  return (
    <div className="space-y-4">
      <div>
        <Link to="/emergency/gestures" className="inline-flex items-center gap-1 text-sm font-bold text-sl-ink-600 hover:text-sl-ink">
          <ChevronLeft className="h-4 w-4" aria-hidden="true" /> My gestures
        </Link>
        <h1 className="font-display text-2xl font-extrabold sm:text-3xl">{g.name}</h1>
        <p className="text-sm text-sl-ink-600">{g.enabled ? 'Turned on' : 'Turned off'} · {g.category_label}</p>
      </div>
      <ErrorNote error={error} onClose={() => setError(null)} />

      {mode === 'assist' ? (
        <AssistPanel gesture={g} contacts={contacts} videoBlob={videoBlob} videoUrl={videoUrl} onClose={() => setMode('view')} onEdit={() => { setForm({ ...g, emergency_contact_id: g.emergency_contact_id || '' }); setMode('edit') }} />
      ) : (
        <div className="grid gap-4 lg:grid-cols-2">
          <Section title="Gesture">
            {mode === 'edit' ? (
              <form onSubmit={saveEdit} className="space-y-3">
                <GestureFields form={form} setForm={setForm} config={config} contacts={contacts} />
                <div className="flex gap-2"><Button type="submit">Save changes</Button><Button variant="ghost" onClick={() => setMode('view')}>Cancel</Button></div>
              </form>
            ) : (
              <dl className="space-y-2 text-sm">
                <div><dt className="font-bold">How you make it, and what it means</dt><dd>{g.description || '—'}</dd></div>
                <div><dt className="font-bold">Emergency category</dt><dd>{g.category_label}</dd></div>
                <div><dt className="font-bold">Emergency message</dt><dd>“{g.message}”</dd></div>
                <div><dt className="font-bold">Emergency service</dt><dd>{g.emergency_contact ? `${g.emergency_contact.number} ${g.emergency_contact.service}` : '112 Unified emergency response (default)'}</dd></div>
                <div><dt className="font-bold">Saved</dt><dd>{new Date(g.created_at).toLocaleString()}{g.updated_at !== g.created_at ? ` · changed ${new Date(g.updated_at).toLocaleString()}` : ''}</dd></div>
              </dl>
            )}
            {mode === 'view' && (
              <div className="mt-4 flex flex-wrap gap-2">
                <Button size="sm" variant="outline" onClick={() => { setForm({ ...g, emergency_contact_id: g.emergency_contact_id || '' }); setMode('edit') }}><Pencil className="h-4 w-4" aria-hidden="true" /> Edit</Button>
                <Button size="sm" variant="outline" onClick={() => { setTested(null); emergencyApi.testGesture(g.id).then(setTested).catch((e) => setError(e.message)) }}><FlaskConical className="h-4 w-4" aria-hidden="true" /> Test gesture</Button>
                <Button size="sm" variant="ghost" onClick={removeGesture}><Trash2 className="h-4 w-4 text-rose-700" aria-hidden="true" /> Delete</Button>
              </div>
            )}
            {tested && mode === 'view' && (
              <div className="mt-3 flex items-start justify-between gap-2 rounded-sl bg-sl-sky-100 p-2 text-xs">
                <span>Test only, nothing sent. A report would say: <strong>{tested.would_report.category_label}</strong>, “{tested.would_report.message}”</span>
                <ProvenanceTag value={tested.would_report.provenance} />
              </div>
            )}
          </Section>

          <Section title="Recorded video" aside={g.has_video && <span className="text-xs text-sl-ink-600">{g.video_duration_seconds} s · consent given {g.recording_consent_at ? new Date(g.recording_consent_at).toLocaleDateString() : ''}</span>}>
            {mode === 'retake' ? (
              pending ? (
                <div className="space-y-3">
                  <label className="flex items-start gap-2 rounded-sl border-2 border-sl-ink bg-white p-3 text-sm">
                    <input type="checkbox" className="mt-0.5 h-5 w-5" checked={retakeConsent} onChange={(e) => setRetakeConsent(e.target.checked)} />
                    <span>I agree that SignLine stores this video for my gesture (it replaces the old one). It is never used to train a model.</span>
                  </label>
                  <div className="flex gap-2">
                    <Button onClick={saveRetake} disabled={!retakeConsent}>Save new video</Button>
                    <Button variant="ghost" onClick={() => { setPending(null); setMode('view') }}>Cancel</Button>
                  </div>
                </div>
              ) : (
                <GestureRecorder keepLabel="Keep this recording" onRecorded={(blob, seconds) => setPending({ blob, seconds })} onSkip={() => setMode('view')} />
              )
            ) : (
              <>
                {videoUrl
                  ? <video src={videoUrl} controls playsInline className="aspect-[4/3] w-full rounded-sl border-2 border-sl-ink bg-black object-contain" aria-label={`Video of the gesture ${g.name}`} />
                  : <p className="text-sm text-sl-ink-600">{g.has_video ? 'Loading video…' : 'No video recorded.'}</p>}
                <div className="mt-3 flex flex-wrap gap-2">
                  <Button size="sm" variant="outline" onClick={() => setMode('retake')}><RotateCcw className="h-4 w-4" aria-hidden="true" /> {g.has_video ? 'Retake video' : 'Record video'}</Button>
                  {g.has_video && <Button size="sm" variant="ghost" onClick={removeVideo}><Trash2 className="h-4 w-4 text-rose-700" aria-hidden="true" /> Delete video</Button>}
                </div>
              </>
            )}
          </Section>

          <div className="lg:col-span-2">
            <Button className="w-full sm:w-auto" onClick={() => setMode('assist')} disabled={!g.enabled}>
              <Siren className="h-5 w-5" aria-hidden="true" /> Request emergency assistance
            </Button>
            {!g.enabled && <p className="mt-1 text-xs text-sl-ink-600">This gesture is turned off. Turn it on with Edit to use it.</p>}
          </div>
        </div>
      )}
    </div>
  )
}

/**
 * Send the report to the SignLine responder desk (demo) through the normal flow: it is prepared and confirmed
 * here, then shared on the session page, which shows delivered / acknowledged / resolved or delivery failed.
 */
function DeskPanel({ gesture, contactId, locationText }) {
  const navigate = useNavigate()
  const [withVideo, setWithVideo] = useState(false)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)
  const prepare = async () => {
    setBusy(true)
    setError(null)
    try {
      const body = { contact_id: contactId, include_video: withVideo, sharing_consent: withVideo }
      if (locationText.trim()) body.location = { source: 'MANUAL', text: locationText.trim() }
      const data = await emergencyApi.requestAssistance(gesture.id, body)
      navigate(`/emergency/session/${data.session.id}`)
    } catch (err) { setError(err.message); setBusy(false) }
  }
  return (
    <Section title="5. Send to the SignLine responder desk (demo)">
      <ErrorNote error={error} onClose={() => setError(null)} />
      <p className="text-sm">The report is prepared and confirmed, then you share it on the next screen. It reaches only the SignLine demo responder desk inside this app, not a real emergency service.</p>
      {gesture.has_video && (
        <label className="mt-2 flex items-start gap-2 text-sm">
          <input type="checkbox" className="mt-0.5 h-5 w-5" checked={withVideo} onChange={(e) => setWithVideo(e.target.checked)} />
          <span>Attach my gesture video. I agree that responders at the desk can watch it with this report (a separate copy, deleted with the report).</span>
        </label>
      )}
      <Button className="mt-3" variant="outline" disabled={busy} onClick={prepare}>{busy ? 'Preparing…' : 'Prepare the report for the desk'}</Button>
    </Section>
  )
}

/** Review screen: destination, message, video, then Call (native dialer) and Share (separate, with consent). */
function AssistPanel({ gesture, contacts, videoBlob, videoUrl, onClose, onEdit }) {
  const [contactId, setContactId] = useState(gesture.emergency_contact_id || 'in_112')
  const [locationText, setLocationText] = useState('')
  const [report, setReport] = useState(null)
  const [confirmCall, setConfirmCall] = useState(false)
  const [shareVideo, setShareVideo] = useState(false)
  const [shareState, setShareState] = useState('idle')
  const [error, setError] = useState(null)

  useEffect(() => {
    emergencyApi.gestureReport(gesture.id, { contact_id: contactId, location_text: locationText || null })
      .then((r) => { setReport(r); setShareState('prepared') }).catch((e) => setError(e.message))
  }, [gesture.id, contactId, locationText])

  const shareMyLocation = () => {
    if (!navigator.geolocation) return setError('This browser cannot share a location. Type a place instead.')
    navigator.geolocation.getCurrentPosition(
      (p) => setLocationText(`${p.coords.latitude.toFixed(5)}, ${p.coords.longitude.toFixed(5)} (±${Math.round(p.coords.accuracy)} m, from this device)`),
      () => setError('Location permission was not given. You can type a place instead.'),
      { enableHighAccuracy: true, timeout: 10000 },
    )
  }
  const file = useMemo(() => (videoBlob ? new File([videoBlob], `gesture.${videoBlob.type.includes('mp4') ? 'mp4' : 'webm'}`, { type: videoBlob.type }) : null), [videoBlob])
  const canFiles = file ? shareSupport(typeof navigator !== 'undefined' ? navigator : null, file) === 'files' : false
  const link = report ? telLink(report.contact.number) : null

  return (
    <div className="space-y-4">
      <ErrorNote error={error} onClose={() => setError(null)} />
      <Section title="1. Who to call">
        <label className="block text-sm font-bold">Emergency service
          <select value={contactId} onChange={(e) => setContactId(e.target.value)} className="mt-1 min-h-11 w-full rounded-sl border-2 border-sl-ink bg-white px-2 font-normal">
            {contacts.map((c) => <option key={c.id} value={c.id}>{c.number} {c.service}{c.availability === 'varies' ? ' (varies by state)' : ''}</option>)}
          </select>
        </label>
        {report && (
          <p className="mt-2 text-sm">
            <strong className="font-display text-2xl">{report.contact.number}</strong> {report.contact.service}. {report.contact.notes}{' '}
            <span className="text-xs text-sl-ink-600">{report.contact.verified ? `Verified ${report.contact.checked_on}` : 'Not verified yet: check that this number fits your situation and state.'}</span>
          </p>
        )}
      </Section>

      <Section title="2. Location (optional)">
        <div className="flex flex-wrap gap-2">
          <input value={locationText} onChange={(e) => setLocationText(e.target.value)} maxLength={300} placeholder="Type a place, e.g. Platform 2, Pune station"
                 className="min-h-11 min-w-0 flex-1 rounded-sl border-2 border-sl-ink bg-white px-3 text-sm" aria-label="Location" />
          <Button size="sm" variant="outline" onClick={shareMyLocation}>Use my location</Button>
        </div>
        <p className="mt-1 text-xs text-sl-ink-600">Only added if you type it or allow location. Never guessed.</p>
      </Section>

      <Section title="3. Review the message">
        {report ? (
          <ul className="space-y-1 text-sm">
            {report.lines.map((l, i) => (
              <li key={i} className="flex items-start justify-between gap-2">
                <span>{l.label ? <><strong>{l.label}:</strong> {l.text}</> : <strong>{l.text}</strong>}</span>
                <ProvenanceTag value={l.provenance} />
              </li>
            ))}
          </ul>
        ) : <p role="status">Preparing…</p>}
        {videoUrl && <video src={videoUrl} controls playsInline className="mt-3 aspect-[4/3] w-full max-w-md rounded-sl border-2 border-sl-ink bg-black object-contain" aria-label="Recorded gesture video" />}
        <div className="mt-3 flex flex-wrap gap-2">
          <Button size="sm" variant="outline" onClick={onEdit}><Pencil className="h-4 w-4" aria-hidden="true" /> Edit details</Button>
          <Button size="sm" variant="ghost" onClick={onClose}>Cancel</Button>
        </div>
      </Section>

      <Section title="4. Call">
        <Hint>{report?.notice}</Hint>
        {!confirmCall ? (
          <Button className="mt-3 w-full bg-rose-600 text-white hover:bg-rose-700 sm:w-auto" onClick={() => setConfirmCall(true)} disabled={!link}>
            <Phone className="h-5 w-5" aria-hidden="true" /> Call emergency number
          </Button>
        ) : (
          <div role="alertdialog" aria-label="Confirm the call" className="mt-3 space-y-2 rounded-sl border-2 border-rose-600 bg-rose-50 p-3">
            <p className="text-sm">Open your phone's dialer with <strong>{report.contact.number}</strong> ({report.contact.service})? You press call yourself.</p>
            <div className="flex flex-wrap gap-2">
              <a href={link} className="inline-flex min-h-11 items-center gap-2 rounded-full border-2 border-sl-ink bg-rose-600 px-5 font-bold text-white hover:bg-rose-700">
                <Phone className="h-5 w-5" aria-hidden="true" /> Open dialer: {report.contact.number}
              </a>
              <Button variant="ghost" onClick={() => setConfirmCall(false)}>Not now</Button>
            </div>
            <p className="text-xs text-sl-ink-600">On a computer without a phone app, nothing may happen: dial {report.contact.number} from a phone.</p>
          </div>
        )}
      </Section>

      <DeskPanel gesture={gesture} contactId={contactId} locationText={locationText} />

      <Section title="6. Share the report yourself (separate from the call)">
        {file && (
          <label className="flex items-start gap-2 text-sm">
            <input type="checkbox" className="mt-0.5 h-5 w-5" checked={shareVideo} onChange={(e) => setShareVideo(e.target.checked)} disabled={!canFiles} />
            <span>Attach the gesture video. {canFiles ? 'Only if you agree: the people you share with can see it.' : 'This device cannot attach videos to a share; the text is shared without it.'}</span>
          </label>
        )}
        <Button className="mt-3" variant="outline" disabled={!report}
                onClick={async () => setShareState(await shareReport({ nav: navigator, text: report.text, file, withVideo: shareVideo && canFiles }))}>
          <Share2 className="h-4 w-4" aria-hidden="true" /> Share emergency report
        </Button>
        {SHARE_STATUS[shareState] && <p role="status" className="mt-2 text-sm font-bold">{SHARE_STATUS[shareState]}</p>}
      </Section>
    </div>
  )
}
