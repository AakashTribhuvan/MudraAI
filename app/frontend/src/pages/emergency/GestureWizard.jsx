import { useState } from 'react'
import Button from '../../components/ui/Button'
import { emergencyApi } from '../../features/emergency/api'
import { ErrorNote, Hint } from '../../features/emergency/components'
import GestureRecorder from './GestureRecorder'

export const intentLabel = (id) => id.replaceAll('_', ' ').toLowerCase().replace(/^./, (c) => c.toUpperCase())
const input = 'mt-1 min-h-11 w-full rounded-sl border-2 border-sl-ink bg-white px-3 font-normal'

/** The details of a gesture (shared by the wizard and the edit form on the detail page). */
export function GestureFields({ form, setForm, config, contacts }) {
  const set = (k) => (e) => setForm((f) => ({ ...f, [k]: e.target.type === 'checkbox' ? e.target.checked : e.target.value }))
  return (
    <div className="space-y-3">
      <label className="block text-sm font-bold">Gesture name
        <input required maxLength={80} value={form.name} onChange={set('name')} placeholder="My Medical SOS" className={input} />
      </label>
      <label className="block text-sm font-bold">How you make it, and what it means
        <input maxLength={500} value={form.description} onChange={set('description')} placeholder="Closed fist on the chest, twice" className={input} />
      </label>
      <label className="block text-sm font-bold">Emergency category
        <select value={form.intent} onChange={set('intent')} className={`${input} px-2`}>
          {(config?.intents || []).map((i) => <option key={i.id} value={i.id}>{intentLabel(i.id)} ({i.category_label})</option>)}
        </select>
      </label>
      <label className="block text-sm font-bold">Emergency message
        <textarea required maxLength={500} rows={3} value={form.message} onChange={set('message')}
                  placeholder="I need urgent medical help. Please communicate with me using text."
                  className="mt-1 w-full rounded-sl border-2 border-sl-ink bg-white p-2 font-normal" />
      </label>
      <label className="block text-sm font-bold">Emergency service to call
        <select value={form.emergency_contact_id || ''} onChange={set('emergency_contact_id')} className={`${input} px-2`}>
          <option value="">112 Unified emergency response (default)</option>
          {(contacts || []).filter((c) => c.id !== 'in_112').map((c) => (
            <option key={c.id} value={c.id}>{c.number} {c.service}{c.availability === 'varies' ? ' (varies by state)' : ''}</option>
          ))}
        </select>
        <span className="mt-1 block text-xs font-normal text-sl-ink-600">Numbers are not verified yet; you can change the service before calling.</span>
      </label>
      <label className="flex items-center gap-2 text-sm"><input type="checkbox" checked={form.enabled} onChange={set('enabled')} className="h-5 w-5" /> Turned on</label>
    </div>
  )
}

export const EMPTY_GESTURE = { name: '', description: '', intent: 'REQUEST_MEDICAL_HELP', message: '', emergency_contact_id: '', enabled: true, require_confirmation: true, cooldown_s: 10 }
export const toBody = (form) => ({ ...form, emergency_contact_id: form.emergency_contact_id || null })

/** Add Gesture: 1 details, 2 record a demonstration (optional), 3 consent and save. */
export default function GestureWizard({ config, contacts, onDone, onCancel }) {
  const [step, setStep] = useState(1)
  const [form, setForm] = useState(EMPTY_GESTURE)
  const [video, setVideo] = useState(null) // {blob, seconds}
  const [consent, setConsent] = useState(false)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)

  const save = async () => {
    setBusy(true)
    setError(null)
    let gesture
    try {
      gesture = await emergencyApi.createGesture(toBody(form))
    } catch (err) { setBusy(false); return setError(err.message) }
    if (video && consent) {
      try {
        gesture = await emergencyApi.uploadGestureVideo(gesture.id, video.blob, video.seconds)
      } catch (err) {
        setBusy(false)
        onDone(gesture, `The gesture is saved, but the video was not: ${err.message} You can record it again on the gesture's page.`)
        return
      }
    }
    setBusy(false)
    onDone(gesture, null)
  }

  return (
    <div className="space-y-3">
      <ol className="flex gap-2 text-xs font-bold" aria-label="Steps">
        {['Details', 'Record', 'Save'].map((s, i) => (
          <li key={s} aria-current={step === i + 1 ? 'step' : undefined}
              className={`rounded-full border-2 px-3 py-1 ${step === i + 1 ? 'border-sl-ink bg-sl-green' : 'border-sl-cream-300 text-sl-ink-600'}`}>{i + 1}. {s}</li>
        ))}
      </ol>
      <ErrorNote error={error} onClose={() => setError(null)} />
      {step === 1 && (
        <form onSubmit={(e) => { e.preventDefault(); setStep(2) }} className="space-y-3">
          <GestureFields form={form} setForm={setForm} config={config} contacts={contacts} />
          <div className="flex gap-2"><Button type="submit">Next: record</Button><Button variant="ghost" onClick={onCancel}>Cancel</Button></div>
        </form>
      )}
      {step === 2 && (
        <GestureRecorder
          onRecorded={(blob, seconds) => { setVideo({ blob, seconds }); setStep(3) }}
          onSkip={() => { setVideo(null); setStep(3) }}
        />
      )}
      {step === 3 && (
        <div className="space-y-3">
          <dl className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-1 text-sm">
            <dt className="font-bold">Name</dt><dd>{form.name}</dd>
            <dt className="font-bold">Message</dt><dd>“{form.message}”</dd>
            <dt className="font-bold">Video</dt><dd>{video ? `${video.seconds} s recording` : 'None (you can add one later)'}</dd>
          </dl>
          {video && (
            <label className="flex items-start gap-2 rounded-sl border-2 border-sl-ink bg-white p-3 text-sm">
              <input type="checkbox" className="mt-0.5 h-5 w-5" checked={consent} onChange={(e) => setConsent(e.target.checked)} />
              <span>I agree that SignLine stores this video for my gesture on this device's account. I can delete it at any time.
                It is not shared with anyone unless I choose to share a report, and it is never used to train a model.</span>
            </label>
          )}
          {video && !consent && <Hint>Without your agreement the gesture is saved without the video.</Hint>}
          <div className="flex flex-wrap gap-2">
            <Button onClick={save} disabled={busy}>{busy ? 'Saving…' : video && consent ? 'Save gesture and video' : 'Save gesture'}</Button>
            <Button variant="outline" onClick={() => setStep(2)} disabled={busy}>Back</Button>
            <Button variant="ghost" onClick={onCancel} disabled={busy}>Cancel</Button>
          </div>
        </div>
      )}
    </div>
  )
}
