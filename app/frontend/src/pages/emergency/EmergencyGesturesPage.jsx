import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { ChevronLeft, ChevronRight, Plus, Video } from 'lucide-react'
import Button from '../../components/ui/Button'
import { emergencyApi } from '../../features/emergency/api'
import { ErrorNote, Hint, Section } from '../../features/emergency/components'
import GestureWizard from './GestureWizard'

/** Personal emergency gestures (a mapping to an intent and message, not a trained recogniser) and communication needs. */
export default function EmergencyGesturesPage() {
  const [config, setConfig] = useState(null)
  const [contacts, setContacts] = useState([])
  const [gestures, setGestures] = useState([])
  const [adding, setAdding] = useState(false)
  const [needs, setNeeds] = useState([])
  const [error, setError] = useState(null)
  const [saved, setSaved] = useState(null)

  useEffect(() => {
    emergencyApi.config().then(setConfig).catch((e) => setError(e.message))
    emergencyApi.gestures().then(setGestures).catch((e) => setError(e.message))
    emergencyApi.contacts().then((c) => setContacts(c.contacts)).catch(() => {})
    emergencyApi.preferences().then((p) => setNeeds(p.communication_needs)).catch(() => {})
  }, [])

  const toggleNeed = async (need) => {
    const next = needs.includes(need) ? needs.filter((n) => n !== need) : [...needs, need]
    setNeeds(next)
    try { await emergencyApi.savePreferences({ communication_needs: next }); setSaved('Saved') } catch (err) { setError(err.message) }
  }

  return (
    <div className="space-y-4">
      <div>
        <Link to="/emergency" className="inline-flex items-center gap-1 text-sm font-bold text-sl-ink-600 hover:text-sl-ink">
          <ChevronLeft className="h-4 w-4" aria-hidden="true" /> Emergency
        </Link>
        <h1 className="font-display text-2xl font-extrabold sm:text-3xl">My emergency gestures and needs</h1>
        <p className="text-sm text-sl-ink-600">Saved on this device only (no account needed).</p>
      </div>
      <ErrorNote error={error} onClose={() => setError(null)} />

      <div className="grid gap-4 lg:grid-cols-2">
        <Section title={adding ? 'Add gesture' : 'Your gestures'}
                 aside={!adding && <Button size="sm" onClick={() => setAdding(true)}><Plus className="h-4 w-4" aria-hidden="true" /> Add gesture</Button>}>
          {adding ? (
            <GestureWizard config={config} contacts={contacts} onCancel={() => setAdding(false)}
                           onDone={(g, warning) => { setGestures((gs) => [...gs, g]); setAdding(false); setError(warning) }} />
          ) : (
            <>
              <Hint>
                A gesture is a shortcut you choose, such as “two taps on the chest”, with an optional video of you making it.
                SignLine does not recognise it from the camera yet: in an emergency you open the gesture and use it yourself.
              </Hint>
              {!gestures.length ? <p className="mt-3 text-sm text-sl-ink-600">None yet.</p> : (
                <ul className="mt-3 space-y-2">
                  {gestures.map((g) => (
                    <li key={g.id}>
                      <Link to={`/emergency/gestures/${g.id}`}
                            className="flex items-start justify-between gap-2 rounded-sl border-2 border-sl-ink bg-white p-3 hover:bg-sl-cream-100 focus-visible:outline focus-visible:outline-4 focus-visible:outline-sl-green-700">
                        <div className="min-w-0">
                          <p className="font-bold">{g.name} {!g.enabled && <span className="text-xs font-normal text-sl-ink-600">(off)</span>}</p>
                          {g.description && <p className="text-xs text-sl-ink-600">{g.description}</p>}
                          <p className="mt-1 text-sm">{g.category_label}: “{g.message}”</p>
                          <p className="mt-1 flex flex-wrap items-center gap-2 text-xs text-sl-ink-600">
                            {g.has_video && <span className="inline-flex items-center gap-1"><Video className="h-3.5 w-3.5" aria-hidden="true" /> Video {g.video_duration_seconds} s</span>}
                            <span>Calls {g.emergency_contact ? `${g.emergency_contact.number} ${g.emergency_contact.service}` : '112'}</span>
                          </p>
                        </div>
                        <ChevronRight className="mt-1 h-5 w-5 shrink-0" aria-hidden="true" />
                        <span className="sr-only">Open {g.name}</span>
                      </Link>
                    </li>
                  ))}
                </ul>
              )}
            </>
          )}
        </Section>

        <Section title="How responders should communicate with me" aside={saved && <span className="text-xs text-sl-green-700">{saved}</span>}>
          <p className="mb-2 text-sm text-sl-ink-600">Added to every report you confirm. Only what you choose here.</p>
          <div className="flex flex-col gap-2">
            {(config?.communication_needs || []).map((n) => (
              <label key={n} className="flex min-h-11 items-center gap-3 rounded-sl border border-sl-cream-300 px-3 text-sm">
                <input type="checkbox" className="h-5 w-5" checked={needs.includes(n)} onChange={() => toggleNeed(n)} /> {n}
              </label>
            ))}
          </div>
        </Section>
      </div>
    </div>
  )
}
