import { useEffect, useRef, useState } from 'react'
import { Link, useLocation } from 'react-router-dom'
import { BookOpenText, Hand, MessageSquareWarning, ShieldCheck, X } from 'lucide-react'
import { EMERGENCY_NUMBER, lastSessionId } from './api'

/**
 * Floating circular SOS button, bottom-right on every page (above the mobile bottom navigation).
 * It only opens a small panel; nothing is started or sent until the user chooses an option.
 */
export default function EmergencySOSButton() {
  const { pathname } = useLocation()
  const [open, setOpen] = useState(false)
  const panelRef = useRef(null)
  const buttonRef = useRef(null)

  useEffect(() => setOpen(false), [pathname])
  useEffect(() => {
    if (!open) return undefined
    panelRef.current?.querySelector('a, button')?.focus()
    const onKey = (e) => { if (e.key === 'Escape') { setOpen(false); buttonRef.current?.focus() } }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [open])

  if (pathname.startsWith('/emergency') || pathname.startsWith('/responder')) return null
  const aboveNav = pathname.startsWith('/app') || pathname.startsWith('/admin')
  const resume = lastSessionId()
  const item = 'flex min-h-12 items-center gap-3 rounded-sl border-2 border-sl-ink bg-white px-4 text-left text-sm font-bold hover:bg-sl-cream-200'

  return (
    <div className={`fixed right-2 sm:right-4 z-50 flex flex-col items-end gap-3 lg:bottom-6 lg:right-6 ${aboveNav ? 'bottom-[4.75rem] sm:bottom-[5.5rem]' : 'bottom-5'}`}
         style={{ marginBottom: 'env(safe-area-inset-bottom)' }}>
      {open && (
        <div ref={panelRef} id="sos-panel" role="dialog" aria-modal="false" aria-labelledby="sos-title"
             className="w-[min(20rem,calc(100vw-2rem))] rounded-sl-lg border-2 border-sl-ink bg-sl-cream-50 p-4 shadow-sl animate-sl-fade-up">
          <div className="flex items-start justify-between gap-2">
            <h2 id="sos-title" className="font-display text-lg font-extrabold text-rose-700">Emergency help</h2>
            <button type="button" onClick={() => setOpen(false)} className="-mr-1 -mt-1 flex h-9 w-9 items-center justify-center rounded-full hover:bg-sl-cream-200">
              <X className="h-5 w-5" aria-hidden="true" /><span className="sr-only">Close</span>
            </button>
          </div>
          <p className="mt-1 text-xs text-sl-ink-600">
            Explain an emergency with signs, buttons or your own gesture. Prototype: in demo mode it contacts no one.
            In a real emergency in India, call <strong className="text-sl-ink">{EMERGENCY_NUMBER}</strong>.
          </p>
          <div className="mt-3 flex flex-col gap-2">
            {resume && (
              <Link to={`/emergency/session/${resume}`} className={`${item} border-rose-700 bg-rose-50`}>
                <MessageSquareWarning className="h-5 w-5 text-rose-700" aria-hidden="true" /> Continue my emergency report
              </Link>
            )}
            <Link to="/emergency" className={`${item} bg-rose-600 text-white hover:bg-rose-700`}>
              <MessageSquareWarning className="h-5 w-5" aria-hidden="true" /> Start emergency communication
            </Link>
            <Link to="/emergency/gestures" className={item}>
              <Hand className="h-5 w-5" aria-hidden="true" /> My emergency gestures
            </Link>
            <Link to="/emergency/about" className={item}>
              <BookOpenText className="h-5 w-5" aria-hidden="true" /> How it works
            </Link>
            <Link to="/responder/login" className={`${item} border-sl-ink-200 text-sl-ink-600`}>
              <ShieldCheck className="h-5 w-5" aria-hidden="true" /> Responder desk
            </Link>
          </div>
        </div>
      )}
      <button
        ref={buttonRef}
        type="button"
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
        aria-controls="sos-panel"
        className="relative flex h-12 w-12 sm:h-16 sm:w-16 items-center justify-center rounded-full border-2 border-sl-ink bg-rose-600 font-display text-sm sm:text-lg font-extrabold tracking-wide text-white shadow-sl transition-transform hover:bg-rose-700 focus-visible:outline focus-visible:outline-4 focus-visible:outline-offset-2 focus-visible:outline-rose-700 active:translate-x-[2px] active:translate-y-[2px] active:shadow-none"
      >
        {!open && <span className="sl-motion absolute inset-0 -z-10 animate-ping rounded-full bg-rose-500/40" aria-hidden="true" />}
        {open ? <X className="h-7 w-7" aria-hidden="true" /> : <span aria-hidden="true">SOS</span>}
        <span className="sr-only">{open ? 'Close emergency help' : 'Emergency help'}</span>
      </button>
    </div>
  )
}
