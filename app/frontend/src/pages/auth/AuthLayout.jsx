import { Link } from 'react-router-dom'
import Logo from '../../components/brand/Logo'
import { DoodleArrow, DoodleSparkle } from '../../components/brand/Doodles'
import { TAGLINE } from '../../components/brand/tagline'

export function DemoNotice() {
  return (
    <p className="rounded-sl border-2 border-dashed border-sl-sky-700/40 bg-sl-sky-100 px-4 py-3 text-sm text-sl-ink-700">
      <strong>Judge preview.</strong> Google verifies your identity. Learner access is open; admin access is granted
      only to email addresses configured by the SignLine team. SignLine requests your name and email address to create
      a session; it does not request access to Google Drive, contacts, or other Google data.
    </p>
  )
}

export default function AuthLayout({ title, subtitle, children }) {
  return (
    <div className="min-h-screen bg-sl-cream">
      <header className="mx-auto flex max-w-6xl items-center justify-between px-4 py-5 sm:px-6">
        <Link to="/welcome" aria-label="SignLine home"><Logo size={40} /></Link>
      </header>
      <main className="mx-auto grid max-w-6xl grid-cols-1 gap-10 px-4 pb-16 sm:px-6 lg:grid-cols-[1.1fr_1fr] lg:items-center">
        <section className="relative hidden lg:block">
          <DoodleSparkle className="absolute -left-2 -top-8 text-sl-purple" size={28} />
          <h1 className="font-display text-5xl font-extrabold leading-[1.05] tracking-tight text-sl-ink xl:text-6xl">
            Your signs,<br />
            <span className="relative inline-block">
              <span className="absolute -inset-x-2 bottom-1 top-1/2 -z-0 rounded-md bg-sl-sun" aria-hidden="true" />
              <span className="relative">understood.</span>
            </span>
          </h1>
          <p className="mt-6 max-w-md text-lg text-sl-ink-600">
            Sign a word in front of your webcam and SignLine turns it into text and speech. Learn new signs step by step.
          </p>
          <p className="mt-6 font-display text-2xl font-bold leading-snug text-sl-ink">
            {TAGLINE[0]}<br /><span className="text-sl-purple-700">{TAGLINE[1]}</span>
          </p>
          <DoodleArrow className="mt-8 text-sl-ink" width={110} />
        </section>
        <section className="w-full min-w-0 max-w-md justify-self-center lg:justify-self-end">
          <p className="mb-2 font-display text-base font-bold text-sl-ink-600 lg:hidden">{TAGLINE.join(' ')}</p>
          <h2 className="font-display text-3xl font-extrabold tracking-tight text-sl-ink">{title}</h2>
          {subtitle && <p className="mt-2 text-sl-ink-600">{subtitle}</p>}
          <div className="mt-6 space-y-5">{children}</div>
        </section>
      </main>
      <footer className="mx-auto flex max-w-6xl flex-wrap gap-x-4 gap-y-1 px-4 pb-8 text-xs text-sl-ink-600 sm:px-6">
        {[['terms', 'Terms'], ['privacy', 'Privacy'], ['accessibility', 'Accessibility'], ['cookies', 'Cookies'], ['contact', 'Contact']].map(([d, t]) => (
          <Link key={d} to={`/legal/${d}`} className="underline underline-offset-4">{t}</Link>
        ))}
      </footer>
    </div>
  )
}
