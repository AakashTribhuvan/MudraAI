import { Link } from 'react-router-dom'
import { Hand, ShieldCheck, ArrowRight } from 'lucide-react'
import AuthLayout, { DemoNotice } from './AuthLayout'

const ROLES = [
  {
    role: 'user', icon: Hand, title: 'I want to sign and learn', color: 'bg-sl-green-50',
    text: 'Communicate with your webcam, learn ISL words step by step and track your progress.',
  },
  {
    role: 'admin', icon: ShieldCheck, title: 'I manage SignLine', color: 'bg-sl-purple-50',
    text: 'For Google accounts approved for the admin workspace.',
  },
]

export default function WelcomePage() {
  return (
    <AuthLayout title="Welcome to SignLine" subtitle="Choose how you want to continue.">
      <ul className="space-y-4">
        {ROLES.map(({ role, icon: Icon, title, text, color }) => (
          <li key={role}>
            <Link
              to={`/signin?role=${role}`}
              className={`group flex items-start gap-4 rounded-sl-lg border-2 border-sl-ink ${color} p-5 shadow-sl-sm transition-transform hover:-translate-y-0.5`}
            >
              <span className="mt-0.5 flex h-11 w-11 shrink-0 items-center justify-center rounded-full border-2 border-sl-ink bg-white">
                <Icon className="h-5 w-5" aria-hidden="true" />
              </span>
              <span className="flex-1">
                <span className="block font-display text-lg font-bold text-sl-ink">{title}</span>
                <span className="mt-1 block text-sm text-sl-ink-600">{text}</span>
              </span>
              <ArrowRight className="mt-3 h-5 w-5 shrink-0 transition-transform group-hover:translate-x-1" aria-hidden="true" />
            </Link>
          </li>
        ))}
      </ul>
      <p className="text-sm text-sl-ink-600">
        New here?{' '}
        <Link to="/register" className="font-bold text-sl-ink underline decoration-sl-green decoration-2 underline-offset-4">
          Create a learner profile
        </Link>
      </p>
      <DemoNotice />
    </AuthLayout>
  )
}
