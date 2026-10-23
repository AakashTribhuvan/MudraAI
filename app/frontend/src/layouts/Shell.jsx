import { useState } from 'react'
import { NavLink, Outlet, useNavigate } from 'react-router-dom'
import { Accessibility, LogOut, Menu, X } from 'lucide-react'
import Logo from '../components/brand/Logo'
import { useAccessibility } from '../context/AccessibilityContext'
import { useSession } from '../context/SessionContext'

/**
 * Shared layout for the User and Admin workspaces: sidebar on desktop; on mobile a top bar with a
 * menu for secondary actions plus a bottom bar with the primary destinations (`primary` items).
 */
export default function Shell({ primary, secondary = [], accent = 'orange', homeLabel }) {
  const { session, signOut } = useSession()
  const { setIsModalOpen } = useAccessibility()
  const navigate = useNavigate()
  const [menuOpen, setMenuOpen] = useState(false)
  const activeCls = accent === 'purple' ? 'bg-sl-purple text-white border-sl-ink' : 'bg-sl-green text-sl-ink border-sl-ink'

  const leave = () => { signOut(); navigate('/welcome', { replace: true }) }
  const linkCls = ({ isActive }) =>
    `flex min-h-11 items-center gap-3 rounded-full border-2 px-4 text-sm font-bold transition-colors ${isActive ? `${activeCls} shadow-sl-sm` : 'border-transparent text-sl-ink hover:bg-sl-cream-200'}`

  const extras = (
    <>
      {secondary.map(({ to, label, icon: Icon }) => (
        <NavLink key={to} to={to} className={linkCls} onClick={() => setMenuOpen(false)}>
          <Icon className="h-5 w-5" aria-hidden="true" /> {label}
        </NavLink>
      ))}
      <button type="button" className={linkCls({ isActive: false })} onClick={() => { setMenuOpen(false); setIsModalOpen(true) }}>
        <Accessibility className="h-5 w-5" aria-hidden="true" /> Accessibility
      </button>
      <button type="button" className={linkCls({ isActive: false })} onClick={leave}>
        <LogOut className="h-5 w-5" aria-hidden="true" /> Sign out
      </button>
    </>
  )

  return (
    <div className="min-h-screen bg-sl-cream">
      <a href="#main" className="sr-only focus:not-sr-only focus:fixed focus:left-3 focus:top-3 focus:z-[60] focus:rounded-full focus:bg-white focus:px-4 focus:py-2">
        Skip to content
      </a>

      {/* Desktop sidebar */}
      <aside className="fixed inset-y-0 left-0 z-40 hidden w-64 flex-col border-r-2 border-sl-ink bg-sl-cream-50 px-4 py-6 lg:flex">
        <NavLink to={primary[0].to} className="px-2" aria-label={`SignLine ${homeLabel}`}><Logo size={40} /></NavLink>
        <nav aria-label="Main" className="mt-8 flex flex-col gap-1.5">
          {primary.map(({ to, label, icon: Icon }) => (
            <NavLink key={to} to={to} end={to === '/admin'} className={linkCls}>
              <Icon className="h-5 w-5" aria-hidden="true" /> {label}
            </NavLink>
          ))}
        </nav>
        <div className="mt-auto flex flex-col gap-1.5 border-t-2 border-dashed border-sl-cream-300 pt-4">
          {extras}
          <div className="mt-2 rounded-sl border border-sl-cream-300 bg-white px-3 py-2 text-xs text-sl-ink-600">
            <div className="flex items-center justify-between gap-2">
              <span className="truncate font-bold text-sl-ink">{session?.name}</span>
              <span className="rounded-full border border-sl-ink/20 px-2 py-1 text-[10px] font-bold uppercase">{session?.role}</span>
            </div>
          </div>
          <p className="px-2 text-[11px] text-sl-ink-600">
            <NavLink to="/legal/privacy" className="underline underline-offset-2">Privacy</NavLink> ·{' '}
            <NavLink to="/legal/terms" className="underline underline-offset-2">Terms</NavLink> ·{' '}
            <NavLink to="/legal/accessibility" className="underline underline-offset-2">Accessibility</NavLink>
          </p>
        </div>
      </aside>

      {/* Mobile top bar */}
      <header className="sticky top-0 z-40 flex items-center justify-between border-b-2 border-sl-ink bg-sl-cream-50 px-4 py-2.5 lg:hidden">
        <NavLink to={primary[0].to} aria-label={`SignLine ${homeLabel}`}><Logo size={32} /></NavLink>
        <button
          type="button"
          onClick={() => setMenuOpen((o) => !o)}
          aria-expanded={menuOpen}
          aria-controls="mobile-menu"
          className="flex h-11 w-11 items-center justify-center rounded-full border-2 border-sl-ink bg-white"
        >
          {menuOpen ? <X className="h-5 w-5" aria-hidden="true" /> : <Menu className="h-5 w-5" aria-hidden="true" />}
          <span className="sr-only">{menuOpen ? 'Close menu' : 'Open menu'}</span>
        </button>
      </header>
      {menuOpen && (
        <div id="mobile-menu" className="fixed inset-x-0 top-[58px] z-40 flex flex-col gap-1.5 border-b-2 border-sl-ink bg-sl-cream-50 p-4 lg:hidden">
          {extras}
        </div>
      )}

      <main id="main" className="lg:pl-64">
        <Outlet />
      </main>

      {/* Mobile bottom navigation */}
      <nav aria-label="Main" className="fixed inset-x-0 bottom-0 z-40 grid border-t-2 border-sl-ink bg-sl-cream-50 lg:hidden"
           style={{ gridTemplateColumns: `repeat(${primary.length}, minmax(0, 1fr))` }}>
        {primary.map(({ to, short, label, icon: Icon }) => (
          <NavLink
            key={to}
            to={to}
            end={to === '/admin'}
            className={({ isActive }) => `flex min-h-16 flex-col items-center justify-center gap-0.5 text-[11px] font-bold ${isActive ? (accent === 'purple' ? 'text-sl-purple-700' : 'text-sl-green-700') : 'text-sl-ink-600'}`}
          >
            {({ isActive }) => (
              <>
                <span className={`flex h-8 w-12 items-center justify-center rounded-full ${isActive ? (accent === 'purple' ? 'bg-sl-purple-100' : 'bg-sl-green-100') : ''}`}>
                  <Icon className="h-5 w-5" aria-hidden="true" />
                </span>
                {short || label}
              </>
            )}
          </NavLink>
        ))}
      </nav>
    </div>
  )
}
