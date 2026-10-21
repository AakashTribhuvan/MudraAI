import { useCallback, useEffect, useRef, useState } from 'react'
import { BrowserRouter, Navigate, Route, Routes, useNavigate, useParams } from 'react-router-dom'
import {
  BarChart3, BookOpen, Disc, Gamepad2, Home, LayoutDashboard, Library, MessagesSquare, ScrollText, Settings,
  Swords, TrendingUp, Users,
} from 'lucide-react'
import { AccessibilityProvider } from './context/AccessibilityContext'
import { SessionProvider, useSession } from './context/SessionContext'
import AccessibilityModal from './components/AccessibilityModal'
import AppLoader from './components/brand/AppLoader'
import Shell from './layouts/Shell'
import WelcomePage from './pages/auth/WelcomePage'
import SignInPage from './pages/auth/SignInPage'
import RegisterPage from './pages/auth/RegisterPage'
import ComingSoonPage from './pages/ComingSoonPage'
import AdminOverviewPage from './pages/admin/AdminOverviewPage'
import { AdminContentPage, AdminGameRulesPage, AdminRecordsPage, AdminReportsPage, AdminUsersPage } from './pages/admin/AdminPages'
import GamePage from './pages/game/GamePage'
import GameLessonPage from './pages/game/GameLessonPage'
import LegalPage from './pages/legal/LegalPage'
import SettingsPage from './pages/SettingsPage'
import UserHomePage from './pages/user/UserHomePage'
import LearningPage from './pages/user/LearningPage'
import ProgressHubPage from './pages/user/ProgressHubPage'
import CommunicatePage from './pages/CommunicatePage'
import PracticePage from './pages/PracticePage'
import RecordPage from './pages/RecordPage'
import { API_BASE_URL } from './services/signRecognizer'
import EmergencySOSButton from './features/emergency/EmergencySOSButton'
import { EmergencyLayout } from './features/emergency/components'
import EmergencyStartPage from './pages/emergency/EmergencyStartPage'
import EmergencySessionPage from './pages/emergency/EmergencySessionPage'
import EmergencyGesturesPage from './pages/emergency/EmergencyGesturesPage'
import EmergencyGestureDetailPage from './pages/emergency/EmergencyGestureDetailPage'
import AvatarQAPage from './pages/dev/AvatarQAPage'
import EmergencyAboutPage from './pages/emergency/EmergencyAboutPage'
import { ResponderDashboardPage, ResponderIncidentPage, ResponderLoginPage } from './pages/emergency/ResponderPages'

const USER_NAV = [
  { to: '/app/home', label: 'Home', icon: Home },
  { to: '/app/communication', label: 'Communication', short: 'Talk', icon: MessagesSquare },
  { to: '/app/learning', label: 'Learning', short: 'Learn', icon: BookOpen },
  { to: '/app/game', label: 'Game Mode', short: 'Game', icon: Gamepad2 },
  { to: '/app/progress', label: 'Progress', icon: TrendingUp },
]
const USER_MORE = [{ to: '/app/record', label: 'Teach a sign', icon: Disc }, { to: '/app/settings', label: 'Settings', icon: Settings }]
const ADMIN_NAV = [
  { to: '/admin', label: 'Overview', icon: LayoutDashboard },
  { to: '/admin/users', label: 'Users', icon: Users },
  { to: '/admin/content', label: 'Content', icon: Library },
  { to: '/admin/records', label: 'Records', icon: ScrollText },
  { to: '/admin/reports', label: 'Reports', icon: BarChart3 },
]
const ADMIN_MORE = [
  { to: '/admin/game', label: 'Game rules', icon: Swords },
  { to: '/admin/settings', label: 'Settings', icon: Settings },
]

// Older pages navigate by tab id; map those ids to routes.
const TAB_ROUTES = {
  home: '/app/home', communicate: '/app/communication', learn: '/app/learning', practice: '/app/practice',
  progress: '/app/progress', record: '/app/record',
}

/** Demo role gate: client-side only (see SessionContext). */
function RequireRole({ role, children }) {
  const { session } = useSession()
  if (!session) return <Navigate to="/welcome" replace />
  if (session.role !== role) return <Navigate to={session.role === 'admin' ? '/admin' : '/app/home'} replace />
  return children
}

function StartRoute() {
  const { session } = useSession()
  return <Navigate to={!session ? '/welcome' : session.role === 'admin' ? '/admin' : '/app/home'} replace />
}

function UserRoutes() {
  const navigate = useNavigate()
  const [practiceSign, setPracticeSign] = useState('HELLO')
  const onNavigate = useCallback((tab, sign = null) => {
    if (sign) setPracticeSign(sign)
    navigate(TAB_ROUTES[tab] || '/app/home')
    window.scrollTo({ top: 0 })
  }, [navigate])

  return (
    <Routes>
      <Route element={<Shell primary={USER_NAV} secondary={USER_MORE} homeLabel="home" />}>
        <Route path="home" element={<UserHomePage />} />
        <Route path="communication" element={<CommunicatePage />} />
        <Route path="learning" element={<LearningPage onNavigate={onNavigate} onSelectPracticeSign={setPracticeSign} />} />
        <Route path="practice" element={<PracticePage targetSignId={practiceSign} onNavigate={onNavigate} />} />
        <Route path="game" element={<GamePage />} />
        <Route path="game/lesson/:lessonId" element={<GameLessonPage />} />
        <Route path="progress" element={<ProgressHubPage onNavigate={onNavigate} />} />
        <Route path="settings" element={<SettingsPage />} />
        <Route path="record" element={<RecordPage />} />
        <Route path="*" element={<Navigate to="home" replace />} />
      </Route>
    </Routes>
  )
}

const ADMIN_SECTIONS = {}

function AdminSection() {
  const { section } = useParams()
  const [title, text] = ADMIN_SECTIONS[section] || ['Not found', 'This admin section does not exist.']
  return <ComingSoonPage eyebrow="Admin" title={title} links={[{ to: '/admin', label: 'Back to overview' }]}><p>{text}</p></ComingSoonPage>
}

function AdminRoutes() {
  return (
    <Routes>
      <Route element={<Shell primary={ADMIN_NAV} secondary={ADMIN_MORE} accent="purple" homeLabel="admin" />}>
        <Route index element={<AdminOverviewPage />} />
        <Route path="users" element={<AdminUsersPage />} />
        <Route path="content" element={<AdminContentPage />} />
        <Route path="records" element={<AdminRecordsPage />} />
        <Route path="reports" element={<AdminReportsPage />} />
        <Route path="game" element={<AdminGameRulesPage />} />
        <Route path="settings" element={<SettingsPage admin />} />
        <Route path=":section" element={<AdminSection />} />
      </Route>
    </Routes>
  )
}

/**
 * Start-up screen timing: it stays until start-up has really run (web fonts, a first contact with the
 * recognition server, capped at 2.5 s) AND the intro animation has been shown: INTRO_MS on every
 * load (owner's choice, 2026-10-16), INTRO_STATIC_MS for the still picture when motion is reduced.
 * "Skip intro" ends the wait at once.
 */
const INTRO_MS = 6000
const INTRO_STATIC_MS = 1500

function introDuration() {
  try {
    const reduced = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches ||
      localStorage.getItem('mudra_reduced_motion') === 'true'
    return reduced ? INTRO_STATIC_MS : INTRO_MS
  } catch {
    return INTRO_STATIC_MS
  }
}

function useStartup() {
  const [phase, setPhase] = useState('loading') // loading | leaving | done | failed
  const skipRef = useRef(null)
  const run = useCallback(() => {
    let cancelled = false
    setPhase('loading')
    const controller = new AbortController()
    const cap = new Promise((resolve) => setTimeout(resolve, 2500))
    const minimum = new Promise((resolve) => {
      const timer = setTimeout(resolve, introDuration())
      skipRef.current = () => { clearTimeout(timer); resolve() }
    })
    const work = Promise.allSettled([
      document.fonts?.ready,
      fetch(`${API_BASE_URL}/health`, { signal: controller.signal }),
    ])
    Promise.all([Promise.race([work, cap]), minimum])
      .then(() => {
        if (cancelled) return
        controller.abort()
        setPhase('leaving')
        setTimeout(() => setPhase('done'), 400)
      })
      .catch(() => !cancelled && setPhase('failed'))
    return () => { cancelled = true } // React may run start-up twice in development: only the last run counts
  }, [])
  useEffect(() => run(), [run])
  const skip = useCallback(() => skipRef.current?.(), [])
  return { phase, retry: run, skip }
}

export default function App() {
  const { phase, retry, skip } = useStartup()
  return (
    <AccessibilityProvider>
      <SessionProvider>
        <BrowserRouter>
          {phase !== 'done' && <AppLoader leaving={phase === 'leaving'} failed={phase === 'failed'} onRetry={retry} onSkip={skip} />}
          <Routes>
            <Route path="/" element={<StartRoute />} />
            <Route path="/welcome" element={<WelcomePage />} />
            <Route path="/signin" element={<SignInPage />} />
            <Route path="/register" element={<RegisterPage />} />
            <Route path="/legal/:doc" element={<LegalPage />} />
            {import.meta.env.DEV && <Route path="/dev/avatar-qa" element={<AvatarQAPage />} />}
            <Route path="/app/*" element={<RequireRole role="user"><UserRoutes /></RequireRole>} />
            <Route path="/admin/*" element={<RequireRole role="admin"><AdminRoutes /></RequireRole>} />
            {/* SignLine Emergency: open without signing in */}
            <Route path="/emergency" element={<EmergencyLayout />}>
              <Route index element={<EmergencyStartPage />} />
              <Route path="session/:sessionId" element={<EmergencySessionPage />} />
              <Route path="gestures" element={<EmergencyGesturesPage />} />
              <Route path="gestures/:gestureId" element={<EmergencyGestureDetailPage />} />
              <Route path="about" element={<EmergencyAboutPage />} />
            </Route>
            <Route path="/settings/emergency-gestures" element={<Navigate to="/emergency/gestures" replace />} />
            <Route path="/responder" element={<EmergencyLayout responder />}>
              <Route index element={<ResponderDashboardPage />} />
              <Route path="login" element={<ResponderLoginPage />} />
              <Route path="incidents/:incidentId" element={<ResponderIncidentPage />} />
            </Route>
            <Route path="*" element={<StartRoute />} />
          </Routes>
          <EmergencySOSButton />
          <AccessibilityModal />
        </BrowserRouter>
      </SessionProvider>
    </AccessibilityProvider>
  )
}
