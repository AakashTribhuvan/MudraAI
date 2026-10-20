import { Link } from 'react-router-dom'
import { ArrowRight, BookOpen, Disc, MessagesSquare, Play, Trophy } from 'lucide-react'
import MudraAvatarViewer from '../../components/avatar/MudraAvatarViewer'
import Card from '../../components/ui/Card'
import StatusPill from '../../components/ui/StatusPill'
import { DoodleArrow, DoodleSparkle } from '../../components/brand/Doodles'
import { useSession } from '../../context/SessionContext'
import { BADGES, LESSON_BY_ID, nextLesson } from '../../features/game/progress'
import { useGameProgress } from '../../features/game/useGameProgress'
import { useVocabulary } from '../../features/vocabulary'
import { GameStats } from '../game/GamePage'

export default function UserHomePage() {
  const { session } = useSession()
  const { progress } = useGameProgress()
  const vocab = useVocabulary()
  const next = nextLesson(progress)
  const recent = Object.entries(progress.lessons)
    .sort((a, b) => (b[1].lastAt || '').localeCompare(a[1].lastAt || ''))
    .slice(0, 3)
    .map(([id, r]) => ({ lesson: LESSON_BY_ID[id], ...r }))
    .filter((r) => r.lesson)
  const nextBadge = BADGES.find((b) => !progress.badges[b.id])
  const isNew = Object.keys(progress.lessons).length === 0

  return (
    <div className="mx-auto max-w-6xl space-y-8 px-4 py-6 sm:px-6 lg:py-10">
      <header className="relative">
        <DoodleSparkle className="absolute -top-3 left-40 text-sl-purple" size={24} />
        <p className="text-sm font-bold uppercase tracking-wider text-sl-green-700">Home</p>
        <h1 className="font-display text-4xl font-extrabold tracking-tight sm:text-5xl">Hi, {session?.name}!</h1>
        <p className="mt-2 text-lg text-sl-ink-600">{isNew ? 'Start with your first lesson: five greetings in a few minutes.' : 'Good to see you again. Keep your streak going.'}</p>
      </header>

      <div className="grid gap-6 lg:grid-cols-[1.4fr_1fr]">
        <Card outlined className="relative overflow-hidden bg-sl-green-50 p-6">
          <p className="text-sm font-bold uppercase tracking-wider text-sl-ink-600">{isNew ? 'Start learning' : 'Continue learning'}</p>
          {next ? (
            <>
              <h2 className="mt-1 font-display text-3xl font-extrabold">{next.unitTitle}</h2>
              <p className="mt-1 text-sl-ink-600">Lesson {next.number} of {next.unitLessons} · {next.words.map(vocab.label).join(', ')}</p>
              <Link to={`/app/game/lesson/${next.id}`} className="btn-primary mt-5 inline-flex">
                <Play className="h-4 w-4" aria-hidden="true" /> {isNew ? 'Start lesson' : 'Continue'}
              </Link>
            </>
          ) : (
            <h2 className="mt-1 font-display text-3xl font-extrabold">Every lesson passed. Amazing!</h2>
          )}
          <DoodleArrow className="absolute bottom-4 right-6 hidden text-sl-ink sm:block" width={90} />
        </Card>

        <Link to="/app/communication" className="group block">
          <Card outlined className="h-full bg-sl-purple p-6 text-white transition-transform group-hover:-translate-y-0.5">
            <MessagesSquare className="h-8 w-8" aria-hidden="true" />
            <h2 className="mt-3 font-display text-2xl font-extrabold">Communicate</h2>
            <p className="mt-1 text-white/90">Sign in front of your webcam: SignLine turns words into text and speech.</p>
            <p className="mt-4 inline-flex items-center gap-2 font-bold">Open Communication <ArrowRight className="h-4 w-4 transition-transform group-hover:translate-x-1" aria-hidden="true" /></p>
            <div className="mt-3">
              <span className="rounded-full bg-white/95 px-2.5 py-1 text-xs font-bold text-sl-ink">
                {vocab.status === 'online' ? '● Recognition server online' : vocab.status === 'loading' ? 'Checking server…' : '○ Server offline: start start_all.bat'}
              </span>
            </div>
          </Card>
        </Link>
      </div>

      <GameStats progress={progress} />

      <div className="grid gap-6 lg:grid-cols-3">
        <Card className="p-5 lg:col-span-1">
          <h2 className="font-display text-lg font-bold">Recent lessons</h2>
          {recent.length === 0 ? (
            <p className="mt-2 text-sm text-sl-ink-600">Nothing yet. Your finished lessons will appear here.</p>
          ) : (
            <ul className="mt-3 space-y-2">
              {recent.map((r) => (
                <li key={r.lesson.id} className="flex items-center justify-between gap-2 text-sm">
                  <span><strong>{r.lesson.unitTitle}</strong> · lesson {r.lesson.number}</span>
                  <span className="text-sl-ink-600">{Math.round(r.best * 100)}% best</span>
                </li>
              ))}
            </ul>
          )}
        </Card>
        <Card className="p-5">
          <h2 className="flex items-center gap-2 font-display text-lg font-bold"><Trophy className="h-5 w-5" aria-hidden="true" /> Next badge</h2>
          {nextBadge ? (
            <p className="mt-2 text-sm"><strong>{nextBadge.title}</strong> · {nextBadge.text}</p>
          ) : <p className="mt-2 text-sm">You have every badge!</p>}
          <p className="mt-2 text-xs text-sl-ink-600">{Object.keys(progress.badges).length} of {BADGES.length} earned</p>
        </Card>
        <Card className="p-5">
          <h2 className="font-display text-lg font-bold">More ways to learn</h2>
          <div className="mt-3 flex flex-col gap-2">
            <Link to="/app/learning" className="inline-flex items-center gap-2 text-sm font-bold underline underline-offset-4"><BookOpen className="h-4 w-4" aria-hidden="true" /> Browse all {vocab.status === 'online' ? Object.keys(vocab.labels).length : 314} words</Link>
            <Link to="/app/record" className="inline-flex items-center gap-2 text-sm font-bold underline underline-offset-4"><Disc className="h-4 w-4" aria-hidden="true" /> Teach SignLine your own sign</Link>
          </div>
        </Card>
      </div>

      <Card className="overflow-hidden">
        <div className="flex flex-wrap items-center justify-between gap-2 border-b border-sl-cream-300 px-5 py-3">
          <h2 className="font-display text-lg font-bold">SignLine avatar</h2>
          <StatusPill status="live">314 recorded signs + A–Z fingerspelling</StatusPill>
        </div>
        <div className="grid gap-4 p-5 md:grid-cols-[1fr_1.2fr] md:items-center">
          <p className="text-sm text-sl-ink-600">
            The 3D avatar signs words typed or spoken on the Communication page (Voice → Avatar). Words SignLine recognises are
            performed from real recorded signs (an approximation of the signer's arm and hand movement); 5 more are
            built in, and other words are fingerspelled with hand shapes from real ISL alphabet photos.
          </p>
          <MudraAvatarViewer signText="HELLO" className="h-[300px] w-full" />
        </div>
      </Card>
    </div>
  )
}
