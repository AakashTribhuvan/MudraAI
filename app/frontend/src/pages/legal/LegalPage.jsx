import { Link, useParams } from 'react-router-dom'
import Logo from '../../components/brand/Logo'
import StatusPill from '../../components/ui/StatusPill'

/**
 * DRAFTS for the project owner to review; not legal advice and not approved. They describe what the
 * code actually does (2026-10-16). Text in [brackets] needs the owner's input.
 */
const OWNER = '[Project owner / organisation name: to be provided]'
const CONTACT = '[Contact email or address: to be provided]'

const DOCS = {
  terms: {
    title: 'Terms and Conditions',
    body: [
      ['What SignLine is', 'SignLine is a prototype that tries to recognise Indian Sign Language (ISL) words from a webcam and turn them into text and speech, and that helps people learn signs. It is provided by ' + OWNER + '.'],
      ['No guarantee of correctness', 'Recognition makes mistakes, especially for people and signing styles it has not seen. Never rely on SignLine in an emergency, for medical, legal or safety-critical communication, or as a replacement for a qualified interpreter.'],
      ['Sign-in', 'SignLine uses Google OpenID Connect to verify your email address. It creates a signed browser session but does not keep a central account directory. Admin access is limited to addresses configured by the SignLine team. SignLine does not receive your Google password.'],
      ['Your recordings', 'If you use Teach a sign, the hand and body landmark coordinates of your recording (not video) and the name you enter are stored on the SignLine server so the word can be recognised and used to improve the model. Only record yourself, and only with your consent.'],
      ['Acceptable use', 'Do not record other people without their consent, and do not use SignLine to harm, harass or identify anyone.'],
      ['Changes', 'These terms may change while SignLine is a prototype. Governing law and jurisdiction: [to be provided].'],
    ],
  },
  privacy: {
    title: 'Privacy Policy',
    body: [
      ['Who is responsible', OWNER + '. Contact: ' + CONTACT + '.'],
      ['Camera', 'Your camera video stays in your browser. MediaPipe models run on your device to find hand, body and face landmarks. No camera image or video is uploaded or stored.'],
      ['What is sent to the SignLine server', 'For live recognition, the coordinates of your hand landmarks and 33 body landmarks (which include points of the face such as the nose, eyes, ears and mouth) are sent to the SignLine server for each camera frame. The server keeps them only in memory while recognising a sign, unless session saving is switched on by the operator (off by default, and only with your consent).'],
      ['What the server stores', 'A log of each recognised sign (time, result, confidence, frame rate, and the camera’s model name, but no landmarks) for diagnosing problems. Recordings you make on Teach a sign (landmark coordinates plus the name you enter), until you delete them.'],
      ['What your browser stores', 'Game Mode progress, practice history, accessibility and camera preferences, and the name used for recordings are kept in your browser’s local storage. SignLine also uses a secure, HTTP-only session cookie after sign-in.'],
      ['Facial cues', 'Facial expression cues are computed on your device and are not sent to the server or stored.'],
      ['Third parties', 'Google receives your browser’s Google sign-in request when you choose Continue with Google. The app also loads its fonts from Google Fonts. No analytics, advertising or tracking services are used.'],
      ['Retention and deletion', 'Recordings stay on the server until deleted (Teach a sign page or Admin). The sign log is kept for diagnosis [retention period: to be decided]. Browser data stays until you clear it or reset your progress in Settings.'],
      ['Your rights', 'See Your data rights.'],
    ],
  },
  accessibility: {
    title: 'Accessibility Statement',
    body: [
      ['Our aim', 'SignLine aims to follow WCAG 2.2 level AA. It has not yet been formally audited, so this is a goal, not a claim of compliance.'],
      ['What is in place', 'Keyboard navigation with visible focus rings; a skip-to-content link; text size, high-contrast, reduced-motion and speech-speed settings; labelled form fields and buttons; states shown with text and icons, not colour alone.'],
      ['Known limitations', 'Sign examples are skeleton animations without text descriptions of the movement. Some older screens have not been fully reviewed for contrast and screen-reader use. Live recognition needs a camera and good lighting.'],
      ['Feedback', 'Tell us about barriers: ' + CONTACT + '.'],
    ],
  },
  cookies: {
    title: 'Cookies and storage',
    body: [
      ['Cookies', 'SignLine uses a secure, HTTP-only session cookie for authentication and a short-lived HTTP-only cookie while completing Google sign-in.'],
      ['Local storage', 'SignLine uses your browser’s local storage for Game Mode progress (signline_progress_v1:<name>), practice history (mudra.practice.sessions.v1), the recorder name (mudra.recorder.signer), accessibility settings (mudra_font_size, mudra_high_contrast, mudra_reduced_motion, mudra_speech_rate), and camera and speech preferences. None of it is used for tracking.'],
      ['Clearing it', 'Clear the site’s data in your browser settings, or reset Game Mode progress in Settings.'],
    ],
  },
  'data-rights': {
    title: 'Your data rights',
    body: [
      ['Sign-in session', 'SignLine uses your verified Google account email to identify your current session. It does not keep a central account directory. Sign out to clear the session cookie. Browser data can be removed by you at any time (see Cookies and storage).'],
      ['Recordings', 'Delete any recording on the Teach a sign page. An administrator can also delete recordings in Admin → Records.'],
      ['Requests', 'To ask what server data relates to you or to have it deleted, contact ' + CONTACT + '. [Applicable law, e.g. India’s Digital Personal Data Protection Act 2023, and response times: to be confirmed by the owner.]'],
    ],
  },
  licences: {
    title: 'Open-source licences and credits',
    body: [
      ['Software', 'React, React Router and Vite (MIT); Tailwind CSS (MIT); lucide-react icons (ISC); three.js (MIT); MediaPipe Tasks (Apache 2.0); FastAPI (MIT); PyTorch (BSD-style). Full licence texts ship with each package.'],
      ['Data', 'The recognition model was trained on INCLUDE (Sridhar et al., ACM Multimedia 2020; CC BY 4.0) and on recordings by the MUDRA team (licence to be confirmed).'],
      ['Assets', 'SignLine logo: provided by the project owner. 3D avatar model (ybot.glb): [source and licence to be confirmed].'],
    ],
  },
  contact: {
    title: 'Contact and support',
    body: [
      ['Contact', CONTACT],
      ['Reporting a problem', 'Describe what you did, what you expected, and what happened. Please do not send videos of people.'],
    ],
  },
}

export default function LegalPage() {
  const { doc } = useParams()
  const d = DOCS[doc]
  return (
    <div className="min-h-screen bg-sl-cream">
      <header className="mx-auto flex max-w-3xl items-center justify-between px-4 py-5 sm:px-6">
        <Link to="/" aria-label="SignLine"><Logo size={36} /></Link>
        <button type="button" onClick={() => window.history.back()} className="text-sm font-bold underline underline-offset-4">← Back</button>
      </header>
      <main className="mx-auto max-w-3xl px-4 pb-16 sm:px-6">
        {!d ? (
          <h1 className="font-display text-3xl font-extrabold">Page not found</h1>
        ) : (
          <article className="space-y-6">
            <div>
              <StatusPill status="placeholder">Draft for review, not legally approved</StatusPill>
              <h1 className="mt-3 font-display text-4xl font-extrabold tracking-tight">{d.title}</h1>
              <p className="mt-1 text-sm text-sl-ink-600">Last updated 16 October 2026. Describes the current prototype.</p>
            </div>
            {d.body.map(([h, p]) => (
              <section key={h}>
                <h2 className="font-display text-xl font-bold">{h}</h2>
                <p className="mt-1 leading-relaxed text-sl-ink-700">{p}</p>
              </section>
            ))}
          </article>
        )}
        <nav aria-label="Legal" className="mt-10 flex flex-wrap gap-x-4 gap-y-2 border-t-2 border-dashed border-sl-cream-300 pt-6 text-sm">
          {Object.entries(DOCS).map(([k, v]) => (
            <Link key={k} to={`/legal/${k}`} aria-current={k === doc ? 'page' : undefined}
                  className={`font-semibold underline underline-offset-4 ${k === doc ? 'text-sl-green-700' : ''}`}>{v.title}</Link>
          ))}
        </nav>
      </main>
    </div>
  )
}
