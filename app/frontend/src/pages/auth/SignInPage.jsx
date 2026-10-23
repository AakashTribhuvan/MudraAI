import { useCallback, useEffect, useRef, useState } from 'react'
import { Link, useSearchParams } from 'react-router-dom'
import Button from '../../components/ui/Button'
import { API_BASE_URL } from '../../services/signRecognizer'
import AuthLayout, { DemoNotice } from './AuthLayout'

const ERRORS = {
  google_not_configured: 'Google sign-in is not configured on this server yet.',
  google_cancelled: 'Sign-in was cancelled or could not be verified. Please try again.',
  google_failed: 'Google sign-in could not be completed. Please try again.',
}

export default function SignInPage() {
  const [params] = useSearchParams()
  const role = params.get('role') === 'admin' ? 'admin' : 'user'
  const [busy, setBusy] = useState(false)
  const [popupError, setPopupError] = useState('')
  const pollRef = useRef(null)
  const timeoutRef = useRef(null)
  const channelRef = useRef(null)
  const messageHandlerRef = useRef(null)
  const error = ERRORS[params.get('error')]
  const next = role === 'admin' ? '/admin' : '/app/home'
  const googleUrl = `${API_BASE_URL}/auth/google/start?next=${encodeURIComponent(next)}`

  const clearWaiting = useCallback(() => {
    if (pollRef.current) window.clearInterval(pollRef.current)
    if (timeoutRef.current) window.clearTimeout(timeoutRef.current)
    if (channelRef.current) channelRef.current.close()
    if (messageHandlerRef.current) window.removeEventListener('message', messageHandlerRef.current)
    pollRef.current = null
    timeoutRef.current = null
    channelRef.current = null
    messageHandlerRef.current = null
  }, [])

  const handleGoogleLogin = useCallback(() => {
    if (busy) return
    setPopupError('')
    setBusy(true)

    const popup = window.open(googleUrl, 'signline-google-auth', 'width=520,height=660,resizable=yes,scrollbars=yes,popup=yes')
    if (!popup) {
      setBusy(false)
      window.location.assign(googleUrl)
      return
    }

    let checkingSession = false
    const finish = (message) => {
      clearWaiting()
      setBusy(false)
      if (message) setPopupError(message)
    }
    const checkSession = async () => {
      if (checkingSession) return
      checkingSession = true
      try {
        const response = await fetch(`${API_BASE_URL}/auth/me`, { credentials: 'include' })
        if (!response.ok) throw new Error(`Session check failed with HTTP ${response.status}`)
        const data = await response.json()
        if (data?.user) {
          clearWaiting()
          setBusy(false)
          window.location.assign(data.user.role === 'admin' ? '/admin' : '/app/home')
        }
      } catch {
        finish(ERRORS.google_failed)
      } finally {
        checkingSession = false
      }
    }
    const onAuthResult = (result) => {
      if (!result || result.type !== 'signline-google-auth') return
      if (result.status === 'success') {
        void checkSession()
      } else {
        finish(result.status === 'cancelled' ? ERRORS.google_cancelled : ERRORS.google_failed)
      }
    }
    const handleMessage = (event) => {
      if (event.origin === window.location.origin) onAuthResult(event.data)
    }
    window.addEventListener('message', handleMessage)
    messageHandlerRef.current = handleMessage
    if ('BroadcastChannel' in window) {
      channelRef.current = new BroadcastChannel('signline-google-auth')
      channelRef.current.onmessage = (event) => onAuthResult(event.data)
    }
    pollRef.current = window.setInterval(() => { void checkSession() }, 1000)
    timeoutRef.current = window.setTimeout(() => finish(ERRORS.google_failed), 10 * 60 * 1000)
  }, [busy, clearWaiting, googleUrl])

  useEffect(() => {
    return clearWaiting
  }, [clearWaiting])

  return (
    <AuthLayout
      title={role === 'admin' ? 'Admin sign in' : 'Sign in to SignLine'}
      subtitle={role === 'admin' ? 'Admin access is limited to approved Google accounts.' : 'Continue securely with your Google account.'}
    >
      {(popupError || error) && <p role="alert" className="rounded-sl border-2 border-rose-700 bg-rose-50 px-4 py-3 text-sm font-semibold text-rose-800">{popupError || error}</p>}
      <Button
        size="lg"
        className="w-full"
        variant={role === 'admin' ? 'purple' : 'primary'}
        onClick={handleGoogleLogin}
        disabled={busy}
      >
        <span aria-hidden="true" className="grid h-6 w-6 place-items-center rounded-full bg-white font-extrabold text-[#4285F4]">G</span>
        {busy ? 'Opening Google…' : 'Continue with Google'}
      </Button>
      <p className="text-sm text-sl-ink-600">
        <Link to="/welcome" className="font-semibold underline underline-offset-4">← Back</Link>
        {role === 'user' && <> · New to SignLine? Your learner profile is created at first sign-in.</>}
      </p>
      <DemoNotice />
    </AuthLayout>
  )
}
