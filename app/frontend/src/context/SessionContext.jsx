import { createContext, useCallback, useContext, useEffect, useState } from 'react'
import { API_BASE_URL } from '../services/signRecognizer'

const SessionContext = createContext(null)

export function SessionProvider({ children }) {
  const [session, setSession] = useState(null)
  const [ready, setReady] = useState(false)

  const refresh = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/auth/me`, { credentials: 'include' })
      const data = response.ok ? await response.json() : null
      setSession(data?.user || null)
    } catch {
      setSession(null)
    } finally {
      setReady(true)
    }
  }, [])

  useEffect(() => { refresh() }, [refresh])

  const signOut = useCallback(async () => {
    try { await fetch(`${API_BASE_URL}/auth/logout`, { method: 'POST', credentials: 'include' }) } catch { /* clear local view even offline */ }
    setSession(null)
  }, [])

  return <SessionContext.Provider value={{ session, ready, signOut, refresh }}>{children}</SessionContext.Provider>
}

export function useSession() {
  const ctx = useContext(SessionContext)
  if (!ctx) throw new Error('useSession must be used within SessionProvider')
  return ctx
}
