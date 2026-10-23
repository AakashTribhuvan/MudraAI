import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig(({ mode }) => ({
  // Electron loads the production bundle from file://, so assets must be relative.
  base: mode === 'electron' ? './' : '/',
  plugins: [react()],
  server: {
    port: 3000,
    open: false,
    // Quick Tunnel URLs vary, so allow only Cloudflare's tunnel host suffix when the
    // judges launcher explicitly enables public preview mode.
    allowedHosts: process.env.MUDRA_PUBLIC_TUNNEL === '1' ? ['.trycloudflare.com'] : [],
    proxy: {
      '^/(health|models|labels|predict|stream|recordings|reference|library|letters|phrases|model|auth|api|docs|openapi\.json)(/|$)': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
        ws: true,
      },
    },
  }
}))
