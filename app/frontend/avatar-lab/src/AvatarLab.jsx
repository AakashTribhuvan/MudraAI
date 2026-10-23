import { useCallback, useEffect, useRef, useState } from 'react'
import { useAvatarRenderer } from '../../src/avatar/useAvatarRenderer.js'
import { useAvatarChoice } from '../../src/avatar/avatarCatalog.js'
import { API_BASE_URL } from '../../src/services/signRecognizer.js'

const EXAMPLES = ['HELLO', 'THANK YOU', 'I NEED WATER', 'GOOD MORNING', 'NAMASTE']

export default function AvatarLab() {
  const hostRef = useRef(null)
  const [text, setText] = useState('HELLO')
  const [subtitle, setSubtitle] = useState('')
  const [speed, setSpeed] = useState(1.2)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [warning, setWarning] = useState('')
  const [api, setApi] = useState({ status: 'checking', references: null, detail: '' })
  const [runtime, setRuntime] = useState({ loaded: false, queue: 0, step: 'idle', frame: null })
  const { avatar, avatars, choose } = useAvatarChoice()

  const onTextUpdate = useCallback((part) => setSubtitle((value) => value + part), [])
  const onLoadingChange = useCallback((value) => {
    setLoading(value)
    if (!value) setError('')
  }, [])
  const onError = useCallback((cause) => setError(cause?.message || 'The avatar failed to load.'), [])
  const onWarning = useCallback((message) => setWarning(message || ''), [])

  const { executeSignSequence, resetToDefaultPose, context } = useAvatarRenderer({
    containerRef: hostRef,
    modelPath: avatar.url,
    modelType: avatar.type,
    animationSpeed: speed,
    pauseDuration: 300,
    onTextUpdate,
    onLoadingChange,
    onError,
    onWarning,
  })

  useEffect(() => {
    let alive = true
    Promise.all([
      fetch(`${API_BASE_URL}/health`).then((response) => {
        if (!response.ok) throw new Error(`Backend returned ${response.status}`)
        return response.json()
      }),
      fetch(`${API_BASE_URL}/labels`).then((response) => {
        if (!response.ok) throw new Error(`Sign list returned ${response.status}`)
        return response.json()
      }),
    ]).then(([health, labels]) => {
      if (alive) setApi({
        status: 'online',
        references: (labels.labels || []).filter((label) => label.has_reference).length,
        detail: `${health.model || 'model ready'} · ${health.num_signs ?? 0} labels`,
      })
    }).catch((cause) => {
      if (alive) setApi({ status: 'offline', references: null, detail: cause.message })
    })
    return () => { alive = false }
  }, [])

  useEffect(() => {
    const timer = window.setInterval(() => {
      const queue = context.animations || []
      const active = queue[0]
      let frame = null
      if (active?.[0] === 'clip' && context.clip) {
        const fps = active[2] || 15
        frame = `${Math.min(context.clip.frames.length, Math.floor((performance.now() - context.clip.start) / 1000 * fps) + 1)} / ${context.clip.frames.length}`
      }
      setRuntime({
        loaded: Boolean(context.avatar),
        queue: queue.length,
        step: Array.isArray(active) ? (active[0] || 'hand pose') : 'idle',
        frame,
      })
    }, 200)
    return () => window.clearInterval(timer)
  }, [context])

  const play = (event) => {
    event?.preventDefault()
    if (!text.trim()) return
    setSubtitle('')
    setWarning('')
    executeSignSequence(text.trim())
  }

  const replay = () => {
    setSubtitle('')
    resetToDefaultPose()
    window.setTimeout(() => executeSignSequence(text.trim()), 120)
  }

  const reset = () => {
    setSubtitle('')
    resetToDefaultPose()
    executeSignSequence('') // invalidate any playback request still fetching its clips
  }

  return (
    <main className="lab-shell">
      <header className="lab-header">
        <div>
          <p className="eyebrow">ISOLATED DEVELOPMENT SITE</p>
          <h1>SignLine <span>Avatar Lab</span></h1>
          <p className="intro">Tune sign playback here. This page uses the production renderer and sign data; it does not change the main website.</p>
        </div>
        <div className={`api-badge ${api.status}`}>
          <i />
          <span>{api.status === 'online' ? 'Backend connected' : api.status === 'checking' ? 'Checking backend' : 'Backend offline'}</span>
          <small>{api.detail}</small>
        </div>
      </header>

      <section className="lab-grid">
        <div className="controls card">
          <div className="section-heading">
            <div><p className="eyebrow">PLAYGROUND</p><h2>Text to sign</h2></div>
            <span className="counter">{api.references ?? '—'} reference clips</span>
          </div>
          <form onSubmit={play}>
            <label htmlFor="sign-text">Enter a word or phrase</label>
            <textarea id="sign-text" rows="3" value={text} onChange={(event) => setText(event.target.value)} placeholder="Type a word or phrase…" />
            <div className="actions">
              <button className="primary" type="submit" disabled={!runtime.loaded}>Play signs</button>
              <button type="button" onClick={replay} disabled={!runtime.loaded}>Replay</button>
              <button type="button" onClick={reset} disabled={!runtime.loaded}>Reset pose</button>
            </div>
          </form>

          <div className="examples">
            <label>Try a sample</label>
            <div className="chips">{EXAMPLES.map((sample) => <button type="button" key={sample} onClick={() => { setText(sample); setSubtitle(''); executeSignSequence(sample) }}>{sample}</button>)}</div>
          </div>

          <div className="settings-row">
            <label htmlFor="avatar-choice">Avatar model</label>
            <select id="avatar-choice" value={avatar.id} onChange={(event) => choose(event.target.value)}>
              {avatars.map((item) => <option key={item.id} value={item.id}>{item.name}</option>)}
            </select>
          </div>
          <div className="settings-row speed-row">
            <label htmlFor="speed">Playback speed <b>{speed.toFixed(1)}×</b></label>
            <input id="speed" type="range" min="0.6" max="1.8" step="0.1" value={speed} onChange={(event) => setSpeed(Number(event.target.value))} />
          </div>

          <div className="subtitle-box">
            <span>PLAYBACK TEXT</span>
            <strong>{subtitle || 'Waiting for playback'}</strong>
          </div>

          <div className="diagnostics" aria-live="polite">
            <div><span>Renderer</span><b>{loading ? 'Loading' : error ? 'Error' : runtime.loaded ? 'Ready' : 'Waiting'}</b></div>
            <div><span>Queue</span><b>{runtime.queue} steps</b></div>
            <div><span>Active step</span><b>{runtime.step}{runtime.frame ? ` · frame ${runtime.frame}` : ''}</b></div>
            <div><span>Rig</span><b>{runtime.loaded ? (context.rig ? 'Mixamo bones found' : 'Rig not detected') : '—'}</b></div>
          </div>

          {warning && <p className="notice warning">{warning}</p>}
          {error && <p className="notice error">{error}</p>}
          {api.status === 'offline' && <p className="notice warning">Start the SignLine backend on port 8000 to load recorded signs. {api.detail}</p>}
          <p className="small-note">The avatar plays available recorded examples first. Unsupported words fall back to fingerspelling or a built-in gesture; ISL grammar and live signing quality still need expert review.</p>
        </div>

        <div className="stage card">
          <div className="stage-heading">
            <div><p className="eyebrow">LIVE PREVIEW</p><h2>{avatar.name}</h2></div>
            <span className={`state-pill ${error ? 'bad' : loading ? 'busy' : 'good'}`}>{error ? 'Error' : loading ? 'Loading' : 'Ready'}</span>
          </div>
          <div className="avatar-stage">
            <div ref={hostRef} className="avatar-canvas" />
            {loading && <div className="overlay">Loading 3D model…</div>}
            {error && <div className="overlay error-overlay">{error}</div>}
          </div>
          <div className="stage-footer"><span>INPUT</span><strong>{text || '—'}</strong></div>
        </div>
      </section>
      <footer>Temporary test surface · localhost:5173 · source: <code>app/frontend/src/avatar</code></footer>
    </main>
  )
}
