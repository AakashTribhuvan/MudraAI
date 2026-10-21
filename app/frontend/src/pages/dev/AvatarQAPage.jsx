import { useEffect, useRef, useState } from 'react'
import * as THREE from 'three'
import { useAvatarRenderer } from '../../avatar/useAvatarRenderer'
import { POSE, letterClip } from '../../avatar/landmarkRetarget'
import { API_BASE_URL } from '../../services/signRecognizer'

const LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'.split('')
const VIEWS = { Front: 0, 'Three-quarter': Math.PI / 4, Side: Math.PI / 2 }
const btn = 'rounded-lg border-2 border-sl-ink bg-white px-2.5 py-1 text-sm font-bold hover:bg-sl-cream-200 disabled:opacity-40'

/**
 * Developer-only avatar QA (route /dev/avatar-qa, development builds only): play any letter or recorded word,
 * pause / scrub, front / side / three-quarter views, skeleton overlay, wrist target markers for letters, the
 * fallbacks a sign used, frame rate and the fastest bone turn while playing, and a snapshot of the canvas.
 */
export default function AvatarQAPage() {
  const hostRef = useRef(null)
  const [words, setWords] = useState([])
  const [current, setCurrent] = useState('A')
  const [info, setInfo] = useState(null)
  const [paused, setPaused] = useState(false)
  const [scrub, setScrub] = useState(0)
  const [skeleton, setSkeleton] = useState(false)
  const [markers, setMarkers] = useState(true)
  const [view, setView] = useState('Front')
  const [stats, setStats] = useState({ fps: 0, maxTurn: 0, frameMs: 0 })
  const [letterNote, setLetterNote] = useState(null)
  const { executeSignSequence, resetToDefaultPose, context: ctx } = useAvatarRenderer({
    containerRef: hostRef, onSequenceInfo: setInfo, animationSpeed: 1.2,
  })

  useEffect(() => {
    fetch(`${API_BASE_URL}/labels`).then((r) => r.json())
      .then((d) => setWords(d.labels.filter((l) => l.has_reference).map((l) => l.sign))).catch(() => setWords([]))
  }, [])

  // frame rate and the fastest bone turn (deg/s) on the animated skeleton, measured in the real render loop
  useEffect(() => {
    let raf, last = performance.now(), frames = 0, windowStart = last, maxTurn = 0, worstFrame = 0
    const prev = new Map()
    const tick = (now) => {
      frames++
      const dt = (now - last) / 1000
      worstFrame = Math.max(worstFrame, now - last)
      last = now
      for (const r of ctx.rig?.restPose || []) {
        const p = prev.get(r.bone)
        if (p && dt > 0) maxTurn = Math.max(maxTurn, THREE.MathUtils.radToDeg(2 * Math.acos(Math.min(1, Math.abs(p.dot(r.bone.quaternion))))) / dt)
        prev.set(r.bone, r.bone.quaternion.clone())
      }
      if (now - windowStart > 1000) {
        setStats({ fps: Math.round((frames * 1000) / (now - windowStart)), maxTurn: Math.round(maxTurn), frameMs: Math.round(worstFrame) })
        frames = 0; windowStart = now; maxTurn = 0; worstFrame = 0
      }
      raf = requestAnimationFrame(tick)
    }
    raf = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf)
  }, [ctx])

  // skeleton overlay
  useEffect(() => {
    if (!skeleton || !ctx.scene) return undefined
    const target = ctx.display || ctx.avatar
    if (!target) return undefined
    const helper = new THREE.SkeletonHelper(target)
    ctx.scene.add(helper)
    return () => { ctx.scene?.remove(helper); helper.dispose?.() }
  }, [skeleton, ctx, info])

  // camera view
  useEffect(() => { ctx.viewYaw = VIEWS[view]; ctx.frameAvatar?.() }, [view, ctx])

  // wrist targets of a letter (where letterClip puts the wrists), as spheres in the reference skeleton's space
  useEffect(() => {
    if (!markers || !ctx.scene || !ctx.rig || current.length !== 1) { setLetterNote(null); return undefined }
    let group = null
    let cancelled = false
    fetch(`${API_BASE_URL}/reference/LETTER_${current}`).then((r) => (r.ok ? r.json() : null)).then((raw) => {
      if (cancelled) return
      if (!raw) { setLetterNote(`No letter reference for ${current} on this server (the avatar uses the hand-made fallback).`); return }
      const clip = letterClip(ctx.rig, raw)
      if (!clip) { setLetterNote(`The letter reference for ${current} has no usable hand.`); return }
      const w = clip.frames[0].world.pose
      group = new THREE.Group()
      for (const side of ['Left', 'Right']) {
        const p = w[POSE[side].wrist]
        const s = w[POSE[side].shoulder]
        if (p[0] === s[0] && p[1] === s[1] && p[2] === s[2]) continue
        const m = new THREE.Mesh(new THREE.SphereGeometry(0.018), new THREE.MeshBasicMaterial({ color: side === 'Left' ? 0x2563eb : 0xdc2626, depthTest: false }))
        m.position.set(p[0], -p[1], -p[2])
        m.renderOrder = 10
        group.add(m)
      }
      ctx.scene.add(group)
      setLetterNote(clip.contact ? `Hands touch in the photo: landmarks L${clip.contact.pair[0]} / R${clip.contact.pair[1]}, ${clip.contact.photo.toFixed(2)} hand sizes apart (matched on the avatar).` : 'Wrist targets shown (blue left, red right).')
    })
    return () => { cancelled = true; if (group) ctx.scene?.remove(group) }
  }, [markers, current, ctx, info])

  useEffect(() => { window.__avatarCtx = ctx }, [ctx]) // dev page only: lets diagnostics inspect the queue

  const play = (sign) => {
    setCurrent(sign)
    ctx.debugPaused = false
    setPaused(false)
    executeSignSequence(sign.length === 1 ? sign : sign.replaceAll('_', ' '))
  }
  const togglePause = () => {
    const now = performance.now()
    if (!paused) { ctx.debugPaused = true; ctx.pausedAt = now } else {
      const d = now - (ctx.pausedAt || now) // the clip's clock skips the paused time
      if (ctx.clip) { ctx.clip.start += d; ctx.clip.tick += d; if (ctx.clip.enterStart) ctx.clip.enterStart += d }
      if (ctx.blend) ctx.blend.start += d
      ctx.debugPaused = false
    }
    setPaused(!paused)
  }
  const scrubTo = (fraction) => {
    setScrub(fraction)
    const frames = ctx.clip?.frames
    if (!frames) return
    if (!paused) togglePause()
    const i = Math.min(frames.length - 1, Math.round(fraction * (frames.length - 1)))
    for (const [bone, q] of frames[i]) bone.quaternion.copy(q)
  }
  const snapshot = () => {
    if (!ctx.renderer) return
    ctx.renderer.render(ctx.scene, ctx.camera)
    const a = document.createElement('a')
    a.href = ctx.renderer.domElement.toDataURL('image/png')
    a.download = `avatar-${current}-${view.toLowerCase()}-${Date.now()}.png`
    a.click()
  }

  return (
    <div className="mx-auto max-w-6xl space-y-4 p-4">
      <h1 className="font-display text-2xl font-extrabold">Avatar QA (development only)</h1>
      <p className="text-sm text-sl-ink-600">Measurements show whether the avatar reproduces its reference; whether a sign is correct ISL still needs a Deaf or qualified ISL signer.</p>
      <div className="grid gap-4 lg:grid-cols-[1fr_22rem]">
        <div className="space-y-2">
          <div ref={hostRef} className="relative h-[460px] w-full overflow-hidden rounded-xl border-2 border-sl-ink bg-white" />
          <div className="flex flex-wrap items-center gap-2">
            <button type="button" className={btn} onClick={() => play(current)}>Play / restart</button>
            <button type="button" className={btn} onClick={togglePause}>{paused ? 'Resume' : 'Pause'}</button>
            <button type="button" className={btn} onClick={() => { ctx.debugPaused = false; setPaused(false); resetToDefaultPose() }}>Rest pose</button>
            <label className="flex items-center gap-2 text-sm">Scrub
              <input type="range" min={0} max={1} step={0.01} value={scrub} onChange={(e) => scrubTo(Number(e.target.value))} disabled={!ctx.clip} />
            </label>
            <button type="button" className={btn} onClick={snapshot}>Snapshot</button>
          </div>
          <div className="flex flex-wrap items-center gap-2 text-sm">
            {Object.keys(VIEWS).map((v) => <button key={v} type="button" className={`${btn} ${view === v ? 'bg-sl-green' : ''}`} onClick={() => setView(v)}>{v}</button>)}
            <label className="flex items-center gap-1"><input type="checkbox" checked={skeleton} onChange={(e) => setSkeleton(e.target.checked)} /> Skeleton</label>
            <label className="flex items-center gap-1"><input type="checkbox" checked={markers} onChange={(e) => setMarkers(e.target.checked)} /> Wrist targets</label>
          </div>
          <p className="font-mono text-xs" role="status">
            {stats.fps} fps · slowest frame {stats.frameMs} ms · fastest bone turn {stats.maxTurn} deg/s (last second)
          </p>
          {letterNote && <p className="text-xs">{letterNote}</p>}
          {info && (
            <p className="text-xs">
              Sequence “{info.text}”: {info.recordedWords} recorded sign(s), {info.letterShapes} letter photo shape(s)
              {info.fallbackLetters.length + info.fallbackWords.length > 0
                ? <strong className="text-amber-800"> · hand-made fallback: {[...info.fallbackWords, ...info.fallbackLetters].join(', ')}</strong>
                : ' · no fallbacks'}
              {info.letterDataMissing && <strong className="text-rose-700"> · letter reference data missing on this server</strong>}
            </p>
          )}
        </div>
        <div className="space-y-3">
          <div>
            <h2 className="text-sm font-bold">Letters</h2>
            <div className="mt-1 flex flex-wrap gap-1">
              {LETTERS.map((l) => <button key={l} type="button" className={`${btn} w-9 ${current === l ? 'bg-sl-green' : ''}`} onClick={() => play(l)}>{l}</button>)}
            </div>
          </div>
          <div>
            <h2 className="text-sm font-bold">Recorded word signs ({words.length})</h2>
            <div className="mt-1 flex max-h-[340px] flex-wrap gap-1 overflow-y-auto">
              {words.map((w) => <button key={w} type="button" className={`${btn} text-xs ${current === w ? 'bg-sl-green' : ''}`} onClick={() => play(w)}>{w}</button>)}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
