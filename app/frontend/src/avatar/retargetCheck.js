import * as THREE from 'three'
import { HAND_SLOT, POSE, buildRig, clipTargets, letterClip, solveClip } from './landmarkRetarget'
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js'
import { API_BASE_URL } from '../services/signRecognizer'

/**
 * How closely the avatar follows a recorded sign: for every frame the solved rotations are applied to the
 * skeleton and each limb's direction is compared with the signer's (same smoothed targets as the solver).
 * Returns { group: { n, meanDeg, over30 } } for upperArm, forearm, hand, palm, knuckle, finger, thumb.
 * Errors come from the anatomical limits (by design) and from bugs; large ones point at bugs.
 */
export function agreement(rig, root, clip) {
  const frames = solveClip(rig, clip)
  if (!frames) return null
  const targets = clipTargets(clip)
  const find = (n) => root.getObjectByName(`mixamorig${n}`)
  const pos = (b) => b.getWorldPosition(new THREE.Vector3())
  const stats = {}
  const add = (group, a, b) => {
    if (a.lengthSq() < 1e-12 || b.lengthSq() < 1e-12) return
    const deg = THREE.MathUtils.radToDeg(a.angleTo(b))
    const s = (stats[group] ||= { n: 0, sum: 0, over30: 0 })
    s.n++
    s.sum += deg
    if (deg > 30) s.over30++
  }
  frames.forEach((locals, fi) => {
    for (const [bone, q] of locals) bone.quaternion.copy(q)
    root.updateMatrixWorld(true)
    const t = targets[fi]
    for (const side of ['Left', 'Right']) {
      if (!t.pose) continue
      const P = t.pose
      const arm = find(`${side}Arm`), fore = find(`${side}ForeArm`), hand = find(`${side}Hand`)
      add('upperArm', pos(fore).sub(pos(arm)), P(POSE[side].elbow).sub(P(POSE[side].shoulder)))
      add('forearm', pos(hand).sub(pos(fore)), P(POSE[side].wrist).sub(P(POSE[side].elbow)))
      const H = t.hand[HAND_SLOT[side]]
      if (!H) continue
      const h = pos(hand)
      const idx = find(`${side}HandIndex1`), mid = find(`${side}HandMiddle1`), pin = find(`${side}HandPinky1`)
      add('hand', pos(mid).sub(h), H(9).sub(H(0)))
      add('palm', new THREE.Vector3().crossVectors(pos(idx).sub(h), pos(pin).sub(h)),
        new THREE.Vector3().crossVectors(H(5).sub(H(0)), H(17).sub(H(0))))
      for (const f of rig[side].fingers) {
        const k = Number(f.bone.name.slice(-1))
        const next = find(f.bone.name.replace(/^mixamorig/, '').replace(/\d$/, String(k + 1)))
        if (!next) continue
        const group = f.kind === 'thumb' ? 'thumb' : f.kind === 'knuckle' ? 'knuckle' : 'finger'
        add(group, pos(next).sub(pos(f.bone)), H(f.to).sub(H(f.from)))
      }
    }
  })
  for (const r of rig.restPose) r.bone.quaternion.copy(r.restLocal) // leave the skeleton at rest
  root.updateMatrixWorld(true)
  return Object.fromEntries(Object.entries(stats).map(([g, s]) => [g, { n: s.n, meanDeg: s.sum / s.n, over30: s.over30 / s.n }]))
}

/** Developer check, from the browser console: (await import('/src/avatar/retargetCheck.js')).checkWords(40) */
export async function checkWords(nWords = 40) {
  const root = (await new GLTFLoader().loadAsync('/ybot.glb')).scene
  const find = (r, n) => r.getObjectByName(n) || r.getObjectByName(n.replace(':', ''))
  const rig = buildRig(root, find)
  const labels = (await (await fetch(`${API_BASE_URL}/labels`)).json()).labels.filter((l) => l.has_reference)
  const step = Math.max(1, Math.floor(labels.length / nWords))
  const words = labels.filter((_, i) => i % step === 0).slice(0, nWords).map((l) => l.sign)
  const total = {}
  const worst = []
  let worldClips = 0
  for (const w of words) {
    const clip = await (await fetch(`${API_BASE_URL}/reference/${w}`)).json()
    if (clip.world) worldClips++
    const a = agreement(rig, root, clip)
    if (!a) continue
    for (const [g, s] of Object.entries(a)) {
      const t = (total[g] ||= { n: 0, sum: 0, over: 0 })
      t.n += s.n; t.sum += s.meanDeg * s.n; t.over += s.over30 * s.n
    }
    worst.push([w, Math.max(...Object.values(a).map((s) => s.meanDeg)).toFixed(0), Object.entries(a).sort((x, y) => y[1].meanDeg - x[1].meanDeg)[0][0]])
  }
  console.warn(`CHECK ${words.length} words (${worldClips} with world landmarks): group meanDeg | >30deg share`)
  for (const [g, t] of Object.entries(total)) console.warn(`CHECK ${g} ${(t.sum / t.n).toFixed(1)} | ${(100 * t.over / t.n).toFixed(1)}%`)
  worst.sort((a, b) => b[1] - a[1])
  console.warn('CHECK worst: ' + worst.slice(0, 6).map((x) => x.join(':')).join(' '))
}

/** The same for the 26 letters (letter photos with made-up arms). */
export async function checkLetters() {
  const root = (await new GLTFLoader().loadAsync('/ybot.glb')).scene
  const find = (r, n) => r.getObjectByName(n) || r.getObjectByName(n.replace(':', ''))
  const rig = buildRig(root, find)
  const total = {}
  const rows = []
  for (const L of 'ABCDEFGHIJKLMNOPQRSTUVWXYZ') {
    const clip = letterClip(rig, await (await fetch(`${API_BASE_URL}/reference/LETTER_${L}`)).json())
    const a = clip && agreement(rig, root, clip)
    if (!a) { rows.push(L + ':none'); continue }
    for (const [g, s] of Object.entries(a)) {
      const t = (total[g] ||= { n: 0, sum: 0, over: 0 })
      t.n += s.n; t.sum += s.meanDeg * s.n; t.over += s.over30 * s.n
    }
    rows.push(`${L}:hand${a.hand.meanDeg.toFixed(0)}/palm${a.palm.meanDeg.toFixed(0)}/fing${a.finger.meanDeg.toFixed(0)}`)
  }
  for (const [g, t] of Object.entries(total)) console.warn(`CHECK ${g} ${(t.sum / t.n).toFixed(1)} | ${(100 * t.over / t.n).toFixed(1)}%`)
  console.warn('CHECK ' + rows.join(' '))
}
