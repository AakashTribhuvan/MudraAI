import * as THREE from 'three'

/**
 * Drive the Mixamo avatar from recorded sign landmarks (GET /reference/{sign}: MediaPipe pose + hands per
 * frame, from real signed videos), so the avatar performs the word as it was signed instead of spelling it.
 *
 * v2 (anatomical constraints; a diagnostic on 25 words found v1's fingers twisting about their own length
 * in 30-39% of frames and wrists / knuckles flipping in 6-18%):
 * - landmarks are smoothed over time (centred 5-frame kernel, no delay: the clip is complete);
 * - upper arm: aimed at the elbow and rolled so the elbow bends in the signer's arm plane;
 *   forearm: a hinge on the elbow (no backward bend);
 * - hand: wrist->middle knuckle with the palm facing as recorded; part of its twist goes to the forearm
 *   (forearm pronation), the rest is limited;
 * - fingers: knuckles bend and spread but never twist; middle and end joints are hinges with limits;
 *   thumb joints swing without twist, limited.
 * Only directions are used, so the avatar's proportions do not matter. Assumes a Mixamo T-pose rest
 * (palms down, elbows bending forward). Depth from MediaPipe is rough; the face is not reproduced.
 */

const POSE = { Left: { shoulder: 11, elbow: 13, wrist: 15 }, Right: { shoulder: 12, elbow: 14, wrist: 16 } }
const HAND_SLOT = { Left: 0, Right: 1 } // /reference hands are [signer-left, signer-right]
const FINGERS = {
  Thumb: [1, 2, 3, 4],
  Index: [5, 6, 7, 8],
  Middle: [9, 10, 11, 12],
  Ring: [13, 14, 15, 16],
  Pinky: [17, 18, 19, 20],
}
const POSE_DEPTH = 0.5 // MediaPipe pose z is noisy: damp it
const HAND_DEPTH = 1.0
const FORWARD = new THREE.Vector3(0, 0, 1) // the avatar faces +z (towards the camera)
const D = Math.PI / 180
const LIMITS = {
  elbow: [-5 * D, 150 * D],
  wristSwing: 80 * D,
  wristTwist: 70 * D,
  forearmTwistShare: 0.6,
  knuckleFlex: [-25 * D, 95 * D],
  knuckleSpread: 25 * D,
  pip: [-5 * D, 120 * D],
  dip: [-5 * D, 90 * D],
  thumbSwing: 70 * D,
  knuckleRoll: 30 * D, // turn of a finger about its length, so its middle joint bends in the signer's plane
  rollMinBend: 15 * D, // below this bend the finger's plane is noise: no turn
}
const SMOOTH = [1, 4, 6, 4, 1]
const UNWRAP_MARGIN = (30 * Math.PI) / 180
const BEND_UNSURE = 0.1 // |upper x lower| below this: the elbow plane is noise
const BEND_SURE = 0.5 // above this: trusted fully
const MAX_DEG_PER_S = { arm: 900, finger: 700 } // fastest joint turn kept (human wrist / finger flicks)
// hand segments measured on the avatar and in a photo (MediaPipe landmark pairs), for the photo's true scale
const HAND_SEGMENTS = {
  palm: [0, 9], width: [5, 17], index1: [5, 6], middle1: [9, 10], ring1: [13, 14], pinky1: [17, 18],
  index2: [6, 7], middle2: [10, 11], ring2: [14, 15], pinky2: [18, 19],
}
const HAND_SEGMENT_BONES = {
  palm: ['Hand', 'Middle1'], width: ['Index1', 'Pinky1'], index1: ['Index1', 'Index2'], middle1: ['Middle1', 'Middle2'],
  ring1: ['Ring1', 'Ring2'], pinky1: ['Pinky1', 'Pinky2'], index2: ['Index2', 'Index3'], middle2: ['Middle2', 'Middle3'],
  ring2: ['Ring2', 'Ring3'], pinky2: ['Pinky2', 'Pinky3'],
}
// landmark index -> bone, for the hand-to-hand contact of two-handed letters
const LANDMARK_BONE = ['Hand', 'Thumb1', 'Thumb2', 'Thumb3', 'Thumb4', 'Index1', 'Index2', 'Index3', 'Index4',
  'Middle1', 'Middle2', 'Middle3', 'Middle4', 'Ring1', 'Ring2', 'Ring3', 'Ring4', 'Pinky1', 'Pinky2', 'Pinky3', 'Pinky4']
const CONTACT_MAX = 1.0 // photo hands closer than this many hand sizes are touching / interacting

const v3 = () => new THREE.Vector3()
const clamp = (x, [lo, hi]) => Math.min(hi, Math.max(lo, x))

function toWorld(p, width, height, depth) {
  return new THREE.Vector3(p[0] * width, -p[1] * height, -(p[2] || 0) * width * depth)
}

function worldPos(bone) {
  return bone.getWorldPosition(v3())
}

const inv = (q) => q.clone().invert()
const local = (q, v) => v.clone().applyQuaternion(inv(q)) // world vector -> frame of world rotation q

/**
 * Rest-pose description of the avatar's arm and hand bones. Call once, with the skeleton at rest
 * (before any pose is applied). `find(root, name)` resolves a Mixamo bone name.
 */
export function buildRig(root, find) {
  root.updateMatrixWorld(true)
  const bone = (n) => find(root, `mixamorig${n}`)
  const rest = (b) => b.getWorldQuaternion(new THREE.Quaternion())
  const sides = {}
  for (const side of ['Left', 'Right']) {
    const shoulder = bone(`${side}Shoulder`)
    const arm = bone(`${side}Arm`)
    const fore = bone(`${side}ForeArm`)
    const hand = bone(`${side}Hand`)
    if (!shoulder || !arm || !fore || !hand) return null
    const armDir = worldPos(fore).sub(worldPos(arm)).normalize()
    const foreDir = worldPos(hand).sub(worldPos(fore)).normalize()
    const elbowHinge = v3().crossVectors(foreDir, FORWARD).normalize() // +angle bends the forearm forward
    const foreQ = rest(fore)
    const middle = bone(`${side}HandMiddle1`)
    const handSeg = {}
    for (const [name, [a, b]] of Object.entries(HAND_SEGMENT_BONES)) {
      const ba = a === 'Hand' ? hand : bone(`${side}Hand${a}`)
      const bb = bone(`${side}Hand${b}`)
      if (ba && bb) handSeg[name] = worldPos(ba).distanceTo(worldPos(bb))
    }
    const entry = {
      // rest-pose sizes in world units, for made-up arm poses (letters)
      geo: { shoulder: worldPos(arm), upper: worldPos(fore).distanceTo(worldPos(arm)),
             fore: worldPos(hand).distanceTo(worldPos(fore)), hand: middle ? worldPos(middle).distanceTo(worldPos(hand)) : 0,
             handSeg },
      shoulder: { bone: shoulder, restQ: rest(shoulder) },
      arm: { bone: arm, restQ: rest(arm), restDir: armDir, restHinge: elbowHinge },
      fore: { bone: fore, restQ: foreQ, restLocal: fore.quaternion.clone(), dirLocal: local(foreQ, foreDir).normalize(),
              hingeLocal: local(foreQ, elbowHinge).normalize() },
      hand: null,
      fingers: [],
    }
    const index1 = bone(`${side}HandIndex1`)
    const middle1 = bone(`${side}HandMiddle1`)
    const pinky1 = bone(`${side}HandPinky1`)
    if (index1 && middle1 && pinky1) {
      const h = worldPos(hand)
      const handQ = rest(hand)
      const handDir = worldPos(middle1).sub(h).normalize()
      const normal = v3().crossVectors(worldPos(index1).sub(h), worldPos(pinky1).sub(h)).normalize()
      const palm = normal.y < 0 ? normal.clone() : normal.clone().negate() // T-pose: palms face down
      entry.hand = { bone: hand, restQ: handQ, restLocal: hand.quaternion.clone(), restDir: handDir, restNormal: normal,
                     dirLocal: local(handQ, handDir).normalize() }
      for (const [finger, lm] of Object.entries(FINGERS)) {
        const chain = [1, 2, 3, 4].map((k) => bone(`${side}Hand${finger}${k}`))
        if (chain.some((b) => !b)) continue
        for (let k = 0; k < 3; k++) {
          const q = rest(chain[k])
          const dir = worldPos(chain[k + 1]).sub(worldPos(chain[k])).normalize()
          const hinge = v3().crossVectors(dir, palm).normalize() // +angle curls towards the palm
          entry.fingers.push({
            bone: chain[k],
            parent: k === 0 ? 'hand' : entry.fingers.length - 1,
            kind: finger === 'Thumb' ? 'thumb' : k === 0 ? 'knuckle' : k === 1 ? 'pip' : 'dip',
            restLocal: chain[k].quaternion.clone(),
            dirLocal: local(q, dir).normalize(),
            hingeLocal: local(q, hinge).normalize(),
            from: lm[k],
            to: lm[k + 1],
          })
        }
      }
    }
    sides[side] = entry
  }
  // every driven bone with its rest local rotation, to return to after a clip
  const all = []
  for (const e of Object.values(sides)) {
    for (const part of [e.arm, e.fore, e.hand, ...e.fingers]) if (part) all.push({ bone: part.bone, restLocal: part.bone.quaternion.clone() })
  }
  sides.restPose = all
  return sides
}

/** World rotation turning a bone from rest to point along `dir`, rolled so `restRoll` lines up with `roll`. */
function aimWithRoll(restQ, restDir, restRoll, dir, roll) {
  const q1 = new THREE.Quaternion().setFromUnitVectors(restDir, dir)
  if (!roll) return q1.multiply(restQ)
  const n1 = restRoll.clone().applyQuaternion(q1).projectOnPlane(dir)
  const n2 = roll.clone().projectOnPlane(dir)
  if (n1.lengthSq() < 1e-8 || n2.lengthSq() < 1e-8) return q1.multiply(restQ)
  return new THREE.Quaternion().setFromUnitVectors(n1.normalize(), n2.normalize()).multiply(q1).multiply(restQ)
}

/** Signed angle of `t` around `axis`, measured from `from` (all in one frame; t need not be unit). */
function hingeAngle(from, axis, t) {
  const curl = v3().crossVectors(axis, from) // where `from` goes when turned +90 deg about `axis`
  return Math.atan2(t.dot(curl), t.dot(from))
}

/** Split q into twist about `axis` and the remaining swing (q = swing * twist). */
function swingTwist(q, axis) {
  const p = new THREE.Vector3(q.x, q.y, q.z).projectOnVector(axis)
  const twist = new THREE.Quaternion(p.x, p.y, p.z, q.w)
  if (twist.lengthSq() < 1e-12) twist.set(0, 0, 0, 1)
  twist.normalize()
  const swing = q.clone().multiply(inv(twist))
  return { swing, twist }
}

function limitAngle(q, max) {
  const angle = 2 * Math.acos(Math.min(1, Math.abs(q.w)))
  return angle <= max ? q : new THREE.Quaternion().slerp(q, max / angle) // identity -> q, scaled down
}

function signedTwist(twist, axis) {
  const angle = 2 * Math.acos(Math.min(1, Math.abs(twist.w)))
  const s = Math.sign(twist.w) * Math.sign(new THREE.Vector3(twist.x, twist.y, twist.z).dot(axis)) || 1
  return angle * s
}

/**
 * Turn of a knuckle about its own bone so the next joint's hinge lies in the signer's bend plane (the plane of
 * the first two finger segments). Without it the middle joint bends sideways from the recorded finger.
 */
function knuckleRoll(f, pip, boneW, seg, H) {
  if (!pip || pip.kind !== 'pip') return 0
  const seg2 = H(pip.to).sub(H(pip.from))
  if (seg2.lengthSq() < 1e-10) return 0
  const n = v3().crossVectors(seg, seg2.normalize()) // = sin(bend) * hinge for a bend towards the palm
  if (n.length() < Math.sin(LIMITS.rollMinBend)) return 0
  const want = local(boneW, n).projectOnPlane(f.dirLocal)
  const have = pip.hingeLocal.clone().applyQuaternion(pip.restLocal).projectOnPlane(f.dirLocal)
  if (want.lengthSq() < 1e-10 || have.lengthSq() < 1e-10) return 0
  let roll = Math.atan2(f.dirLocal.dot(v3().crossVectors(have, want)), have.dot(want))
  if (Math.abs(roll) > Math.PI / 2) roll -= Math.sign(roll) * Math.PI // bent backwards: same plane, other side
  return clamp(roll, [-LIMITS.knuckleRoll, LIMITS.knuckleRoll])
}

/**
 * A letter (/reference/LETTER_X: one photo's hand shapes, no body) as a world-landmark clip the solver can play:
 * the hands keep the photo's shape, orientation and placement relative to each other, sized to this avatar's
 * hands; the arms are made up with two-bone IK so the wrists sit in front of the chest (one-handed letters:
 * in front of that shoulder; the other arm hangs relaxed). Null when the photo has no hand.
 */
export function letterClip(rig, clip) {
  const photo = clip?.frames?.[0]?.hands
  if (!rig || !photo) return null
  const sides = ['Left', 'Right'].filter((side) => photo[HAND_SLOT[side]] && rig[side]?.geo?.hand)
  if (!sides.length) return null
  // photo -> avatar axes (image x = signer-left = avatar +x, image y down, z away from the camera)
  const A = (q) => new THREE.Vector3(q[0], -q[1], -(q[2] || 0))
  const hands = Object.fromEntries(sides.map((side) => [side, photo[HAND_SLOT[side]].map(A)]))
  const k = photoScale(rig, sides, hands)
  if (!k) return null
  const L = rig.Left.geo
  const Rg = rig.Right.geo
  const reach = L.upper + L.fore
  const chest = L.shoulder.clone().add(Rg.shoulder).multiplyScalar(0.5).add(v3().set(0, -0.2 * reach, 0.5 * reach))
  const wrists = {}
  if (sides.length === 2) {
    const mid = hands.Left[0].clone().add(hands.Right[0]).multiplyScalar(0.5)
    for (const side of sides) wrists[side] = chest.clone().add(hands[side][0].clone().sub(mid).multiplyScalar(k))
  } else {
    const side = sides[0]
    wrists[side] = chest.clone().setX(rig[side].geo.shoulder.x * 0.6).add(v3().set(0, 0.08 * reach, 0))
  }
  const build = () => buildLetterFrame(rig, sides, hands, wrists, k)
  let frame = build()
  // two-handed letters whose hands touch in the photo: move the wrists until the same points touch on the avatar
  const contact = sides.length === 2 ? photoContact(hands, k, rig.Left.geo.hand) : null
  if (contact) {
    for (let it = 0; it < 3; it++) {
      const gap = avatarGap(rig, frame, contact)
      if (!gap) break
      const delta = contact.vec.clone().sub(gap) // wanted (photo) gap minus the avatar's gap, avatar units
      if (delta.length() < 0.05 * rig.Left.geo.hand) break
      wrists.Right.add(delta.clone().multiplyScalar(0.5))
      wrists.Left.sub(delta.clone().multiplyScalar(0.5))
      frame = build()
    }
  }
  return { ...clip, width: 1, height: 1, frames: clip.frames.map(() => frame), contact: contact ? { pair: contact.pair, photo: contact.dist } : null }
}

/** Avatar units per photo unit: the 25th percentile over many hand segments (a tilted palm or finger looks short in a photo, never long). */
function photoScale(rig, sides, hands) {
  const ratios = []
  for (const side of sides) {
    const seg = rig[side].geo.handSeg || {}
    for (const [name, [a, b]] of Object.entries(HAND_SEGMENTS)) {
      const photoLen = hands[side][a].distanceTo(hands[side][b])
      if (seg[name] && photoLen > 1e-6) ratios.push(seg[name] / photoLen)
    }
  }
  if (!ratios.length) {
    const size = sides.reduce((sum, side) => sum + hands[side][9].distanceTo(hands[side][0]), 0) / sides.length
    return size > 1e-6 ? rig[sides[0]].geo.hand / size : null
  }
  ratios.sort((x, y) => x - y)
  return ratios[Math.floor(ratios.length * 0.25)]
}

/** The closest landmark pair between the two photo hands, if they touch: {pair, vec (Left -> Right, avatar units), dist (hand sizes)}. */
function photoContact(hands, k, handLen) {
  let best = null
  for (let a = 0; a < 21; a++) {
    for (let b = 0; b < 21; b++) {
      const d = hands.Right[a].distanceTo(hands.Left[b])
      if (!best || d < best.d) best = { a, b, d }
    }
  }
  const dist = (best.d * k) / handLen
  if (dist > CONTACT_MAX) return null
  return { pair: [best.b, best.a], vec: hands.Right[best.a].clone().sub(hands.Left[best.b]).multiplyScalar(k), dist }
}

/** Solve one letter frame on the rig's skeleton and measure the contact pair (the skeleton's pose is restored). */
function avatarGap(rig, frame, contact) {
  const solved = solveClip(rig, { frames: [frame], width: 1, height: 1 })
  if (!solved) return null
  const bones = rig.restPose.map((r) => r.bone)
  const top = topOf(bones[0])
  const saved = bones.map((b) => b.quaternion.clone())
  for (const [b, q] of solved[0]) b.quaternion.copy(q)
  top.updateMatrixWorld(true)
  const at = (side, i) => {
    const name = LANDMARK_BONE[i] === 'Hand' ? `${side}Hand` : `${side}Hand${LANDMARK_BONE[i]}`
    const b = top.getObjectByName(`mixamorig${name}`) || top.getObjectByName(`mixamorig:${name}`)
    return b ? b.getWorldPosition(v3()) : null
  }
  const l = at('Left', contact.pair[0])
  const r = at('Right', contact.pair[1])
  bones.forEach((b, i) => b.quaternion.copy(saved[i]))
  top.updateMatrixWorld(true)
  return l && r ? r.sub(l) : null
}

function topOf(o) {
  let x = o
  while (x.parent) x = x.parent
  return x
}

function buildLetterFrame(rig, sides, hands, wrists, k) {
  const toMp = (v) => [v.x, -v.y, -v.z] // back to MediaPipe world axes, as recorded clips have them
  const pose = Array.from({ length: 33 }, () => [0, 0, 0])
  const worldHands = [null, null]
  for (const side of ['Left', 'Right']) {
    const g = rig[side].geo
    const ids = POSE[side]
    pose[ids.shoulder] = toMp(g.shoulder)
    if (!wrists[side]) { // unused arm: hangs at the side, elbow a little bent
      const out = side === 'Left' ? 1 : -1
      const elbow = g.shoulder.clone().add(v3().set(0.12 * out, -1, 0.05).normalize().multiplyScalar(g.upper))
      pose[ids.elbow] = toMp(elbow)
      pose[ids.wrist] = toMp(elbow.clone().add(v3().set(0.05 * out, -0.9, 0.4).normalize().multiplyScalar(g.fore)))
      continue
    }
    const hand = hands[side]
    const handDir = hand[9].clone().sub(hand[0]).normalize()
    const { elbow, wrist } = bestElbow(g, wrists[side], handDir, side === 'Left' ? 1 : -1)
    pose[ids.elbow] = toMp(elbow)
    pose[ids.wrist] = toMp(wrist)
    worldHands[HAND_SLOT[side]] = hand.map((q) => toMp(wrist.clone().add(q.clone().sub(hand[0]).multiplyScalar(k))))
  }
  return { pose: null, hands: [null, null], world: { pose, hands: worldHands } }
}

/**
 * Of the elbow positions that put the wrist on target, a relaxed one: hanging low, on its own side of the body,
 * with the forearm lined up with the hand where the wrist alone could not bend that far (beyond ~50 deg).
 */
function bestElbow(g, target, handDir, outward) {
  let best = null
  for (let a = 0; a < 24; a++) {
    const t = (a / 24) * 2 * Math.PI
    const pole = v3().set(Math.cos(t), Math.sin(t) * 0.7, Math.sin(t) * 0.7)
    const ik = twoBoneIK(g.shoulder, target, g.upper, g.fore, pole)
    const fore = ik.wrist.clone().sub(ik.elbow).normalize()
    let cost = Math.max(0, fore.angleTo(handDir) - 50 * D)
    const up = ik.elbow.y - (g.shoulder.y - 0.55 * g.upper)
    if (up > 0) cost += 2.5 * up / g.upper // raised elbows look forced
    if ((ik.elbow.x - g.shoulder.x) * outward < -0.7 * g.upper) cost += 1 // elbow far across the body
    const wide = (ik.elbow.x - g.shoulder.x) * outward - 0.3 * g.upper
    if (wide > 0) cost += 2 * wide / g.upper // signers keep the elbows near the body, not flared out
    if (!best || cost < best.cost) best = { ...ik, cost }
  }
  return best
}

/** Elbow for a shoulder -> wrist reach with the given bone lengths, bent towards `pole`; the wrist is pulled in when out of reach. */
function twoBoneIK(shoulder, target, upper, fore, pole) {
  const u = target.clone().sub(shoulder)
  const d = Math.min(Math.max(u.length(), Math.abs(upper - fore) + 1e-4), (upper + fore) * 0.985)
  u.normalize()
  const wrist = shoulder.clone().add(u.clone().multiplyScalar(d))
  let side = pole.clone().projectOnPlane(u)
  if (side.lengthSq() < 1e-8) side = v3().set(0, -1, 0).projectOnPlane(u)
  side.normalize()
  const x = (upper * upper - fore * fore + d * d) / (2 * d)
  const h = Math.sqrt(Math.max(0, upper * upper - x * x))
  return { elbow: shoulder.clone().add(u.multiplyScalar(x)).add(side.multiplyScalar(h)), wrist }
}

/**
 * The clip without the signer's resting frames at its start and end (both wrists low, near the hips): recordings
 * begin and end with the hands down, about 40% of the frames, which made the avatar stand still between words.
 * Keeps `pad` frames either side; a clip whose hands never rise is returned whole.
 */
export function trimRest(clip, pad = 2) {
  const frames = clip?.frames || []
  const raised = frames.map((f) => {
    const p = f.pose || f.world?.pose // both have y pointing down
    if (!p?.[24]) return true
    const shoulder = (p[11][1] + p[12][1]) / 2
    const hip = (p[23][1] + p[24][1]) / 2
    const low = hip - 0.3 * (hip - shoulder)
    return !(p[15][1] > low && p[16][1] > low)
  })
  const first = raised.indexOf(true)
  if (first < 0) return clip
  const last = raised.lastIndexOf(true)
  const from = Math.max(0, first - pad)
  const to = Math.min(frames.length, last + pad + 1)
  return from === 0 && to === frames.length ? clip : { ...clip, frames: frames.slice(from, to) }
}

export { POSE, HAND_SLOT, FINGERS }

/** Centred smoothing of one landmark array across frames (null frames are skipped). */
/**
 * ``angle`` (radians, in (-pi, pi]) or the same angle one turn up / down, whichever is closest to ``prev``, but
 * never further than UNWRAP_MARGIN past half a turn: a twist crossing +-180 deg stays continuous for a moment
 * instead of flipping sign, without drifting into angles the joint limits would then clamp into a wrong pose.
 */
export function unwrapAngle(prev, angle) {
  if (prev === undefined || !Number.isFinite(prev)) return angle
  const bound = Math.PI + UNWRAP_MARGIN
  return [angle, angle + 2 * Math.PI, angle - 2 * Math.PI]
    .filter((a) => Math.abs(a) <= bound)
    .reduce((best, a) => (Math.abs(a - prev) < Math.abs(best - prev) ? a : best), angle)
}

export function despike(tracks) {
  // a 3-frame median per coordinate: one mis-detected frame (a hand jumping away and back) is replaced, while
  // steady motion passes unchanged (measured: 1-2% of wrist / forearm steps above 720 deg/s came from such frames)
  return tracks.map((own, i) => {
    const a = tracks[i - 1]
    const b = tracks[i + 1]
    if (!own || !a || !b) return own
    return own.map((p, j) => p.map((v, c) => {
      const x = a[j]?.[c] ?? v
      const y = b[j]?.[c] ?? v
      return Math.max(Math.min(x, y), Math.min(Math.max(x, y), v))
    }))
  })
}

function smoothTrack(frames, get) {
  const out = frames.map(() => null)
  const r = (SMOOTH.length - 1) / 2
  const tracks = despike(frames.map((f) => get(f) || null))
  for (let i = 0; i < frames.length; i++) {
    const own = tracks[i]
    if (!own) continue
    const acc = own.map(() => [0, 0, 0])
    let wsum = 0
    for (let k = -r; k <= r; k++) {
      const pts = tracks[i + k]
      if (!pts) continue
      const w = SMOOTH[k + r]
      wsum += w
      for (let j = 0; j < pts.length; j++) for (let c = 0; c < 3; c++) acc[j][c] += w * (pts[j][c] || 0)
    }
    out[i] = acc.map((p) => p.map((v) => v / wsum))
  }
  return out
}

/**
 * Local bone rotations for every frame of a /reference clip: [Map bone -> Quaternion], or null when the
 * clip has no usable pose. Missing hands keep their previous frame (or the rest pose).
 */
/**
 * The smoothed landmark positions the solver aims at, per frame: { pose(i), hand[slot](i) } as world-like
 * vectors (functions, or null when missing). Exported so the agreement check measures the same targets.
 */
export function clipTargets(clip) {
  const width = clip.width || 640
  const height = clip.height || 480
  // world landmarks (metres, real depth) when the clip has them, else image landmarks
  const useWorld = clip.frames.some((f) => f.world?.pose)
  const poses = smoothTrack(clip.frames, (f) => (useWorld ? f.world?.pose : f.pose))
  const hands = [0, 1].map((slot) => smoothTrack(clip.frames, (f) => (useWorld ? f.world?.hands?.[slot] : f.hands?.[slot])))
  const W = useWorld ? (p) => toWorld(p, 1, 1, 1) : null // world: x right, y down, z away from the camera
  return clip.frames.map((_, fi) => ({
    pose: poses[fi] ? (i) => (W ? W(poses[fi][i]) : toWorld(poses[fi][i], width, height, POSE_DEPTH)) : null,
    hand: [0, 1].map((slot) => (hands[slot][fi]
      ? (i) => (W ? W(hands[slot][fi][i]) : toWorld(hands[slot][fi][i], width, height, HAND_DEPTH))
      : null)),
    world: useWorld,
  }))
}

export function solveClip(rig, clip) {
  if (!rig || !clip?.frames?.length) return null
  const targets = clipTargets(clip)
  const out = []
  const last = new Map()
  const lastBend = {}
  const prev = new Map() // last angle per joint: each new angle is unwrapped next to it (no 360 deg flips)
  const cont = (key, angle) => { const a = unwrapAngle(prev.get(key), angle); prev.set(key, a); return a }
  let any = false
  for (let fi = 0; fi < clip.frames.length; fi++) {
    const locals = new Map()
    for (const side of ['Left', 'Right']) {
      const s = rig[side]
      const P = targets[fi].pose
      if (!P) continue
      const sh = P(POSE[side].shoulder)
      const el = P(POSE[side].elbow)
      const wr = P(POSE[side].wrist)
      const upper = el.clone().sub(sh)
      const lower = wr.clone().sub(el)
      if (upper.lengthSq() < 1e-6 || lower.lengthSq() < 1e-6) continue
      upper.normalize()
      lower.normalize()
      any = true

      // upper arm: aim at the elbow, roll so the elbow hinge matches the signer's arm plane
      // elbow plane: ill-defined when the arm is nearly straight or fully folded (|bend| -> 0), where it can turn
      // 100+ deg in one frame (measured: YOU frame 17); blend it with the last plane by how well it is defined
      let bend = v3().crossVectors(upper, lower)
      const sure = Math.min(1, Math.max(0, (bend.length() - BEND_UNSURE) / (BEND_SURE - BEND_UNSURE)))
      const before = lastBend[side]
      if (bend.lengthSq() > 1e-10) bend.normalize()
      if (before && sure < 1) bend = sure > 0 ? before.clone().lerp(bend, sure).normalize() : before.clone()
      else if (!before && sure === 0) bend = null
      if (bend) lastBend[side] = bend.clone()
      const armW = aimWithRoll(s.arm.restQ, s.arm.restDir, s.arm.restHinge, upper, bend)
      locals.set(s.arm.bone, inv(s.shoulder.restQ).multiply(armW))

      // forearm: hinge on the elbow
      const foreFrame = armW.clone().multiply(s.fore.restLocal)
      const elbow = clamp(hingeAngle(s.fore.dirLocal, s.fore.hingeLocal, local(foreFrame, lower)), LIMITS.elbow)
      let foreLocal = s.fore.restLocal.clone().multiply(new THREE.Quaternion().setFromAxisAngle(s.fore.hingeLocal, elbow))
      let foreW = armW.clone().multiply(foreLocal)

      // hand: palm frame as recorded; share its twist with the forearm, limit the rest
      const H = targets[fi].hand[HAND_SLOT[side]]
      let handW = null
      if (s.hand && H) {
        const w = H(0)
        const dir = H(9).sub(w)
        const normal = v3().crossVectors(H(5).sub(w), H(17).sub(w))
        if (dir.lengthSq() > 1e-8 && normal.lengthSq() > 1e-8) {
          const target = aimWithRoll(s.hand.restQ, s.hand.restDir, s.hand.restNormal, dir.normalize(), normal.normalize())
          const rel = inv(foreW.clone().multiply(s.hand.restLocal)).multiply(target)
          const { twist } = swingTwist(rel, s.hand.dirLocal)
          const share = cont(`${side}:twist`, signedTwist(twist, s.hand.dirLocal)) * LIMITS.forearmTwistShare
          foreLocal = foreLocal.multiply(new THREE.Quaternion().setFromAxisAngle(s.fore.dirLocal, share))
          foreW = armW.clone().multiply(foreLocal)
          const rel2 = inv(foreW.clone().multiply(s.hand.restLocal)).multiply(target)
          const parts = swingTwist(rel2, s.hand.dirLocal)
          const tw = clamp(cont(`${side}:wrist`, signedTwist(parts.twist, s.hand.dirLocal)), [-LIMITS.wristTwist, LIMITS.wristTwist])
          const limited = limitAngle(parts.swing, LIMITS.wristSwing).multiply(new THREE.Quaternion().setFromAxisAngle(s.hand.dirLocal, tw))
          const handLocal = s.hand.restLocal.clone().multiply(limited)
          handW = foreW.clone().multiply(handLocal)
          locals.set(s.hand.bone, handLocal)

          // fingers, parents first
          const worlds = []
          s.fingers.forEach((f, i) => {
            const parentW = f.parent === 'hand' ? handW : worlds[f.parent]
            const frame = parentW.clone().multiply(f.restLocal)
            const seg = H(f.to).sub(H(f.from))
            let S = new THREE.Quaternion()
            if (seg.lengthSq() > 1e-10) {
              const t = local(frame, seg.normalize())
              if (f.kind === 'pip' || f.kind === 'dip') {
                const angle = clamp(cont(f.bone.uuid, hingeAngle(f.dirLocal, f.hingeLocal, t)), LIMITS[f.kind])
                S.setFromAxisAngle(f.hingeLocal, angle)
              } else if (f.kind === 'knuckle') {
                const curl = v3().crossVectors(f.hingeLocal, f.dirLocal)
                const flex = clamp(cont(f.bone.uuid, Math.atan2(t.dot(curl), t.dot(f.dirLocal))), LIMITS.knuckleFlex)
                const spread = clamp(Math.asin(Math.max(-1, Math.min(1, t.dot(f.hingeLocal)))), [-LIMITS.knuckleSpread, LIMITS.knuckleSpread])
                const aimed = f.dirLocal.clone().multiplyScalar(Math.cos(flex)).add(curl.multiplyScalar(Math.sin(flex)))
                  .multiplyScalar(Math.cos(spread)).add(f.hingeLocal.clone().multiplyScalar(Math.sin(spread))).normalize()
                S.setFromUnitVectors(f.dirLocal, aimed) // swing only, then a limited turn about the finger:
                const roll = knuckleRoll(f, s.fingers[i + 1], frame.clone().multiply(S), seg, H)
                if (roll) S.multiply(new THREE.Quaternion().setFromAxisAngle(f.dirLocal, roll))
              } else {
                S = limitAngle(new THREE.Quaternion().setFromUnitVectors(f.dirLocal, t), LIMITS.thumbSwing)
              }
            }
            const fl = f.restLocal.clone().multiply(S)
            worlds.push(parentW.clone().multiply(fl))
            locals.set(f.bone, fl)
          })
        }
      }
      locals.set(s.fore.bone, foreLocal)
    }
    for (const [b, q] of last) if (!locals.has(b)) locals.set(b, q) // hold the last known pose
    for (const [b, q] of locals) last.set(b, q)
    out.push(locals)
  }
  return any ? rateLimit(out, clip.fps || 15) : null
}

/**
 * No bone turns faster than a person can move it: per frame, each local rotation goes at most MAX_DEG_PER_S / fps
 * towards the solved one. Removes the remaining one-frame flips (hidden fingers flickering between curled and
 * straight in the landmarks, arm planes near straight), which then glide over a few frames instead of snapping.
 */
function rateLimit(frames, fps) {
  const last = new Map()
  for (const locals of frames) {
    for (const [bone, q] of locals) {
      const p = last.get(bone)
      if (p) {
        const finger = /Hand(Thumb|Index|Middle|Ring|Pinky)/.test(bone.name)
        const maxRad = ((finger ? MAX_DEG_PER_S.finger : MAX_DEG_PER_S.arm) * D) / fps
        const angle = 2 * Math.acos(Math.min(1, Math.abs(p.dot(q))))
        if (angle > maxRad) q.copy(p.clone().slerp(q, maxRad / angle))
      }
      last.set(bone, q)
    }
  }
  return frames
}
