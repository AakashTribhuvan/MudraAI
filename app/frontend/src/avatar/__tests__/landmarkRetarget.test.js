import { describe, expect, it } from 'vitest'
import * as THREE from 'three'
import { buildRig, despike, letterClip, solveClip, trimRest, unwrapAngle } from '../landmarkRetarget'
import { glideDuration } from '../useAvatarRenderer'
import { SYNTH_SIGNS } from '../synthSigns'

/** A tiny Mixamo-named skeleton in T-pose: shoulders at +-0.2, arms along +-x, hands with three fingers. */
function makeSkeleton() {
  const root = new THREE.Group()
  const bone = (name, parent, x, y = 0, z = 0) => {
    const b = new THREE.Bone()
    b.name = name
    b.position.set(x, y, z)
    parent.add(b)
    return b
  }
  const spine = bone('mixamorigSpine2', root, 0, 1.4)
  for (const [side, s] of [['Left', 1], ['Right', -1]]) {
    const shoulder = bone(`mixamorig${side}Shoulder`, spine, 0.1 * s)
    const arm = bone(`mixamorig${side}Arm`, shoulder, 0.1 * s)
    const fore = bone(`mixamorig${side}ForeArm`, arm, 0.3 * s)
    const hand = bone(`mixamorig${side}Hand`, fore, 0.25 * s)
    for (const [finger, z] of [['Index', 0.03], ['Middle', 0], ['Pinky', -0.04]]) {
      let parent = hand
      for (let k = 1; k <= 4; k++) parent = bone(`mixamorig${side}Hand${finger}${k}`, parent, (k === 1 ? 0.08 : 0.03) * s, 0, k === 1 ? z : 0)
    }
  }
  return root
}

const find = (root, name) => root.getObjectByName(name)

function pose(points) {
  const p = Array.from({ length: 33 }, () => [0.5, 0.5, 0, 1])
  for (const [i, xy] of Object.entries(points)) p[i] = [xy[0], xy[1], 0, 1]
  return p
}

describe('landmark retargeting', () => {
  it('builds a rig for both arms and their fingers', () => {
    const rig = buildRig(makeSkeleton(), find)
    expect(rig.Left.fingers.length).toBe(9) // 3 fingers x 3 joints
    expect(rig.restPose.length).toBeGreaterThan(20)
  })

  it('points the arm where the signer pointed (arm hanging down, forearm forward-up)', () => {
    const root = makeSkeleton()
    const rig = buildRig(root, find)
    // image coordinates (y down): signer's left shoulder (index 11) at x 0.6, elbow straight below, wrist up-right
    const frame = { pose: pose({ 11: [0.6, 0.4], 13: [0.6, 0.6], 15: [0.7, 0.5], 12: [0.4, 0.4], 14: [0.4, 0.6], 16: [0.3, 0.6] }), hands: [null, null] }
    const [solved] = solveClip(rig, { frames: [frame], width: 100, height: 100 })
    for (const [b, q] of solved) b.quaternion.copy(q)
    root.updateMatrixWorld(true)
    const at = (n) => root.getObjectByName(n).getWorldPosition(new THREE.Vector3())
    const upper = at('mixamorigLeftForeArm').sub(at('mixamorigLeftArm')).normalize()
    const lower = at('mixamorigLeftHand').sub(at('mixamorigLeftForeArm')).normalize()
    expect(upper.y).toBeLessThan(-0.99) // straight down
    expect(lower.x).toBeGreaterThan(0.6) // out to the signer's left (avatar +x) ...
    expect(lower.y).toBeGreaterThan(0.6) // ... and up
  })

  it('returns null without usable pose data', () => {
    const rig = buildRig(makeSkeleton(), find)
    expect(solveClip(rig, { frames: [{ pose: null, hands: [null, null] }] })).toBeNull()
  })

  it('trims the resting frames (hands down at the hips) from both ends of a clip', () => {
    const down = { pose: pose({ 11: [0.6, 0.3], 12: [0.4, 0.3], 23: [0.55, 0.7], 24: [0.45, 0.7], 15: [0.6, 0.75], 16: [0.4, 0.75] }), hands: [null, null] }
    const up = { pose: pose({ 11: [0.6, 0.3], 12: [0.4, 0.3], 23: [0.55, 0.7], 24: [0.45, 0.7], 15: [0.55, 0.35], 16: [0.4, 0.75] }), hands: [null, null] }
    const clip = { frames: [down, down, down, down, up, up, down, down, down, down] }
    expect(trimRest(clip, 1).frames.length).toBe(4) // 2 raised + 1 either side
    const still = { frames: [down, down, down] }
    expect(trimRest(still)).toBe(still) // hands never rise: kept whole
  })

  it('places a two-handed letter in front of the chest, keeping the hands apart as in the photo', () => {
    const rig = buildRig(makeSkeleton(), find)
    const clip = letterClip(rig, SYNTH_SIGNS.NAMASTE())
    const f = clip.frames[0].world
    const [lw, rw] = [f.pose[15], f.pose[16]] // MediaPipe world axes: x signer-left, y down, z away
    expect(lw[0]).toBeGreaterThan(rw[0]) // left hand on the signer's left
    expect(Math.abs(lw[0] - rw[0])).toBeLessThan(0.1) // palms together
    expect(lw[2]).toBeLessThan(f.pose[11][2]) // in front of the shoulders
    expect(solveClip(rig, clip)).not.toBeNull()
  })

  it('lets the unused arm hang for a one-handed letter', () => {
    const rig = buildRig(makeSkeleton(), find)
    const right = SYNTH_SIGNS.NAMASTE().frames[0].hands[1]
    const clip = letterClip(rig, { fps: 15, frames: [{ hands: [null, right], pose: null }] })
    const p = clip.frames[0].world.pose
    expect(p[15][1]).toBeGreaterThan(p[13][1]) // left wrist below the left elbow (y down)
    expect(p[13][1]).toBeGreaterThan(p[11][1]) // elbow below the shoulder
    expect(clip.frames[0].world.hands[0]).toBeNull()
  })

  it('keeps the true hand scale when a photo foreshortens the palm (letter A cause)', () => {
    const rig = buildRig(makeSkeleton(), find)
    // the same flat hand, once as seen straight on and once with the palm (wrist -> knuckles) squashed to 40%,
    // as when the palm tilts towards the camera; the fingers keep their length
    const hand = (dx, squash) => {
      const pts = Array.from({ length: 21 }, () => [0, 0, 0])
      const knuckle = (i, x) => { pts[i] = [x, -0.09 * squash, 0] }
      knuckle(5, 0.02); knuckle(9, 0); knuckle(13, -0.015); knuckle(17, -0.03)
      for (const [base, x] of [[5, 0.02], [9, 0], [13, -0.015], [17, -0.03]]) {
        for (let k = 1; k <= 3; k++) pts[base + k] = [x, pts[base][1] - 0.03 * k, 0]
      }
      for (let k = 1; k <= 4; k++) pts[k] = [0.03 + 0.01 * k, -0.02 * k, 0]
      return pts.map(([x, y, z]) => [0.5 + x + dx, 0.5 + y, z])
    }
    const gap = (squash) => {
      const clip = letterClip(rig, { fps: 15, frames: [{ hands: [hand(0.15, squash), hand(-0.15, squash)], pose: null }] })
      const p = clip.frames[0].world.pose
      return Math.abs(p[15][0] - p[16][0])
    }
    expect(gap(0.4)).toBeLessThan(gap(1) * 1.15) // before the fix the squashed palm made the gap 2.5x larger
    expect(gap(0.4)).toBeGreaterThan(gap(1) * 0.85)
  })

  it('returns no letter clip without hand data (missing or invalid reference)', () => {
    const rig = buildRig(makeSkeleton(), find)
    expect(letterClip(rig, null)).toBeNull()
    expect(letterClip(rig, { frames: [] })).toBeNull()
    expect(letterClip(rig, { fps: 15, frames: [{ hands: [null, null], pose: null }] })).toBeNull()
  })

  it('keeps twist angles continuous across +-180 deg without drifting', () => {
    const near = Math.PI - 0.05
    expect(unwrapAngle(near, -near)).toBeCloseTo(Math.PI + 0.05) // crossing 180 deg: no jump to -175 deg
    expect(unwrapAngle(Math.PI + 0.5, 0.1)).toBeCloseTo(0.1) // never beyond the bound, so limits stay meaningful
    expect(unwrapAngle(undefined, 1)).toBe(1)
  })

  it('removes one-frame landmark spikes but keeps steady motion', () => {
    const track = [0, 0.1, 5, 0.3, 0.4].map((v) => [[v, 0, 0]])
    const out = despike(track)
    expect(out[2][0][0]).toBeCloseTo(0.3) // the spike is replaced by its neighbours' range
    expect(out[1][0][0]).toBeCloseTo(0.1)
    const ramp = despike([0, 1, 2, 3].map((v) => [[v, 0, 0]]))
    expect(ramp.map((f) => f[0][0])).toEqual([0, 1, 2, 3])
  })

  it('never turns a bone faster than a person can (no one-frame flips)', () => {
    const root = makeSkeleton()
    const rig = buildRig(root, find)
    // the left forearm points forward, then straight back the next frame (a 180 deg flip)
    const fwd = pose({ 11: [0.6, 0.4], 13: [0.7, 0.4], 15: [0.8, 0.4], 12: [0.4, 0.4], 14: [0.4, 0.6], 16: [0.4, 0.8] })
    const back = pose({ 11: [0.6, 0.4], 13: [0.7, 0.4], 15: [0.6, 0.41], 12: [0.4, 0.4], 14: [0.4, 0.6], 16: [0.4, 0.8] })
    const frames = solveClip(rig, { fps: 15, width: 100, height: 100, frames: [fwd, fwd, back, back].map((p) => ({ pose: p, hands: [null, null] })) })
    const fore = root.getObjectByName('mixamorigLeftForeArm')
    const deg = (a, b) => THREE.MathUtils.radToDeg(2 * Math.acos(Math.min(1, Math.abs(a.dot(b)))))
    for (let i = 1; i < frames.length; i++) expect(deg(frames[i - 1].get(fore), frames[i].get(fore))).toBeLessThanOrEqual(900 / 15 + 0.5)
  })

  it('times the glide between signs by how far the joints travel', () => {
    expect(glideDuration(0)).toBe(160)
    expect(glideDuration(120)).toBeGreaterThan(glideDuration(30))
    expect(glideDuration(1000)).toBe(620)
  })
})

