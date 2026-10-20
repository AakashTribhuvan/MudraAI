import { describe, expect, it } from 'vitest'
import * as THREE from 'three'
import { buildRig, letterClip, solveClip, trimRest } from '../landmarkRetarget'
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
})
