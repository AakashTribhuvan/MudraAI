/**
 * Signs without a recording whose form is unambiguous, written as hand landmarks in the same format as a letter
 * photo (/reference/LETTER_X: image x = signer-left, y down, z away from the camera), so letterClip() places them
 * in front of the chest and the landmark solver poses the avatar exactly as it does for recorded data.
 * Only add a sign here when its form is certain; otherwise record it on the Teach a sign page.
 */

// A flat right hand, fingers up, palm facing the signer's left, thumb towards the body; metres, wrist at 0,
// written in avatar axes (x = signer-left, y up, z forward)
const FLAT_RIGHT = (() => {
  const pts = [[0, 0, 0]]
  pts.push([0.008, 0.02, -0.025], [0.01, 0.04, -0.045], [0.01, 0.06, -0.055], [0.01, 0.08, -0.06]) // thumb 1-4
  const fingers = [[-0.03, 0.085], [-0.01, 0.09], [0.01, 0.085], [0.03, 0.075]] // index, middle, ring, pinky: z, MCP height
  const segs = [0.04, 0.025, 0.02]
  for (const [z, base] of fingers) {
    let y = base
    pts.push([0, y, z])
    for (const s of segs) {
      y += s
      pts.push([0, y, z])
    }
  }
  return pts
})()

const toPhoto = ([x, y, z], dx) => [0.5 + x + dx, 0.5 - y, -z]

/** Palms pressed together in front of the chest, fingers up. */
function namaste() {
  const gap = 0.012 // half the hands' thickness
  const right = FLAT_RIGHT.map((p) => toPhoto(p, -gap))
  const left = FLAT_RIGHT.map(([x, y, z]) => toPhoto([-x, y, z], gap)) // mirror image: the left hand
  return { sign: 'NAMASTE', fps: 15, width: 100, height: 100, kind: 'synth', frames: Array(18).fill({ hands: [left, right], pose: null }) }
}

export const SYNTH_SIGNS = { NAMASTE: namaste }
