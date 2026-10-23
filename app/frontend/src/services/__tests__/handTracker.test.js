import { describe, expect, it, vi } from 'vitest'

vi.mock('@mediapipe/tasks-vision', () => ({ FilesetResolver: {}, HandLandmarker: {}, PoseLandmarker: {} }))
const { describeGpu, fromCrop, handCrop, handMissCounts, mergeHands } = await import('../handTracker')

describe('browser GPU detection', () => {
  it('names the GPU and tells on-chip graphics from a graphics card (Chrome on Windows strings)', () => {
    expect(describeGpu('ANGLE (Intel, Intel(R) UHD Graphics (0x0000A7A8) Direct3D11 vs_5_0 ps_5_0, D3D11)'))
      .toEqual({ name: 'Intel(R) UHD Graphics', integrated: true })
    expect(describeGpu('ANGLE (NVIDIA, NVIDIA GeForce RTX 5060 Laptop GPU (0x00002D19) Direct3D11 vs_5_0 ps_5_0, D3D11)'))
      .toEqual({ name: 'NVIDIA GeForce RTX 5060 Laptop GPU', integrated: false })
    expect(describeGpu('ANGLE (AMD, AMD Radeon(TM) Graphics (0x00001638) Direct3D11 vs_5_0 ps_5_0, D3D11)').integrated).toBe(true)
    expect(describeGpu('')).toEqual({ name: '', integrated: false })
  })
})

// body landmarks: shoulders at y 0.4, hips at y 0.8, both wrists raised to y 0.5 (above the middle of the torso)
const body = () => {
  const p = Array.from({ length: 33 }, () => ({ x: 0.5, y: 0.9, visibility: 1 }))
  p[11] = { x: 0.6, y: 0.4, visibility: 1 }
  p[12] = { x: 0.4, y: 0.4, visibility: 1 }
  p[23] = { x: 0.58, y: 0.8, visibility: 1 }
  p[24] = { x: 0.42, y: 0.8, visibility: 1 }
  p[15] = { x: 0.65, y: 0.5, visibility: 1 }
  p[16] = { x: 0.35, y: 0.5, visibility: 1 }
  p[17] = p[19] = { x: 0.66, y: 0.46, visibility: 1 }
  p[18] = p[20] = { x: 0.34, y: 0.46, visibility: 1 }
  return p
}
const handAt = (x, y) => Array.from({ length: 21 }, () => ({ x, y, z: 0 }))

describe('lost hands', () => {
  it('counts a raised body wrist without a hand near it as missed', () => {
    const one = handMissCounts([handAt(0.65, 0.5)], body())
    expect(one).toMatchObject({ raised: 2, missed: 1 })
    expect(one.where[0].wrist.x).toBe(0.35)
    expect(handMissCounts([handAt(0.65, 0.5), handAt(0.35, 0.5)], body())).toMatchObject({ raised: 2, missed: 0 })
    const low = body()
    low[15] = { ...low[15], y: 0.75 } // hand resting low: not raised
    expect(handMissCounts([], low)).toMatchObject({ raised: 1, missed: 1 })
  })

  it('searches a square around the lost hand, inside the picture, and maps landmarks back', () => {
    const where = handMissCounts([], body()).where[0]
    const crop = handCrop(where, 640, 480)
    expect(crop.side).toBeCloseTo(192) // 0.4 of the picture height here
    expect(crop.x).toBeGreaterThanOrEqual(0)
    expect(crop.x + crop.side).toBeLessThanOrEqual(640)
    const back = fromCrop([{ x: 0.5, y: 0.5, z: 0 }], crop, 640, 480)[0]
    expect(back.x).toBeCloseTo((crop.x + 96) / 640)
    expect(back.y).toBeCloseTo((crop.y + 96) / 480)
  })

  it('adds a found hand only when it is not one already tracked, and never more than two', () => {
    const tracked = { landmarks: [handAt(0.65, 0.5)], handedness: [[{ categoryName: 'Left' }]] }
    expect(mergeHands(tracked, { landmarks: [handAt(0.66, 0.5)], handedness: [[]] }).added).toBe(0)
    const two = mergeHands(tracked, { landmarks: [handAt(0.35, 0.5), handAt(0.1, 0.1)], handedness: [[], []] })
    expect(two.added).toBe(1)
    expect(two.landmarks).toHaveLength(2)
  })
})
