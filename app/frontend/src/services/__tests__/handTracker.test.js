import { describe, expect, it, vi } from 'vitest'

vi.mock('@mediapipe/tasks-vision', () => ({ FilesetResolver: {}, HandLandmarker: {}, PoseLandmarker: {} }))
const { describeGpu } = await import('../handTracker')

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
