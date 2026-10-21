import { describe, expect, it } from 'vitest'
import { cameraErrorKind, createRecorder, pickMimeType } from '../recorder'
import { SHARE_STATUS, shareReport, shareSupport, telLink } from '../dial'

/** A fake MediaRecorder: start() records, stop() delivers one chunk and fires onstop. */
class FakeRecorder {
  static isTypeSupported(t) { return t === 'video/webm' }
  constructor(stream, opts) { this.stream = stream; this.mimeType = opts?.mimeType || ''; this.state = 'inactive' }
  start() { this.state = 'recording' }
  stop() { this.state = 'inactive'; this.ondataavailable({ data: new Blob(['frame'], { type: 'video/webm' }) }); this.onstop() }
}
const stream = () => ({ getTracks: () => [{ stopped: false, stop() { this.stopped = true } }] })
function setup({ getUserMedia, MediaRecorderImpl = FakeRecorder } = {}) {
  let t = 0
  const s = stream()
  const rec = createRecorder({
    mediaDevices: { getUserMedia: getUserMedia || (async () => s) },
    MediaRecorderImpl, now: () => t, setTimer: () => 1, clearTimer: () => {},
  })
  return { rec, advance: (sec) => { t += sec * 1000 }, s }
}

describe('gesture recorder', () => {
  it('asks for the camera only when requested, then previews, records, stops and keeps the take', async () => {
    let asked = 0
    const { rec, advance } = setup({ getUserMedia: async () => { asked++; return stream() } })
    expect(asked).toBe(0)
    expect(rec.state.status).toBe('idle')
    await rec.requestCamera()
    expect(asked).toBe(1)
    expect(rec.state.status).toBe('preview')
    rec.start()
    expect(rec.state.status).toBe('recording')
    advance(3.24)
    rec.stop()
    expect(rec.state.status).toBe('recorded')
    expect(rec.state.duration).toBe(3.2)
    expect(rec.state.mimeType).toBe('video/webm')
    expect(rec.state.blob.size).toBeGreaterThan(0)
  })

  it('retake goes back to the live preview and drops the take', async () => {
    const { rec, advance } = setup()
    await rec.requestCamera(); rec.start(); advance(2); rec.stop()
    rec.retake()
    expect(rec.state.status).toBe('preview')
    expect(rec.state.blob).toBeNull()
  })

  it('rejects a take that is too short', async () => {
    const { rec, advance } = setup()
    await rec.requestCamera(); rec.start(); advance(0.3); rec.stop()
    expect(rec.state.status).toBe('preview')
    expect(rec.state.error).toBe('too_short')
  })

  it('handles permission denied, missing camera and unsupported browsers', async () => {
    const deny = setup({ getUserMedia: async () => { throw Object.assign(new Error('no'), { name: 'NotAllowedError' }) } })
    await deny.rec.requestCamera()
    expect(deny.rec.state.status).toBe('denied')
    const none = setup({ getUserMedia: async () => { throw Object.assign(new Error('no'), { name: 'NotFoundError' }) } })
    await none.rec.requestCamera()
    expect(none.rec.state.error).toBe('no_camera')
    const old = createRecorder({ mediaDevices: undefined, MediaRecorderImpl: undefined })
    await old.requestCamera()
    expect(old.state.status).toBe('unsupported')
    expect(cameraErrorKind({ name: 'NotReadableError' })).toBe('busy')
  })

  it('turns the camera off on release', async () => {
    const { rec, s } = setup()
    await rec.requestCamera()
    const track = s.getTracks()[0]
    rec.state.stream = { getTracks: () => [track] }
    rec.release()
    expect(track.stopped).toBe(true)
    expect(rec.state.stream).toBeNull()
  })

  it('picks a recordable type', () => {
    expect(pickMimeType(FakeRecorder)).toBe('video/webm')
    expect(pickMimeType(undefined)).toBeNull()
    expect(pickMimeType(class {})).toBe('')
  })
})

describe('dial and share', () => {
  it('builds dialer links with digits only', () => {
    expect(telLink('112')).toBe('tel:112')
    expect(telLink(' 1 0 8 ')).toBe('tel:108')
    expect(telLink('javascript:alert(1)')).toBeNull()
    expect(telLink('')).toBeNull()
  })

  it('knows whether a video can be attached', () => {
    const file = new Blob(['x'])
    expect(shareSupport(null, file)).toBe('none')
    expect(shareSupport({ share: () => {} }, file)).toBe('text')
    expect(shareSupport({ share: () => {}, canShare: () => true }, file)).toBe('files')
    expect(shareSupport({ share: () => {}, canShare: () => false }, file)).toBe('text')
  })

  it('never claims delivery: the share sheet only reaches "sharing started"', async () => {
    let shared = null
    const nav = { share: async (d) => { shared = d }, canShare: () => true }
    expect(await shareReport({ nav, text: 'R', file: new Blob(['v']), withVideo: true })).toBe('sharing')
    expect(shared.files).toHaveLength(1)
    expect(await shareReport({ nav, text: 'R', file: new Blob(['v']), withVideo: false })).toBe('sharing')
    expect(shared.files).toBeUndefined() // no video without consent
    const cancel = { share: async () => { throw Object.assign(new Error('x'), { name: 'AbortError' }) } }
    expect(await shareReport({ nav: cancel, text: 'R' })).toBe('cancelled')
    const fails = { share: async () => { throw new Error('x') } }
    expect(await shareReport({ nav: fails, text: 'R' })).toBe('failed')
    let copied = null
    expect(await shareReport({ nav: { clipboard: { writeText: async (t) => { copied = t } } }, text: 'R' })).toBe('copied')
    expect(copied).toBe('R')
    for (const label of Object.values(SHARE_STATUS).filter(Boolean)) expect(label).not.toMatch(/^(delivered|received|acknowledged)/i)
    expect(SHARE_STATUS.sharing).toMatch(/cannot see whether it was delivered/)
  })
})
