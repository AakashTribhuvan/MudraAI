import { useEffect, useMemo, useRef, useState } from 'react'
import { Camera, Circle, RotateCcw, Square } from 'lucide-react'
import Button from '../../components/ui/Button'
import { Hint } from '../../features/emergency/components'
import { MAX_SECONDS, RECORDER_MESSAGES, createRecorder } from '../../features/emergency/recorder'

/**
 * Camera panel for a gesture demonstration: explains why the camera is needed, asks for it only when the user
 * presses "Record gesture", shows a live preview, then Start / Stop / Preview / Retake. Calls onRecorded(blob,
 * seconds) when a take is kept, or onSkip() to continue without a recording.
 */
export default function GestureRecorder({ onRecorded, onSkip, keepLabel = 'Use this recording' }) {
  const [st, setSt] = useState({ status: 'idle' })
  const [elapsed, setElapsed] = useState(0)
  const liveRef = useRef(null)
  const recorder = useMemo(() => createRecorder({
    mediaDevices: typeof navigator !== 'undefined' ? navigator.mediaDevices : undefined,
    MediaRecorderImpl: typeof window !== 'undefined' ? window.MediaRecorder : undefined,
    onChange: setSt,
  }), [])
  const previewUrl = useMemo(() => (st.blob ? URL.createObjectURL(st.blob) : null), [st.blob])

  useEffect(() => () => recorder.release(), [recorder]) // camera off when the panel closes
  useEffect(() => () => { if (previewUrl) URL.revokeObjectURL(previewUrl) }, [previewUrl])
  useEffect(() => {
    if (liveRef.current && st.stream && liveRef.current.srcObject !== st.stream) liveRef.current.srcObject = st.stream
  }, [st.stream, st.status])
  useEffect(() => {
    if (st.status !== 'recording') return undefined
    const t = setInterval(() => setElapsed((Date.now() - st.startedAt) / 1000), 200)
    return () => clearInterval(t)
  }, [st.status, st.startedAt])

  const failed = ['unsupported', 'denied', 'error'].includes(st.status)
  return (
    <div className="space-y-3">
      {st.status === 'idle' && (
        <>
          <Hint>
            The camera records a short video of you making the gesture (up to {MAX_SECONDS} seconds, no sound), so you and a
            responder can see what it looks like. The video can show your face. It is stored only after you agree, only for
            this device, and is never used to train a model. Recording does not teach SignLine to recognise the gesture.
          </Hint>
          <div className="flex flex-wrap gap-2">
            <Button onClick={recorder.requestCamera}><Camera className="h-4 w-4" aria-hidden="true" /> Record gesture</Button>
            <Button variant="outline" onClick={onSkip}>Save without a recording</Button>
          </div>
        </>
      )}
      {st.status === 'requesting' && <p className="text-sm font-bold" role="status">Waiting for camera permission…</p>}
      {failed && (
        <div role="alert" className="space-y-2 rounded-sl border-2 border-amber-500 bg-amber-50 p-3 text-sm">
          <p>{RECORDER_MESSAGES[st.error] || RECORDER_MESSAGES.error}</p>
          <div className="flex flex-wrap gap-2">
            {st.status !== 'unsupported' && <Button size="sm" variant="outline" onClick={recorder.requestCamera}>Try again</Button>}
            <Button size="sm" onClick={onSkip}>Save without a recording</Button>
          </div>
        </div>
      )}
      {(st.status === 'preview' || st.status === 'recording') && (
        <div className="space-y-2">
          <div className="relative overflow-hidden rounded-sl border-2 border-sl-ink bg-black">
            <video ref={liveRef} autoPlay muted playsInline className="aspect-[4/3] w-full -scale-x-100 object-cover" aria-label="Live camera preview" />
            {st.status === 'recording' && (
              <span role="status" className="absolute left-2 top-2 inline-flex items-center gap-1.5 rounded-full bg-rose-600 px-3 py-1 text-xs font-extrabold text-white">
                <Circle className="h-3 w-3 animate-pulse fill-white" aria-hidden="true" /> Recording {elapsed.toFixed(0)} / {MAX_SECONDS} s
              </span>
            )}
          </div>
          {st.error && <p role="alert" className="text-sm text-rose-700">{RECORDER_MESSAGES[st.error]}</p>}
          <div className="flex flex-wrap gap-2">
            {st.status === 'preview'
              ? <Button onClick={recorder.start}><Circle className="h-4 w-4 fill-rose-600 text-rose-600" aria-hidden="true" /> Start recording</Button>
              : <Button variant="dark" onClick={recorder.stop}><Square className="h-4 w-4" aria-hidden="true" /> Stop recording</Button>}
            {st.status === 'preview' && <Button variant="ghost" onClick={() => { recorder.release(); onSkip() }}>Save without a recording</Button>}
          </div>
        </div>
      )}
      {st.status === 'recorded' && previewUrl && (
        <div className="space-y-2">
          <p className="text-sm font-bold">Preview ({st.duration} s). Nothing is saved until you choose to keep it.</p>
          <video src={previewUrl} controls playsInline className="aspect-[4/3] w-full rounded-sl border-2 border-sl-ink bg-black object-contain" aria-label="Recorded gesture preview" />
          <div className="flex flex-wrap gap-2">
            <Button onClick={() => { recorder.release(); onRecorded(st.blob, st.duration) }}>{keepLabel}</Button>
            <Button variant="outline" onClick={recorder.retake}><RotateCcw className="h-4 w-4" aria-hidden="true" /> Retake</Button>
          </div>
        </div>
      )}
    </div>
  )
}
