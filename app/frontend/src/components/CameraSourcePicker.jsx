import React, { useCallback, useEffect, useState } from 'react'

/** Lists cameras after permission is granted so browser-provided device labels are available. */
export function useCameraSources(cameraActive) {
  const [sources, setSources] = useState([])
  const [deviceId, setDeviceId] = useState('')

  const refresh = useCallback(async () => {
    if (!navigator.mediaDevices?.enumerateDevices) return
    const devices = await navigator.mediaDevices.enumerateDevices()
    setSources(devices.filter((device) => device.kind === 'videoinput'))
  }, [])

  useEffect(() => {
    refresh().catch(() => {})
    navigator.mediaDevices?.addEventListener?.('devicechange', refresh)
    return () => navigator.mediaDevices?.removeEventListener?.('devicechange', refresh)
  }, [refresh])

  useEffect(() => {
    if (cameraActive) refresh().catch(() => {})
  }, [cameraActive, refresh])

  return { sources, deviceId, setDeviceId }
}

export default function CameraSourcePicker({ sources, deviceId, onChange, disabled = false }) {
  if (sources.length < 2) return null
  return (
    <label className="flex items-center gap-2 text-xs">
      <span className="shrink-0">Camera</span>
      <select
        aria-label="Camera source"
        className="max-w-52 rounded-lg border border-white/20 bg-slate-900 px-2 py-1.5 text-white"
        value={deviceId}
        onChange={(event) => onChange(event.target.value)}
        disabled={disabled}
      >
        <option value="">Default camera</option>
        {sources.map((source, index) => (
          <option key={source.deviceId} value={source.deviceId}>
            {source.label || `Camera ${index + 1}`}
          </option>
        ))}
      </select>
    </label>
  )
}
