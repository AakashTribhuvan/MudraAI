import React, { useState, useRef, useEffect } from 'react'
import { Sparkles, Play, RotateCcw, Gauge, AlertCircle, RefreshCw } from 'lucide-react'
import { useAvatarRenderer } from '../../avatar/useAvatarRenderer'
import { useAvatarChoice } from '../../avatar/avatarCatalog'

export default function MudraAvatarViewer({
  signText = 'HELLO',
  autoPlay = true,
  className = ''
}) {
  const canvasHostRef = useRef(null)
  const [isModelLoading, setIsModelLoading] = useState(true)
  const [loadError, setLoadError] = useState(null)
  const [activeSubtitle, setActiveSubtitle] = useState('')
  const [speed, setSpeed] = useState(1.2)
  const [retryKey, setRetryKey] = useState(0)
  const [warning, setWarning] = useState(null)
  const [seqInfo, setSeqInfo] = useState(null) // which signs used hand-made fallbacks (shown, never silent)
  const { avatar, avatars, choose } = useAvatarChoice()
  useEffect(() => { setWarning(null); setIsModelLoading(true); setLoadError(null) }, [avatar.id])

  const { executeSignSequence, resetToDefaultPose } = useAvatarRenderer({
    containerRef: canvasHostRef,
    modelPath: avatar.url,
    modelType: avatar.type,
    onWarning: setWarning,
    onSequenceInfo: setSeqInfo,
    animationSpeed: speed,
    pauseDuration: 300,
    onTextUpdate: (char) => {
      setActiveSubtitle((prev) => prev + char)
    },
    onLoadingChange: (loading) => {
      setIsModelLoading(loading)
      if (!loading) setLoadError(null)
    },
    onError: (err) => {
      console.error('MudraAvatarViewer error:', err)
      setLoadError(err?.message || 'Failed to load 3D ISL avatar')
    }
  })

  // Play animation whenever signText changes or model finishes loading
  useEffect(() => {
    if (!isModelLoading && !loadError && signText && autoPlay) {
      setActiveSubtitle('')
      const timer = setTimeout(() => {
        executeSignSequence(signText)
      }, 150)
      return () => clearTimeout(timer)
    }
  }, [signText, isModelLoading, loadError, autoPlay, executeSignSequence, retryKey])

  const handleReplay = () => {
    setActiveSubtitle('')
    resetToDefaultPose()
    setTimeout(() => {
      executeSignSequence(signText)
    }, 100)
  }

  const handleSpeedToggle = () => {
    const nextSpeed = speed === 1.0 ? 1.3 : speed === 1.3 ? 1.6 : 1.0
    setSpeed(nextSpeed)
  }

  const handleRetry = () => {
    setLoadError(null)
    setIsModelLoading(true)
    setRetryKey((prev) => prev + 1)
  }

  return (
    <div className={`relative rounded-3xl overflow-hidden glass-card border border-white/80 shadow-xl flex flex-col ${className}`}>
      
      {/* Top Header & MUDRA Avatar Badge */}
      <div className="absolute top-3.5 left-3.5 right-3.5 flex items-center justify-between z-20 pointer-events-auto">
        <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-mudra-indigo-900/85 backdrop-blur-md border border-white/15 text-xs text-white shadow-xs">
          <Sparkles className="w-3.5 h-3.5 text-mudra-peach-300 animate-pulse" />
          <span className="font-display font-bold tracking-wide">SignLine avatar</span>
        </div>

        <div className="flex items-center gap-1.5">
          {avatars.length > 1 && (
            <label className="flex items-center">
              <span className="sr-only">Avatar model</span>
              <select value={avatar.id} onChange={(e) => choose(e.target.value)} disabled={isModelLoading && !loadError}
                      title="Choose the avatar model (files in app/frontend/src/assets/avatars)"
                      className="max-w-[9.5rem] truncate rounded-full border border-mudra-lavender-200 bg-white/85 px-2.5 py-1 text-[11px] font-bold text-mudra-indigo-800 backdrop-blur-md">
                {avatars.map((a) => <option key={a.id} value={a.id}>{a.name}</option>)}
              </select>
            </label>
          )}
          <button
            type="button"
            onClick={handleSpeedToggle}
            className="flex items-center gap-1 px-2.5 py-1 rounded-full bg-white/80 hover:bg-white text-mudra-indigo-800 text-[11px] font-mono font-bold border border-mudra-lavender-200 backdrop-blur-md transition-all cursor-pointer"
            title="Toggle playback speed"
          >
            <Gauge className="w-3 h-3 text-mudra-lavender-600" />
            <span>{speed}x</span>
          </button>

          <button
            type="button"
            onClick={handleReplay}
            disabled={isModelLoading || !!loadError}
            className="p-1.5 rounded-full bg-white/80 hover:bg-white text-mudra-indigo-800 border border-mudra-lavender-200 backdrop-blur-md transition-all cursor-pointer disabled:opacity-50"
            title="Replay 3D demonstration"
          >
            <RotateCcw className="w-3.5 h-3.5 text-mudra-lavender-600" />
          </button>
        </div>
      </div>

      {/* Main 3D Canvas Stage */}
      <div className="relative w-full h-full min-h-[320px] sm:min-h-[380px] flex-1 bg-gradient-to-b from-mudra-lavender-100/50 via-white/80 to-mudra-peach-50/40 flex items-center justify-center overflow-hidden">
        
        {/* Dedicated Three.js canvas mount container with ZERO React children */}
        <div
          ref={canvasHostRef}
          key={retryKey}
          className="absolute inset-0 w-full h-full"
        />

        {/* React Loading Overlay - as independent sibling */}
        {isModelLoading && !loadError && (
          <div className="absolute inset-0 flex flex-col items-center justify-center p-6 text-center z-10 bg-white/80 backdrop-blur-sm pointer-events-auto">
            <div className="w-10 h-10 rounded-full border-4 border-mudra-lavender-200 border-t-mudra-lavender-600 animate-spin mb-3"></div>
            <div className="font-display font-bold text-sm text-mudra-indigo-900">
              Loading 3D ISL Avatar...
            </div>
            <div className="text-[11px] text-mudra-indigo-500 font-mono mt-0.5">
              Mixamo Rig Skeleton
            </div>
          </div>
        )}

        {!warning && !loadError && !isModelLoading && seqInfo && (seqInfo.fallbackLetters.length > 0 || seqInfo.fallbackWords.length > 0 || seqInfo.spelledWords?.length > 0) && (
          <div role="status" className="absolute bottom-2 left-2 right-2 z-10 rounded-xl border border-amber-300 bg-amber-50/95 px-3 py-2 text-[11px] font-semibold text-amber-900">
            {seqInfo.letterDataMissing
              ? 'Letter shapes are not available on this server, so letters use approximate hand-made animations.'
              : seqInfo.fallbackLetters.length || seqInfo.fallbackWords.length
                ? `Approximate hand-made animation for: ${[...seqInfo.fallbackWords, ...seqInfo.fallbackLetters.map((l) => `letter ${l}`)].join(', ')}.`
                : `Spelled letter by letter (no recorded sign yet): ${seqInfo.spelledWords.join(', ')}. Record it on Teach a sign and the avatar will sign it.`}
          </div>
        )}

        {warning && !loadError && !isModelLoading && (
          <div role="status" className="absolute bottom-2 left-2 right-2 z-10 rounded-xl border border-amber-300 bg-amber-50/95 px-3 py-2 text-[11px] font-semibold text-amber-900">
            {warning}
          </div>
        )}

        {/* Error State Overlay */}
        {loadError && (
          <div className="absolute inset-0 flex flex-col items-center justify-center p-6 text-center z-10 bg-white/95 backdrop-blur-sm pointer-events-auto space-y-3">
            <div className="w-12 h-12 rounded-2xl bg-rose-100 border border-rose-200 flex items-center justify-center text-rose-600">
              <AlertCircle className="w-6 h-6" />
            </div>
            <div className="space-y-1">
              <div className="font-display font-bold text-sm text-mudra-indigo-950">
                Avatar Failed to Load
              </div>
              <div className="text-xs text-mudra-indigo-600 max-w-xs font-mono">
                {loadError}
              </div>
            </div>
            <button
              type="button"
              onClick={handleRetry}
              className="btn-primary py-2 px-4 text-xs font-bold flex items-center gap-1.5 shadow-md"
            >
              <RefreshCw className="w-3.5 h-3.5" />
              <span>Retry Loading</span>
            </button>
          </div>
        )}
      </div>

      {/* Bottom Subtitle / Synchronized Text Bar */}
      <div className="p-3 bg-white/90 backdrop-blur-md border-t border-mudra-lavender-200/80 flex items-center justify-between z-20">
        <div className="flex items-center gap-2 min-w-0">
          <span className="text-[10px] uppercase font-mono font-bold text-mudra-indigo-400">
            Sign:
          </span>
          <span className="font-display font-bold text-sm text-mudra-indigo-950 truncate">
            {signText}
          </span>
          {activeSubtitle && (
            <span className="text-xs font-mono text-mudra-lavender-700 bg-mudra-lavender-100 px-2 py-0.5 rounded-md">
              {activeSubtitle}
            </span>
          )}
        </div>

        <button
          type="button"
          onClick={handleReplay}
          disabled={isModelLoading || !!loadError}
          className="flex items-center gap-1 text-xs font-bold text-mudra-lavender-700 hover:text-mudra-lavender-900 transition-colors disabled:opacity-50 cursor-pointer"
        >
          <Play className="w-3 h-3 fill-current" />
          <span>Play</span>
        </button>
      </div>

    </div>
  )
}
