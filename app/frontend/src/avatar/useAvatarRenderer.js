import { useRef, useEffect, useCallback } from 'react'
import * as THREE from 'three'
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js'
import { FBXLoader } from 'three/examples/jsm/loaders/FBXLoader.js'
import { defaultPoseFrame, setDefaultPose } from './Animations/defaultPose'
import * as AlphabetGestures from './Animations'
import { buildRig, letterClip, solveClip, trimRest } from './landmarkRetarget'
import { SYNTH_SIGNS } from './synthSigns'
import { fetchVocabulary } from '../features/vocabulary'
import { API_BASE_URL } from '../services/signRecognizer'

// Recorded sign clips (GET /reference/{sign}) are fetched once per page and reused
const clipCache = new Map()
function fetchClip(sign) {
  if (!clipCache.has(sign)) {
    clipCache.set(sign, fetch(`${API_BASE_URL}/reference/${encodeURIComponent(sign)}`)
      .then((r) => (r.ok ? r.json() : null))
      .catch(() => null))
  }
  return clipCache.get(sign)
}
const LETTER_FOLLOW = 0.12 // letters glide into their hand shape (about 0.3 s), recorded words follow closely
const CLIP_FOLLOW = 0.55 // (both per 1/60 s, so the glide takes the same time at any frame rate)

const WORD_ALIASES = { THANKS: 'THANK_YOU', THANKYOU: 'THANK_YOU', HI: 'HELLO', HOME: 'HOUSE' }
// hand-made word animations that do not show the sign (checked frame by frame on Y Bot): spelled instead
// until the word is recorded. NAMASTE is built from landmarks (synthSigns.js), HOME uses the HOUSE recording.
const HANDMADE_OFF = new Set(['Water', 'Person', 'Namaste', 'Home'])
const MAX_PHRASE_WORDS = 4

/**
 * MUDRA 3D Avatar Renderer Hook
 * Manages Three.js WebGL scene, Mixamo skeletal bone interpolation, and ISL animations.
 * 
 * Features:
 * - Completely removes SIH / external hackathon branding decal (Mesh/Material 'Asset 1')
 * - Supports verified whole-word gestures and fallback fingerspelling
 * - Clean ResizeObserver and unified continuous render loop
 * - Safe idempotent lifecycle
 */
export function useAvatarRenderer({
  containerRef,
  modelPath = '/ybot.glb',
  modelType, // 'glb' | 'fbx'; taken from the file extension when not given
  animationSpeed = 1.2,
  pauseDuration = 350,
  onTextUpdate,
  onLoadingChange,
  onError,
  onWarning // non-fatal: e.g. the model has no Mixamo skeleton, so it cannot sign
}) {
  const boneTransitionsRef = useRef(new Map())

  const contextRef = useRef({
    animations: [],
    avatar: null,
    pending: false,
    animate: () => {},
    scene: null,
    camera: null,
    renderer: null,
    flag: false,
    boneNameMap: {}
  })

  const speedRef = useRef(animationSpeed)
  const pauseRef = useRef(pauseDuration)
  const onTextUpdateRef = useRef(onTextUpdate)
  const onLoadingChangeRef = useRef(onLoadingChange)
  const onErrorRef = useRef(onError)
  const onWarningRef = useRef(onWarning)
  onWarningRef.current = onWarning
  const isMountedRef = useRef(true)

  useEffect(() => {
    speedRef.current = animationSpeed
  }, [animationSpeed])

  useEffect(() => {
    pauseRef.current = pauseDuration
  }, [pauseDuration])

  useEffect(() => {
    onTextUpdateRef.current = onTextUpdate
  }, [onTextUpdate])

  useEffect(() => {
    onLoadingChangeRef.current = onLoadingChange
  }, [onLoadingChange])

  useEffect(() => {
    onErrorRef.current = onError
  }, [onError])

  // Easing function for smooth organic motion
  const easeInOutCubic = (t) => {
    return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2
  }

  const addHumanVariation = (value, intensity = 0.02) => {
    return value * (1 + (Math.random() - 0.5) * intensity)
  }

  // Setup and run Three.js scene
  useEffect(() => {
    isMountedRef.current = true
    const hostElement = containerRef.current
    if (!hostElement) return
    // This run's own lifetime. React may mount twice in development: a model loaded by a discarded run must
    // not become ctx.avatar (animations would then move an avatar that is never drawn: frozen T-pose).
    let disposed = false

    const ctx = contextRef.current
    ctx.flag = false
    ctx.pending = false
    ctx.animations = []
    ctx.animate = () => {
      ctx.pending = true
    }

    // 1. Create Three.js Scene
    const scene = new THREE.Scene()
    scene.background = null
    ctx.scene = scene

    // 2. High-Quality 5-Point Lighting Rig
    const ambientLight = new THREE.AmbientLight(0xffffff, 1.4)
    scene.add(ambientLight)

    const keyLight = new THREE.DirectionalLight(0xffffff, 1.6)
    keyLight.position.set(0, 2.5, 3.5)
    scene.add(keyLight)

    const fillLight = new THREE.DirectionalLight(0xe8e3f5, 1.0)
    fillLight.position.set(-3, 3, 3)
    scene.add(fillLight)

    const rimLight = new THREE.DirectionalLight(0xfdba74, 0.8)
    rimLight.position.set(3, 3, -2)
    scene.add(rimLight)

    const bounceLight = new THREE.DirectionalLight(0xfdfbf7, 0.6)
    bounceLight.position.set(0, -2, 2)
    scene.add(bounceLight)

    // 3. Camera Setup: Upper Torso, Chest, Hands & Face Framing
    const initialWidth = hostElement.clientWidth || 400
    const initialHeight = hostElement.clientHeight || 450

    const camera = new THREE.PerspectiveCamera(
      38,
      initialWidth / Math.max(initialHeight, 1),
      0.1,
      1000
    )
    camera.position.set(0, 1.38, 1.45)
    camera.lookAt(0, 1.28, 0)
    ctx.camera = camera

    // Fit the camera to the loaded model: head (with margin) down to the waist, plus the signing space
    // beside the body, for any container shape. Re-run on resize.
    const frameAvatar = () => {
      const f = ctx.framing
      if (!f) return
      const tanHalf = Math.tan(THREE.MathUtils.degToRad(camera.fov / 2))
      const aspect = camera.aspect || 1
      const distV = (f.height / 2) / tanHalf
      const distH = f.halfWidth / (tanHalf * aspect)
      const dist = Math.max(distV, distH)
      camera.position.set(0, f.centerY, f.centerZ + dist)
      camera.lookAt(0, f.centerY, f.centerZ)
      camera.updateProjectionMatrix()
    }

    // 4. WebGL Renderer
    const renderer = new THREE.WebGLRenderer({
      antialias: true,
      alpha: true,
      powerPreference: 'high-performance'
    })
    renderer.setSize(initialWidth, initialHeight)
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    renderer.shadowMap.enabled = false
    renderer.outputColorSpace = THREE.SRGBColorSpace
    ctx.renderer = renderer

    if (!hostElement.contains(renderer.domElement)) {
      renderer.domElement.style.width = '100%'
      renderer.domElement.style.height = '100%'
      renderer.domElement.style.display = 'block'
      hostElement.appendChild(renderer.domElement)
    }

    // 5. Responsive Resize Observer
    const resizeObserver = new ResizeObserver((entries) => {
      if (!isMountedRef.current || !ctx.camera || !ctx.renderer) return
      for (const entry of entries) {
        const width = entry.contentRect.width || hostElement.clientWidth
        const height = entry.contentRect.height || hostElement.clientHeight
        if (width > 0 && height > 0) {
          ctx.camera.aspect = width / height
          ctx.camera.updateProjectionMatrix()
          ctx.renderer.setSize(width, height)
          frameAvatar()
        }
      }
    })
    resizeObserver.observe(hostElement)

    // 6. Bone Finder Helper
    const findBoneByName = (root, name) => {
      if (!root) return null
      let bone = root.getObjectByName(name) || root.getObjectByName(name.replace(':', ''))
      if (bone) return bone

      const map = ctx.boneNameMap
      if (map && map[name]) {
        bone = root.getObjectByName(map[name])
        if (bone) return bone
      }

      if (name.startsWith('mixamorig') && !name.includes(':')) {
        const colonName = `mixamorig:${name.substring('mixamorig'.length)}`
        bone = root.getObjectByName(colonName)
        if (bone) return bone
      }

      const suffix = name.replace(/^.*mixamorig:?/, '')
      let found = null
      root.traverse((obj) => {
        if (obj?.name === suffix || obj?.name?.endsWith?.(suffix)) {
          found = obj
        }
      })
      return found
    }

    // 7. Load the 3D model (GLB or rigged FBX) & strip SIH branding.
    // The sign animations are written for Y Bot (REFERENCE_MODEL), whose bones rest at rotation 0. Any other
    // model is shown while a hidden Y Bot skeleton plays the animations; every frame each bone's movement is
    // copied to the shown model in world space (retargeting), so rigs with other rest angles sign correctly.
    const REFERENCE_MODEL = '/ybot.glb'
    const type = (modelType || modelPath.split('?')[0].split('.').pop() || 'glb').toLowerCase()
    const retarget = modelPath !== REFERENCE_MODEL
    const loadModel = (path, kind) => new Promise((resolve, reject) => {
      const loader = kind === 'fbx' ? new FBXLoader() : new GLTFLoader()
      loader.load(path, (loaded) => resolve(kind === 'fbx' ? loaded : loaded.scene), undefined, reject)
    })
    const prepareMeshes = (root) => {
      const decals = []
      root.traverse((child) => {
        // Remove SIH decal mesh/texture
        if (child.name === 'Asset 1' || child.material?.name === 'Asset 1' || child.name?.toLowerCase().includes('asset')) {
          decals.push(child)
          return
        }
        if (child.type === 'SkinnedMesh' || child.isMesh) {
          child.frustumCulled = false
          const materials = Array.isArray(child.material) ? child.material : [child.material]
          materials.forEach((mat) => {
            if (!mat) return
            mat.roughness = 0.6
            mat.metalness = 0.05
            mat.needsUpdate = true
          })
        }
      })
      decals.forEach((d) => d.parent?.remove(d))
    }
    // the real bone of each name (Mixamo FBX nests empty helper nodes with the same name inside each bone)
    const realBones = (root) => {
      const seen = new Set()
      const bones = []
      root.traverse((obj) => {
        if (obj.isBone && obj.name && !seen.has(obj.name)) {
          seen.add(obj.name)
          bones.push(obj)
        }
      })
      return bones
    }
    onLoadingChangeRef.current?.(true)

    Promise.all([loadModel(modelPath, type), retarget ? loadModel(REFERENCE_MODEL, 'glb') : Promise.resolve(null)])
      .then(([root, reference]) => {
        if (disposed || !isMountedRef.current) return
        try {
          prepareMeshes(root)
          ctx.display = root
          scene.add(root)

          // Models in other units (Mixamo FBX is in centimetres) are scaled to the default avatar's height
          root.updateMatrixWorld(true)
          const rawHeight = new THREE.Box3().setFromObject(root).getSize(new THREE.Vector3()).y
          if (rawHeight > 0 && (rawHeight < 0.5 || rawHeight > 4)) {
            root.scale.multiplyScalar(1.8 / rawHeight)
          }

          // Framing from the model's real size (T-pose bounding box and hip height)
          root.updateMatrixWorld(true)
          const box = new THREE.Box3().setFromObject(root)
          const size = box.getSize(new THREE.Vector3())
          const hips = root.getObjectByName('mixamorig:Hips') || root.getObjectByName('mixamorigHips')
          const hipsY = hips ? hips.getWorldPosition(new THREE.Vector3()).y : box.min.y + size.y * 0.53
          if (!hips) {
            onWarningRef.current?.('This model has no Mixamo skeleton (mixamorig bones), so it cannot play the signs.')
          }
          const top = box.max.y + size.y * 0.03
          const bottom = hipsY - size.y * 0.12 // resting hands sit around the hips
          ctx.framing = {
            centerY: (top + bottom) / 2,
            centerZ: (box.min.z + box.max.z) / 2,
            height: top - bottom,
            halfWidth: size.y * 0.3 // hands reach about this far to the side while signing
          }
          frameAvatar()

          // The animated skeleton: the shown model itself, or the hidden Y Bot reference
          ctx.avatar = reference || root
          ctx.retarget = null
          if (reference) {
            reference.updateMatrixWorld(true)
            const pairs = []
            for (const tgt of realBones(root)) {
              const ref = reference.getObjectByName(tgt.name)
              if (!ref?.isBone) continue
              pairs.push({
                ref,
                tgt,
                refRestInv: ref.getWorldQuaternion(new THREE.Quaternion()).invert(),
                tgtRest: tgt.getWorldQuaternion(new THREE.Quaternion())
              })
            }
            ctx.retarget = { reference, root, pairs }
            if (hips && pairs.length < 10) {
              onWarningRef.current?.('Few bones match the Mixamo skeleton, so the signs may look wrong on this model.')
            }
          }

          // Build bone name lookup table
          const boneMap = {}
          ctx.avatar.traverse((obj) => {
            if (!obj.name) return
            boneMap[obj.name] = obj.name
            if (obj.name.startsWith('mixamorig:')) {
              const keyNoColon = obj.name.replace('mixamorig:', 'mixamorig')
              boneMap[keyNoColon] = obj.name
              const suffix = obj.name.replace(/^.*?:/, '')
              boneMap[suffix] = obj.name
            }
          })
          ctx.boneNameMap = boneMap

          // Recorded sign clips drive this skeleton (built at rest, before the neutral pose)
          ctx.rig = buildRig(ctx.avatar, findBoneByName)
          // the neutral pose as local rotations: where a recorded clip returns to (not the T-pose rest)
          ctx.neutral = new Map()
          for (const [name, action, axis, value] of defaultPoseFrame()) {
            const b = findBoneByName(ctx.avatar, name)
            if (!b || action !== 'rotation') continue
            const e = new THREE.Euler().setFromQuaternion(ctx.neutral.get(b) || b.quaternion)
            e[axis] = value
            ctx.neutral.set(b, new THREE.Quaternion().setFromEuler(e))
          }

          // Set neutral starting posture
          setDefaultPose(ctx)
        } catch (initErr) {
          console.warn('3D Avatar: Warning during pose initialization:', initErr)
        } finally {
          if (isMountedRef.current) {
            onLoadingChangeRef.current?.(false)
          }
        }
      })
      .catch((err) => {
        if (disposed) return
        console.error('3D Avatar: Failed to load model from:', modelPath, err)
        if (isMountedRef.current) {
          onLoadingChangeRef.current?.(false)
          onErrorRef.current?.(err)
        }
      })

    // Copy the reference skeleton's pose to the shown model: same world-space rotation change per bone
    const qRef = new THREE.Quaternion()
    const qParent = new THREE.Quaternion()
    const qDelta = new THREE.Quaternion()
    const applyRetarget = () => {
      const r = ctx.retarget
      if (!r) return
      r.reference.updateMatrixWorld(true)
      r.root.updateMatrixWorld(true)
      for (const { ref, tgt, refRestInv, tgtRest } of r.pairs) {
        ref.getWorldQuaternion(qRef)
        qDelta.multiplyQuaternions(qRef, refRestInv) // how far this bone turned from its rest pose (world)
        qDelta.multiply(tgtRest) // the same turn from the shown model's rest pose
        tgt.parent.getWorldQuaternion(qParent)
        tgt.quaternion.copy(qParent.invert().multiply(qDelta))
        tgt.updateMatrixWorld(true)
      }
    }

    // 8. Continuous Unified Render & Animation Loop
    const qClip = new THREE.Quaternion()
    let animLoopId = null
    const unifiedRenderLoop = () => {
      if (disposed || !isMountedRef.current) return
      animLoopId = requestAnimationFrame(unifiedRenderLoop)

      if (ctx.avatar && ctx.animations.length > 0 && !ctx.flag) {
        const currentFrame = ctx.animations[0]

        if (Array.isArray(currentFrame) && currentFrame[0] === 'blend-rest') {
          // after a recorded clip: arms and hands glide back to the rest pose (the neutral pose builds on it)
          const now = performance.now()
          if (!ctx.blend) ctx.blend = { start: now, from: (ctx.rig?.restPose || []).map((r) => r.bone.quaternion.clone()) }
          const a = Math.min((now - ctx.blend.start) / 280, 1)
          ;(ctx.rig?.restPose || []).forEach((r, k) =>
            r.bone.quaternion.slerpQuaternions(ctx.blend.from[k], ctx.neutral?.get(r.bone) || r.restLocal, a))
          if (a >= 1) {
            ctx.blend = null
            boneTransitionsRef.current.clear()
            ctx.animations.shift()
          }
        } else if (Array.isArray(currentFrame) && currentFrame[0] === 'clip') {
          const [, frames, fps, follow = CLIP_FOLLOW] = currentFrame
          const now = performance.now()
          if (!ctx.clip || ctx.clip.frames !== frames) {
            ctx.clip = { frames, start: now, tick: now }
          }
          const ticks = Math.min((now - ctx.clip.tick) / (1000 / 60), 30)
          ctx.clip.tick = now
          const step = 1 - Math.pow(1 - follow, ticks)
          // position in the clip (frames are 1/fps apart), played at the chosen speed
          const t = ((now - ctx.clip.start) / 1000) * fps * Math.max(0.5, speedRef.current / 1.2)
          const i = Math.min(Math.floor(t), frames.length - 1)
          const j = Math.min(i + 1, frames.length - 1)
          const a = Math.min(t - i, 1)
          for (const [bone, q] of frames[i]) {
            const next = frames[j].get(bone) || q
            qClip.slerpQuaternions(q, next, a)
            bone.quaternion.slerp(qClip, step) // light smoothing of landmark jitter
          }
          if (t >= frames.length - 1) {
            ctx.clip = null
            boneTransitionsRef.current.clear() // the next step starts from the clip's last pose
            ctx.animations.shift()
          }
        } else if (Array.isArray(currentFrame) && currentFrame.length > 0) {
          if (currentFrame[0] === 'add-text') {
            if (onTextUpdateRef.current && typeof currentFrame[1] === 'string') {
              onTextUpdateRef.current(currentFrame[1])
            }
            ctx.animations.shift()
          } else {
            const currentTime = performance.now()
            const transitions = boneTransitionsRef.current

            for (let i = 0; i < currentFrame.length; ) {
              const instruction = currentFrame[i]
              const [boneName, action, axis, targetValue] = instruction

              const bone = findBoneByName(ctx.avatar, boneName)

              if (!bone) {
                currentFrame.splice(i, 1)
                continue
              }

              const transitionKey = `${boneName}_${action}_${axis}`

              if (!transitions.has(transitionKey)) {
                const currentValue = bone[action][axis]
                const baseDuration = 300
                const calculatedDuration = Math.max(
                  180,
                  Math.min(750, baseDuration * (1 / speedRef.current))
                )
                const duration = addHumanVariation(calculatedDuration, 0.05)

                transitions.set(transitionKey, {
                  startValue: currentValue,
                  targetValue,
                  startTime: currentTime,
                  duration
                })
              }

              const transition = transitions.get(transitionKey)
              const elapsed = currentTime - transition.startTime
              const progress = Math.min(elapsed / transition.duration, 1)
              const easedProgress = easeInOutCubic(progress)

              const range = transition.targetValue - transition.startValue
              const newValue = transition.startValue + range * easedProgress

              bone[action][axis] = newValue

              if (progress >= 1) {
                bone[action][axis] = transition.targetValue
                transitions.delete(transitionKey)
                currentFrame.splice(i, 1)
              } else {
                i++
              }
            }
          }
        } else {
          ctx.flag = true
          setTimeout(() => {
            if (isMountedRef.current) {
              ctx.flag = false
            }
          }, pauseRef.current)
          ctx.animations.shift()
        }
      }

      if (renderer && scene && camera) {
        applyRetarget()
        renderer.render(scene, camera)
      }
    }

    unifiedRenderLoop()

    // 9. Cleanup
    return () => {
      disposed = true
      isMountedRef.current = false
      resizeObserver.disconnect()

      if (animLoopId) {
        cancelAnimationFrame(animLoopId)
        animLoopId = null
      }

      ctx.pending = false

      ctx.retarget?.reference.traverse((object) => {
        if (object.isMesh) object.geometry?.dispose()
      })
      ctx.retarget = null
      ctx.display = null

      if (scene) {
        scene.traverse((object) => {
          if (object.isMesh) {
            object.geometry?.dispose()
            const materials = Array.isArray(object.material) ? object.material : [object.material]
            materials.forEach((m) => m?.dispose?.())
          }
        })
      }

      if (renderer && renderer.domElement) {
        if (renderer.domElement.parentNode === hostElement) {
          hostElement.removeChild(renderer.domElement)
        }
        renderer.dispose()
      }

      ctx.scene = null
      ctx.camera = null
      ctx.renderer = null
      ctx.avatar = null
      ctx.framing = null
      ctx.animations = []
      boneTransitionsRef.current.clear()
    }
  }, [containerRef, modelPath, modelType])

  /**
   * Triggers ISL gesture sequence for a word, phrase, or sentence
   */
  const sequenceIdRef = useRef(0)
  const executeSignSequence = useCallback(async (inputText) => {
    const ctx = contextRef.current
    if (!ctx.avatar || !isMountedRef.current) return
    const sequenceId = ++sequenceIdRef.current

    // Words that have a recorded sign (the trained vocabulary's /reference clips): longest match first,
    // so "thank you" plays THANK_YOU and "good morning" GOOD_MORNING
    const vocab = await fetchVocabulary()
    const words = inputText.match(/[a-zA-Z]+/g) || []
    const recorded = new Map() // index of the first word -> { sign, length }
    for (let w = 0; w < words.length; ) {
      let match = null
      for (let n = Math.min(MAX_PHRASE_WORDS, words.length - w); n >= 1 && !match; n--) {
        const raw = words.slice(w, w + n).join('_').toUpperCase()
        const sign = WORD_ALIASES[raw] || raw
        if (vocab.labels?.[sign]?.has_reference) match = { sign, length: n }
      }
      if (match) recorded.set(w, match)
      w += match ? match.length : 1
    }
    const clips = new Map()
    await Promise.all([...recorded.values()].map(async ({ sign }) => {
      const clip = await fetchClip(sign)
      try {
        const solved = clip && ctx.rig ? solveClip(ctx.rig, trimRest(clip)) : null
        if (solved) clips.set(sign, { frames: solved, fps: clip.fps || 15 })
      } catch (err) {
        console.error('3D Avatar: could not use the recorded sign for', sign, err) // falls back to spelling
      }
    }))
    const synth = new Map()
    for (const [w, word] of words.entries()) {
      const make = SYNTH_SIGNS[word.toUpperCase()]
      if (!make || [...recorded].some(([start, r]) => w >= start && w < start + r.length)) continue
      try {
        const clip = letterClip(ctx.rig, make())
        const solved = clip && solveClip(ctx.rig, clip)
        if (solved) synth.set(word.toUpperCase(), { frames: solved, fps: clip.fps })
      } catch (err) {
        console.error('3D Avatar: could not build the sign for', word, err)
      }
    }
    for (const { frames } of synth.values()) {
      for (const frame of frames) for (const r of ctx.rig.restPose) if (!frame.has(r.bone)) frame.set(r.bone, ctx.neutral?.get(r.bone) || r.restLocal)
    }

    // Letters for the words without a recorded sign: the alphabet photos' hand shapes (/reference/LETTER_X),
    // arms placed in front of the chest; bones a letter does not use go back to the neutral pose
    const letters = new Map()
    const spelled = new Set(words.filter((_, w) => ![...recorded].some(([start, r]) => w >= start && w < start + r.length))
      .join('').toUpperCase())
    await Promise.all([...spelled].map(async (ch) => {
      try {
        const clip = letterClip(ctx.rig, await fetchClip(`LETTER_${ch}`))
        const solved = clip && solveClip(ctx.rig, clip)
        if (!solved) return
        for (const frame of solved) {
          for (const r of ctx.rig.restPose) if (!frame.has(r.bone)) frame.set(r.bone, ctx.neutral?.get(r.bone) || r.restLocal)
        }
        letters.set(ch, { frames: solved, fps: clip.fps || 15 })
      } catch (err) {
        console.error('3D Avatar: could not use the letter photo for', ch, err) // falls back to the hand-made letter
      }
    }))
    if (sequenceId !== sequenceIdRef.current || !isMountedRef.current) return // a newer request replaced this one

    ctx.animations = []
    ctx.clip = null
    ctx.blend = null
    boneTransitionsRef.current.clear()

    const tokens = inputText.match(/([a-zA-Z]+|[^a-zA-Z]+)/g) || []
    let wordIndex = -1

    const wordAliasMap = {
      thank: 'ThankYou',
      thanks: 'ThankYou',
      thankyou: 'ThankYou',
      'thank you': 'ThankYou',
      namaste: 'Namaste',
      hello: 'Hello',
      goodbye: 'Goodbye',
      bye: 'Goodbye',
      yes: 'Yes',
      no: 'No',
      home: 'Home',
      person: 'Person',
      time: 'Time',
      you: 'You',
      water: 'Water',
      pani: 'Water',
      help: 'Help',
      madad: 'Help',
      peace: 'Peace',
      shanti: 'Peace',
      victory: 'Peace'
    }

    for (let i = 0; i < tokens.length; i++) {
      const token = tokens[i]
      const isWord = /^[a-zA-Z]+$/.test(token)

      if (isWord) {
        wordIndex++
        const rec = recorded.get(wordIndex)
        const clip = rec && clips.get(rec.sign)
        if (clip) {
          // the recorded sign covers this word and the next rec.length - 1 words
          let consumed = 0
          let text = ''
          while (i < tokens.length && consumed < rec.length) {
            if (/^[a-zA-Z]+$/.test(tokens[i])) consumed++
            text += tokens[i]
            if (consumed < rec.length) i++
          }
          wordIndex += rec.length - 1
          ctx.animations.push(['add-text', text])
          ctx.animations.push(['clip', clip.frames, clip.fps])
          ctx.animations.push(['blend-rest'])
          continue
        }
        const made = synth.get(token.toUpperCase())
        if (made) {
          ctx.animations.push(['add-text', token])
          ctx.animations.push(['clip', made.frames, made.fps, LETTER_FOLLOW])
          ctx.animations.push(['blend-rest'])
          continue
        }
        const lowerToken = token.toLowerCase()
        const titleCaseWord =
          wordAliasMap[lowerToken] ||
          token.charAt(0).toUpperCase() + token.slice(1).toLowerCase()
        
        const wordFunctionName = `createWord${titleCaseWord}`
        const wordAnimation = HANDMADE_OFF.has(titleCaseWord) ? null : AlphabetGestures[wordFunctionName]

        // Check 2-word combinations
        if (
          !wordAnimation &&
          i + 2 < tokens.length &&
          /^[^a-zA-Z]+$/.test(tokens[i + 1]) &&
          /^[a-zA-Z]+$/.test(tokens[i + 2])
        ) {
          const nextWord = tokens[i + 2]
          const combined = (token + ' ' + nextWord).toLowerCase()
          const combinedAlias = wordAliasMap[combined] || 
            token.charAt(0).toUpperCase() + token.slice(1).toLowerCase() + nextWord.charAt(0).toUpperCase() + nextWord.slice(1).toLowerCase()
          const combinedFuncName = `createWord${combinedAlias}`
          const combinedAnimation = HANDMADE_OFF.has(combinedAlias) ? null : AlphabetGestures[combinedFuncName]

          if (typeof combinedAnimation === 'function') {
            ctx.animations.push(['add-text', token.charAt(0)])
            combinedAnimation(ctx)
            ctx.animations.push(['add-text', token.slice(1) + tokens[i + 1] + nextWord])
            i += 3
            continue
          }
        }

        if (typeof wordAnimation === 'function') {
          ctx.animations.push(['add-text', token.charAt(0)])
          wordAnimation(ctx)
          if (token.length > 1) {
            ctx.animations.push(['add-text', token.slice(1)])
          }
        } else {
          // Spell letter by letter: the letter photos' hand shapes, else the hand-made A-Z keyframes
          let usedLetterClip = false
          for (const char of token) {
            const upperChar = char.toUpperCase()
            const letter = letters.get(upperChar)
            if (letter) {
              ctx.animations.push(['add-text', char])
              ctx.animations.push(['clip', letter.frames, letter.fps, LETTER_FOLLOW])
              usedLetterClip = true
              continue
            }
            const letterFuncName = `createLetter${upperChar}`
            const letterAnimation = AlphabetGestures[letterFuncName]

            if (typeof letterAnimation === 'function') {
              ctx.animations.push(['add-text', char])
              letterAnimation(ctx)
            } else {
              ctx.animations.push(['add-text', char])
              ctx.animations.push([])
            }
          }
          if (usedLetterClip) ctx.animations.push(['blend-rest'])
        }
      } else {
        for (const char of token) {
          ctx.animations.push(['add-text', char])
          ctx.animations.push([])
        }
      }
    }

    setDefaultPose(ctx) // back to the neutral pose after the last sign

    if (!ctx.pending) {
      ctx.pending = true
    }
  }, [])

  const resetToDefaultPose = useCallback(() => {
    const ctx = contextRef.current
    if (ctx.avatar && isMountedRef.current) {
      ctx.animations = []
      boneTransitionsRef.current.clear()
      setDefaultPose(ctx)
    }
  }, [])

  return {
    executeSignSequence,
    resetToDefaultPose,
    context: contextRef.current
  }
}
