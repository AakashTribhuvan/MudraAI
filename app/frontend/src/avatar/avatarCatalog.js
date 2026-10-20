import { useCallback, useEffect, useState } from 'react'

/**
 * Avatar models for the 3D signing avatar.
 *
 * To add one: put a rigged model in app/frontend/src/assets/avatars/ (FBX with embedded textures,
 * e.g. Mixamo "FBX Binary" + "With Skin", or a .glb). It appears in the avatar dropdown on the next
 * page load. The sign animations drive Mixamo bones (mixamorig:Hips, mixamorig:RightArm, ...), so
 * the model needs a Mixamo skeleton; without one it loads but cannot sign.
 */
const found = import.meta.glob('../assets/avatars/*.{fbx,FBX,glb,GLB}', { eager: true, query: '?url', import: 'default' })

export const DEFAULT_AVATAR = { id: 'ybot', name: 'Y Bot (default)', url: '/ybot.glb', type: 'glb' }

function displayName(file) {
  const base = file.replace(/\.[^.]+$/, '').replace(/[_-]+/g, ' ').trim()
  return base.charAt(0).toUpperCase() + base.slice(1)
}

export const AVATARS = [
  DEFAULT_AVATAR,
  ...Object.entries(found)
    .map(([path, url]) => {
      const file = path.split('/').pop()
      return { id: file, name: displayName(file), url, type: file.split('.').pop().toLowerCase() }
    })
    .sort((a, b) => a.name.localeCompare(b.name)),
]

const KEY = 'signline_avatar'
const EVENT = 'signline-avatar-change'

function stored() {
  try {
    const id = localStorage.getItem(KEY)
    return AVATARS.find((a) => a.id === id) || DEFAULT_AVATAR
  } catch {
    return DEFAULT_AVATAR
  }
}

/** The chosen avatar, remembered in this browser and shared by every avatar viewer on the page. */
export function useAvatarChoice() {
  const [avatar, setAvatar] = useState(stored)
  useEffect(() => {
    const onChange = () => setAvatar(stored())
    window.addEventListener(EVENT, onChange)
    return () => window.removeEventListener(EVENT, onChange)
  }, [])
  const choose = useCallback((id) => {
    try { localStorage.setItem(KEY, id) } catch { /* private mode: this page only */ }
    setAvatar(AVATARS.find((a) => a.id === id) || DEFAULT_AVATAR)
    window.dispatchEvent(new Event(EVENT))
  }, [])
  return { avatar, avatars: AVATARS, choose }
}
