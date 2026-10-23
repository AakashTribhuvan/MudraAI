/**
 * Dial and share helpers for the emergency confirmation screen. Nothing here calls or sends by itself: the
 * dialer opens only from the user's tap on the final "Call" button, and the share sheet only from "Share".
 */
export function telLink(number) {
  let digits = String(number || '').replace(/[^\d+]/g, '')
  digits = digits ? digits[0] + digits.slice(1).replace(/\+/g, '') : ''
  return /^\+?\d{2,15}$/.test(digits) ? `tel:${digits}` : null
}

/** Which sharing the device offers: native share sheet with files, text only, or none (copy instead). */
export function shareSupport(nav, file) {
  if (!nav?.share) return 'none'
  if (file && typeof nav.canShare === 'function') {
    try { if (nav.canShare({ files: [file] })) return 'files' } catch { /* not supported */ }
  }
  return 'text'
}

/** Share status labels: the share sheet can never confirm delivery, so it stops at "Sharing started". */
export const SHARE_STATUS = {
  idle: null,
  prepared: 'Report prepared (not shared yet)',
  sharing: 'Sharing started: your phone opened the share sheet. SignLine cannot see whether it was delivered.',
  cancelled: 'Sharing cancelled: nothing was shared.',
  copied: 'Copied to the clipboard: paste it into a message yourself.',
  failed: 'Sharing failed: nothing was shared.',
}

export async function shareReport({ nav, text, file, withVideo }) {
  const support = shareSupport(nav, withVideo ? file : null)
  if (support === 'none') {
    try { await nav?.clipboard?.writeText(text); return 'copied' } catch { return 'failed' }
  }
  try {
    await nav.share(support === 'files' ? { title: 'Emergency report', text, files: [file] } : { title: 'Emergency report', text })
    return 'sharing'
  } catch (err) {
    return err?.name === 'AbortError' ? 'cancelled' : 'failed'
  }
}
