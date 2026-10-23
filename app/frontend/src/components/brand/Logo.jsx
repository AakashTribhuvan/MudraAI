/**
 * SignLine logo: the supplied mark (public/brand/signline-mark.png, background removed from
 * signline-logo-original.png without redrawing) plus the wordmark.
 */
export default function Logo({ size = 40, withName = true, className = '', nameClassName = '' }) {
  return (
    <span className={`inline-flex items-center gap-2.5 ${className}`}>
      <img
        src={`${import.meta.env.BASE_URL}brand/signline-mark.png`}
        alt={withName ? '' : 'SignLine'}
        width={Math.round(size * 0.82)}
        height={size}
        className="shrink-0 select-none"
        style={{ width: Math.round(size * 0.82), height: size }}
        draggable="false"
      />
      {withName && (
        <span className={`font-display font-extrabold tracking-tight text-sl-ink ${nameClassName}`} style={{ fontSize: size * 0.55 }}>
          SignLine
        </span>
      )}
    </span>
  )
}
