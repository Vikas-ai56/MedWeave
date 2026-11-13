/** Line icons for navigation and controls (1.5px stroke, 20px grid) and the institution signature. */

const PATHS = {
  weave: (
    <>
      <circle cx="10" cy="10" r="2.6" />
      <circle cx="4" cy="4.5" r="1.6" />
      <circle cx="16" cy="4.5" r="1.6" />
      <circle cx="16" cy="15.5" r="1.6" />
      <circle cx="4" cy="15.5" r="1.6" />
      <path d="M5.3 5.6 8 8.2M14.7 5.6 12 8.2M14.7 14.4 12 11.8M5.3 14.4 8 11.8" />
    </>
  ),
  experiments: (
    <>
      <path d="M3 16.5h14" />
      <path d="M4 13.5 8 9l3 2.5L16.5 5" />
      <circle cx="16.5" cy="5" r="1.2" />
    </>
  ),
  institutions: (
    <>
      <circle cx="10" cy="10" r="7" strokeDasharray="1.6 2.2" />
      <circle cx="10" cy="10" r="2.2" />
      <circle cx="15" cy="6" r="1.2" />
      <circle cx="5.5" cy="13.5" r="1.2" />
    </>
  ),
  protection: (
    <>
      <path d="M10 2.8 15.8 5v4.6c0 3.6-2.5 6.3-5.8 7.6-3.3-1.3-5.8-4-5.8-7.6V5Z" />
      <path d="M7.4 10.2 9.3 12l3.4-3.8" />
    </>
  ),
  evidence: (
    <>
      <path d="M6 3v14" />
      <circle cx="6" cy="5.5" r="1.3" />
      <circle cx="6" cy="10" r="1.3" />
      <circle cx="6" cy="14.5" r="1.3" />
      <path d="M9.5 5.5H16M9.5 10H14M9.5 14.5H15" />
    </>
  ),
  copy: (
    <>
      <rect x="7" y="7" width="9" height="9" rx="2" />
      <path d="M13 7V5.5A1.5 1.5 0 0 0 11.5 4h-6A1.5 1.5 0 0 0 4 5.5v6A1.5 1.5 0 0 0 5.5 13H7" />
    </>
  ),
  refresh: (
    <>
      <path d="M16 10a6 6 0 1 1-1.8-4.3" />
      <path d="M16 3.5v3.6h-3.6" />
    </>
  ),
  close: <path d="m5.5 5.5 9 9m0-9-9 9" />,
  lock: (
    <>
      <rect x="4.5" y="9" width="11" height="8" rx="2" />
      <path d="M7 9V6.5a3 3 0 0 1 6 0V9" />
    </>
  ),
  download: (
    <>
      <path d="M10 3.5v9M6.5 9 10 12.5 13.5 9" />
      <path d="M4 16.5h12" />
    </>
  ),
  arrowUp: <path d="M10 15.5v-11M5.5 9 10 4.5 14.5 9" />,
  arrowRight: <path d="M4 10h11.5M11 5.5l4.5 4.5-4.5 4.5" />,
  chevron: <path d="m7.5 5 5 5-5 5" />,
  image: (
    <>
      <rect x="3" y="4" width="14" height="12" rx="2" />
      <circle cx="7.5" cy="8.5" r="1.4" />
      <path d="m4 15 4.5-4 3 2.5 2-1.7L17 14.5" />
    </>
  ),
  grid: <path d="M4 4h12v12H4zM4 8h12M4 12h12M8 4v12M12 4v12" />,
  noise: <path d="M3 10h2l1.2-4 2 8 2-10 2 12 1.6-6H17" />,
};

export function Glyph({ name, size = 18, className = "" }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 20 20"
      aria-hidden="true"
      className={`glyph ${className}`}
      fill="none"
      stroke="currentColor"
      strokeWidth="1.5"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      {PATHS[name]}
    </svg>
  );
}

/**
 * An institution's signature: a hexagonal facade whose inner pattern is fixed by its
 * position, so each institution keeps the same mark everywhere it appears.
 */
const HEXAGON = Array.from({ length: 6 }, (_, i) => {
  const a = (Math.PI / 3) * i - Math.PI / 2;
  return `${(12 + 10 * Math.cos(a)).toFixed(2)},${(12 + 10 * Math.sin(a)).toFixed(2)}`;
}).join(" ");

/** The mark's shapes on a 24-unit grid, for use inside any SVG. */
export function MarkShapes({ index }) {
  const variant = index % 4;
  return (
    <>
      <polygon points={HEXAGON} className="mark-hex" />
      {variant === 0 && <path d="M8 16V10M12 16V8M16 16V10M7 16h10" />}
      {variant === 1 && <path d="M12 6.5v11M7.5 9.5l9 5M16.5 9.5l-9 5" />}
      {variant === 2 && <path d="M8.5 15.5V11a3.5 3.5 0 0 1 7 0v4.5M12 7.5v2" />}
      {variant === 3 && (
        <>
          <circle cx="12" cy="12" r="3.4" />
          <path d="M12 6.5v2M12 15.5v2" />
        </>
      )}
    </>
  );
}

export function InstitutionMark({ index, size = 28 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" aria-hidden="true" className="inst-mark" fill="none" stroke="currentColor" strokeWidth="1.2">
      <MarkShapes index={index} />
    </svg>
  );
}
