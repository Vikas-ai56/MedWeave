import { useRef, useState } from "react";

const WIDTH = 760;
const HEIGHT = 340;
const PAD = { top: 18, right: 132, bottom: 38, left: 52 };
const LABEL_GAP = 15;

/** Marker shapes give each series a second encoding besides colour. */
function Marker({ shape, cx, cy, r, color, hollow }) {
  const common = { fill: hollow ? "var(--chart-surface)" : color, stroke: color, strokeWidth: hollow ? 2 : 0, className: "series-marker" };
  if (shape === "square") return <rect x={cx - r} y={cy - r} width={r * 2} height={r * 2} rx="1.5" {...common} />;
  if (shape === "diamond") return <path d={`M${cx} ${cy - r * 1.3}L${cx + r * 1.3} ${cy}L${cx} ${cy + r * 1.3}L${cx - r * 1.3} ${cy}Z`} {...common} />;
  if (shape === "triangle") return <path d={`M${cx} ${cy - r * 1.3}L${cx + r * 1.2} ${cy + r}L${cx - r * 1.2} ${cy + r}Z`} {...common} />;
  return <circle cx={cx} cy={cy} r={r} {...common} />;
}

/** Spreads end labels so they never overlap, keeping each near its line's last value. */
function placeLabels(items) {
  const sorted = [...items].sort((a, b) => a.y - b.y);
  for (let i = 1; i < sorted.length; i += 1) {
    if (sorted[i].y - sorted[i - 1].y < LABEL_GAP) sorted[i].y = sorted[i - 1].y + LABEL_GAP;
  }
  return sorted;
}

/**
 * Several series over a shared x axis (rounds or turns). One y scale, recessive grid,
 * crosshair tooltip listing every visible series, direct labels at line ends.
 */
export function SeriesChart({ series, xCount, xName, domain, format, ariaLabel, tickFormat = format }) {
  const ref = useRef(null);
  const [hover, setHover] = useState(null);
  const [yMin, yMax] = domain;
  const innerW = WIDTH - PAD.left - PAD.right;
  const innerH = HEIGHT - PAD.top - PAD.bottom;
  const x = (i) => PAD.left + (xCount === 1 ? innerW / 2 : (i / (xCount - 1)) * innerW);
  const y = (v) => PAD.top + innerH - ((Math.min(yMax, Math.max(yMin, v)) - yMin) / (yMax - yMin)) * innerH;
  const ticks = Array.from({ length: 5 }, (_, i) => yMin + ((yMax - yMin) * i) / 4);
  const visible = series.filter((s) => s.values?.length);

  function onPointerMove(event) {
    const rect = ref.current.getBoundingClientRect();
    const px = ((event.clientX - rect.left) / rect.width) * WIDTH;
    const index = Math.round(((px - PAD.left) / innerW) * (xCount - 1));
    setHover(Math.min(xCount - 1, Math.max(0, index)));
  }

  function onKeyDown(event) {
    const step = { ArrowRight: 1, ArrowLeft: -1 }[event.key];
    if (!step) return;
    event.preventDefault();
    setHover((h) => Math.min(xCount - 1, Math.max(0, (h ?? (step > 0 ? -1 : xCount)) + step)));
  }

  const labels = placeLabels(visible.map((s) => ({ id: s.id, text: s.short ?? s.label, color: s.color, y: y(s.values.at(-1)) })));
  const rows = hover == null ? [] : visible.map((s) => ({ ...s, v: s.values[hover] })).sort((a, b) => b.v - a.v);

  return (
    <div className="chart chart-wide">
      <svg
        ref={ref}
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
        role="img"
        aria-label={`${ariaLabel}. Use the arrow keys to read each ${xName.toLowerCase()}.`}
        tabIndex={0}
        onKeyDown={onKeyDown}
        onBlur={() => setHover(null)}
        onPointerMove={onPointerMove}
        onPointerLeave={() => setHover(null)}
      >
        {ticks.map((t) => (
          <g key={t}>
            <line className={t === yMin ? "baseline" : "gridline"} x1={PAD.left} x2={PAD.left + innerW} y1={y(t)} y2={y(t)} />
            <text className="tick" x={PAD.left - 10} y={y(t)} dy="0.32em" textAnchor="end">
              {tickFormat(t)}
            </text>
          </g>
        ))}
        {Array.from({ length: xCount }, (_, i) => (
          <text key={i} className="tick" x={x(i)} y={HEIGHT - 14} textAnchor="middle">
            {i + 1}
          </text>
        ))}
        <text className="axis-title" x={PAD.left + innerW / 2} y={HEIGHT - 1} textAnchor="middle">
          {xName}
        </text>
        {hover != null && <line className="crosshair" x1={x(hover)} x2={x(hover)} y1={PAD.top} y2={PAD.top + innerH} />}
        {visible.map((s) => (
          <g key={s.id} className="series-group">
            <polyline
              points={s.values.map((v, i) => `${x(i)},${y(v)}`).join(" ")}
              fill="none"
              stroke={s.color}
              strokeWidth={s.emphasis ? 2.5 : 2}
              strokeDasharray={s.dashed ? "5 5" : undefined}
              strokeLinejoin="round"
              strokeLinecap="round"
            />
            {s.values.map((v, i) => (
              <Marker key={i} shape={s.marker} cx={x(i)} cy={y(v)} r={hover === i ? 5 : 3.4} color={s.color} hollow={s.dashed} />
            ))}
          </g>
        ))}
        {labels.map((l) => (
          <g key={l.id}>
            <line x1={PAD.left + innerW + 6} x2={PAD.left + innerW + 14} y1={l.y} y2={l.y} stroke={l.color} strokeWidth="2" />
            <text className="direct-label" x={PAD.left + innerW + 18} y={l.y} dy="0.32em">
              {l.text}
            </text>
          </g>
        ))}
        <rect className="hit" x={PAD.left - 10} y={0} width={innerW + 20} height={HEIGHT - PAD.bottom + 6} />
      </svg>
      {hover != null && rows.length > 0 && (
        <div
          className="tooltip"
          style={{
            left: `${(x(hover) / WIDTH) * 100}%`,
            top: 0,
            transform: `translateX(${hover > xCount / 2 ? "calc(-100% - 12px)" : "12px"})`,
          }}
        >
          <div className="tooltip-label">
            {xName} {hover + 1}
          </div>
          {rows.map((r) => (
            <div key={r.id} className="tooltip-row">
              <span className="tooltip-key" style={{ background: r.color }} />
              <span>{r.short ?? r.label}</span>
              <span className="mono">{format(r.v)}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
