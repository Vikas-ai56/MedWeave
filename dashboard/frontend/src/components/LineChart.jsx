import { useMemo, useRef, useState } from "react";

const WIDTH = 560;
const HEIGHT = 230;
const PAD = { top: 14, right: 18, bottom: 30, left: 46 };

/**
 * Single-series line chart of one value per training turn.
 * 2px line, 8px markers, recessive grid, crosshair that snaps to the nearest turn.
 * Failed turns have no value: the line breaks there and a hollow marker sits on the baseline.
 * Arrow keys step the crosshair when the chart has focus.
 */
export function LineChart({ turns, value, format, domain, ariaLabel }) {
  const ref = useRef(null);
  const [hover, setHover] = useState(null);

  const points = useMemo(
    () => turns.map((turn, index) => ({ index, turn, y: turn.status === "completed" ? value(turn) : null })),
    [turns, value],
  );
  const values = points.map((p) => p.y).filter((y) => y != null);

  if (turns.length === 0 || values.length === 0) {
    return <div className="empty chart-empty">No completed turns yet</div>;
  }

  const [yMin, yMax] = domain ?? niceDomain(values);
  const innerW = WIDTH - PAD.left - PAD.right;
  const innerH = HEIGHT - PAD.top - PAD.bottom;
  const x = (i) => PAD.left + (turns.length === 1 ? innerW / 2 : (i / (turns.length - 1)) * innerW);
  const y = (v) => PAD.top + innerH - ((v - yMin) / (yMax - yMin || 1)) * innerH;
  const ticks = Array.from({ length: 5 }, (_, i) => yMin + ((yMax - yMin) * i) / 4);
  const xLabelEvery = Math.max(1, Math.ceil(turns.length / 8));

  // Break the line at failed turns.
  const segments = [];
  let current = [];
  for (const p of points) {
    if (p.y == null) {
      if (current.length) segments.push(current);
      current = [];
    } else current.push(p);
  }
  if (current.length) segments.push(current);

  function onPointerMove(event) {
    const rect = ref.current.getBoundingClientRect();
    const px = ((event.clientX - rect.left) / rect.width) * WIDTH;
    let nearest = 0;
    for (const p of points) if (Math.abs(x(p.index) - px) < Math.abs(x(nearest) - px)) nearest = p.index;
    setHover(nearest);
  }

  function onKeyDown(event) {
    const last = points.length - 1;
    const step = { ArrowRight: 1, ArrowLeft: -1 }[event.key];
    if (step) {
      event.preventDefault();
      setHover((h) => Math.min(last, Math.max(0, (h ?? (step > 0 ? -1 : last + 1)) + step)));
    } else if (event.key === "Home" || event.key === "End") {
      event.preventDefault();
      setHover(event.key === "Home" ? 0 : last);
    }
  }

  const hovered = hover == null ? null : points[hover];

  return (
    <div className="chart">
      <svg
        ref={ref}
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
        role="img"
        aria-label={`${ariaLabel}. Use the arrow keys to read each turn.`}
        tabIndex={0}
        onKeyDown={onKeyDown}
        onBlur={() => setHover(null)}
        onPointerMove={onPointerMove}
        onPointerLeave={() => setHover(null)}
      >
        {ticks.map((t) => (
          <g key={t}>
            <line className={t === yMin ? "baseline" : "gridline"} x1={PAD.left} x2={WIDTH - PAD.right} y1={y(t)} y2={y(t)} />
            <text className="tick" x={PAD.left - 10} y={y(t)} dy="0.32em" textAnchor="end">
              {format(t)}
            </text>
          </g>
        ))}
        {points.map((p) =>
          p.index % xLabelEvery === 0 || p.index === turns.length - 1 ? (
            <text key={p.index} className="tick" x={x(p.index)} y={HEIGHT - 8} textAnchor="middle">
              {p.index + 1}
            </text>
          ) : null,
        )}
        {hovered && <line className="crosshair" x1={x(hovered.index)} x2={x(hovered.index)} y1={PAD.top} y2={PAD.top + innerH} />}
        {segments.map((seg, i) => (
          <polyline key={i} className="series" points={seg.map((p) => `${x(p.index)},${y(p.y)}`).join(" ")} />
        ))}
        {points.map((p) =>
          p.y == null ? (
            <circle key={p.index} className="marker-failed" cx={x(p.index)} cy={y(yMin)} r={4.5} />
          ) : (
            <circle key={p.index} className="marker" cx={x(p.index)} cy={y(p.y)} r={hovered?.index === p.index ? 6 : 4} />
          ),
        )}
        <rect className="hit" x={PAD.left - 12} y={0} width={innerW + 24} height={HEIGHT - PAD.bottom + 8} />
      </svg>
      {hovered && (
        <div
          className="tooltip"
          style={{
            left: `${(x(hovered.index) / WIDTH) * 100}%`,
            top: 0,
            transform: `translateX(${hovered.index > turns.length / 2 ? "calc(-100% - 12px)" : "12px"})`,
          }}
        >
          <div className="tooltip-value">
            <span className={hovered.y == null ? "tooltip-key is-failed" : "tooltip-key"} />
            {hovered.y == null ? "Turn failed" : format(hovered.y)}
          </div>
          <div className="tooltip-label">
            Turn {hovered.index + 1} · Round {hovered.turn.round}
          </div>
          <div className="tooltip-label">
            {hovered.turn.owner} trained, {hovered.turn.peer} computed
          </div>
        </div>
      )}
    </div>
  );
}

function niceDomain(values) {
  const max = Math.max(...values);
  const step = 10 ** Math.floor(Math.log10(max || 1));
  return [0, Math.ceil((max * 1.1) / (step / 2)) * (step / 2)];
}
