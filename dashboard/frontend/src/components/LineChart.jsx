import { useMemo, useRef, useState } from "react";

const WIDTH = 560;
const HEIGHT = 220;
const PAD = { top: 12, right: 16, bottom: 30, left: 44 };

/**
 * Single-series line chart of one value per training turn.
 * 2px line, 8px markers, recessive grid, crosshair that snaps to the nearest turn.
 * Failed turns have no value: the line breaks there and a hollow red marker sits on the baseline.
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
    return <div className="empty">No completed turns yet</div>;
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

  const hovered = hover == null ? null : points[hover];

  return (
    <div className="chart">
      <svg
        ref={ref}
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
        role="img"
        aria-label={ariaLabel}
        onPointerMove={onPointerMove}
        onPointerLeave={() => setHover(null)}
      >
        {ticks.map((t) => (
          <g key={t}>
            <line className={t === yMin ? "baseline" : "gridline"} x1={PAD.left} x2={WIDTH - PAD.right} y1={y(t)} y2={y(t)} />
            <text className="tick" x={PAD.left - 8} y={y(t)} dy="0.32em" textAnchor="end">
              {format(t)}
            </text>
          </g>
        ))}
        {points.map((p) =>
          p.index % xLabelEvery === 0 || p.index === turns.length - 1 ? (
            <text key={p.index} className="tick" x={x(p.index)} y={HEIGHT - 10} textAnchor="middle">
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
            <circle key={p.index} className="marker-failed" cx={x(p.index)} cy={y(yMin)} r={4} />
          ) : (
            <circle key={p.index} className="marker" cx={x(p.index)} cy={y(p.y)} r={hovered?.index === p.index ? 5.5 : 4} />
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
            transform: `translateX(${hovered.index > turns.length / 2 ? "calc(-100% - 10px)" : "10px"})`,
          }}
        >
          <div className="tooltip-value">
            <span className="tooltip-key" />
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
