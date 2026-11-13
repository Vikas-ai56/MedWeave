const WIDTH = 280;
const HEIGHT = 56;
const PAD = 5;

/** A compact trend of finished turns: line for completed ones, a hollow mark for failures. */
export function Sparkline({ turns, value = (t) => t.accuracy, label }) {
  const finished = turns.filter((t) => t.status === "completed" || t.status === "failed");
  const values = finished.map((t) => (t.status === "completed" ? value(t) : null)).filter((v) => v != null);
  if (values.length < 2) return <div className="spark-empty muted">Not enough finished turns for a trend yet</div>;
  const min = Math.min(...values);
  const max = Math.max(...values);
  const x = (i) => PAD + (i / Math.max(finished.length - 1, 1)) * (WIDTH - PAD * 2);
  const y = (v) => HEIGHT - PAD - ((v - min) / (max - min || 1)) * (HEIGHT - PAD * 2);
  const points = finished.map((t, i) => (t.status === "completed" ? `${x(i)},${y(value(t))}` : null)).filter(Boolean);
  const lastIndex = finished.length - 1;
  const last = finished[lastIndex];
  return (
    <svg className="sparkline" viewBox={`0 0 ${WIDTH} ${HEIGHT}`} role="img" aria-label={label}>
      <polyline points={points.join(" ")} />
      {finished.map((t, i) => (t.status === "failed" ? <circle key={i} cx={x(i)} cy={HEIGHT - PAD} r="3" className="spark-failed" /> : null))}
      {last.status === "completed" && <circle cx={x(lastIndex)} cy={y(value(last))} r="3.5" className="spark-last" />}
    </svg>
  );
}
