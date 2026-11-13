export function clock(iso) {
  if (!iso) return "—";
  return new Date(iso).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

export function ago(iso, now = Date.now()) {
  if (!iso) return "—";
  const seconds = Math.max(0, Math.round((now - new Date(iso).getTime()) / 1000));
  if (seconds < 60) return `${seconds}s ago`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
  return `${Math.floor(seconds / 3600)}h ago`;
}

export function percent(value) {
  return value == null ? "—" : `${Math.round(value * 100)}%`;
}

export function decimal(value, digits = 3) {
  return value == null ? "—" : value.toFixed(digits);
}

export function seconds(value) {
  return value == null ? "—" : `${value.toFixed(1)}s`;
}
