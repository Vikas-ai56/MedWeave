/** Status indicator: always an icon plus a label, never colour alone. */

const ICONS = {
  check: <path d="M3.5 7.2 6 9.7l4.5-5" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />,
  cross: <path d="m4.5 4.5 5 5m0-5-5 5" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />,
  dot: <circle cx="7" cy="7" r="3" fill="currentColor" />,
  ring: <circle cx="7" cy="7" r="3.2" fill="none" stroke="currentColor" strokeWidth="1.6" />,
  pulse: (
    <>
      <circle cx="7" cy="7" r="2.6" fill="currentColor" />
      <circle cx="7" cy="7" r="5.4" fill="none" stroke="currentColor" strokeWidth="1.2" opacity="0.5" />
    </>
  ),
  alert: (
    <>
      <path d="M7 3.5v4" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
      <circle cx="7" cy="10.2" r="1" fill="currentColor" />
    </>
  ),
  diamond: <path d="M7 2.8 11.2 7 7 11.2 2.8 7Z" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinejoin="round" />,
  dash: <path d="M4 7h6" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />,
};

// What each backend status looks like.
const STATUS = {
  online: { tone: "good", icon: "dot", label: "Online" },
  offline: { tone: "critical", icon: "ring", label: "Offline" },
  training: { tone: "active", icon: "pulse", label: "Busy" },
  running: { tone: "active", icon: "pulse", label: "Running" },
  completed: { tone: "good", icon: "check", label: "Completed" },
  failed: { tone: "critical", icon: "cross", label: "Failed" },
  pending: { tone: "neutral", icon: "ring", label: "Pending" },
  active: { tone: "active", icon: "pulse", label: "In progress" },
  unreachable: { tone: "critical", icon: "alert", label: "Backend unreachable" },
  live: { tone: "good", icon: "dot", label: "Live" },
  simulated: { tone: "sim", icon: "diamond", label: "Simulated" },
  skipped: { tone: "neutral", icon: "dash", label: "Skipped" },
  connecting: { tone: "neutral", icon: "ring", label: "Connecting" },
};

export function Icon({ name, size = 14 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 14 14" aria-hidden="true" className="icon">
      {ICONS[name]}
    </svg>
  );
}

export function Status({ status, label }) {
  const spec = STATUS[status] ?? { tone: "neutral", icon: "ring", label: status };
  return (
    <span className={`status status-${spec.tone}`}>
      <Icon name={spec.icon} />
      {label ?? spec.label}
    </span>
  );
}
