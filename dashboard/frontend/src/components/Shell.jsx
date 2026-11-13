import { clock } from "../format.js";
import { useCopy } from "../lib/hooks.js";
import { VIEWS, routeHref } from "../lib/route.js";
import { Glyph } from "./Glyphs.jsx";
import { Status } from "./Status.jsx";

export function BrandMark({ size = 30 }) {
  return (
    <svg className="brand-mark" width={size} height={size} viewBox="0 0 32 32" aria-hidden="true">
      <circle cx="16" cy="16" r="15" fill="none" stroke="currentColor" strokeOpacity="0.18" />
      <path d="M5 11c5.5 0 5.5 10 11 10s5.5-10 11-10" fill="none" stroke="var(--mint)" strokeWidth="1.8" strokeLinecap="round" />
      <path d="M5 21c5.5 0 5.5-10 11-10s5.5 10 11 10" fill="none" stroke="var(--violet)" strokeWidth="1.4" strokeLinecap="round" opacity="0.85" />
      <circle cx="16" cy="16" r="2.2" fill="var(--ink)" />
    </svg>
  );
}

/** Desktop rail; collapses to a top bar with a scrollable nav strip on narrow screens. */
export function Sidebar({ view, simulated, connection, updatedAt }) {
  return (
    <nav className="sidebar" aria-label="Primary">
      <a className="brand" href={routeHref("weave", { simulated })}>
        <BrandMark />
        <span className="brand-name">MedWeave</span>
      </a>
      <ul className="nav">
        {VIEWS.map((v) => (
          <li key={v.id}>
            <a href={routeHref(v.id, { simulated })} className={view === v.id ? "is-active" : ""} aria-current={view === v.id ? "page" : undefined}>
              <Glyph name={v.icon} size={18} />
              <span>{v.label}</span>
            </a>
          </li>
        ))}
      </ul>
      <div className="sidebar-foot">
        <div className="health">
          <Status status={connection} label={connection === "unreachable" ? "Offline" : undefined} />
          <span className="mono muted">{updatedAt ? `updated ${clock(new Date(updatedAt).toISOString())}` : "waiting for data"}</span>
          <a className="muted health-link" href="/api/docs">
            API docs
          </a>
        </div>
        <p className="sdg">
          <span className="sdg-badge">SDG 9</span>
          Building inclusive, resilient and collaborative AI infrastructure.
        </p>
      </div>
    </nav>
  );
}

export function CommandBar({ session, sessions, sessionId, onSessionChange, onRefresh, simulated, onSimulatedChange, connection, updatedAt }) {
  const { copied, copy } = useCopy();
  return (
    <div className="command-bar">
      <div className="command-left">
        <span className="eyebrow">MedWeave / Coordinator</span>
        {session && (
          <button type="button" className="id-pill" onClick={() => copy(session.id)} aria-label={`Copy session ID ${session.id}`}>
            <span className="muted">Session</span>
            <span className="mono">{session.id}</span>
            <Glyph name="copy" size={14} />
            <span className="copied" aria-live="polite">
              {copied === session.id ? "Copied" : ""}
            </span>
          </button>
        )}
        {session && <Status status={session.status} />}
      </div>
      <div className="command-right">
        <Status status={connection} label={connection === "unreachable" ? "Offline" : undefined} />
        <span className="mono muted updated">{updatedAt ? `Updated ${clock(new Date(updatedAt).toISOString())}` : ""}</span>
        {!simulated && sessions.length > 1 && (
          <label className="select">
            <span className="sr-only">Session</span>
            <select value={sessionId ?? ""} onChange={(e) => onSessionChange(e.target.value || null)}>
              <option value="">Latest session</option>
              {sessions.map((s) => (
                <option key={s.session_id} value={s.session_id}>
                  {s.session_id.slice(0, 12)} · {s.status}
                </option>
              ))}
            </select>
          </label>
        )}
        {!simulated && (
          <button type="button" className="icon-button" onClick={onRefresh} aria-label="Refresh now">
            <Glyph name="refresh" size={18} />
          </button>
        )}
        <div className="source-toggle" role="radiogroup" aria-label="Data source">
          <button type="button" role="radio" aria-checked={!simulated} className={!simulated ? "is-on" : ""} onClick={() => onSimulatedChange(false)}>
            Live
          </button>
          <button type="button" role="radio" aria-checked={simulated} className={simulated ? "is-on" : ""} onClick={() => onSimulatedChange(true)}>
            Simulated
          </button>
        </div>
      </div>
    </div>
  );
}

/** Shown while a snapshot is still on screen but the latest request failed. */
export function StaleAlert({ error, updatedAt }) {
  return (
    <div className="alert-strip" role="alert">
      <Status status="unreachable" />
      <span>
        Showing the last data received{updatedAt ? ` at ${clock(new Date(updatedAt).toISOString())}` : ""}. <span className="mono">({error})</span>
      </span>
    </div>
  );
}
