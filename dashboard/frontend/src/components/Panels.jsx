import { useMemo, useRef, useState } from "react";

import { ago, clock, decimal, percent, seconds } from "../format.js";
import { downloadText, turnsToCsv } from "../lib/csv.js";
import { connectedCount, pad2, percent1, sessionProgress } from "../lib/derive.js";
import { Glyph, InstitutionMark } from "./Glyphs.jsx";
import { Icon, Status } from "./Status.jsx";

/* A. Overview: the instrument readout ------------------------------------------- */

export function Overview({ session, institutions, model, accuracy, noise }) {
  const online = connectedCount(institutions);
  const running = session.status === "running";
  const progress = sessionProgress(session);
  return (
    <dl className="readout">
      <Reading label="Round" value={session.current_round ? pad2(session.current_round) : "—"} unit={`/ ${pad2(session.num_rounds)}`}>
        <span className="readout-rail" aria-hidden="true">
          <span style={{ transform: `scaleX(${progress})` }} />
        </span>
      </Reading>
      <Reading label="Turns completed" value={session.completed_turns} unit={`/ ${session.total_turns}`} />
      <Reading label="Institutions" value={pad2(online)} unit={`/ ${pad2(institutions.length)}`} note="connected" />
      <Reading label="Data owner" value={running ? session.current_owner ?? "—" : "—"} note={running ? "training on local data" : "no turn running"} small />
      <Reading label="Compute peer" value={running ? session.current_peer ?? "—" : "—"} note={running ? "running the Backbone" : "no turn running"} small />
      <Reading label="Backbone" value={model?.backbone_version == null ? "—" : `v${model.backbone_version}`} note="latest checkpoint" />
      {accuracy !== undefined && (
        <Reading label="Accuracy" value={accuracy ? percent1(accuracy.value) : "—"} note={accuracy ? `training · ${accuracy.basis}` : "no completed turn"} />
      )}
      {noise !== undefined && (
        <Reading label="Activation noise" value={noise ? `σ ${noise.noise_scale}` : "Off"} note={noise ? noise.mechanism : "not configured"} />
      )}
      <Reading label="Raw data moved" value="None" note="by design" tone="good" />
    </dl>
  );
}

function Reading({ label, value, unit, note, small, tone, children }) {
  return (
    <div className={`reading${tone ? ` tone-${tone}` : ""}`}>
      <dt>{label}</dt>
      <dd>
        <span className={`reading-value${small ? " is-small" : ""}`}>
          {value}
          {unit && <small> {unit}</small>}
        </span>
        {note && <span className="reading-note">{note}</span>}
        {children}
      </dd>
    </div>
  );
}

/* B. Institution network: ledger card -------------------------------------------- */

export const ROLE = {
  owner: { label: "Data owner", hint: "Head + Tail on local data" },
  peer: { label: "Compute peer", hint: "Running the shared Backbone" },
  idle: { label: "Idle", hint: "Waiting for its turn" },
};

export function InstitutionCard({ institution, now, index = 0, onOpen }) {
  const role = ROLE[institution.role];
  const last = institution.last_contribution;
  const progress = institution.progress;
  return (
    <article className={`ledger-card is-${institution.role}`}>
      <div className="ledger-top">
        <span className="ledger-mark">
          <InstitutionMark index={index} size={34} />
        </span>
        <div className="ledger-id">
          <div className="ledger-name">{institution.name}</div>
          <div className="mono muted">{institution.id}</div>
        </div>
        <Status status={institution.status} />
      </div>
      <span className={`role role-${institution.role}`} title={role.hint}>
        {role.label}
        <span className="role-hint">{role.hint}</span>
      </span>
      {progress && <Meter done={progress.done} total={progress.total} label="Training progress" />}
      <dl className="kv">
        <dt>Latest contribution</dt>
        <dd>{last ? `Round ${last.round}` : "None yet"}</dd>
        {last && (
          <>
            <dt>Loss · accuracy</dt>
            <dd className="mono">
              {decimal(last.loss)} · {percent(last.accuracy)}
            </dd>
          </>
        )}
        <dt>Turns trained · computed</dt>
        <dd className="mono">
          {institution.contributions} · {institution.computations}
        </dd>
        <dt>Last seen</dt>
        <dd className="mono">{ago(institution.last_seen_at, now)}</dd>
      </dl>
      {onOpen && (
        <button type="button" className="text-button" onClick={() => onOpen(institution.id)}>
          Open in the constellation <Glyph name="arrowRight" size={14} />
        </button>
      )}
    </article>
  );
}

export function Meter({ done, total, label }) {
  const share = done / Math.max(total, 1);
  return (
    <div className="meter" role="progressbar" aria-label={label} aria-valuemin={0} aria-valuemax={total} aria-valuenow={done}>
      <div className="meter-row">
        <span>{label}</span>
        <span className="mono">
          {done} / {total} batches
        </span>
      </div>
      <div className="meter-track">
        <div className="meter-fill" style={{ transform: `scaleX(${share})` }} />
      </div>
    </div>
  );
}

/* C. Training activity: the execution rail --------------------------------------- */

const STATE_LABEL = { done: "Done", active: "In progress", pending: "Pending", skipped: "Skipped", failed: "Failed" };

export function TurnTimeline({ turn, sessionStatus, compact = false }) {
  if (!turn) {
    return (
      <section className={`rail-card${compact ? " is-compact" : ""}`}>
        <div className="eyebrow">Current turn</div>
        <div className="empty">No turn has started yet</div>
      </section>
    );
  }
  const live = sessionStatus === "running" && turn.status === "active";
  return (
    <section className={`rail-card${compact ? " is-compact" : ""}`} aria-label={live ? "Current turn" : "Latest turn"}>
      <header className="rail-head">
        <div className="eyebrow">{live ? "Current turn" : "Latest turn"}</div>
        <div className="rail-title">
          <span className="display">Round {turn.round}</span>
          <Status status={turn.status} />
        </div>
        <div className="rail-pair">
          <span className="role role-owner">{turn.owner}</span>
          <span className="muted">trains</span>
          <span className="role role-peer">{turn.peer}</span>
          <span className="muted">computes</span>
        </div>
        <div className="mono muted">Backbone v{turn.backbone_version}</div>
      </header>
      <ol className="rail">
        {turn.steps.map((step, index) => (
          <li key={step.key} className={`rail-step step-${step.state}`}>
            <span className="rail-node" aria-hidden="true">
              {step.state === "done" ? <Icon name="check" size={12} /> : step.state === "failed" ? <Icon name="cross" size={12} /> : index + 1}
            </span>
            <span className="rail-text">
              <span className="rail-label">{step.label}</span>
              <span className="rail-state">{STATE_LABEL[step.state] ?? step.state}</span>
            </span>
            <span className="rail-detail mono">{stepDetail(step, turn)}</span>
            {step.key === "backbone" && step.state === "active" && turn.batches_total > 0 && (
              <span className="rail-progress" aria-hidden="true">
                <span style={{ transform: `scaleX(${turn.batches_done / turn.batches_total})` }} />
              </span>
            )}
          </li>
        ))}
      </ol>
      {turn.error && (
        <p className="rail-error" role="alert">
          <Status status="failed" label="Turn failed" />
          <span>{turn.error}</span>
        </p>
      )}
    </section>
  );
}

function stepDetail(step, turn) {
  if (step.state === "skipped") return step.detail ?? "Skipped";
  if (step.key === "backbone") return step.detail;
  if (step.key === "checkpoint" && step.state === "done") return `v${turn.version_after}`;
  if (step.key === "completed" && step.state === "done") return seconds(turn.seconds);
  return step.state === "failed" ? "Failed" : "";
}

/* E. Model status: the artifact card --------------------------------------------- */

export function ModelStatus({ model, session }) {
  const total = model?.total_turns ?? 0;
  const done = model?.completed_contributions ?? 0;
  return (
    <section className="artifact">
      <div className="eyebrow">Shared Backbone · model artifact</div>
      <div className="artifact-version" aria-label={`Backbone version ${model?.backbone_version ?? "none"}`}>
        <span className="artifact-v">v</span>
        {model?.backbone_version ?? "—"}
      </div>
      {total > 0 && (
        <div className="artifact-lineage" aria-hidden="true">
          {Array.from({ length: total }, (_, i) => (
            <span key={i} className={i < done ? "is-done" : ""} />
          ))}
        </div>
      )}
      <dl className="kv">
        <dt>Latest checkpoint</dt>
        <dd className="mono">{clock(model?.latest_checkpoint_at)}</dd>
        <dt>Saved by</dt>
        <dd className="mono">{model?.latest_checkpoint_by ?? "—"}</dd>
        <dt>Contributions</dt>
        <dd className="mono">
          {done} / {total}
        </dd>
        <dt>Session</dt>
        <dd>
          <Status status={session.status} />
        </dd>
      </dl>
    </section>
  );
}

/* Turn history (table view of the charts) ------------------------------------------ */

const TURN_FILTERS = [
  ["all", "All"],
  ["completed", "Completed"],
  ["active", "In progress"],
  ["failed", "Failed"],
];

export function TurnTable({ turns, sessionId }) {
  const [filter, setFilter] = useState("all");
  const rows = useMemo(
    () => turns.map((turn, index) => ({ turn, index })).filter(({ turn }) => filter === "all" || turn.status === filter),
    [turns, filter],
  );
  if (turns.length === 0) return <div className="empty">No turns yet</div>;
  return (
    <div className="table-block">
      <div className="toolbar">
        <Segmented label="Filter turns by status" options={TURN_FILTERS} value={filter} onChange={setFilter} />
        <button type="button" className="ghost-button" onClick={() => downloadText(`medweave-turns-${sessionId ?? "session"}.csv`, turnsToCsv(turns))}>
          <Glyph name="download" size={16} /> Export view as CSV
        </button>
      </div>
      <div className="table-wrap" tabIndex={0} role="region" aria-label="Turn history">
        <table>
          <thead>
            <tr>
              <th className="num">Turn</th>
              <th className="num">Round</th>
              <th>Data owner</th>
              <th>Compute peer</th>
              <th>Status</th>
              <th className="num">Loss</th>
              <th className="num">Accuracy</th>
              <th className="num">Backbone</th>
              <th className="num">Duration</th>
            </tr>
          </thead>
          <tbody>
            {rows.map(({ turn, index }) => (
              <tr key={index}>
                <td className="num">{index + 1}</td>
                <td className="num">{turn.round}</td>
                <td className="mono">{turn.owner}</td>
                <td className="mono">{turn.peer}</td>
                <td>
                  <Status status={turn.status} />
                </td>
                <td className="num">{decimal(turn.loss)}</td>
                <td className="num">{percent(turn.accuracy)}</td>
                <td className="num">
                  v{turn.backbone_version}
                  {turn.version_after != null && ` → v${turn.version_after}`}
                </td>
                <td className="num">{seconds(turn.seconds)}</td>
              </tr>
            ))}
            {rows.length === 0 && (
              <tr>
                <td colSpan={9} className="empty">
                  No turns match this filter
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export function Segmented({ label, options, value, onChange }) {
  return (
    <div className="segmented" role="radiogroup" aria-label={label}>
      {options.map(([id, text]) => (
        <button key={id} type="button" role="radio" aria-checked={value === id} className={value === id ? "is-on" : ""} onClick={() => onChange(id)}>
          {text}
        </button>
      ))}
    </div>
  );
}

/* D. Privacy: the guarded preview -------------------------------------------------- */

const BOUNDARY_TILES = [
  { icon: "image", title: "Original image", where: "Stays at the institution" },
  { icon: "grid", title: "Activations before noise", where: "Exist only inside the data owner" },
  { icon: "noise", title: "Activations after noise", where: "Sent to the compute peer, never stored" },
];

export function PrivacyPanel({ noise }) {
  return (
    <section className="privacy">
      <div className="boundary-tiles" role="list">
        {BOUNDARY_TILES.map((tile) => (
          <div key={tile.title} className="boundary-tile" role="listitem">
            <div className="boundary-art" aria-hidden="true">
              <Glyph name={tile.icon} size={26} />
              <span className="boundary-lock">
                <Glyph name="lock" size={14} />
              </span>
            </div>
            <div className="boundary-title">{tile.title}</div>
            <div className="boundary-where">{tile.where}</div>
            <div className="boundary-na">Not available in the coordinator dashboard</div>
          </div>
        ))}
      </div>
      <p className="note">
        Noise:{" "}
        {noise ? (
          <strong className="mono">
            {noise.mechanism}, scale {noise.noise_scale}
          </strong>
        ) : (
          <span className="muted">not configured yet (privacy layer, Member 3)</span>
        )}
      </p>
      <p className="note muted">
        Relative change = ‖A<sub>noisy</sub> − A<sub>original</sub>‖₂ / ‖A<sub>original</sub>‖₂ measures perturbation, not a privacy guarantee.
      </p>
    </section>
  );
}

/* Activity log: the event stream --------------------------------------------------- */

const EVENT_ICON = { registration: "dot", assignment: "diamond", training: "pulse", checkpoint: "ring", completion: "check", failure: "cross" };

export function EventLog({ events }) {
  const [type, setType] = useState("all");
  const list = useRef(null);
  const types = useMemo(() => [...new Set(events.map((e) => e.type))], [events]);
  const shown = type === "all" ? events : events.filter((e) => e.type === type);
  if (events.length === 0) return <div className="empty">No events yet</div>;
  return (
    <div className="stream-block">
      <div className="toolbar">
        <label className="select">
          <span>Event type</span>
          <select value={type} onChange={(e) => setType(e.target.value)}>
            <option value="all">All ({events.length})</option>
            {types.map((t) => (
              <option key={t} value={t}>
                {t} ({events.filter((e) => e.type === t).length})
              </option>
            ))}
          </select>
        </label>
        <button type="button" className="ghost-button" onClick={() => list.current?.scrollTo({ top: 0, behavior: "smooth" })}>
          <Glyph name="arrowUp" size={16} /> Jump to newest
        </button>
      </div>
      <ul className="stream" ref={list} tabIndex={0} aria-label="Latest training events, newest first">
        {shown.map((event, index) => (
          <li key={`${event.at}-${index}`} className={`stream-item type-${event.type}`}>
            <time dateTime={event.at} className="mono">
              {clock(event.at)}
            </time>
            <span className="stream-node" aria-hidden="true">
              <Icon name={EVENT_ICON[event.type] ?? "dot"} size={12} />
            </span>
            <span className="stream-body">
              <span className="stream-type">{event.type}</span>
              <span className="stream-message">{event.message}</span>
            </span>
          </li>
        ))}
      </ul>
    </div>
  );
}
