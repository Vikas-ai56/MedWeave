import { ago, clock, decimal, percent, seconds } from "../format.js";
import { Icon, Status } from "./Status.jsx";

/* A. Overview ------------------------------------------------------------------ */

export function Overview({ session, institutions, model }) {
  const online = institutions.filter((i) => i.status !== "offline").length;
  const running = session.status === "running";
  return (
    <div className="grid grid-tiles">
      <Tile label="Round" value={session.current_round || "—"} unit={`/ ${session.num_rounds}`} />
      <Tile label="Turns completed" value={session.completed_turns} unit={`/ ${session.total_turns}`} />
      <Tile label="Institutions connected" value={online} unit={`/ ${institutions.length}`} />
      <Tile label="Data owner" value={running ? session.current_owner ?? "—" : "—"} note={running ? "training on local data" : "no turn running"} />
      <Tile label="Compute peer" value={running ? session.current_peer ?? "—" : "—"} note={running ? "running the Backbone" : "no turn running"} />
      <Tile label="Backbone" value={model?.backbone_version == null ? "—" : `v${model.backbone_version}`} note="latest checkpoint" />
    </div>
  );
}

function Tile({ label, value, unit, note }) {
  return (
    <div className="card tile">
      <div className="tile-label">{label}</div>
      <div className="tile-value">
        {value} {unit && <small>{unit}</small>}
      </div>
      {note && <div className="tile-note">{note}</div>}
    </div>
  );
}

/* B. Institution network ------------------------------------------------------- */

const ROLE = {
  owner: { label: "Data owner", hint: "Head + Tail on local data" },
  peer: { label: "Compute peer", hint: "Running the shared Backbone" },
  idle: { label: "Idle", hint: "Waiting for its turn" },
};

export function InstitutionCard({ institution, now }) {
  const role = ROLE[institution.role];
  const last = institution.last_contribution;
  const progress = institution.progress;
  return (
    <article className={`card inst is-${institution.role}`}>
      <div className="inst-top">
        <div>
          <div className="inst-name">{institution.name}</div>
          <div className="inst-id">{institution.id}</div>
        </div>
        <Status status={institution.status} />
      </div>
      <span className={`role role-${institution.role}`} title={role.hint}>
        {role.label}
      </span>
      {progress && (
        <div className="meter" aria-label={`Training progress ${progress.done} of ${progress.total} batches`}>
          <div className="meter-row">
            <span>Training progress</span>
            <span>
              {progress.done} / {progress.total} batches
            </span>
          </div>
          <div className="meter-track">
            <div className="meter-fill" style={{ width: `${(100 * progress.done) / Math.max(progress.total, 1)}%` }} />
          </div>
        </div>
      )}
      <dl className="kv">
        <dt>Latest contribution</dt>
        <dd>{last ? `Round ${last.round}` : "None yet"}</dd>
        {last && (
          <>
            <dt>Loss · accuracy</dt>
            <dd>
              {decimal(last.loss)} · {percent(last.accuracy)}
            </dd>
          </>
        )}
        <dt>Turns trained · computed</dt>
        <dd>
          {institution.contributions} · {institution.computations}
        </dd>
        <dt>Last seen</dt>
        <dd>{ago(institution.last_seen_at, now)}</dd>
      </dl>
    </article>
  );
}

/* C. Training activity --------------------------------------------------------- */

const STEP_ICON = { done: "check", failed: "cross", skipped: "ring", pending: "ring", active: null };

export function TurnTimeline({ turn, sessionStatus }) {
  if (!turn) return <div className="card"><h3>Current turn</h3><div className="empty">No turn has started yet</div></div>;
  const live = sessionStatus === "running" && turn.status === "active";
  return (
    <div className="card">
      <h3>
        {live ? "Current turn" : "Latest turn"}
        <span className="card-sub">Backbone v{turn.backbone_version}</span>
      </h3>
      <div className="turn-head">
        <strong>Round {turn.round}</strong>
        <span>
          {turn.owner} trains · {turn.peer} computes
        </span>
        <Status status={turn.status} />
      </div>
      <ol className="steps">
        {turn.steps.map((step) => (
          <li key={step.key} className={`step step-${step.state}`}>
            <span className="step-icon">
              {step.state === "active" ? (
                <svg width="14" height="14" viewBox="0 0 14 14" aria-hidden="true">
                  <path className="spin" d="M7 2a5 5 0 1 1-5 5" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
                </svg>
              ) : (
                <Icon name={STEP_ICON[step.state]} size={12} />
              )}
            </span>
            <span className="step-label">
              {step.label}
              <span className="sr-only"> ({step.state})</span>
            </span>
            <span className="step-detail">{stepDetail(step, turn)}</span>
          </li>
        ))}
      </ol>
      {turn.error && <p className="note status-critical">{turn.error}</p>}
    </div>
  );
}

function stepDetail(step, turn) {
  if (step.state === "skipped") return step.detail ?? "Skipped";
  if (step.key === "backbone") return step.detail;
  if (step.key === "checkpoint" && step.state === "done") return `v${turn.version_after}`;
  if (step.key === "completed" && step.state === "done") return seconds(turn.seconds);
  return step.state === "failed" ? "Failed" : "";
}

/* E. Model status -------------------------------------------------------------- */

export function ModelStatus({ model, session }) {
  return (
    <div className="card model">
      <h3>Shared Backbone</h3>
      <div className="hero">
        <small>v</small>
        {model?.backbone_version ?? "—"}
      </div>
      <dl className="kv">
        <dt>Latest checkpoint</dt>
        <dd>{clock(model?.latest_checkpoint_at)}</dd>
        <dt>Saved by</dt>
        <dd>{model?.latest_checkpoint_by ?? "—"}</dd>
        <dt>Contributions</dt>
        <dd>
          {model?.completed_contributions ?? 0} / {model?.total_turns ?? 0}
        </dd>
        <dt>Session</dt>
        <dd>
          <Status status={session.status} />
        </dd>
      </dl>
    </div>
  );
}

/* Turn history (table view of the charts) --------------------------------------- */

export function TurnTable({ turns }) {
  if (turns.length === 0) return <div className="empty">No turns yet</div>;
  return (
    <div className="table-wrap">
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
          {turns.map((turn, index) => (
            <tr key={index}>
              <td className="num">{index + 1}</td>
              <td className="num">{turn.round}</td>
              <td>{turn.owner}</td>
              <td>{turn.peer}</td>
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
        </tbody>
      </table>
    </div>
  );
}

/* D. Privacy visualisation (later phase) ----------------------------------------- */

export function PrivacyPanel({ noise }) {
  return (
    <div className="card">
      <h3>
        Activation privacy
        <span className="card-sub">later phase</span>
      </h3>
      <div className="privacy-grid" aria-hidden="true">
        <div className="privacy-slot">Original image (stays at the institution)</div>
        <div className="privacy-slot">Activations before noise</div>
        <div className="privacy-slot">Activations after noise</div>
      </div>
      <p className="note">
        Noise:{" "}
        {noise ? (
          <strong>
            {noise.mechanism}, scale {noise.noise_scale}
          </strong>
        ) : (
          <span className="muted">not configured yet (privacy layer, Member 3)</span>
        )}
      </p>
      <p className="note muted">
        Relative change = ‖A<sub>noisy</sub> − A<sub>original</sub>‖₂ / ‖A<sub>original</sub>‖₂ measures perturbation, not a
        privacy guarantee.
      </p>
    </div>
  );
}

/* Activity log ------------------------------------------------------------------ */

export function EventLog({ events }) {
  if (events.length === 0) return <div className="empty">No events yet</div>;
  return (
    <ul className="log">
      {events.map((event, index) => (
        <li key={`${event.at}-${index}`}>
          <time dateTime={event.at}>{clock(event.at)}</time>
          <span className={`type type-${event.type}`}>{event.type}</span>
          <span className="message">{event.message}</span>
        </li>
      ))}
    </ul>
  );
}
