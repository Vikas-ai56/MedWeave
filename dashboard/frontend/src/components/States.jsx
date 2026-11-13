import { BrandMark } from "./Shell.jsx";
import { Status } from "./Status.jsx";

function StateCard({ eyebrow, title, children, onSimulate, tone }) {
  return (
    <div className={`state-card glass${tone ? ` tone-${tone}` : ""}`}>
      <div className="state-mark" aria-hidden="true">
        <BrandMark size={44} />
      </div>
      <div className="eyebrow">{eyebrow}</div>
      <h2 className="state-title">{title}</h2>
      {children}
      {onSimulate && (
        <div className="state-actions">
          <button type="button" className="primary-button" onClick={onSimulate}>
            Preview a simulated session
          </button>
          <span className="muted small">Clearly labelled, generated in the browser, never mixed with live data.</span>
        </div>
      )}
    </div>
  );
}

export function LoadingState() {
  return (
    <div className="state" role="status" aria-live="polite">
      <div className="loom" aria-hidden="true">
        <span />
        <span />
        <span />
      </div>
      <p className="muted">Loading…</p>
    </div>
  );
}

export function UnreachableState({ error, onSimulate }) {
  const notReady = /^503/.test(error ?? "");
  return (
    <div className="state">
      <StateCard
        eyebrow={notReady ? "Database not ready" : "Backend unreachable"}
        title={notReady ? "The coordinator is still preparing its database" : "The dashboard API is not answering"}
        onSimulate={onSimulate}
        tone="critical"
      >
        <p>
          <Status status="unreachable" /> <span className="mono">{error}</span>
        </p>
        {notReady ? (
          <p className="muted">This usually clears within a few seconds once the tables exist. The page keeps retrying every 1.5 seconds.</p>
        ) : (
          <>
            <p>Start the dashboard backend from the repository root:</p>
            <pre className="code">dashboard/run_dashboard.sh</pre>
            <p className="muted">The page keeps retrying every 1.5 seconds and recovers on its own.</p>
          </>
        )}
      </StateCard>
    </div>
  );
}

export function EmptyState({ databaseExists, onSimulate }) {
  return (
    <div className="state">
      <StateCard eyebrow={databaseExists ? "No session" : "Waiting for the database"} title="No training session yet" onSimulate={onSimulate}>
        <p>Start one from the repository root:</p>
        <pre className="code">coordinator/flower_app/run_local.sh</pre>
        <p className="muted">{databaseExists ? "The dashboard will update as soon as the session starts." : "Waiting for the coordinator's database."}</p>
      </StateCard>
    </div>
  );
}
