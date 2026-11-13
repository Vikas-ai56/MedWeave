import { useEffect, useMemo, useState } from "react";

import { LineChart } from "./components/LineChart.jsx";
import {
  EventLog,
  InstitutionCard,
  ModelStatus,
  Overview,
  PrivacyPanel,
  TurnTable,
  TurnTimeline,
} from "./components/Panels.jsx";
import { Status } from "./components/Status.jsx";
import { decimal, percent } from "./format.js";
import { usePolling } from "./usePolling.js";

const POLL_MS = 1500;
const loss = (turn) => turn.loss;
const accuracy = (turn) => turn.accuracy;

export default function App() {
  const { data, error } = usePolling("/api/dashboard", POLL_MS);
  const now = useNow(5000);

  return (
    <div className="app">
      <Header data={data} error={error} />
      {error && (
        <div className="banner" role="alert">
          <Status status="unreachable" /> {data ? "Showing the last data received." : ""} ({error})
        </div>
      )}
      {!data ? (
        !error && <div className="empty">Loading…</div>
      ) : !data.session ? (
        <EmptyState databaseExists={data.database_exists} />
      ) : (
        <Dashboard data={data} now={now} />
      )}
    </div>
  );
}

function Header({ data, error }) {
  const session = data?.session;
  return (
    <header className="header">
      <div className="brand">
        <svg className="brand-mark" viewBox="0 0 28 28" aria-hidden="true">
          <rect width="28" height="28" rx="7" fill="#22304a" />
          <path d="M6 9c4 0 4 10 8 10s4-10 8-10" fill="none" stroke="#22a898" strokeWidth="2.2" strokeLinecap="round" />
          <path d="M6 19c4 0 4-10 8-10s4 10 8 10" fill="none" stroke="#b4bfcf" strokeWidth="1.6" strokeLinecap="round" opacity="0.7" />
        </svg>
        <div>
          <h1>MedWeave</h1>
          <p>Collaborative split-learning training</p>
        </div>
      </div>
      <div className="session-meta">
        {session && (
          <>
            <span className="chip">
              Session <code>{session.id}</code>
            </span>
            <span className="chip">
              <Status status={session.status} />
            </span>
          </>
        )}
        <span className="chip">{error ? <Status status="unreachable" label="Offline" /> : <Status status="live" />}</span>
      </div>
    </header>
  );
}

function Dashboard({ data, now }) {
  const { session, institutions, current_turn, turns, model, noise, events } = data;
  // Charts plot finished turns only; a turn still in progress has no value yet.
  const finishedTurns = useMemo(() => turns.filter((t) => t.status === "completed" || t.status === "failed"), [turns]);
  return (
    <main>
      <Overview session={session} institutions={institutions} model={model} />

      <section className="section">
        <h2 className="section-title">Institution network</h2>
        <div className="grid grid-institutions">
          {institutions.map((institution) => (
            <InstitutionCard key={institution.id} institution={institution} now={now} />
          ))}
        </div>
      </section>

      <section className="section">
        <h2 className="section-title">Training activity</h2>
        <div className="grid grid-turn">
          <TurnTimeline turn={current_turn} sessionStatus={session.status} />
          <ModelStatus model={model} session={session} />
        </div>
      </section>

      <section className="section">
        <h2 className="section-title">Training metrics per turn</h2>
        <div className="grid grid-2">
          <div className="card">
            <h3>
              Training loss<span className="card-sub">data owner's batches, lower is better</span>
            </h3>
            <LineChart turns={finishedTurns} value={loss} format={(v) => decimal(v, 2)} ariaLabel="Training loss per turn" />
          </div>
          <div className="card">
            <h3>
              Training accuracy<span className="card-sub">data owner's batches</span>
            </h3>
            <LineChart
              turns={finishedTurns}
              value={accuracy}
              format={percent}
              domain={[0, 1]}
              ariaLabel="Training accuracy per turn"
            />
          </div>
        </div>
        <div className="card section">
          <h3>Turn history</h3>
          <TurnTable turns={turns} />
        </div>
      </section>

      <section className="section">
        <div className="grid grid-2">
          <div>
            <h2 className="section-title">Privacy</h2>
            <PrivacyPanel noise={noise} />
          </div>
          <div>
            <h2 className="section-title">Activity log</h2>
            <div className="card">
              <EventLog events={events} />
            </div>
          </div>
        </div>
      </section>
    </main>
  );
}

function EmptyState({ databaseExists }) {
  return (
    <div className="card empty-state">
      <h2>No training session yet</h2>
      <p>Start one from the repository root:</p>
      <p>
        <code>coordinator/flower_app/run_local.sh</code>
      </p>
      <p className="muted">
        {databaseExists ? "The dashboard will update as soon as the session starts." : "Waiting for the coordinator's database."}
      </p>
    </div>
  );
}

function useNow(intervalMs) {
  const [now, setNow] = useState(Date.now());
  useEffect(() => {
    const timer = setInterval(() => setNow(Date.now()), intervalMs);
    return () => clearInterval(timer);
  }, [intervalMs]);
  return now;
}
