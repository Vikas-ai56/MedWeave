import { useEffect, useRef, useState } from "react";

import { CommandBar, Sidebar, StaleAlert } from "./components/Shell.jsx";
import { EmptyState, LoadingState, UnreachableState } from "./components/States.jsx";
import { useNow, usePrefersReducedMotion, useUpdatedAt } from "./lib/hooks.js";
import { VIEWS, routeHref, useHashRoute } from "./lib/route.js";
import { SIMULATION_TICK_MS, simulateSnapshot } from "./lib/simulate.js";
import { usePolling } from "./usePolling.js";
import { EvidenceView } from "./views/EvidenceView.jsx";
import { ExperimentsView } from "./views/ExperimentsView.jsx";
import { InstitutionsView } from "./views/InstitutionsView.jsx";
import { ProtectionView } from "./views/ProtectionView.jsx";
import { WeaveView } from "./views/WeaveView.jsx";

const POLL_MS = 1500;
const SESSIONS_POLL_MS = 10000;
const SESSION_VIEWS = { weave: WeaveView, institutions: InstitutionsView, evidence: EvidenceView };
const STATIC_VIEWS = { experiments: ExperimentsView, protection: ProtectionView };

/** A new `refresh` value changes the URL, which restarts usePolling's cycle immediately. */
function dashboardUrl(sessionId, refresh) {
  const params = new URLSearchParams();
  if (sessionId) params.set("session_id", sessionId);
  if (refresh) params.set("_refresh", String(refresh));
  const query = params.toString();
  return `/api/dashboard${query ? `?${query}` : ""}`;
}

function useSimulation(enabled) {
  const origin = useRef(Date.now());
  const [snapshot, setSnapshot] = useState(null);
  useEffect(() => {
    if (!enabled) {
      setSnapshot(null);
      return undefined;
    }
    origin.current = Date.now();
    const tick = () => setSnapshot(simulateSnapshot(Date.now(), origin.current));
    tick();
    const timer = setInterval(tick, SIMULATION_TICK_MS);
    return () => clearInterval(timer);
  }, [enabled]);
  return snapshot;
}

export default function App() {
  const { view, params } = useHashRoute();
  const simulated = params.get("sim") === "1";
  const [sessionId, setSessionId] = useState(null);
  const [refresh, setRefresh] = useState(0);

  const live = usePolling(dashboardUrl(sessionId, refresh), POLL_MS);
  const sessions = usePolling("/api/sessions", SESSIONS_POLL_MS);
  const simulation = useSimulation(simulated);

  const data = simulated ? simulation : live.data;
  const error = simulated ? null : live.error;
  const updatedAt = useUpdatedAt(data);
  const now = useNow(simulated ? 1000 : 5000);
  const reducedMotion = usePrefersReducedMotion();
  const connection = simulated ? "simulated" : error ? "unreachable" : live.data ? "live" : "connecting";

  const first = useRef(true);
  useEffect(() => {
    document.title = `${VIEWS.find((v) => v.id === view).label} · MedWeave`;
    if (first.current) {
      first.current = false;
      return;
    }
    window.scrollTo(0, 0);
    document.getElementById("view-title")?.focus({ preventScroll: true });
  }, [view]);

  const setSimulated = (on) => {
    window.location.hash = routeHref(view, { simulated: on });
  };
  const simulate = () => setSimulated(true);

  return (
    <div className={`app view-${view}${simulated ? " is-simulated" : ""}`}>
      <a
        className="skip-link"
        href="#main"
        onClick={(event) => {
          event.preventDefault();
          document.getElementById("main")?.focus();
        }}
      >
        Skip to content
      </a>
      <Sidebar view={view} simulated={simulated} connection={connection} updatedAt={updatedAt} />
      <div className="workspace">
        <CommandBar
          session={data?.session}
          sessions={sessions.data ?? []}
          sessionId={sessionId}
          onSessionChange={setSessionId}
          onRefresh={() => setRefresh((n) => n + 1)}
          simulated={simulated}
          onSimulatedChange={setSimulated}
          connection={connection}
          updatedAt={updatedAt}
        />
        {error && data && <StaleAlert error={error} updatedAt={updatedAt} />}
        <main id="main" tabIndex={-1}>
          <Content view={view} data={data} error={error} now={now} reducedMotion={reducedMotion} onSimulate={simulate} />
        </main>
      </div>
    </div>
  );
}

function Content({ view, data, error, now, reducedMotion, onSimulate }) {
  const Static = STATIC_VIEWS[view];
  if (Static) return <Static snapshot={data} />;
  if (!data) return error ? <UnreachableState error={error} onSimulate={onSimulate} /> : <LoadingState />;
  if (!data.session) return <EmptyState databaseExists={data.database_exists} onSimulate={onSimulate} />;
  const View = SESSION_VIEWS[view];
  return <View snapshot={data} now={now} reducedMotion={reducedMotion} />;
}
