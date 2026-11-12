import { useState } from "react";

import { DetailLayer } from "../components/DetailLayer.jsx";
import { Overview, TurnTimeline } from "../components/Panels.jsx";
import { WeaveCanvas } from "../components/weave/WeaveCanvas.jsx";
import { clock } from "../format.js";
import { headlineAccuracy, statusSentence } from "../lib/derive.js";
import { useTurnCompletion } from "../lib/hooks.js";

export function WeaveView({ snapshot, now, reducedMotion }) {
  const [focusId, setFocusId] = useState(null);
  const [selectedId, setSelectedId] = useState(null);
  const { session, institutions, current_turn: turn, turns, model, noise, events } = snapshot;
  const pulse = useTurnCompletion(session.completed_turns, session.id);
  const latest = events[0];
  const toggle = (id) => setSelectedId((current) => (current === id ? null : id));

  return (
    <div className="weave-view">
      <header className="weave-intro">
        <div className="eyebrow">The Living Weave</div>
        <h1 className="hero-title" id="view-title" tabIndex={-1}>
          Intelligence, <em>without</em> centralization.
        </h1>
        <p className="hero-sub">Different hospitals. Shared innovation. Stronger digital infrastructure.</p>
        <p className="status-line">
          <span>{statusSentence(session, institutions)}</span> The data remains where care happens.
        </p>
        <WeaveLegend />
      </header>

      <div className="weave-canvas-cell">
        <WeaveCanvas
          snapshot={snapshot}
          focusId={focusId}
          onFocus={setFocusId}
          selectedId={selectedId}
          onSelect={toggle}
          pulse={pulse}
          reducedMotion={reducedMotion}
          now={now}
        />
        {latest && (
          <div className="audit-line" key={`${latest.at}-${latest.message}`} aria-live="polite">
            <span className="audit-dot" aria-hidden="true" />
            <span className="mono">{clock(latest.at)}</span>
            <span className="audit-type">{latest.type}</span>
            <span className="audit-message">{latest.message}</span>
          </div>
        )}
      </div>

      <aside className="weave-rail" aria-label="Training activity">
        <TurnTimeline turn={turn} sessionStatus={session.status} compact />
      </aside>

      <div className="weave-readout">
        <Overview session={session} institutions={institutions} model={model} accuracy={headlineAccuracy(turns, institutions.length)} noise={noise} />
      </div>

      <DetailLayer snapshot={snapshot} selectedId={selectedId} onClose={() => setSelectedId(null)} now={now} />
    </div>
  );
}

function WeaveLegend() {
  return (
    <ul className="legend" aria-label="How to read The Weave">
      <li>
        <span className="legend-pulse is-activation" aria-hidden="true" /> Noised activations, owner to Backbone
      </li>
      <li>
        <span className="legend-pulse is-gradient" aria-hidden="true" /> Gradients returning
      </li>
      <li>
        <span className="legend-boundary" aria-hidden="true" /> Data boundary: nothing raw crosses it
      </li>
    </ul>
  );
}
