import { EventLog, ModelStatus, TurnTimeline } from "../components/Panels.jsx";
import { Status } from "../components/Status.jsx";
import { connectedCount, percent1 } from "../lib/derive.js";
import { RUNS, bytesLabel, communicationPerRound } from "../lib/research.js";

const best = RUNS.find((r) => r.id === "medweave");
const central = RUNS.find((r) => r.id === "centralized");
const splitRound = communicationPerRound().find((r) => r.id === "medweave");

export function EvidenceView({ snapshot }) {
  const { session, institutions, current_turn: turn, model, events } = snapshot;
  const maxWork = Math.max(1, ...institutions.map((i) => Math.max(i.contributions, i.computations)));

  return (
    <div className="page">
      <header className="page-head">
        <div className="eyebrow">Evidence</div>
        <h1 className="page-title" id="view-title" tabIndex={-1}>
          A calm record of <em>what happened.</em>
        </h1>
        <p className="page-sub">The live audit trail of this session, the measured research behind it, and what it adds up to.</p>
      </header>

      <div className="evidence-grid">
        <section aria-labelledby="proof-title">
          <div className="eyebrow">Proof rail</div>
          <h2 className="section-title" id="proof-title">
            This turn, step by step
          </h2>
          <TurnTimeline turn={turn} sessionStatus={session.status} />
        </section>
        <section aria-labelledby="artifact-title">
          <div className="eyebrow">Artifact</div>
          <h2 className="section-title" id="artifact-title">
            The shared model
          </h2>
          <ModelStatus model={model} session={session} />
        </section>
        <section className="evidence-log" aria-labelledby="log-title">
          <div className="eyebrow">Activity log · latest 40</div>
          <h2 className="section-title" id="log-title">
            Event stream
          </h2>
          <EventLog events={events} />
        </section>
      </div>

      <section className="section" aria-labelledby="participation-title">
        <div className="eyebrow">Hospital participation</div>
        <h2 className="section-title" id="participation-title">
          Who trained, who computed
        </h2>
        <div className="participation" role="table" aria-label="Turns trained and computed per institution">
          {institutions.map((inst) => (
            <div key={inst.id} className="participation-row" role="row">
              <span role="rowheader">
                {inst.name} <span className="mono muted">{inst.id}</span>
              </span>
              <span className="participation-bars" role="cell">
                <span className="bar is-trained" style={{ transform: `scaleX(${inst.contributions / maxWork})` }} />
                <span className="bar is-computed" style={{ transform: `scaleX(${inst.computations / maxWork})` }} />
              </span>
              <span className="mono small" role="cell">
                {inst.contributions} trained · {inst.computations} computed
              </span>
            </div>
          ))}
        </div>
        <ul className="legend inline">
          <li>
            <span className="legend-bar is-trained" aria-hidden="true" /> Turns trained on local data (upper bar)
          </li>
          <li>
            <span className="legend-bar is-computed" aria-hidden="true" /> Turns computing the Backbone for a peer (lower bar)
          </li>
        </ul>
      </section>

      <section className="section" aria-labelledby="compare-title">
        <div className="eyebrow">Measured comparison · PathMNIST test split</div>
        <h2 className="section-title" id="compare-title">
          Centralized and split, side by side
        </h2>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Approach</th>
                <th className="num">Test accuracy</th>
                <th className="num">AUC</th>
                <th className="num">Macro F1</th>
                <th className="num">Hospital gap</th>
                <th className="num">Seeds</th>
              </tr>
            </thead>
            <tbody>
              {RUNS.map((r) => (
                <tr key={r.id}>
                  <td>{r.label}</td>
                  <td className="num">
                    {percent1(r.test.accuracy)} <span className="muted">± {(r.test.accuracyStd * 100).toFixed(1)}</span>
                  </td>
                  <td className="num">{r.test.auc.toFixed(3)}</td>
                  <td className="num">{r.test.macroF1.toFixed(3)}</td>
                  <td className="num">{r.hospitalGap == null ? "—" : `${r.hospitalGap.toFixed(1)} pts`}</td>
                  <td className="num">{r.seeds}</td>
                </tr>
              ))}
              <tr className="is-muted">
                <td>FedAvg</td>
                <td colSpan={5} className="muted">
                  Not run in this project
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </section>

      <section className="section" aria-labelledby="attack-title">
        <div className="eyebrow">Attack tests</div>
        <h2 className="section-title" id="attack-title">
          What has been tried against it
        </h2>
        <div className="attack-grid">
          <div className="attack-card">
            <Status status="failed" label="Leak found" />
            <h3>Label inference from gradients</h3>
            <p>320 of 320 labels recovered from returned gradients in a tested configuration. Activation noise does not mitigate it. Open issue.</p>
          </div>
          <div className="attack-card">
            <Status status="pending" label="Not run" />
            <h3>Reconstruction and membership inference</h3>
            <p>Out of the initial scope (SPEC §1.7). No claim is made either way.</p>
          </div>
        </div>
      </section>

      <section className="section sdg-section" aria-labelledby="sdg-title">
        <div className="eyebrow">SDG 9 · Industry, innovation and infrastructure</div>
        <h2 className="section-title" id="sdg-title">
          MedWeave as resilient AI infrastructure
        </h2>
        <div className="sdg-grid">
          <ul className="sdg-statements">
            <li>Hospitals collaborate on one model without centralizing sensitive data.</li>
            <li>Training rotates across institutions, so no single site is the system.</li>
            <li>Institutions that cannot pool datasets can still contribute to and use advanced AI.</li>
            <li>A measurable digital-infrastructure prototype on public data, not a claim of real hospital deployment.</li>
          </ul>
          <dl className="indicators">
            <Indicator label="Connected institutions" value={`${connectedCount(institutions)} / ${institutions.length}`} source="live" />
            <Indicator label="Successful training turns" value={`${session.completed_turns} / ${session.total_turns}`} source="live" />
            <Indicator label="Test accuracy vs centralized" value={`${percent1(best.test.accuracy)} vs ${percent1(central.test.accuracy)}`} source="measured · baseline-recipe reference" />
            <Indicator label="Communication per round" value={bytesLabel(splitRound.bytes)} source="computed from model shapes" />
            <Indicator label="Raw data retained locally" value="100%" source="by design: no image or label is sent" />
            <Indicator label="Privacy budget consumed" value="Not tracked" source="no formal DP accounting" />
          </dl>
        </div>
      </section>
    </div>
  );
}

function Indicator({ label, value, source }) {
  return (
    <div className="indicator">
      <dt>{label}</dt>
      <dd>
        <span className="indicator-value">{value}</span>
        <span className="indicator-source">{source}</span>
      </dd>
    </div>
  );
}
