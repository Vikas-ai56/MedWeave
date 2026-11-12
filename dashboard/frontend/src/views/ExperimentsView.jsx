import { useMemo, useState } from "react";

import { LineChart } from "../components/LineChart.jsx";
import { Segmented, TurnTable } from "../components/Panels.jsx";
import { SeriesChart } from "../components/SeriesChart.jsx";
import { decimal, percent } from "../format.js";
import { finishedTurns, percent1 } from "../lib/derive.js";
import { NOISE_COLORS, NOISE_SWEEP, NOT_MEASURED, RUNS, SERIES_COLORS, bytesLabel, communicationPerRound } from "../lib/research.js";

const TABS = [
  ["accuracy", "Accuracy"],
  ["metrics", "Test metrics"],
  ["privacy", "Privacy cost"],
  ["communication", "Communication"],
];
const MARKERS = { centralized: "circle", baseline: "square", augmented: "triangle", medweave: "diamond" };
const loss = (turn) => turn.loss;
const accuracy = (turn) => turn.accuracy;
const best = RUNS.find((r) => r.id === "medweave");

export function ExperimentsView({ snapshot }) {
  const [tab, setTab] = useState("accuracy");
  const session = snapshot?.session;
  const turns = snapshot?.turns ?? [];
  const finished = useMemo(() => finishedTurns(turns), [turns]);

  return (
    <div className="page">
      <header className="page-head">
        <div className="eyebrow">Research Observatory</div>
        <h1 className="page-title" id="view-title" tabIndex={-1}>
          PathMNIST <span className="title-sep">/</span> Split-learning CNN
        </h1>
        <p className="page-sub">The shared model improves without pooling the source images.</p>
      </header>

      <dl className="figures" aria-label="Research summary">
        <Figure label="Rounds" value="10" />
        <Figure label="Hospitals" value="04" />
        <Figure label="Test accuracy" value={percent1(best.test.accuracy)} note={`± ${(best.test.accuracyStd * 100).toFixed(1)} pts, ${best.seeds} seeds`} />
        <Figure label="Test AUC" value={best.test.auc.toFixed(3)} />
        <Figure label="Hospital gap" value={`${best.hospitalGap.toFixed(1)} pts`} note="best vs worst institution" />
        <Figure label="Live noise" value={snapshot?.noise ? `σ ${snapshot.noise.noise_scale}` : "—"} note={snapshot?.noise ? "this session" : "no session or not set"} />
      </dl>

      <section className="observatory glass-soft" aria-label="Comparison">
        <div className="observatory-bar">
          <Segmented label="Measure" options={TABS} value={tab} onChange={setTab} />
        </div>
        {tab === "accuracy" && <AccuracyPanel />}
        {tab === "metrics" && <MetricsPanel />}
        {tab === "privacy" && <PrivacyCostPanel />}
        {tab === "communication" && <CommunicationPanel />}
      </section>

      <section className="section" aria-labelledby="live-title">
        <div className="section-head">
          <div>
            <div className="eyebrow">This session{session ? ` · ${session.id}` : ""}</div>
            <h2 className="section-title" id="live-title">
              Training metrics per turn
            </h2>
          </div>
          <ul className="legend inline" aria-label="Chart legend">
            <li>
              <span className="legend-line" aria-hidden="true" /> Completed turn
            </li>
            <li>
              <span className="legend-failed" aria-hidden="true" /> Failed turn (no value)
            </li>
          </ul>
        </div>
        <p className="note muted">Training metrics on the data owner's batches, not held-out test accuracy. Finished turns only.</p>
        {session ? (
          <>
            <div className="chart-pair">
              <figure className="chart-card">
                <figcaption>
                  Training loss<span className="card-sub">lower is better</span>
                </figcaption>
                <LineChart turns={finished} value={loss} format={(v) => decimal(v, 2)} ariaLabel="Training loss per turn" />
              </figure>
              <figure className="chart-card">
                <figcaption>
                  Training accuracy<span className="card-sub">data owner's batches</span>
                </figcaption>
                <LineChart turns={finished} value={accuracy} format={percent} domain={[0, 1]} ariaLabel="Training accuracy per turn" />
              </figure>
            </div>
            <div className="table-card">
              <h3 className="card-title">Turn history</h3>
              <TurnTable turns={turns} sessionId={session.id} />
            </div>
          </>
        ) : (
          <div className="empty">No training session yet. Live metrics appear here once one starts.</div>
        )}
      </section>
    </div>
  );
}

function Figure({ label, value, note }) {
  return (
    <div className="figure">
      <dt>{label}</dt>
      <dd>
        <span className="figure-value">{value}</span>
        {note && <span className="figure-note">{note}</span>}
      </dd>
    </div>
  );
}

function SeriesToggles({ items, visible, onToggle }) {
  return (
    <div className="toggles" role="group" aria-label="Series">
      {items.map((s) => (
        <button key={s.id} type="button" className="toggle" aria-pressed={visible.has(s.id)} onClick={() => onToggle(s.id)}>
          <span className="toggle-swatch" style={{ background: s.color }} aria-hidden="true" />
          {s.label}
        </button>
      ))}
      {NOT_MEASURED.map((s) => (
        <span key={s.id} className="toggle is-disabled" title={s.reason}>
          <span className="toggle-swatch" aria-hidden="true" />
          {s.label} · not measured
        </span>
      ))}
    </div>
  );
}

function AccuracyPanel() {
  const [visible, setVisible] = useState(() => new Set(RUNS.map((r) => r.id)));
  const toggle = (id) =>
    setVisible((prev) => {
      const next = new Set(prev);
      if (next.has(id)) {
        if (next.size > 1) next.delete(id);
      } else next.add(id);
      return next;
    });
  const series = RUNS.map((r) => ({
    id: r.id,
    label: r.label,
    short: r.short,
    color: SERIES_COLORS[r.id],
    values: r.byRound,
    marker: MARKERS[r.id],
    dashed: r.reference,
    emphasis: r.id === "medweave",
  }));
  return (
    <div className="panel-body">
      <SeriesToggles items={series} visible={visible} onToggle={toggle} />
      <SeriesChart
        series={series.filter((s) => visible.has(s.id))}
        xCount={10}
        xName="Round"
        domain={[0.65, 0.96]}
        format={percent1}
        tickFormat={percent}
        ariaLabel="Validation accuracy by round for each training approach"
      />
      <div className="run-notes">
        {RUNS.filter((r) => visible.has(r.id)).map((r) => (
          <div key={r.id} className="run-note">
            <span className="toggle-swatch" style={{ background: SERIES_COLORS[r.id] }} aria-hidden="true" />
            <div>
              <div className="run-note-title">
                {r.label} <span className="mono muted">test {percent1(r.test.accuracy)}</span>
              </div>
              <div className="muted small">{r.what}</div>
            </div>
          </div>
        ))}
      </div>
      <p className="note muted">
        Validation accuracy per round, mean of {RUNS.map((r) => r.seeds).join("/")} seeds. Test accuracy is taken at the round with the best mean validation accuracy. The centralized line uses the baseline recipe, so the MedWeave recipe passing it reflects augmentation and normalisation, not an advantage of splitting.
      </p>
    </div>
  );
}

const METRICS = [
  ["accuracy", "Accuracy"],
  ["balanced", "Balanced accuracy"],
  ["macroF1", "Macro F1"],
  ["auc", "AUC"],
];
const DOT_MIN = 0.7;

function MetricsPanel() {
  const x = (v) => `${((v - DOT_MIN) / (1 - DOT_MIN)) * 100}%`;
  return (
    <div className="panel-body">
      <div className="dotplot" role="table" aria-label="Test metrics by approach">
        <div className="dotplot-scale" aria-hidden="true">
          {[0.7, 0.8, 0.9, 1].map((t) => (
            <span key={t} style={{ left: x(t) }}>
              {t.toFixed(1)}
            </span>
          ))}
        </div>
        {METRICS.map(([key, label]) => (
          <div key={key} className="dotplot-row" role="row">
            <span className="dotplot-label" role="rowheader">
              {label}
            </span>
            <span className="dotplot-track" role="cell">
              {RUNS.map((r) => (
                <span
                  key={r.id}
                  className={`dot dot-${MARKERS[r.id]}`}
                  style={{ left: x(r.test[key]), "--c": SERIES_COLORS[r.id] }}
                  title={`${r.label}: ${r.test[key].toFixed(3)}`}
                  aria-label={`${r.label}: ${r.test[key].toFixed(3)}`}
                  role="img"
                />
              ))}
            </span>
          </div>
        ))}
      </div>
      <ul className="legend inline">
        {RUNS.map((r) => (
          <li key={r.id}>
            <span className={`dot dot-static dot-${MARKERS[r.id]}`} style={{ "--c": SERIES_COLORS[r.id] }} aria-hidden="true" /> {r.label}
          </li>
        ))}
      </ul>
      <div className="gap-bars">
        <div className="eyebrow">Hospital gap · best minus worst institution, test accuracy points</div>
        {RUNS.filter((r) => r.hospitalGap != null).map((r) => (
          <div key={r.id} className="gap-row">
            <span>{r.label}</span>
            <span className="gap-track">
              <span style={{ transform: `scaleX(${r.hospitalGap / 14})`, background: SERIES_COLORS[r.id] }} />
            </span>
            <span className="mono">{r.hospitalGap.toFixed(1)} pts</span>
          </div>
        ))}
        <p className="note muted">A smaller gap means every institution gets a model that works about as well on its own data.</p>
      </div>
    </div>
  );
}

function PrivacyCostPanel() {
  const series = NOISE_SWEEP.map((n, i) => ({
    id: `s${n.sigma}`,
    label: `σ ${n.sigma}`,
    short: `σ ${n.sigma}`,
    color: NOISE_COLORS[i],
    values: n.byTurn,
    marker: ["circle", "square", "triangle"][i],
  }));
  return (
    <div className="panel-body">
      <SeriesChart series={series} xCount={8} xName="Turn" domain={[0.35, 0.85]} format={percent1} tickFormat={percent} ariaLabel="Validation accuracy per turn at three activation noise levels" />
      <div className="noise-cells">
        {NOISE_SWEEP.map((n, i) => (
          <div key={n.sigma} className="noise-cell">
            <span className="toggle-swatch" style={{ background: NOISE_COLORS[i] }} aria-hidden="true" />
            <div className="mono">σ {n.sigma}</div>
            <div className="noise-change">{n.change.toFixed(0)}%</div>
            <div className="muted small">mean activation change</div>
            <div className="mono small">final {percent1(n.byTurn.at(-1))} val.</div>
          </div>
        ))}
      </div>
      <p className="note muted">
        Two-round sweep on the early placeholder CNN, clean evaluation on the official validation split. Low noise changed every outgoing activation by about a quarter with no measurable accuracy cost. Relative change measures perturbation, not a privacy guarantee, and there is no ε: this is not differential privacy.
      </p>
    </div>
  );
}

function CommunicationPanel() {
  const rows = communicationPerRound();
  const max = Math.log10(Math.max(...rows.map((r) => r.bytes)));
  const min = Math.log10(Math.min(...rows.map((r) => r.bytes))) - 0.6;
  return (
    <div className="panel-body">
      <div className="comm" role="table" aria-label="Bytes moved per training round">
        {rows.map((r) => (
          <div key={r.id} className={`comm-row${r.ran ? "" : " is-analytical"}`} role="row">
            <div className="comm-label" role="rowheader">
              <div>{r.label}</div>
              <div className="muted small">{r.moves}</div>
            </div>
            <div className="comm-track" role="cell">
              <span style={{ transform: `scaleX(${(Math.log10(r.bytes) - min) / (max - min)})` }} />
            </div>
            <div className="comm-value mono" role="cell">
              {bytesLabel(r.bytes)}
            </div>
            <div className="comm-note muted small">{r.note}</div>
          </div>
        ))}
      </div>
      <p className="note muted">
        Computed from tensor shapes for MedWeavePathCNN-v1 (float32, batch 64, 351 batches per turn, 4 turns per round); log scale. Split learning is the most bandwidth-hungry option here: that is the price of no institution holding the full model and labels never leaving their owner.
      </p>
    </div>
  );
}
