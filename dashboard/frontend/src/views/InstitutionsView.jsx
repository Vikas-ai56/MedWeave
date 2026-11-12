import { useState } from "react";

import { DetailLayer } from "../components/DetailLayer.jsx";
import { InstitutionCard, Segmented } from "../components/Panels.jsx";
import { WeaveCanvas } from "../components/weave/WeaveCanvas.jsx";

const MODES = [
  ["contribution", "Contribution"],
  ["performance", "Performance"],
  ["protection", "Protection"],
];

const MODE_NOTE = {
  contribution: "Mark size grows with total work. Each outer ring is one turn trained on local data; each tick is one turn spent computing the Backbone for someone else.",
  performance: "The outer arc fills to the institution's latest training accuracy, measured on its own batches.",
  protection: "Every boundary is closed: images and labels stay inside. Threads carry only noised activations and gradients.",
};

const FILTERS = [
  ["all", "All", () => true],
  ["active", "Active", (i) => i.status === "training"],
  ["owner", "Owner", (i) => i.role === "owner"],
  ["peer", "Peer", (i) => i.role === "peer"],
  ["idle", "Idle", (i) => i.role === "idle"],
];

export function InstitutionsView({ snapshot, now, reducedMotion }) {
  const [mode, setMode] = useState("contribution");
  const [filter, setFilter] = useState("all");
  const [focusId, setFocusId] = useState(null);
  const [selectedId, setSelectedId] = useState(null);
  const { institutions } = snapshot;
  const test = FILTERS.find(([id]) => id === filter)[2];
  const options = FILTERS.map(([id, label, f]) => [id, `${label} ${institutions.filter(f).length}`]);

  const open = (id) => {
    setSelectedId(id);
    document.getElementById("view-title")?.scrollIntoView({ behavior: reducedMotion ? "auto" : "smooth" });
  };

  return (
    <div className="page">
      <header className="page-head">
        <div className="eyebrow">Institutions</div>
        <h1 className="page-title" id="view-title" tabIndex={-1}>
          A constellation of <em>independent</em> institutions.
        </h1>
        <p className="page-sub">Each keeps its own data and its own boundary. Together they train one shared Backbone.</p>
      </header>

      <section className="constellation glass-soft" aria-label="Institution constellation">
        <div className="constellation-bar">
          <Segmented label="Constellation mode" options={MODES} value={mode} onChange={setMode} />
          <p className="muted small">{MODE_NOTE[mode]}</p>
        </div>
        <div className="constellation-stage">
          <WeaveCanvas
            snapshot={snapshot}
            mode={mode}
            focusId={focusId}
            onFocus={setFocusId}
            selectedId={selectedId}
            onSelect={(id) => setSelectedId((current) => (current === id ? null : id))}
            reducedMotion={reducedMotion}
            now={now}
          />
        </div>
        <DetailLayer snapshot={snapshot} selectedId={selectedId} onClose={() => setSelectedId(null)} now={now} />
      </section>

      <section className="section" aria-labelledby="ledger-title">
        <div className="section-head">
          <div>
            <div className="eyebrow">Institution network</div>
            <h2 className="section-title" id="ledger-title">
              Ledger
            </h2>
          </div>
          <Segmented label="Filter institutions" options={options} value={filter} onChange={setFilter} />
        </div>
        <div className="ledger-grid">
          {institutions.map((institution, index) =>
            test(institution) ? <InstitutionCard key={institution.id} institution={institution} index={index} now={now} onOpen={open} /> : null,
          )}
        </div>
        {!institutions.some(test) && <div className="empty">No institutions match this filter</div>}
      </section>
    </div>
  );
}
