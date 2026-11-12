import { useState } from "react";

import { Glyph } from "../components/Glyphs.jsx";
import { PrivacyPanel } from "../components/Panels.jsx";

const STAGES = ["Local data", "Protected update", "Shared model"];

const steps = (noise) => [
  {
    title: "Local data stays local",
    stage: 0,
    icon: "lock",
    body: "Images and labels are read only inside the institution. They are never sent to the coordinator, the relay or another institution.",
  },
  {
    title: "The Head runs on site",
    stage: 0,
    icon: "grid",
    body: "The first layers of the network run at the hospital and turn each image into a 32 × 14 × 14 feature map. That map, not the image, is what can leave.",
  },
  {
    title: "Activation noise",
    stage: 1,
    icon: "noise",
    body: `Gaussian noise${noise ? ` (σ ${noise.noise_scale} in this session)` : ""} is added to every outgoing activation. In the team's sweep, σ 0.25 changed activations by about 26% with no measurable accuracy cost.`,
  },
  {
    title: "Blind relay",
    stage: 1,
    icon: "arrowRight",
    body: "The relay forwards tensors between the data owner and the compute peer and counts messages for the progress bar. It never opens them.",
  },
  {
    title: "Audit record",
    stage: 2,
    icon: "evidence",
    body: "The coordinator records IDs, statuses, versions, counts, training metrics and events, nothing else. Each completed turn saves a numbered Backbone checkpoint.",
  },
];

export function ProtectionView({ snapshot }) {
  const [active, setActive] = useState(0);
  const list = steps(snapshot?.noise);
  const step = list[active];

  const onKeyDown = (event) => {
    const move = { ArrowDown: 1, ArrowRight: 1, ArrowUp: -1, ArrowLeft: -1 }[event.key];
    if (!move) return;
    event.preventDefault();
    const next = (active + move + list.length) % list.length;
    setActive(next);
    document.getElementById(`protect-step-${next}`)?.focus();
  };

  return (
    <div className="page">
      <header className="page-head">
        <div className="eyebrow">Protection</div>
        <h1 className="page-title" id="view-title" tabIndex={-1}>
          What leaves, and <em>what never does.</em>
        </h1>
        <p className="page-sub">Follow one training step from a hospital's images to the shared model.</p>
      </header>

      <div className="transform" aria-label="Transformation stages">
        {STAGES.map((label, i) => (
          <div key={label} className={`transform-stage${step.stage === i ? " is-on" : ""}${step.stage > i ? " is-past" : ""}`}>
            <span className="transform-index mono">0{i + 1}</span>
            <span className="transform-label">{label}</span>
            {i < STAGES.length - 1 && (
              <span className="transform-arrow" aria-hidden="true">
                <Glyph name="arrowRight" size={22} />
              </span>
            )}
          </div>
        ))}
      </div>

      <section className="protect glass-soft">
        <ol className="protect-steps" role="tablist" aria-label="Protection steps" aria-orientation="vertical" onKeyDown={onKeyDown}>
          {list.map((s, i) => (
            <li key={s.title} role="presentation">
              <button
                id={`protect-step-${i}`}
                type="button"
                role="tab"
                aria-selected={active === i}
                aria-controls="protect-panel"
                tabIndex={active === i ? 0 : -1}
                className={active === i ? "is-on" : ""}
                onClick={() => setActive(i)}
              >
                <span className="protect-num mono">{i + 1}</span>
                <span>{s.title}</span>
              </button>
            </li>
          ))}
        </ol>
        <div className="protect-panel" id="protect-panel" role="tabpanel" aria-labelledby={`protect-step-${active}`} key={active}>
          <span className="protect-icon" aria-hidden="true">
            <Glyph name={step.icon} size={28} />
          </span>
          <h2 className="protect-title">{step.title}</h2>
          <p>{step.body}</p>
        </div>
      </section>

      <section className="section" aria-labelledby="boundary-title">
        <div className="eyebrow">Data boundary</div>
        <h2 className="section-title" id="boundary-title">
          Two sides of one line
        </h2>
        <div className="boundary-columns">
          <div className="boundary-col is-visible">
            <h3>Visible to the coordinator</h3>
            <ul>
              <li>Institution IDs and statuses</li>
              <li>Backbone versions and checkpoints</li>
              <li>Batch counts and turn timings</li>
              <li>Training loss and accuracy per turn</li>
              <li>Events: registrations, assignments, failures</li>
            </ul>
          </div>
          <div className="boundary-col is-local">
            <h3>Stays at the institution</h3>
            <ul>
              <li>Images</li>
              <li>Labels</li>
              <li>Activations before noise</li>
              <li>The Head and Tail of the model</li>
            </ul>
            <p className="muted small">Noised activations and gradients travel only between the data owner and its compute peer, through the relay, and are never stored by the coordinator.</p>
          </div>
        </div>
      </section>

      <section className="section" aria-labelledby="preview-title">
        <div className="eyebrow">Activation privacy</div>
        <h2 className="section-title" id="preview-title">
          A guarded preview
        </h2>
        <PrivacyPanel noise={snapshot?.noise ?? null} />
      </section>

      <section className="section limits" aria-labelledby="limits-title">
        <div className="eyebrow">Limits</div>
        <h2 className="section-title" id="limits-title">
          What this does not protect
        </h2>
        <ul className="limit-list">
          <li>
            <strong>Not differential privacy.</strong> There is no clipping, no privacy accountant and no ε. Noise scale is a standard deviation in activation units.
          </li>
          <li>
            <strong>The gradient channel is open.</strong> Gradients sent back to the peer are derived from the labels. The team's audit recovered 320 of 320 labels from them in a tested configuration, and activation noise does not change that.
          </li>
          <li>
            <strong>No secure aggregation or blockchain.</strong> The Backbone moves turn by turn between institutions; the audit trail is the coordinator's own database.
          </li>
        </ul>
      </section>
    </div>
  );
}
