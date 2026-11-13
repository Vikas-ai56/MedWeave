import { useEffect, useRef, useState } from "react";

import { ago, clock, decimal, percent, seconds } from "../format.js";
import { completedTurns, institutionName, turnsOwnedBy } from "../lib/derive.js";
import { useCopy, useEscape } from "../lib/hooks.js";
import { Glyph, InstitutionMark } from "./Glyphs.jsx";
import { Meter, ROLE, TurnTimeline } from "./Panels.jsx";
import { Sparkline } from "./Sparkline.jsx";
import { Status } from "./Status.jsx";

/** A floating glass layer over the canvas. Escape or the close button dismisses it. */
export function DetailLayer({ snapshot, selectedId, onClose, now }) {
  const heading = useRef(null);
  useEscape(onClose, selectedId != null);
  useEffect(() => {
    if (selectedId != null) heading.current?.focus();
  }, [selectedId]);
  if (selectedId == null) return null;

  const index = snapshot.institutions.findIndex((i) => i.id === selectedId);
  const institution = snapshot.institutions[index];
  return (
    <aside className="detail glass" role="dialog" aria-modal="false" aria-labelledby="detail-title">
      <button type="button" className="icon-button detail-close" onClick={onClose} aria-label="Close details">
        <Glyph name="close" size={18} />
      </button>
      {institution ? (
        <InstitutionDetail institution={institution} index={index} snapshot={snapshot} heading={heading} now={now} />
      ) : (
        <CoreDetail snapshot={snapshot} heading={heading} />
      )}
    </aside>
  );
}

function InstitutionDetail({ institution, index, snapshot, heading, now }) {
  const { copied, copy } = useCopy();
  const [evidence, setEvidence] = useState(false);
  const own = turnsOwnedBy(snapshot.turns, institution.id);
  const last = institution.last_contribution;
  const latestOwn = own.filter((t) => t.status === "completed" || t.status === "failed").at(-1);
  const role = ROLE[institution.role];
  const noise = snapshot.noise;

  return (
    <>
      <header className="detail-head">
        <span className={`detail-mark is-${institution.role}`}>
          <InstitutionMark index={index} size={40} />
        </span>
        <div>
          <div className="eyebrow">Institution</div>
          <h2 id="detail-title" ref={heading} tabIndex={-1} className="detail-title">
            {institution.name}
          </h2>
          <button type="button" className="id-pill" onClick={() => copy(institution.id)} aria-label={`Copy institution ID ${institution.id}`}>
            <span className="mono">{institution.id}</span>
            <Glyph name="copy" size={14} />
            <span className="copied" aria-live="polite">
              {copied === institution.id ? "Copied" : ""}
            </span>
          </button>
        </div>
      </header>

      <div className="detail-badges">
        <Status status={institution.status} />
        <span className={`role role-${institution.role}`} title={role.hint}>
          {role.label}
        </span>
      </div>

      {institution.progress ? (
        <Meter done={institution.progress.done} total={institution.progress.total} label="Training progress" />
      ) : (
        <p className="detail-quiet">Not training this turn. {role.hint}.</p>
      )}

      <dl className="kv">
        <dt>Latest contribution</dt>
        <dd>{last ? `Round ${last.round}` : "None yet"}</dd>
        <dt>Training loss · accuracy</dt>
        <dd className="mono">{last ? `${decimal(last.loss)} · ${percent(last.accuracy)}` : "—"}</dd>
        <dt>Last update</dt>
        <dd>{latestOwn ? <Status status={latestOwn.status} label={latestOwn.status === "completed" ? "Accepted · Backbone advanced" : "Rejected · turn failed"} /> : "—"}</dd>
        <dt>Turns trained · computed</dt>
        <dd className="mono">
          {institution.contributions} · {institution.computations}
        </dd>
        <dt>Last seen</dt>
        <dd className="mono">{ago(institution.last_seen_at, now)}</dd>
      </dl>

      <div className="detail-section">
        <div className="eyebrow">Training accuracy across its turns</div>
        <Sparkline turns={own} label={`Training accuracy over ${institution.name}'s turns`} />
      </div>

      <div className="detail-section protection-summary">
        <div className="eyebrow">Protection</div>
        <ul className="checks">
          <li>
            <Glyph name="lock" size={16} /> Images and labels never leave {institution.name}.
          </li>
          <li>
            <Glyph name="noise" size={16} />
            {noise ? `Activations leave only after ${noise.mechanism} noise, σ ${noise.noise_scale}.` : "No activation noise is configured for this session."}
          </li>
        </ul>
      </div>

      <button type="button" className="ghost-button wide" aria-expanded={evidence} onClick={() => setEvidence((v) => !v)}>
        {evidence ? "Hide evidence" : "View evidence"}
        <Glyph name="chevron" size={14} className={evidence ? "rot-90" : ""} />
      </button>
      {evidence && (
        <div className="detail-evidence">
          <div className="eyebrow">Turns where {institution.id} was the data owner</div>
          {own.length === 0 ? (
            <div className="empty">No turns yet</div>
          ) : (
            <div className="table-wrap compact">
              <table>
                <thead>
                  <tr>
                    <th className="num">Round</th>
                    <th>Peer</th>
                    <th>Status</th>
                    <th className="num">Loss</th>
                    <th className="num">Acc.</th>
                    <th className="num">Backbone</th>
                    <th className="num">Time</th>
                  </tr>
                </thead>
                <tbody>
                  {own.map((t, i) => (
                    <tr key={i}>
                      <td className="num">{t.round}</td>
                      <td className="mono">{t.peer}</td>
                      <td>
                        <Status status={t.status} />
                      </td>
                      <td className="num">{decimal(t.loss)}</td>
                      <td className="num">{percent(t.accuracy)}</td>
                      <td className="num">
                        v{t.backbone_version}
                        {t.version_after != null && ` → v${t.version_after}`}
                      </td>
                      <td className="num">{seconds(t.seconds)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
          <p className="note muted">
            This is coordination metadata from the coordinator's database. The coordinator has no access to this institution's images, labels, activations or gradients.
          </p>
        </div>
      )}
    </>
  );
}

function CoreDetail({ snapshot, heading }) {
  const [evidence, setEvidence] = useState(false);
  const { session, model, institutions, turns, current_turn: turn } = snapshot;
  const done = completedTurns(turns);
  const hostedBy = session?.status === "running" ? session.current_peer : null;
  const lineage = done.slice(-6).reverse();

  return (
    <>
      <header className="detail-head">
        <span className="detail-mark is-core" aria-hidden="true">
          <Glyph name="weave" size={30} />
        </span>
        <div>
          <div className="eyebrow">Global model</div>
          <h2 id="detail-title" ref={heading} tabIndex={-1} className="detail-title">
            Shared Backbone
          </h2>
          <div className="mono muted">{hostedBy ? `computed this turn by ${hostedBy}` : "at rest with the coordinator"}</div>
        </div>
      </header>

      <div className="detail-hero">
        <span className="artifact-v">v</span>
        {model?.backbone_version ?? "—"}
      </div>

      <dl className="kv">
        <dt>Training round</dt>
        <dd className="mono">{session ? `${session.current_round || "—"} / ${session.num_rounds}` : "—"}</dd>
        <dt>Contributions</dt>
        <dd className="mono">
          {model?.completed_contributions ?? 0} / {model?.total_turns ?? 0}
        </dd>
        <dt>Latest checkpoint</dt>
        <dd className="mono">
          {clock(model?.latest_checkpoint_at)} · {model?.latest_checkpoint_by ?? "—"}
        </dd>
        <dt>Checkpoint status</dt>
        <dd>
          {turn ? (
            <Status
              status={turn.status === "active" ? "active" : turn.status}
              label={turn.status === "active" ? "Next checkpoint in progress" : turn.status === "completed" ? "Up to date" : turn.status === "failed" ? "Last turn failed; version kept" : undefined}
            />
          ) : (
            "—"
          )}
        </dd>
      </dl>

      <div className="detail-section">
        <div className="eyebrow">Training accuracy progression · all turns</div>
        <Sparkline turns={turns} label="Training accuracy across all finished turns" />
      </div>

      <div className="detail-section">
        <div className="eyebrow">Participating institutions</div>
        <ul className="participants">
          {institutions.map((inst) => (
            <li key={inst.id}>
              <span>{inst.name}</span>
              <span className="mono muted">
                {inst.contributions} trained · {inst.computations} computed
              </span>
            </li>
          ))}
        </ul>
      </div>

      <button type="button" className="ghost-button wide" aria-expanded={evidence} onClick={() => setEvidence((v) => !v)}>
        {evidence ? "Hide evidence" : "View evidence"}
        <Glyph name="chevron" size={14} className={evidence ? "rot-90" : ""} />
      </button>
      {evidence && (
        <div className="detail-evidence">
          <TurnTimeline turn={turn} sessionStatus={session?.status} compact />
          <div className="eyebrow">Recent checkpoint lineage</div>
          {lineage.length === 0 ? (
            <div className="empty">No checkpoints from this session yet</div>
          ) : (
            <ol className="lineage">
              {lineage.map((t, i) => (
                <li key={i}>
                  <span className="mono">
                    v{t.backbone_version} → v{t.version_after}
                  </span>
                  <span className="muted">
                    {institutionName(institutions, t.owner)} trained, {t.peer} computed
                  </span>
                </li>
              ))}
            </ol>
          )}
          <p className="note muted">
            The Backbone moves between institutions one turn at a time; there is no averaging step. Each completed turn saves a new version.
          </p>
        </div>
      )}
    </>
  );
}
