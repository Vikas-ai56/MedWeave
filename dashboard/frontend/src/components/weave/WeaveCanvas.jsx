import { useEffect, useMemo, useRef, useState } from "react";

import { ago, percent } from "../../format.js";
import { thread, weaveLayout } from "../../lib/geometry.js";
import { useElementSize } from "../../lib/hooks.js";
import { MarkShapes } from "../Glyphs.jsx";
import { Status } from "../Status.jsx";
import { Core } from "./Core.jsx";

export const ROLE_LABEL = { owner: "Data owner", peer: "Compute peer", idle: "Idle" };
const STATUS_WORD = { online: "Online", offline: "Offline", training: "Busy" };
const DATA_DOTS = 18;
const REWEAVE_MS = 900;

function stepState(turn, key) {
  return turn?.steps?.find((s) => s.key === key)?.state;
}

/** What the threads are doing right now, from the current turn's steps. */
function flowOf(session, turn) {
  if (session?.status !== "running" || turn?.status !== "active") return "quiet";
  if (stepState(turn, "backbone") === "active") return "carrying";
  if (stepState(turn, "checkpoint") === "active") return "saving";
  return "preparing";
}

function metaLine(mode, institution, noise) {
  const { status, role, contributions, computations, last_contribution: last } = institution;
  if (mode === "contribution") return `${contributions} trained · ${computations} computed`;
  if (mode === "performance") return last?.accuracy != null ? `${percent(last.accuracy)} latest training acc.` : "No contribution yet";
  if (mode === "protection") {
    if (status === "offline") return "Offline · data still local";
    return noise ? `Data local · noise σ ${noise.noise_scale}` : "Data local · no noise set";
  }
  return `${ROLE_LABEL[role]} · ${STATUS_WORD[status] ?? status}`;
}

/**
 * The institutions arranged around the Shared Backbone. SVG draws; transparent HTML
 * buttons sit over each node so every institution is a real, focusable control.
 */
export function WeaveCanvas({ snapshot, mode = "weave", focusId, onFocus, selectedId, onSelect, pulse = 0, reducedMotion, now, labels = true }) {
  const stage = useRef(null);
  const svg = useRef(null);
  const { width, height } = useElementSize(stage);
  const [reweaving, setReweaving] = useState(false);
  const { institutions, session, current_turn: turn, model, noise } = snapshot;

  const layout = useMemo(
    () => (width > 0 && height > 0 ? weaveLayout(width, height, institutions.length) : null),
    [width, height, institutions.length],
  );
  const threads = useMemo(
    () => (layout ? layout.nodes.map((node) => thread(node, layout.core, layout.boundaryR)) : []),
    [layout],
  );

  useEffect(() => {
    if (pulse === 0 || reducedMotion) return undefined;
    svg.current?.querySelectorAll("[data-burst]").forEach((el) => el.beginElement?.());
    setReweaving(true);
    const timer = setTimeout(() => setReweaving(false), REWEAVE_MS);
    return () => clearTimeout(timer);
  }, [pulse, reducedMotion]);

  const flow = flowOf(session, turn);
  const ownerId = session?.status === "running" ? session.current_owner : null;
  const peerId = session?.status === "running" ? session.current_peer : null;
  const dim = (id) => focusId != null && focusId !== id;
  const maxWork = Math.max(1, ...institutions.map((i) => i.contributions + i.computations));
  const focused = institutions.find((i) => i.id === focusId);
  const focusedIndex = institutions.indexOf(focused);

  return (
    <div className={`weave-stage mode-${mode}${reweaving ? " is-reweaving" : ""}`} ref={stage}>
      {layout && (
        <svg ref={svg} className="weave-svg" width={width} height={height} viewBox={`0 0 ${width} ${height}`} aria-hidden="true">
          <defs>
            <radialGradient id="core-aura">
              <stop offset="0%" stopColor="var(--violet)" stopOpacity="0.22" />
              <stop offset="45%" stopColor="var(--violet)" stopOpacity="0.07" />
              <stop offset="100%" stopColor="var(--violet)" stopOpacity="0" />
            </radialGradient>
            <radialGradient id="halo-owner">
              <stop offset="0%" stopColor="var(--mint)" stopOpacity="0.3" />
              <stop offset="100%" stopColor="var(--mint)" stopOpacity="0" />
            </radialGradient>
            <radialGradient id="halo-peer">
              <stop offset="0%" stopColor="var(--violet)" stopOpacity="0.3" />
              <stop offset="100%" stopColor="var(--violet)" stopOpacity="0" />
            </radialGradient>
          </defs>

          {institutions.map((inst, i) => {
            const t = threads[i];
            const isOwner = inst.id === ownerId;
            const isPeer = inst.id === peerId;
            const carrying = isOwner && flow === "carrying";
            const classes = ["thread", isOwner && "is-owner", isPeer && "is-peer", carrying && "is-carrying", dim(inst.id) && "is-dim", inst.status === "offline" && "is-offline"]
              .filter(Boolean)
              .join(" ");
            const [mx, my] = t.at(0.5);
            return (
              <g key={inst.id} className={classes}>
                <path d={t.d} className="thread-base" />
                {(isOwner || isPeer) && <path d={t.d} className="thread-live" />}
                {carrying && !reducedMotion && (
                  <>
                    {[0, 0.8, 1.6].map((begin) => (
                      <circle key={`a${begin}`} r="3.2" className="pulse pulse-activation">
                        <animateMotion dur="2.4s" begin={`${begin}s`} repeatCount="indefinite" path={t.d} />
                      </circle>
                    ))}
                    {[0.4, 1.6].map((begin) => (
                      <circle key={`g${begin}`} r="2.4" className="pulse pulse-gradient">
                        <animateMotion dur="2.4s" begin={`${begin}s`} repeatCount="indefinite" path={t.d} keyPoints="1;0" keyTimes="0;1" calcMode="linear" />
                      </circle>
                    ))}
                  </>
                )}
                {carrying && reducedMotion && (
                  <path d="M-5 -4 L2 0 L-5 4" className="thread-arrow" transform={`translate(${mx} ${my}) rotate(${t.tangent(0.5)})`} />
                )}
                {!reducedMotion && (
                  <circle r="3" className="pulse pulse-burst" opacity="0">
                    <animateMotion data-burst="" dur="0.7s" begin="indefinite" fill="freeze" path={t.d} />
                    <animate data-burst="" attributeName="opacity" values="0;1;1;0" keyTimes="0;0.15;0.8;1" dur="0.7s" begin="indefinite" fill="freeze" />
                  </circle>
                )}
                {labels && mode === "weave" && width > 720 && (isOwner || isPeer) && (
                  <text x={mx} y={my} dy={-10} className="thread-label" textAnchor="middle">
                    {isOwner ? (carrying ? "noised activations ⇄ gradients" : "waiting for activations") : "hosts the Backbone"}
                  </text>
                )}
                {mode === "protection" && (
                  <text x={mx} y={my} dy={-8} className="thread-label" textAnchor="middle">
                    {noise ? `σ ${noise.noise_scale}` : "no noise"}
                  </text>
                )}
              </g>
            );
          })}

          <Core
            x={layout.core.x}
            y={layout.core.y}
            r={layout.core.r}
            active={flow !== "quiet"}
            pulse={pulse}
            reducedMotion={reducedMotion}
            dimmed={focusId != null && focusId !== "core"}
            selected={selectedId === "core"}
          />
          {labels && (
            <g className={`core-label${dim("core") ? " is-dim" : ""}`}>
              <text x={layout.core.x} y={layout.core.y + layout.core.r * 1.12 + 22} textAnchor="middle" className="core-title">
                Shared Backbone
              </text>
              <text x={layout.core.x} y={layout.core.y + layout.core.r * 1.12 + 38} textAnchor="middle" className="core-version">
                {model?.backbone_version == null ? "no checkpoint yet" : `v${model.backbone_version}`}
              </text>
            </g>
          )}

          {institutions.map((inst, i) => (
            <Node
              key={inst.id}
              institution={inst}
              index={i}
              point={layout.nodes[i]}
              boundaryR={layout.boundaryR}
              above={layout.nodes[i].y < layout.core.y}
              mode={mode}
              noise={noise}
              labels={labels}
              dimmed={dim(inst.id)}
              focused={focusId === inst.id}
              selected={selectedId === inst.id}
              workShare={(inst.contributions + inst.computations) / maxWork}
            />
          ))}
        </svg>
      )}

      {layout &&
        institutions.map((inst, i) => {
          const { x, y } = layout.nodes[i];
          const size = layout.boundaryR * 2 + 8;
          return (
            <button
              key={inst.id}
              type="button"
              className="node-hit"
              style={{ left: x - size / 2, top: y - size / 2, width: size, height: size }}
              aria-label={`${inst.name} (${inst.id}): ${STATUS_WORD[inst.status] ?? inst.status}, ${ROLE_LABEL[inst.role]}. Open details.`}
              aria-pressed={selectedId === inst.id}
              onPointerEnter={() => onFocus(inst.id)}
              onPointerLeave={() => onFocus(null)}
              onFocus={() => onFocus(inst.id)}
              onBlur={() => onFocus(null)}
              onClick={() => onSelect(inst.id)}
            />
          );
        })}
      {layout && (
        <button
          type="button"
          className="node-hit is-core"
          style={{
            left: layout.core.x - layout.core.r * 1.2,
            top: layout.core.y - layout.core.r * 1.2,
            width: layout.core.r * 2.4,
            height: layout.core.r * 2.4,
          }}
          aria-label={`Shared Backbone, version ${model?.backbone_version ?? "none"}. Open the global model.`}
          aria-pressed={selectedId === "core"}
          onPointerEnter={() => onFocus("core")}
          onPointerLeave={() => onFocus(null)}
          onFocus={() => onFocus("core")}
          onBlur={() => onFocus(null)}
          onClick={() => onSelect("core")}
        />
      )}

      {layout && focused && selectedId !== focused.id && (
        <NodeTooltip institution={focused} point={layout.nodes[focusedIndex]} boundaryR={layout.boundaryR} stageWidth={width} now={now} />
      )}
    </div>
  );
}

function Node({ institution, index, point, boundaryR: B, above, mode, noise, labels, dimmed, focused, selected, workShare }) {
  const { role, status, progress } = institution;
  const markSize = B * 0.66 * (mode === "contribution" ? 0.75 + 0.45 * Math.sqrt(workShare) : 1);
  const progressShare = progress ? progress.done / Math.max(progress.total, 1) : null;
  const accuracy = institution.last_contribution?.accuracy;
  const classes = [
    "node",
    `role-${role}`,
    `status-${status}`,
    dimmed && "is-dim",
    focused && "is-focus",
    selected && "is-selected",
  ]
    .filter(Boolean)
    .join(" ");
  const nameY = above ? -B - 30 : B + 22;

  return (
    <g className={classes} transform={`translate(${point.x.toFixed(1)} ${point.y.toFixed(1)})`}>
      {role !== "idle" && <circle r={B * 1.25} fill={`url(#halo-${role})`} className="node-halo" />}
      <circle r={B} className="node-boundary" />
      {Array.from({ length: DATA_DOTS }, (_, i) => {
        const a = (i / DATA_DOTS) * Math.PI * 2;
        return <circle key={i} cx={Math.cos(a) * B * 0.8} cy={Math.sin(a) * B * 0.8} r="1.1" className="node-data" />;
      })}
      {progressShare != null && (
        <>
          <circle r={B * 0.56} className="node-progress-track" />
          <circle r={B * 0.56} className="node-progress" pathLength="1" strokeDasharray={`${progressShare} 1`} transform="rotate(-90)" />
        </>
      )}
      <g className="node-mark" transform={`translate(${-markSize / 2} ${-markSize / 2}) scale(${markSize / 24})`}>
        <MarkShapes index={index} />
      </g>

      {mode === "contribution" &&
        Array.from({ length: Math.min(institution.contributions, 12) }, (_, i) => (
          <circle key={`c${i}`} r={B + 5 + i * 3} className="node-ring" />
        ))}
      {mode === "contribution" &&
        Array.from({ length: Math.min(institution.computations, 24) }, (_, i) => {
          const a = (i / 24) * Math.PI * 2 - Math.PI / 2;
          return <line key={`k${i}`} x1={Math.cos(a) * (B - 4)} y1={Math.sin(a) * (B - 4)} x2={Math.cos(a) * (B + 2)} y2={Math.sin(a) * (B + 2)} className="node-tick" />;
        })}
      {mode === "performance" && (
        <>
          <circle r={B + 7} className="node-score-track" />
          {accuracy != null && (
            <circle r={B + 7} className="node-score" pathLength="1" strokeDasharray={`${accuracy} 1`} transform="rotate(-90)" />
          )}
        </>
      )}
      {mode === "protection" && (
        <g transform={`translate(-7 ${-B - 8})`} className="node-lock">
          <rect x="1.5" y="6" width="11" height="8" rx="2" />
          <path d="M4 6V4.5a3 3 0 0 1 6 0V6" />
        </g>
      )}

      {labels && (
        <g className="node-label">
          <text y={nameY} textAnchor="middle" className="node-name">
            {institution.name}
          </text>
          <text y={nameY + 16} textAnchor="middle" className="node-meta">
            {metaLine(mode, institution, noise)}
          </text>
        </g>
      )}
    </g>
  );
}

function NodeTooltip({ institution, point, boundaryR, stageWidth, now }) {
  const right = point.x > stageWidth / 2;
  const last = institution.last_contribution;
  const style = {
    top: point.y - 24,
    left: right ? undefined : point.x + boundaryR + 16,
    right: right ? stageWidth - point.x + boundaryR + 16 : undefined,
  };
  return (
    <div className="weave-tooltip glass" style={style} role="status">
      <div className="weave-tooltip-name">{institution.name}</div>
      <div className="mono muted">{institution.id}</div>
      <dl className="kv kv-tight">
        <dt>Status</dt>
        <dd>
          <Status status={institution.status} />
        </dd>
        <dt>Role</dt>
        <dd>{ROLE_LABEL[institution.role]}</dd>
        <dt>Turns trained · computed</dt>
        <dd className="mono">
          {institution.contributions} · {institution.computations}
        </dd>
        <dt>Latest training acc.</dt>
        <dd className="mono">{last ? percent(last.accuracy) : "—"}</dd>
        <dt>Last seen</dt>
        <dd className="mono">{ago(institution.last_seen_at, now)}</dd>
      </dl>
      <div className="weave-tooltip-hint">Select for details</div>
    </div>
  );
}
