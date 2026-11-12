import { useEffect, useRef } from "react";

import { CORE_MESH, meshPaths } from "../../lib/geometry.js";

const IDLE_SPEED = 0.11;
const ACTIVE_SPEED = 0.2;
const REWEAVE_SPEED = 1.5;
const REWEAVE_MS = 900;

/**
 * The Shared Backbone as a slowly turning geodesic mesh. Rotation is written straight to
 * the SVG in an animation frame, so React re-renders only when the data changes.
 * A completed turn (`pulse`) briefly spins the mesh faster: the model reorganising.
 */
export function Core({ x, y, r, active, pulse, reducedMotion, dimmed, selected }) {
  const front = useRef(null);
  const back = useRef(null);
  const dots = useRef([]);
  const state = useRef({ yaw: 0.6, boostUntil: 0, active });
  state.current.active = active;

  useEffect(() => {
    if (pulse > 0) state.current.boostUntil = performance.now() + REWEAVE_MS;
  }, [pulse]);

  useEffect(() => {
    const draw = (yaw) => {
      const pitch = -0.38 + Math.sin(yaw * 0.7) * 0.08;
      const mesh = meshPaths(yaw, pitch, x, y, r);
      front.current?.setAttribute("d", mesh.front);
      back.current?.setAttribute("d", mesh.back);
      mesh.points.forEach((p, i) => {
        const dot = dots.current[i];
        if (!dot) return;
        dot.setAttribute("cx", (x + p[0] * r).toFixed(1));
        dot.setAttribute("cy", (y + p[1] * r).toFixed(1));
        dot.setAttribute("opacity", (0.25 + 0.75 * Math.max(0, p[2])).toFixed(2));
      });
    };

    if (reducedMotion) {
      draw(state.current.yaw);
      return undefined;
    }
    let frame;
    let last = performance.now();
    const tick = (now) => {
      const dt = Math.min(0.05, (now - last) / 1000);
      last = now;
      const boost = Math.max(0, state.current.boostUntil - now) / REWEAVE_MS;
      const base = state.current.active ? ACTIVE_SPEED : IDLE_SPEED;
      state.current.yaw += dt * (base + boost * boost * REWEAVE_SPEED);
      draw(state.current.yaw);
      frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [x, y, r, reducedMotion]);

  return (
    <g className={`core${dimmed ? " is-dim" : ""}${selected ? " is-selected" : ""}`}>
      <circle cx={x} cy={y} r={r * 2.1} fill="url(#core-aura)" className="core-aura" />
      <circle cx={x} cy={y} r={r * 1.08} className="core-shell" />
      <g key={reducedMotion ? 0 : pulse} className={pulse > 0 && !reducedMotion ? "core-mesh is-reweaving" : "core-mesh"} style={{ transformOrigin: `${x}px ${y}px` }}>
        <path ref={back} className="core-back" />
        <path ref={front} className="core-front" />
        {CORE_MESH.vertices.map((_, i) => (
          <circle key={i} ref={(el) => (dots.current[i] = el)} r={i < 12 ? 1.9 : 1.2} className="core-vertex" />
        ))}
      </g>
      <circle cx={x} cy={y} r={r * 0.2} className="core-nucleus" />
    </g>
  );
}
