/** Geometry for The Weave: a geodesic core and the institution layout around it. */

const PHI = (1 + Math.sqrt(5)) / 2;

const normalize = ([x, y, z]) => {
  const length = Math.hypot(x, y, z);
  return [x / length, y / length, z / length];
};

const ICOSAHEDRON = [
  [-1, PHI, 0], [1, PHI, 0], [-1, -PHI, 0], [1, -PHI, 0],
  [0, -1, PHI], [0, 1, PHI], [0, -1, -PHI], [0, 1, -PHI],
  [PHI, 0, -1], [PHI, 0, 1], [-PHI, 0, -1], [-PHI, 0, 1],
].map(normalize);

const FACES = [
  [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
  [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9], [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
];

/** Frequency-2 geodesic sphere: 42 vertices, 120 edges. */
function geodesic() {
  const vertices = [...ICOSAHEDRON];
  const midpoints = new Map();
  const midpoint = (a, b) => {
    const key = a < b ? `${a}-${b}` : `${b}-${a}`;
    if (!midpoints.has(key)) {
      const [p, q] = [vertices[a], vertices[b]];
      vertices.push(normalize([(p[0] + q[0]) / 2, (p[1] + q[1]) / 2, (p[2] + q[2]) / 2]));
      midpoints.set(key, vertices.length - 1);
    }
    return midpoints.get(key);
  };
  const edges = new Set();
  const addEdge = (a, b) => edges.add(a < b ? `${a}-${b}` : `${b}-${a}`);
  for (const [a, b, c] of FACES) {
    const ab = midpoint(a, b);
    const bc = midpoint(b, c);
    const ca = midpoint(c, a);
    for (const [p, q, r] of [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]) {
      addEdge(p, q);
      addEdge(q, r);
      addEdge(r, p);
    }
  }
  return { vertices, edges: [...edges].map((key) => key.split("-").map(Number)) };
}

export const CORE_MESH = geodesic();

export function rotate([x, y, z], yaw, pitch) {
  const cosY = Math.cos(yaw);
  const sinY = Math.sin(yaw);
  const x1 = x * cosY + z * sinY;
  const z1 = -x * sinY + z * cosY;
  const cosP = Math.cos(pitch);
  const sinP = Math.sin(pitch);
  return [x1, y * cosP - z1 * sinP, y * sinP + z1 * cosP];
}

/** Edge paths split by depth so the far side of the mesh can sit back. */
export function meshPaths(yaw, pitch, cx, cy, r) {
  const points = CORE_MESH.vertices.map((v) => rotate(v, yaw, pitch));
  let front = "";
  let back = "";
  for (const [a, b] of CORE_MESH.edges) {
    const [p, q] = [points[a], points[b]];
    const segment = `M${(cx + p[0] * r).toFixed(1)} ${(cy + p[1] * r).toFixed(1)}L${(cx + q[0] * r).toFixed(1)} ${(cy + q[1] * r).toFixed(1)}`;
    if (p[2] + q[2] > 0) front += segment;
    else back += segment;
  }
  return { front, back, points };
}

const clamp = (value, min, max) => Math.min(max, Math.max(min, value));

/** Places the core at the centre of a `width` × `height` stage and institutions on an ellipse around it. */
export function weaveLayout(width, height, count) {
  const cx = width / 2;
  const cy = height / 2;
  const rx = clamp(width * 0.33, 110, 380);
  const ry = clamp(height * 0.3, 100, 240);
  const boundaryR = clamp(Math.min(rx, ry) * 0.26, 34, 60);
  const start = -Math.PI * 0.75;
  const nodes = Array.from({ length: count }, (_, i) => {
    const angle = start + (i * 2 * Math.PI) / Math.max(count, 1);
    return { x: cx + Math.cos(angle) * rx, y: cy + Math.sin(angle) * ry, angle };
  });
  return { core: { x: cx, y: cy, r: clamp(Math.min(rx, ry) * 0.36, 34, 84) }, nodes, boundaryR };
}

/**
 * A gently curved thread from an institution's data boundary to the core's surface.
 * Threads start at the boundary, never the node: raw data has no path out.
 */
export function thread(node, core, boundaryR) {
  const dx = core.x - node.x;
  const dy = core.y - node.y;
  const length = Math.hypot(dx, dy) || 1;
  const [ux, uy] = [dx / length, dy / length];
  const s = [node.x + ux * (boundaryR + 3), node.y + uy * (boundaryR + 3)];
  const e = [core.x - ux * core.r * 1.12, core.y - uy * core.r * 1.12];
  const bend = 0.14 * (length - boundaryR - core.r);
  const q = [(s[0] + e[0]) / 2 - uy * bend, (s[1] + e[1]) / 2 + ux * bend];
  const at = (t) => {
    const u = 1 - t;
    return [u * u * s[0] + 2 * u * t * q[0] + t * t * e[0], u * u * s[1] + 2 * u * t * q[1] + t * t * e[1]];
  };
  const tangent = (t) => {
    const dxt = 2 * (1 - t) * (q[0] - s[0]) + 2 * t * (e[0] - q[0]);
    const dyt = 2 * (1 - t) * (q[1] - s[1]) + 2 * t * (e[1] - q[1]);
    return (Math.atan2(dyt, dxt) * 180) / Math.PI;
  };
  const f = (n) => n.toFixed(1);
  return { d: `M${f(s[0])} ${f(s[1])}Q${f(q[0])} ${f(q[1])} ${f(e[0])} ${f(e[1])}`, at, tangent };
}
