/**
 * A simulated session for demos without a coordinator.
 *
 * `simulateSnapshot(now, origin)` is a pure function of time that returns the exact
 * /api/dashboard shape, so every view renders it through the same code path as live
 * data. Turn steps and roles mirror dashboard/backend/snapshot.py.
 */

const INSTITUTIONS = [
  { id: "inst-1", name: "City General Hospital" },
  { id: "inst-2", name: "Northside Medical Center" },
  { id: "inst-3", name: "St. Mary's Hospital" },
  { id: "inst-4", name: "Regional Health Institute" },
];
const ROUNDS = 10;
const TOTAL_TURNS = ROUNDS * INSTITUTIONS.length;
const TURN_MS = 6400;
const COOLDOWN_MS = 16000;
const CYCLE_MS = TOTAL_TURNS * TURN_MS + COOLDOWN_MS;
const BATCHES = 351;
const START_TURN = 12;
const FAILED_TURN = 21;
const FAIL_AT = 0.46;
const NOISE = { mechanism: "gaussian", noise_scale: 0.25 };

// Phase boundaries inside one turn, as fractions of TURN_MS.
const FIRST_ACTIVATION = 0.07;
const BATCHES_END = 0.82;
const COMPLETE = 0.94;

const STEP_LABELS = [
  ["selected", "Institution selected"],
  ["assigned", "Compute peer assigned"],
  ["activations", "Activations generated"],
  ["noise", "Noise applied"],
  ["backbone", "Backbone computation"],
  ["checkpoint", "Backbone checkpoint updated"],
  ["completed", "Contribution completed"],
];

const unit = (i, salt) => {
  const x = Math.sin(i * 12.9898 + salt * 78.233) * 43758.5453;
  return x - Math.floor(x);
};
const iso = (ms) => new Date(ms).toISOString();
const clamp01 = (v) => Math.min(1, Math.max(0, v));

const accuracyAt = (k) => 0.885 - 0.43 * Math.exp(-k / 8) + (unit(k, 1) - 0.5) * 0.03;
const lossAt = (k) => 0.32 + 1.25 * Math.exp(-k / 7) + (unit(k, 2) - 0.5) * 0.05;

function sessionId(cycle) {
  return `sim-${Math.floor(unit(cycle + 3, 9) * 0xffffffff).toString(16).padStart(8, "0")}`;
}

/** One row of compute_assignments as it stands at `phase` (0..1) through its slot. */
function assignment(k, phase, sessionStart) {
  const n = INSTITUTIONS.length;
  const assignedAt = sessionStart + k * TURN_MS;
  const failed = k === FAILED_TURN && phase >= FAIL_AT;
  const completed = !failed && phase >= COMPLETE;
  const started = phase >= FIRST_ACTIVATION;
  const progress = clamp01((Math.min(phase, failed ? FAIL_AT : 1) - FIRST_ACTIVATION) / (BATCHES_END - FIRST_ACTIVATION));
  const endPhase = failed ? FAIL_AT : COMPLETE;
  return {
    k,
    round_number: Math.floor(k / n) + 1,
    data_owner_id: INSTITUTIONS[k % n].id,
    compute_peer_id: INSTITUTIONS[(k + 1) % n].id,
    status: failed ? "failed" : completed ? "completed" : "active",
    backbone_version: k > FAILED_TURN ? k - 1 : k,
    batches_done: Math.floor(progress * BATCHES),
    batches_total: BATCHES,
    assigned_at: iso(assignedAt),
    first_activation_at: started ? iso(assignedAt + FIRST_ACTIVATION * TURN_MS) : null,
    completed_at: failed || completed ? iso(assignedAt + endPhase * TURN_MS) : null,
    loss: completed ? lossAt(k) : null,
    accuracy: completed ? accuracyAt(k) : null,
    error: failed ? `Compute peer ${INSTITUTIONS[(k + 1) % n].id} stopped responding during the Backbone computation` : null,
  };
}

function toTurn(a) {
  const seconds = a.completed_at ? (Date.parse(a.completed_at) - Date.parse(a.assigned_at)) / 1000 : null;
  return {
    round: a.round_number,
    owner: a.data_owner_id,
    peer: a.compute_peer_id,
    status: a.status,
    backbone_version: a.backbone_version,
    version_after: a.status === "completed" ? a.backbone_version + 1 : null,
    loss: a.loss,
    accuracy: a.accuracy,
    seconds,
    error: a.error,
    completed_at: a.completed_at,
  };
}

function currentTurn(a) {
  if (!a) return null;
  const doneAll = a.status === "completed";
  const started = a.first_activation_at != null;
  const finishedBatches = a.batches_done >= a.batches_total;
  const states = {
    selected: "done",
    assigned: "done",
    activations: started ? "done" : "active",
    noise: started ? "done" : "pending",
    backbone: finishedBatches || doneAll ? "done" : started ? "active" : "pending",
    checkpoint: doneAll ? "done" : finishedBatches ? "active" : "pending",
    completed: doneAll ? "done" : "pending",
  };
  if (a.status === "failed") {
    const firstOpen = STEP_LABELS.find(([key]) => states[key] === "active" || states[key] === "pending")?.[0];
    for (const [key] of STEP_LABELS) {
      if (states[key] === "active" || states[key] === "pending") states[key] = key === firstOpen ? "failed" : "pending";
    }
  }
  const details = { noise: null, backbone: `${a.batches_done} / ${a.batches_total} batches` };
  return {
    ...toTurn(a),
    batches_done: a.batches_done,
    batches_total: a.batches_total,
    steps: STEP_LABELS.map(([key, label]) => ({ key, label, state: states[key], detail: details[key] ?? null })),
  };
}

function eventsFor(assignments, sessionStart, sessionDone, completedAt) {
  const events = INSTITUTIONS.map((inst, i) => ({
    at: iso(sessionStart - 4000 + i * 600),
    institution: inst.id,
    type: "registration",
    message: `${inst.name} registered as ${inst.id}`,
  }));
  for (const a of assignments) {
    const owner = a.data_owner_id;
    events.push({
      at: a.assigned_at,
      institution: owner,
      type: "assignment",
      message: `Round ${a.round_number}: ${owner} trains, ${a.compute_peer_id} computes the Backbone`,
    });
    if (a.first_activation_at) {
      events.push({ at: a.first_activation_at, institution: owner, type: "training", message: `${owner} sent its first noised activations` });
    }
    if (a.status === "completed") {
      events.push({ at: a.completed_at, institution: a.compute_peer_id, type: "checkpoint", message: `Backbone v${a.backbone_version + 1} saved` });
      events.push({
        at: a.completed_at,
        institution: owner,
        type: "completion",
        message: `${owner} finished its turn: loss ${a.loss.toFixed(3)}, accuracy ${Math.round(a.accuracy * 100)}%`,
      });
    }
    if (a.status === "failed") {
      events.push({ at: a.completed_at, institution: a.compute_peer_id, type: "failure", message: a.error });
    }
  }
  if (sessionDone) {
    const done = assignments.filter((a) => a.status === "completed").length;
    events.push({ at: completedAt, institution: null, type: "completion", message: `Session finished: ${done} of ${TOTAL_TURNS} turns completed` });
  }
  return events.slice(-40).reverse();
}

function institutionsFor(assignments, running, current, now) {
  const failedNow = current?.status === "failed";
  return INSTITUTIONS.map((inst, i) => {
    let role = "idle";
    if (running && current && inst.id === current.data_owner_id) role = "owner";
    else if (running && current && inst.id === current.compute_peer_id) role = "peer";
    const own = assignments.filter((a) => a.data_owner_id === inst.id && a.status === "completed");
    const offline = failedNow && inst.id === current.compute_peer_id;
    const busy = running && current?.status === "active" && role !== "idle";
    return {
      id: inst.id,
      name: inst.name,
      status: offline ? "offline" : busy ? "training" : "online",
      role,
      last_seen_at: iso(offline ? Date.parse(current.completed_at) : now - (busy ? 400 : 1500 + unit(i, 5) * 4000)),
      progress: role === "owner" ? { done: current.batches_done, total: current.batches_total } : null,
      contributions: own.length,
      computations: assignments.filter((a) => a.compute_peer_id === inst.id && a.status === "completed").length,
      last_contribution: own.length ? toTurn(own.at(-1)) : null,
    };
  });
}

/**
 * @param {number} now     current time in ms
 * @param {number} origin  when the simulation was started, in ms
 * @returns {import("./types.js").Snapshot}
 */
export function simulateSnapshot(now, origin) {
  const elapsed = Math.max(0, now - origin) + START_TURN * TURN_MS;
  const cycle = Math.floor(elapsed / CYCLE_MS);
  const t = elapsed % CYCLE_MS;
  const sessionStart = origin - START_TURN * TURN_MS + cycle * CYCLE_MS;
  const turnIndex = Math.min(Math.floor(t / TURN_MS), TOTAL_TURNS);
  const phase = turnIndex < TOTAL_TURNS ? (t % TURN_MS) / TURN_MS : 1;
  const sessionDone = turnIndex >= TOTAL_TURNS;

  const assignments = Array.from({ length: Math.min(turnIndex + 1, TOTAL_TURNS) }, (_, k) =>
    assignment(k, k < turnIndex ? 1 : phase, sessionStart),
  );
  const current = assignments.at(-1) ?? null;
  const completed = assignments.filter((a) => a.status === "completed");
  const lastCheckpoint = completed.at(-1);
  const completedAt = sessionDone ? iso(sessionStart + TOTAL_TURNS * TURN_MS) : null;
  const running = !sessionDone;

  return {
    session: {
      id: sessionId(cycle),
      status: running ? "running" : "completed",
      current_round: current?.round_number ?? 0,
      num_rounds: ROUNDS,
      total_turns: TOTAL_TURNS,
      completed_turns: completed.length,
      current_owner: current?.data_owner_id ?? null,
      current_peer: current?.compute_peer_id ?? null,
      created_at: iso(sessionStart - 5000),
      completed_at: completedAt,
    },
    institutions: institutionsFor(assignments, running, current, now),
    current_turn: currentTurn(current),
    turns: assignments.map(toTurn),
    model: {
      backbone_version: completed.length,
      latest_checkpoint_at: lastCheckpoint ? lastCheckpoint.completed_at : iso(sessionStart - 2000),
      latest_checkpoint_by: lastCheckpoint ? lastCheckpoint.compute_peer_id : "coordinator",
      completed_contributions: completed.length,
      total_turns: TOTAL_TURNS,
      session_status: running ? "running" : "completed",
    },
    noise: NOISE,
    events: eventsFor(assignments, sessionStart, sessionDone, completedAt),
    database_exists: true,
  };
}

export const SIMULATION_TICK_MS = 500;
