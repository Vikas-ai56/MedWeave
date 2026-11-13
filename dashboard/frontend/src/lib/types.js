/**
 * The /api/dashboard contract (dashboard/backend/snapshot.py). The frontend derives
 * display values from these fields and never requires new ones.
 *
 * @typedef {"online" | "offline" | "training"} InstitutionStatus
 * @typedef {"owner" | "peer" | "idle"} Role
 * @typedef {"pending" | "active" | "completed" | "failed"} TurnStatus
 * @typedef {"done" | "active" | "pending" | "skipped" | "failed"} StepState
 *
 * @typedef {object} Turn
 * @property {number} round
 * @property {string} owner
 * @property {string} peer
 * @property {TurnStatus} status
 * @property {number} backbone_version
 * @property {number | null} version_after
 * @property {number | null} loss
 * @property {number | null} accuracy
 * @property {number | null} seconds
 * @property {string | null} error
 * @property {string | null} completed_at
 *
 * @typedef {Turn & {
 *   batches_done: number,
 *   batches_total: number,
 *   steps: {key: string, label: string, state: StepState, detail: string | null}[],
 * }} CurrentTurn
 *
 * @typedef {object} Institution
 * @property {string} id
 * @property {string} name
 * @property {InstitutionStatus} status
 * @property {Role} role
 * @property {string} last_seen_at
 * @property {{done: number, total: number} | null} progress
 * @property {number} contributions
 * @property {number} computations
 * @property {Turn | null} last_contribution
 *
 * @typedef {object} Session
 * @property {string} id
 * @property {"pending" | "running" | "completed" | "failed"} status
 * @property {number} current_round
 * @property {number} num_rounds
 * @property {number} total_turns
 * @property {number} completed_turns
 * @property {string | null} current_owner
 * @property {string | null} current_peer
 * @property {string} created_at
 * @property {string | null} completed_at
 *
 * @typedef {object} Model
 * @property {number | null} backbone_version
 * @property {string | null} latest_checkpoint_at
 * @property {string | null} latest_checkpoint_by
 * @property {number} completed_contributions
 * @property {number} total_turns
 * @property {string | null} session_status
 *
 * @typedef {{at: string, institution: string | null, type: string, message: string}} TrainingEvent
 *
 * @typedef {object} Snapshot
 * @property {Session | null} session
 * @property {Institution[]} institutions
 * @property {CurrentTurn | null} current_turn
 * @property {Turn[]} turns
 * @property {Model | null} model
 * @property {{mechanism: string, noise_scale: number} | null} noise
 * @property {TrainingEvent[]} events
 * @property {boolean} database_exists
 */

export {};
