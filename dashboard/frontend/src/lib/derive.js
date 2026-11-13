/** Display values derived from a snapshot. Pure functions; no new API fields. */

const NUMBER_WORDS = ["No", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten", "Eleven", "Twelve"];

export const numberWord = (n) => NUMBER_WORDS[n] ?? String(n);

export const pad2 = (n) => (n == null ? "—" : String(n).padStart(2, "0"));

export const percent1 = (value) => (value == null ? "—" : `${(value * 100).toFixed(1)}%`);

/** Charts plot finished turns only; a turn still in progress has no value yet. */
export const finishedTurns = (turns) => turns.filter((t) => t.status === "completed" || t.status === "failed");

export const completedTurns = (turns) => turns.filter((t) => t.status === "completed");

export const connectedCount = (institutions) => institutions.filter((i) => i.status !== "offline").length;

export const turnsOwnedBy = (turns, id) => turns.filter((t) => t.owner === id);

export const institutionName = (institutions, id) => institutions.find((i) => i.id === id)?.name ?? id ?? "—";

export const isLive = (session) => session?.status === "running";

/**
 * The accuracy shown on The Weave's readout.
 *
 * Every value in `turns` is a *training* accuracy measured on one data owner's batches,
 * so any single number here is a summary choice, not a held-out test score.
 *
 * @param {import("./types.js").Turn[]} turns  all turns, oldest first
 * @param {number} institutionCount            participating institutions (turns per round)
 * @returns {{value: number, basis: string} | null}  null when nothing has completed yet
 */
export function headlineAccuracy(turns, institutionCount) {
  // TODO(you): choose how the headline accuracy summarises recent turns.
  const latest = completedTurns(turns).at(-1);
  return latest ? { value: latest.accuracy, basis: "latest turn" } : null;
}

/** "Four institutions are learning together." adapted to what the snapshot says. */
export function statusSentence(session, institutions) {
  const connected = connectedCount(institutions);
  if (!session) return "Waiting for the first session to begin.";
  if (session.status === "running") {
    return `${numberWord(connected)} ${connected === 1 ? "institution is" : "institutions are"} learning together.`;
  }
  if (session.status === "completed") return `${numberWord(institutions.length)} institutions finished training together.`;
  if (session.status === "failed") return "The session stopped before it finished.";
  return "The session is preparing to start.";
}

export function sessionProgress(session) {
  if (!session?.total_turns) return 0;
  return Math.min(1, session.completed_turns / session.total_turns);
}
