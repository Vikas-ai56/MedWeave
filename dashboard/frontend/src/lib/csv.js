const COLUMNS = [
  ["turn", (_, i) => i + 1],
  ["round", (t) => t.round],
  ["data_owner", (t) => t.owner],
  ["compute_peer", (t) => t.peer],
  ["status", (t) => t.status],
  ["loss", (t) => t.loss],
  ["accuracy", (t) => t.accuracy],
  ["backbone_version", (t) => t.backbone_version],
  ["version_after", (t) => t.version_after],
  ["seconds", (t) => t.seconds],
  ["error", (t) => t.error],
];

const cell = (value) => {
  if (value == null) return "";
  const text = String(value);
  return /[",\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
};

/** The turn history exactly as the API returned it, one row per turn. */
export function turnsToCsv(turns) {
  const header = COLUMNS.map(([name]) => name).join(",");
  const rows = turns.map((turn, index) => COLUMNS.map(([, get]) => cell(get(turn, index))).join(","));
  return [header, ...rows].join("\n");
}

export function downloadText(filename, text, type = "text/csv") {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const link = Object.assign(document.createElement("a"), { href: url, download: filename });
  link.click();
  URL.revokeObjectURL(url);
}
