/**
 * The project's measured research record, copied from its sources so the
 * observatory works without MLflow running. Update it when the curated runs change.
 *
 *   Curated runs:  mlflow.db, runs tagged curated=true (PathMNIST, 10 rounds, 4 institutions)
 *   Noise sweep:   experiments/results/noise_comparison_clean_eval.csv (feature/data-privacy)
 *   Model shapes:  split_learning/path_cnn.py, MedWeavePathCNN-v1 with GroupNorm, batch 64
 */

export const DATASET = {
  name: "PathMNIST",
  description: "28×28 colon histology, 9 tissue classes",
  imagesPerInstitution: 22499,
  institutions: 4,
  imageBytes: 28 * 28 * 3,
};

/** Validation accuracy by round (mean over seeds) and test metrics at the selected round. */
export const RUNS = [
  {
    id: "centralized",
    label: "Centralized reference",
    short: "Centralized",
    reference: true,
    what: "Same baseline CNN, all images pooled in one place. Baseline recipe, so it is not a tuned upper bound.",
    seeds: 3,
    byRound: [0.839, 0.8516, 0.9125, 0.9238, 0.8826, 0.8786, 0.9127, 0.9162, 0.9416, 0.9229],
    test: { accuracy: 0.838, accuracyStd: 0.0112, auc: 0.9707, macroF1: 0.7856, balanced: 0.7858 },
    hospitalGap: null,
  },
  {
    id: "baseline",
    label: "Split · baseline",
    short: "Baseline",
    what: "CNN split across 4 institutions. BatchNorm, no augmentation, Adam.",
    seeds: 3,
    byRound: [0.7987, 0.8278, 0.8171, 0.8792, 0.8867, 0.8953, 0.8942, 0.9044, 0.9008, 0.9206],
    test: { accuracy: 0.791, accuracyStd: 0.0112, auc: 0.9607, macroF1: 0.7362, balanced: 0.7522 },
    hospitalGap: 13.52,
  },
  {
    id: "augmented",
    label: "Split · + augmentation",
    short: "+ Augmentation",
    what: "Adds flips, 90° rotations and colour jitter 0.2.",
    seeds: 3,
    byRound: [0.7662, 0.829, 0.8502, 0.8729, 0.8758, 0.8997, 0.8948, 0.9108, 0.8863, 0.9032],
    test: { accuracy: 0.8733, accuracyStd: 0.0053, auc: 0.9862, macroF1: 0.8307, balanced: 0.8384 },
    hospitalGap: 3.43,
  },
  {
    id: "medweave",
    label: "MedWeave recipe",
    short: "MedWeave",
    what: "GroupNorm (8 groups), AdamW with weight decay 1e-4, cosine schedule with 5% warm-up.",
    seeds: 2,
    byRound: [0.7058, 0.8117, 0.8495, 0.8723, 0.8699, 0.8921, 0.8995, 0.9066, 0.9096, 0.9114],
    test: { accuracy: 0.8832, accuracyStd: 0.0092, auc: 0.9875, macroF1: 0.8342, balanced: 0.842 },
    hospitalGap: 0.7,
  },
];

export const NOT_MEASURED = [
  { id: "fedavg", label: "FedAvg", reason: "Not run in this project. Its communication cost below is analytical only." },
];

/** Validation accuracy per turn at three activation-noise levels (2 rounds, early placeholder CNN). */
export const NOISE_SWEEP = [
  { sigma: 0, change: 0, byTurn: [0.6267, 0.6854, 0.6949, 0.7519, 0.7508, 0.8001, 0.7854, 0.7125] },
  { sigma: 0.25, change: 26.1, byTurn: [0.6397, 0.6653, 0.6709, 0.7124, 0.7563, 0.7746, 0.788, 0.7443] },
  { sigma: 1.25, change: 122.1, byTurn: [0.3936, 0.465, 0.4819, 0.5183, 0.5516, 0.512, 0.5466, 0.6478] },
];

export const MODEL = {
  name: "MedWeavePathCNN-v1 · GroupNorm",
  params: { head: 960, backbone: 92736, tail: 1161 },
  cutFloatsPerBatch: 64 * 32 * 14 * 14,
  batchSize: 64,
  batchesPerTurn: 351,
};

const FLOAT_BYTES = 4;
const totalParams = () => MODEL.params.head + MODEL.params.backbone + MODEL.params.tail;

/**
 * Bytes moved per training round, from tensor shapes (float32), for what each approach sends.
 * U-shaped split learning crosses the network four times per batch: activations, Backbone
 * features, and the two gradients back. Each turn also hands the Backbone to the next peer.
 */
export function communicationPerRound() {
  const turns = DATASET.institutions;
  const splitPerBatch = 4 * MODEL.cutFloatsPerBatch * FLOAT_BYTES;
  const handoff = 2 * MODEL.params.backbone * FLOAT_BYTES;
  return [
    {
      id: "centralized",
      label: "Centralized",
      moves: "Raw images, once",
      bytes: DATASET.institutions * DATASET.imagesPerInstitution * DATASET.imageBytes,
      note: "Patient images leave every institution.",
      ran: true,
    },
    {
      id: "fedavg",
      label: "FedAvg",
      moves: "Full model weights",
      bytes: turns * 2 * totalParams() * FLOAT_BYTES,
      note: "Analytical. Each institution would download and upload the whole model.",
      ran: false,
    },
    {
      id: "medweave",
      label: "MedWeave split",
      moves: "Noised activations and gradients",
      bytes: turns * (MODEL.batchesPerTurn * splitPerBatch + handoff),
      note: "No institution ever holds the full model, and labels stay with their owner.",
      ran: true,
    },
  ];
}

export function bytesLabel(bytes) {
  const units = ["B", "KB", "MB", "GB", "TB"];
  let value = bytes;
  let i = 0;
  while (value >= 1000 && i < units.length - 1) {
    value /= 1000;
    i += 1;
  }
  return `${value >= 100 ? value.toFixed(0) : value.toFixed(1)} ${units[i]}`;
}

export const SERIES_COLORS = { baseline: "#d4704a", augmented: "#1f9fbf", medweave: "#9a6cf0", centralized: "#8a929c" };
export const NOISE_COLORS = ["#7b63e0", "#a597ff", "#d9d2ff"];
