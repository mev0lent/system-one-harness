// Shared metrics: bootstrap intervals, confusion matrix, macro-F1, calibration (ECE, Brier, risk-coverage).
import { CLASSES, makeRng } from "./dataset.mjs";

export const K = CLASSES.length;
export const argmax = (a) => a.reduce((bi, v, i) => (v > a[bi] ? i : bi), 0);
export const mean = (a) => a.reduce((s, v) => s + v, 0) / (a.length || 1);
export const pct = (a, q) => { const s = [...a].sort((x, y) => x - y); return s[Math.min(s.length - 1, Math.floor(q * s.length))]; };
export const f2 = (x) => (x * 100).toFixed(1) + "%";

// ---- bootstrap: 95% percentile intervals over resampled examples (seeded, so reruns match) ----
const B_REPS = 2000;
export function boot(n, stat, seed = 42) {
  const g = makeRng(seed);
  const all = Array.from({ length: n }, (_, i) => i);
  const vals = [];
  for (let b = 0; b < B_REPS; b++) vals.push(stat(all.map(() => Math.floor(g.r() * n))));
  vals.sort((x, y) => x - y);
  return { est: stat(all), lo: vals[Math.floor(0.025 * B_REPS)], hi: vals[Math.floor(0.975 * B_REPS)] };
}
export const accOn = (rows, pred) => (idx) => mean(idx.map((i) => (pred[i] === rows[i].label ? 1 : 0)));
export const f1On = (rows, pred) => (idx) => macroF1(confusion(idx.map((i) => rows[i]), idx.map((i) => pred[i])));

// ---- metrics ----
export function confusion(rows, pred) {
  const m = Array.from({ length: K }, () => Array(K).fill(0));
  rows.forEach((r, i) => m[CLASSES.indexOf(r.label)][CLASSES.indexOf(pred[i])]++);
  return m;
}
export function macroF1(m) {
  const fs_ = [];
  for (let k = 0; k < K; k++) {
    const tp = m[k][k], fp = m.reduce((s, r) => s + r[k], 0) - tp, fn = m[k].reduce((s, v) => s + v, 0) - tp;
    const p = tp / Math.max(1, tp + fp), r = tp / Math.max(1, tp + fn);
    fs_.push(p + r === 0 ? 0 : (2 * p * r) / (p + r));
  }
  return mean(fs_);
}
export function calibration(rows, probs) {
  // probs: array of K-vectors (normalised)
  const conf = probs.map((p) => Math.max(...p));
  const pred = probs.map(argmax);
  const hit = rows.map((r, i) => (CLASSES[pred[i]] === r.label ? 1 : 0));
  let ece = 0;
  for (let b = 0; b < 10; b++) {
    const idx = conf.map((c, i) => i).filter((i) => conf[i] > b / 10 && conf[i] <= (b + 1) / 10);
    if (idx.length) ece += (idx.length / rows.length) * Math.abs(mean(idx.map((i) => hit[i])) - mean(idx.map((i) => conf[i])));
  }
  let brier = 0;
  rows.forEach((r, i) => { for (let k = 0; k < K; k++) brier += (probs[i][k] - (CLASSES[k] === r.label ? 1 : 0)) ** 2; });
  const rc = [0.5, 0.7, 0.8, 0.9, 0.95].map((tau) => {
    const idx = conf.map((c, i) => i).filter((i) => conf[i] >= tau);
    return { tau, coverage: idx.length / rows.length, risk: idx.length ? 1 - mean(idx.map((i) => hit[i])) : null };
  });
  return { ece, brier: brier / rows.length, riskCoverage: rc };
}
