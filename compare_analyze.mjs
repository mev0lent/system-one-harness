// Cross-model analysis (PROTOCOL.md sections 8, 9, 14): zero-shot decisions, heads S1, S1-b, S2, S3, S4,
// gates and the pre-registered endpoints P1-P3 / H-a..H-c, for every model with results on all three splits.
// Writes results/compare_summary.json.
import fs from "node:fs";
import { generate, generateHard, CLASSES } from "./dataset.mjs";
import { parseResponse } from "./preprocess.mjs";
import { heuristic } from "./heuristic.mjs";
import { trainLogReg, layaFeatures } from "./adapt.mjs";
import { K, argmax, mean, pct, boot, accOn, confusion, macroF1, calibration } from "./stats.mjs";

const load = (f) => (fs.existsSync(`results/${f}.json`) ? JSON.parse(fs.readFileSync(`results/${f}.json`, "utf8")) : null);
// Laya's diff-input runs predate compare.mjs and live under their original names
const FILES = { laya: ["dev", "diff", "hard"], gliner: ["gliner_dev", "gliner_test", "gliner_hard"], jev: ["jev_dev", "jev_test", "jev_hard"], openai: ["openai_dev", "openai_test", "openai_hard"] };
const models = Object.entries(FILES).map(([m, fs_]) => [m, fs_.map(load)]).filter(([, d]) => d.every(Boolean)).map(([m, [dev, test, hard]]) => ({ m, dev, test, hard }));
const ex = { dev: generate({ perClass: 100, seed: 2024 }), test: generate({ perClass: 100, seed: 1337 }), hard: generateHard({ perClass: 50, seed: 4242 }) };
const y = (split) => ex[split].map((e) => CLASSES.indexOf(e.label));
const rowsOf = (split) => ex[split].map((e) => ({ label: e.label }));

// ---- deterministic features z(s), PROTOCOL.md 8.1 ----
const VOLATILE = new Set(["Date", "X-Request-Id", "Content-Length"]);
const mask = (s) => s.replace(/\b[0-9a-f]{6,}\b/gi, "<hex>").replace(/\d+/g, "<n>");
const NON_NORMAL = CLASSES.filter((c) => c !== "normal");
function detFeatures(state) {
  const b = parseResponse(state.baseline_response), c = parseResponse(state.candidate_response);
  const verdict = heuristic(state);
  const hv = NON_NORMAL.map((k) => (verdict === k ? 1 : 0));
  if (!c) return [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, Math.log(1 / (b.body.length + 1)), 0, 0, 1, 1, ...hv];
  const cls = Math.floor(c.status / 100);
  const added = Object.keys(c.headers).filter((k) => !(k in b.headers)).length;
  const removed = Object.keys(b.headers).filter((k) => !(k in c.headers)).length;
  const changed = Object.keys(c.headers).filter((k) => k in b.headers && !VOLATILE.has(k) && mask(c.headers[k]) !== mask(b.headers[k])).length;
  return [
    c.status !== b.status ? 1 : 0,
    0, cls === 1 ? 1 : 0, cls === 2 ? 1 : 0, cls === 3 ? 1 : 0, cls === 4 ? 1 : 0, cls === 5 ? 1 : 0,
    Math.log1p(added), Math.log1p(removed), Math.log1p(changed),
    Math.log((c.body.length + 1) / (b.body.length + 1)),
    mask(c.body) === mask(b.body) ? 1 : 0,
    c.body.startsWith(b.body) && c.body.length > b.body.length ? 1 : 0,
    c.body.length === 0 ? 1 : 0,
    0,
    ...hv,
  ];
}
const Z = Object.fromEntries(Object.entries(ex).map(([s, e]) => [s, e.map((x) => detFeatures(x.state))]));
// z-score with D_dev statistics
const D = Z.dev[0].length;
const mu = Array.from({ length: D }, (_, d) => mean(Z.dev.map((r) => r[d])));
const sd = Array.from({ length: D }, (_, d) => Math.sqrt(mean(Z.dev.map((r) => (r[d] - mu[d]) ** 2))) || 1);
const zs = (r) => r.map((v, d) => (v - mu[d]) / sd[d]);
for (const s of Object.keys(Z)) Z[s] = Z[s].map(zs);

// ---- heads ----
const broadOnly = (r) => layaFeatures(r).slice(0, 6);
function featureSets(split) {
  const sets = {};
  for (const { m, ...d } of models) {
    const rows = d[split];
    sets[`${m}:S1`] = rows.map(layaFeatures);
    sets[`${m}:S1-b`] = rows.map(broadOnly);
    sets[`${m}:S3`] = rows.map((r, i) => [...layaFeatures(r), ...Z[split][i]]);
  }
  sets["S2"] = Z[split];
  if (models.length > 1) {
    const stack = (skip) => Z[split].map((z, i) => [...models.filter((x) => x.m !== skip).flatMap((x) => layaFeatures(x[split][i])), ...z]);
    sets["S4"] = stack(null);
    for (const { m } of models) sets[`S4-loo:${m}`] = stack(m);
  }
  return sets;
}
const F = { dev: featureSets("dev"), test: featureSets("test"), hard: featureSets("hard") };

// 5-fold cross-fitted dev predictions (folds by index mod 5, and since classes are contiguous, folds are stratified)
function crossFit(X, yy) {
  const out = new Array(X.length);
  for (let f = 0; f < 5; f++) {
    const tr = X.map((_, i) => i).filter((i) => i % 5 !== f);
    const head = trainLogReg(tr.map((i) => X[i]), tr.map((i) => yy[i]), K);
    X.forEach((x, i) => { if (i % 5 === f) out[i] = head(x); });
  }
  return out;
}

// ---- gates (PROTOCOL.md 9) ----
const isAnom = (split) => ex[split].map((e) => e.label !== "normal");
function gate(split, escalate) {
  const a = isAnom(split), idx = a.map((_, i) => i);
  const an = idx.filter((i) => a[i]), no = idx.filter((i) => !a[i]);
  return {
    escalated: mean(idx.map((i) => (escalate[i] ? 1 : 0))),
    recall: mean(an.map((i) => (escalate[i] ? 1 : 0))),
    false_escalation: mean(no.map((i) => (escalate[i] ? 1 : 0))),
  };
}
const TAUS = Array.from({ length: 70 }, (_, i) => +(0.3 + i * 0.01).toFixed(2));
const thresholdEsc = (P, tau) => P.map((p) => !(argmax(p) === 0 && p[0] >= tau));
// most aggressive tau with recall >= 0.95 on (cross-fitted) dev predictions. 1.01 means "never archive"
function pickTau(Pdev) {
  for (const tau of TAUS) if (gate("dev", thresholdEsc(Pdev, tau)).recall >= 0.95) return tau;
  return 1.01;
}
// ranking: escalate the top fraction B by r = 1 - P(normal). Ties broken by index
function rankEsc(P, B) {
  const order = P.map((p, i) => [1 - p[0], i]).sort((a, b) => b[0] - a[0] || a[1] - b[1]).map((x) => x[1]);
  const esc = new Array(P.length).fill(false);
  order.slice(0, Math.ceil(B * P.length)).forEach((i) => (esc[i] = true));
  return esc;
}
const recallAtB = (split, P, B) => gate(split, rankEsc(P, B)).recall;
const aurc = (split, P) => mean(Array.from({ length: 101 }, (_, k) => recallAtB(split, P, k / 100)));
// post hoc (P2 saturates at 5/6 prevalence: escalating 50% can reach at most 60% recall):
// share of normal observations spared while keeping >= 95% of anomalies, threshold set on the same split (upper bound)
function sparedAtR95(split, P) {
  const a = isAnom(split), nA = a.filter(Boolean).length, nN = a.length - nA;
  const order = P.map((p, i) => [1 - p[0], i]).sort((x, z) => z[0] - x[0] || x[1] - z[1]).map((x) => x[1]);
  let caught = 0, normalsEscalated = 0;
  for (const i of order) { if (caught >= Math.ceil(0.95 * nA)) break; if (a[i]) caught++; else normalsEscalated++; }
  return 1 - normalsEscalated / nN;
}
// escalated fraction if anomalies made up 1% of traffic (importance weighting, PROTOCOL.md 10.5)
const escAtPrev = (g, prev = 0.01) => prev * g.recall + (1 - prev) * g.false_escalation;

// ---- evaluation ----
const pred = (P) => P.map((p) => CLASSES[argmax(p)]);
function evalProbs(split, P) {
  const rows = rowsOf(split), pr = pred(P), c = calibration(rows, P);
  return { accuracy: boot(rows.length, accOn(rows, pr)), macroF1: macroF1(confusion(rows, pr)), brier: c.brier, ece: c.ece, recall_at_B50: recallAtB(split, P, 0.5), aurc: aurc(split, P), spared_normals_at_R95: sparedAtR95(split, P) };
}
const paired = (split, PA, PB) => { const rows = rowsOf(split), a = pred(PA), b = pred(PB); return boot(rows.length, (idx) => accOn(rows, a)(idx) - accOn(rows, b)(idx)); };
const norm = (p) => { const s = p.reduce((a, b) => a + b, 0) || 1; return p.map((v) => v / s); };

const summary = { models: models.map((x) => x.m), zero_shot: {}, heads: {}, endpoints: {}, gates: {}, cost: {} };
const P = { zs: {}, heads: {} };

for (const { m, dev, test, hard } of models) {
  const d = { dev, test, hard };
  P.zs[m] = Object.fromEntries(["dev", "test", "hard"].map((s) => [s, d[s].map((r) => r.broad.probs)]));
  summary.zero_shot[m] = {};
  for (const s of ["test", "hard"]) {
    summary.zero_shot[m][s] = {
      broad: evalProbs(s, P.zs[m][s]),
      narrow_accuracy: boot(d[s].length, accOn(rowsOf(s), pred(d[s].map((r) => norm(r.narrow.p))))),
      refusals: { broad: d[s].filter((r) => r.refused?.broad).length, narrow: d[s].filter((r) => r.refused?.narrow?.length).length },
    };
  }
  // cost and latency
  const ms = [...dev, ...test, ...hard].map((r) => r.broad.ms).filter((v) => typeof v === "number");
  const tok = [...dev, ...test, ...hard].map((r) => r.broad.tokens).filter((v) => typeof v === "number");
  summary.cost[m] = { broad_ms_p50: ms.length ? pct(ms, 0.5) : null, broad_ms_p95: ms.length ? pct(ms, 0.95) : null, broad_input_tokens_mean: tok.length ? mean(tok) : null };
}

for (const name of Object.keys(F.dev)) {
  const head = trainLogReg(F.dev[name], y("dev"), K);
  const Pd = crossFit(F.dev[name], y("dev"));
  P.heads[name] = { dev_cv: Pd, test: F.test[name].map(head), hard: F.hard[name].map(head) };
  const tau = pickTau(Pd);
  summary.heads[name] = {
    dev_cv_accuracy: mean(pred(Pd).map((p, i) => (p === ex.dev[i].label ? 1 : 0))),
    test: evalProbs("test", P.heads[name].test),
    hard: evalProbs("hard", P.heads[name].hard),
    dev_selected_tau: tau,
    gate_at_dev_tau: { test: gate("test", thresholdEsc(P.heads[name].test, tau)), hard: gate("hard", thresholdEsc(P.heads[name].hard, tau)) },
  };
  for (const s_ of ["test", "hard"]) summary.heads[name].gate_at_dev_tau[s_].escalated_at_1pct_prevalence = escAtPrev(summary.heads[name].gate_at_dev_tau[s_]);
}

// zero-shot gates: dev-selected tau, plus the H-c check over the whole tau grid on D_hard
for (const { m } of models) {
  const tau = pickTau(P.zs[m].dev);
  const hc = TAUS.map((t) => gate("hard", thresholdEsc(P.zs[m].hard, t))).filter((g) => g.recall >= 0.95);
  summary.gates[m] = {
    dev_selected_tau: tau,
    test: gate("test", thresholdEsc(P.zs[m].test, tau)),
    hard: gate("hard", thresholdEsc(P.zs[m].hard, tau)),
    hard_min_escalation_at_recall_95: hc.length ? Math.min(...hc.map((g) => g.escalated)) : null,
  };
  for (const s_ of ["test", "hard"]) summary.gates[m][s_].escalated_at_1pct_prevalence = escAtPrev(summary.gates[m][s_]);
}
summary.gates.heuristic = Object.fromEntries(["test", "hard"].map((s) => { const g = gate(s, ex[s].map((e) => heuristic(e.state) !== "normal")); return [s, { ...g, escalated_at_1pct_prevalence: escAtPrev(g) }]; }));

// ---- pre-registered endpoints (PROTOCOL.md 14) ----
for (const { m } of models) {
  const s3 = P.heads[`${m}:S3`].hard;
  summary.endpoints[m] = {
    P1_S3_minus_zero_shot_hard: paired("hard", s3, P.zs[m].hard),
    P2_recall_at_B50_S3_hard: recallAtB("hard", s3, 0.5),
    P2_with_exploration: Object.fromEntries([0.02, 0.05].map((e) => { const r = recallAtB("hard", s3, 0.5); return [`eps_${e}`, { recall: r + e * (1 - r), escalated: 0.5 + e * 0.5 }]; })),
    Hb_S3_minus_S2_hard: paired("hard", s3, P.heads["S2"].hard),
    Hc_zero_shot_min_escalation_at_recall_95: summary.gates[m].hard_min_escalation_at_recall_95,
  };
  // exploratory, not pre-registered: the small broad-only head against zero-shot
  summary.endpoints[m].exploratory_S1b_minus_zero_shot = { test: paired("test", P.heads[`${m}:S1-b`].test, P.zs[m].test), hard: paired("hard", P.heads[`${m}:S1-b`].hard, P.zs[m].hard) };
}
if (P.heads["S4"]) {
  const best = models.map(({ m }) => m).sort((a, b) => summary.heads[`${b}:S3`].dev_cv_accuracy - summary.heads[`${a}:S3`].dev_cv_accuracy)[0];
  summary.endpoints.P3 = { best_single_by_dev_cv: best, S4_minus_best_S3_hard: paired("hard", P.heads["S4"].hard, P.heads[`${best}:S3`].hard) };
}

fs.writeFileSync("results/compare_summary.json", JSON.stringify(summary, null, 2));

// ---- compact console report ----
const f = (o) => (o && o.est !== undefined ? `${(o.est * 100).toFixed(1)} [${(o.lo * 100).toFixed(1)}, ${(o.hi * 100).toFixed(1)}]` : o === null || o === undefined ? "-" : typeof o === "number" ? (o * 100).toFixed(1) : JSON.stringify(o));
console.log("models:", summary.models.join(", "));
for (const m of summary.models) {
  const z = summary.zero_shot[m];
  console.log(`\n${m}  zero-shot broad  test ${f(z.test.broad.accuracy)}  hard ${f(z.hard.broad.accuracy)} | narrow test ${f(z.test.narrow_accuracy)} hard ${f(z.hard.narrow_accuracy)} | refusals ${JSON.stringify(z.hard.refusals)}`);
  console.log(`  zero-shot hard: spared@R95 ${f(z.hard.broad.spared_normals_at_R95)} | dev-tau gate hard ${JSON.stringify(summary.gates[m].hard, (k, v) => (typeof v === "number" ? +v.toFixed(3) : v))}`);
  for (const h of ["S1-b", "S1", "S3"]) { const r = summary.heads[`${m}:${h}`]; console.log(`  ${h.padEnd(5)} test ${f(r.test.accuracy)}  hard ${f(r.hard.accuracy)}  hard spared@R95 ${f(r.hard.spared_normals_at_R95)}  hard ECE ${r.hard.ece.toFixed(3)}  dev-tau gate hard ${JSON.stringify(r.gate_at_dev_tau.hard, (k, v) => (typeof v === "number" ? +v.toFixed(3) : v))}`); }
  const e = summary.endpoints[m];
  console.log(`  P1 ${f(e.P1_S3_minus_zero_shot_hard)} | P2 ${f(e.P2_recall_at_B50_S3_hard)} | H-b ${f(e.Hb_S3_minus_S2_hard)} | H-c min esc@R95 ${f(e.Hc_zero_shot_min_escalation_at_recall_95)}`);
  console.log(`  S1-b - zero-shot: test ${f(e.exploratory_S1b_minus_zero_shot.test)} hard ${f(e.exploratory_S1b_minus_zero_shot.hard)}`);
  console.log(`  cost: ${JSON.stringify(summary.cost[m])}`);
}
const s2 = summary.heads["S2"]; console.log(`\nS2 (deterministic only)  test ${f(s2.test.accuracy)}  hard ${f(s2.hard.accuracy)}  hard R@B50 ${f(s2.hard.recall_at_B50)}`);
if (summary.heads["S4"]) { const s4 = summary.heads["S4"]; console.log(`S4 (stack)  test ${f(s4.test.accuracy)}  hard ${f(s4.hard.accuracy)}  hard R@B50 ${f(s4.hard.recall_at_B50)} | P3 ${JSON.stringify(summary.endpoints.P3.best_single_by_dev_cv)} ${f(summary.endpoints.P3.S4_minus_best_S3_hard)}`);
  for (const { m } of models) console.log(`  S4 without ${m}: hard ${f(summary.heads[`S4-loo:${m}`].hard.accuracy)}`); }
console.log(`heuristic gate: test ${JSON.stringify(summary.gates.heuristic.test)} hard ${JSON.stringify(summary.gates.heuristic.hard)}`);
