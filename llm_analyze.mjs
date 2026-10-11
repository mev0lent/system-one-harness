// Generative LLM baseline analysis (PROTOCOL.md 18.9). Compares the generative settings with the decision models
// on identical splits, questions and gate rules. Writes results/llm_summary.json.
import fs from "node:fs";
import { generate, generateHard, CLASSES } from "./dataset.mjs";
import { trainLogReg } from "./adapt.mjs";
import { K, argmax, mean, pct, boot, accOn, confusion, macroF1, calibration } from "./stats.mjs";
import { broadFeat, archived, gateStats, escAtPrev, pickTau, crossFit, template, fingerprint } from "./gatelib.mjs";

const load = (f) => (fs.existsSync(`results/${f}.json`) ? JSON.parse(fs.readFileSync(`results/${f}.json`, "utf8")) : null);
const ex = { dev: generate({ perClass: 100, seed: 2024 }), test: generate({ perClass: 100, seed: 1337 }), hard: generateHard({ perClass: 50, seed: 4242 }) };
const rowsOf = (s) => ex[s].map((e) => ({ label: e.label }));
const yDev = ex.dev.map((e) => CLASSES.indexOf(e.label));
const pred = (P) => P.map((p) => CLASSES[argmax(p)]);

// prices per million tokens, 2026-10-11 (PROTOCOL 18.5, 18.9). Decisions/Jev bill input only
const PRICE = { openai: [0.10, 0], jev: [0.042, 0], gen_none: [0.10, 0.50], gen_medium: [0.10, 0.50], gen_sol: [2.00, 10.00] };
const FILES = {
  jev: ["jev_dev", "jev_test", "jev_hard"],
  openai: ["openai_dev", "openai_test", "openai_hard"],
  gen_none: ["gen_none_dev", "gen_none_test", "gen_none_hard"],
  gen_medium: ["gen_medium_dev", "gen_medium_test", "gen_medium_hard"],
  gen_sol: ["gen_sol_dev", "gen_sol_test", "gen_sol_hard"],
};
const models = Object.entries(FILES).map(([m, f]) => [m, f.map(load)]).filter(([, d]) => d.every(Boolean));

const out = { models: models.map(([m]) => m), per_model: {}, paired_vs_decisions: {} };
const P = {};
for (const [m, [dev, test, hard]] of models) {
  const d = { dev, test, hard };
  P[m] = Object.fromEntries(Object.entries(d).map(([s, rows]) => [s, rows.map((r) => r.broad.probs)]));
  const head = trainLogReg(dev.map((r) => broadFeat(r.broad.probs)), yDev, K);
  const Ph = { test: P[m].test.map((p) => head(broadFeat(p))), hard: P[m].hard.map((p) => head(broadFeat(p))) };
  const PdevHead = crossFit(dev.map((r) => broadFeat(r.broad.probs)), yDev);
  const tauZs = pickTau(ex.dev, P[m].dev), tauHead = pickTau(ex.dev, PdevHead);
  const all = [...dev, ...test, ...hard];
  const ms = all.map((r) => r.broad.ms).filter((v) => typeof v === "number");
  const tin = mean(all.map((r) => r.broad.tokens ?? 0)), tout = mean(all.map((r) => r.broad.output_tokens ?? 0));
  const res = { refusals: Object.fromEntries(Object.entries(d).map(([s, rows]) => [s, rows.filter((r) => r.refused?.broad).length])) };
  for (const s of ["test", "hard"]) {
    const rows = rowsOf(s), c = calibration(rows, P[m][s]), ch = calibration(rows, Ph[s]);
    const escZs = P[m][s].map((p) => !archived(p, tauZs)), escHead = Ph[s].map((p) => !archived(p, tauHead));
    const gz = gateStats(ex[s], escZs), gh = gateStats(ex[s], escHead);
    const gfp = gateStats(ex[s], escHead.map((e, i) => e || fingerprint(ex[s][i])));
    res[s] = {
      zero_shot: { accuracy: boot(rows.length, accOn(rows, pred(P[m][s]))), macroF1: macroF1(confusion(rows, pred(P[m][s]))), ece: c.ece, brier: c.brier, gate: { tau: tauZs, ...gz, at_1pct: escAtPrev(gz) } },
      head_S1b: { accuracy: boot(rows.length, accOn(rows, pred(Ph[s]))), ece: ch.ece, brier: ch.brier, gate: { tau: tauHead, ...gh, at_1pct: escAtPrev(gh) }, gate_with_fingerprints: { ...gfp, at_1pct: escAtPrev(gfp) } },
    };
    if (s === "hard") {
      const t = {};
      ex.hard.forEach((e, i) => { const k = template(e); t[k] ??= { n: 0, zero_shot: 0, head: 0 }; t[k].n++; t[k].zero_shot += escZs[i]; t[k].head += escHead[i]; });
      res.hard.by_template = t;
    }
  }
  res.latency_ms = { p50: pct(ms, 0.5), p95: pct(ms, 0.95) };
  res.tokens_mean = { input: tin, output: tout, reasoning: mean(all.map((r) => r.broad.reasoning_tokens ?? 0)) };
  res.cost_per_million_decisions = tin * PRICE[m][0] + tout * PRICE[m][1];
  out.per_model[m] = res;
}
// L-P1 / L-P2: paired accuracy difference against the Decisions API, zero-shot and S1-b
if (P.openai) for (const [m] of models) {
  if (m === "openai") continue;
  out.paired_vs_decisions[m] = Object.fromEntries(["test", "hard"].map((s) => {
    const rows = rowsOf(s), a = pred(P[m][s]), b = pred(P.openai[s]);
    return [s, boot(rows.length, (idx) => accOn(rows, a)(idx) - accOn(rows, b)(idx))];
  }));
}
fs.writeFileSync("results/llm_summary.json", JSON.stringify(out, null, 2));

const f = (o) => (o && o.est !== undefined ? `${(o.est * 100).toFixed(1)} [${(o.lo * 100).toFixed(1)}, ${(o.hi * 100).toFixed(1)}]` : o == null ? "-" : (o * 100).toFixed(1));
for (const [m, r] of Object.entries(out.per_model)) {
  console.log(`\n${m}  refusals ${JSON.stringify(r.refusals)}  latency p50/p95 ${r.latency_ms.p50.toFixed(0)}/${r.latency_ms.p95.toFixed(0)} ms  tokens in/out/reasoning ${r.tokens_mean.input.toFixed(0)}/${r.tokens_mean.output.toFixed(0)}/${r.tokens_mean.reasoning.toFixed(0)}  $/M decisions ${r.cost_per_million_decisions.toFixed(0)}`);
  for (const s of ["test", "hard"]) {
    const z = r[s].zero_shot, h = r[s].head_S1b;
    console.log(`  ${s}: zs acc ${f(z.accuracy)} F1 ${z.macroF1.toFixed(3)} ECE ${z.ece.toFixed(3)} Brier ${z.brier.toFixed(3)} | gate tau ${z.gate.tau} esc ${f(z.gate.escalated)} rec ${f(z.gate.recall)} FE ${f(z.gate.false_escalation)} @1% ${f(z.gate.at_1pct)}`);
    console.log(`  ${s}: S1-b acc ${f(h.accuracy)} ECE ${h.ece.toFixed(3)} | gate tau ${h.gate.tau} esc ${f(h.gate.escalated)} rec ${f(h.gate.recall)} FE ${f(h.gate.false_escalation)} @1% ${f(h.gate.at_1pct)} | +fp rec ${f(h.gate_with_fingerprints.recall)} FE ${f(h.gate_with_fingerprints.false_escalation)}`);
  }
  console.log(`  hard by template (escalated zs/head of n): ` + Object.entries(r.hard.by_template).filter(([k]) => /hostname|second response|two HTML|truncated|normal/.test(k)).map(([k, v]) => `${k} ${v.zero_shot}/${v.head}/${v.n}`).join("; "));
}
for (const [m, d] of Object.entries(out.paired_vs_decisions)) console.log(`paired ${m} - Decisions: test ${f(d.test)} hard ${f(d.hard)}`);
