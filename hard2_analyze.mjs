// Frozen-setting analysis for PROTOCOL.md §18.11. Reads stored results. No API calls.
import fs from "node:fs";
import { CLASSES, generate, generateHard2 } from "./dataset.mjs";
import { trainLogReg } from "./adapt.mjs";
import { K, argmax } from "./stats.mjs";
import { archived, broadFeat, fingerprint } from "./gatelib.mjs";
import { heuristic } from "./heuristic.mjs";

const load = (name) => JSON.parse(fs.readFileSync(`results/${name}.json`, "utf8"));
const examples = generateHard2();
const loadHard2 = (name) => {
  const rows = load(`hard2_${name}`);
  if (rows.length !== examples.length || rows.some((r, i) => r.id !== examples[i].id || r.label !== examples[i].label || r.template !== examples[i].template)) {
    throw new Error(`hard2_${name} does not align with the frozen generator`);
  }
  return rows;
};
const decisions = loadHard2("openai");
const jev = loadHard2("jev");
const luna = loadHard2("medium");
const dev = generate({ perClass: 100, seed: 2024 });
const devRows = load("openai_dev");
if (devRows.length !== dev.length || devRows.some((r, i) => r.id !== dev[i].id)) throw new Error("D_dev rows do not align");
const head = trainLogReg(devRows.map((r) => broadFeat(r.broad.probs)), dev.map((e) => CLASSES.indexOf(e.label)), K);
const P = decisions.map((r) => head(broadFeat(r.broad.probs)));
const fp = examples.map(fingerprint);

// Exact equal-tailed Clopper–Pearson interval, obtained by inverting binomial tails.
const logFact = [0];
for (let i = 1; i <= examples.length; i++) logFact[i] = logFact[i - 1] + Math.log(i);
const logChoose = (n, k) => logFact[n] - logFact[k] - logFact[n - k];
function binomTail(n, p, from, to) {
  if (p === 0) return from <= 0 && to >= 0 ? 1 : 0;
  if (p === 1) return from <= n && to >= n ? 1 : 0;
  const terms = [];
  for (let k = from; k <= to; k++) terms.push(logChoose(n, k) + k * Math.log(p) + (n - k) * Math.log1p(-p));
  const m = Math.max(...terms);
  return Math.exp(m) * terms.reduce((a, x) => a + Math.exp(x - m), 0);
}
function cp(k, n, confidence = 0.95) {
  const alpha = 1 - confidence;
  const invert = (target, tail, increasing) => {
    let lo = 0, hi = 1;
    for (let i = 0; i < 60; i++) {
      const mid = (lo + hi) / 2;
      const value = tail(mid);
      if ((value < target) === increasing) lo = mid;
      else hi = mid;
    }
    return (lo + hi) / 2;
  };
  return {
    estimate: k / n,
    lo: k === 0 ? 0 : invert(alpha / 2, (p) => binomTail(n, p, k, n), true),
    hi: k === n ? 1 : invert(alpha / 2, (p) => binomTail(n, p, 0, k), false),
    successes: k, total: n,
  };
}
const anomaly = examples.map((e, i) => i).filter((i) => examples[i].label !== "normal");
const normal = examples.map((e, i) => i).filter((i) => examples[i].label === "normal");
function report(escalate, calls = examples.map(() => false)) {
  const recall = cp(anomaly.filter((i) => escalate[i]).length, anomaly.length);
  const falseEscalation = cp(normal.filter((i) => escalate[i]).length, normal.length);
  const callAnomaly = cp(anomaly.filter((i) => calls[i]).length, anomaly.length, 0.975);
  const callNormal = cp(normal.filter((i) => calls[i]).length, normal.length, 0.975);
  const weightedCalls = {
    estimate: 0.01 * callAnomaly.estimate + 0.99 * callNormal.estimate,
    lo: calls.every((x) => !x) ? 0 : calls.every(Boolean) ? 1 : 0.01 * callAnomaly.lo + 0.99 * callNormal.lo,
    hi: calls.every((x) => !x) ? 0 : calls.every(Boolean) ? 1 : 0.01 * callAnomaly.hi + 0.99 * callNormal.hi,
    call_rate_anomaly: callAnomaly,
    call_rate_normal: callNormal,
  };
  return { recall, false_escalation: falseEscalation, reasoning_calls_at_1pct: weightedCalls,
    escalated_at_1pct: 0.01 * recall.estimate + 0.99 * falseEscalation.estimate };
}

const heuristicEsc = examples.map((e) => heuristic(e.state) !== "normal");
const decisionsZero = decisions.map((r) => !archived(r.broad.probs, 0.3));
const decisionsHead = P.map((p) => !archived(p, 0.3));
const fpHead = P.map((p, i) => fp[i] || !archived(p, 0.3));
const lunaEsc = luna.map((r) => argmax(r.broad.probs) !== 0);
const jevZero = jev.map((r) => !archived(r.broad.probs, 0.3));
const reasoningCalls = P.map((p, i) => !fp[i] && archived(p, 0.3) && !archived(p, 0.6));
const cascadeEsc = P.map((p, i) => fpHead[i] || (reasoningCalls[i] && lunaEsc[i]));
const policies = {
  cascade: report(cascadeEsc, reasoningCalls),
  heuristic: report(heuristicEsc),
  decisions_zero_shot: report(decisionsZero),
  decisions_s1b: report(decisionsHead),
  fingerprints_plus_decisions_s1b: report(fpHead),
  luna_reasoning_alone: report(lunaEsc, examples.map(() => true)),
  jev_zero_shot: report(jevZero),
};
const perTemplate = Object.fromEntries([...new Set(examples.map((e) => e.template))].map((name) => {
  const indices = examples.map((e, i) => i).filter((i) => eTemplate(i) === name);
  const count = indices.filter((i) => cascadeEsc[i]).length;
  return [name, { label: examples[indices[0]].label, ...cp(count, indices.length) }];
}));
function eTemplate(i) { return examples[i].template; }
const misses = examples.filter((e, i) => e.label !== "normal" && !cascadeEsc[i]).map((e) => ({ id: e.id, template: e.template }));
const refusals = Object.fromEntries([["openai", decisions], ["jev", jev], ["medium", luna]].map(([name, rows]) => [name, rows.filter((r) => r.refused.broad).length]));
const targets = policies.cascade;
const verdict = targets.recall.estimate >= 0.95 && targets.false_escalation.estimate <= 0.1 && targets.reasoning_calls_at_1pct.estimate <= 0.1
  ? "targets met" : targets.recall.estimate >= 0.9 && targets.false_escalation.estimate <= 0.2 ? "targets partly met" : "targets missed";
const summary = { protocol: "18.11", status: "frozen-setting exploratory; template author saw first-tier results", n: examples.length,
  refusals, verdict, policies, per_template: perTemplate, misses };
fs.writeFileSync("results/hard2_summary.json", JSON.stringify(summary, null, 2) + "\n");
const fmt = (x) => `${(100 * x.estimate).toFixed(1)} [${(100 * x.lo).toFixed(1)}, ${(100 * x.hi).toFixed(1)}]`;
console.log(`D_hard2: ${examples.length} examples; refusals ${JSON.stringify(refusals)}; ${verdict}`);
for (const [name, p] of Object.entries(policies)) console.log(`${name.padEnd(35)} recall ${fmt(p.recall)}  false esc ${fmt(p.false_escalation)}  reasoning@1% ${fmt(p.reasoning_calls_at_1pct)}`);
console.log("Cascade recall by template:");
for (const [name, p] of Object.entries(perTemplate)) if (p.label !== "normal") console.log(`  ${name.padEnd(35)} ${p.successes}/${p.total}  ${fmt(p)}`);
console.log(`Misses: ${JSON.stringify(misses)}`);
