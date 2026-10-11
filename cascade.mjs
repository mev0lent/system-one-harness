// Post-hoc cascade on D_hard (PROTOCOL.md 18.10): fingerprints, then Decisions + S1-b head, then a reasoning
// model only for the band where the head is unsure. Reads results/ only. Prints a report.
import fs from "node:fs";
import { generate, generateHard, CLASSES } from "./dataset.mjs";
import { trainLogReg } from "./adapt.mjs";
import { K, argmax, mean } from "./stats.mjs";
import { broadFeat, archived, fingerprint, template } from "./gatelib.mjs";

const load = (f) => JSON.parse(fs.readFileSync(`results/${f}.json`, "utf8"));
const yDev = generate({ perClass: 100, seed: 2024 }).map((e) => CLASSES.indexOf(e.label));
const head = trainLogReg(load("openai_dev").map((r) => broadFeat(r.broad.probs)), yDev, K);
const hard = generateHard({ perClass: 50, seed: 4242 });
const P = load("openai_hard").map((r) => head(broadFeat(r.broad.probs)));
const PRICE_PER_CALL = { gen_medium: 100 / 1e6, gen_sol: 1863 / 1e6 }; // from llm_summary.json, $ per decision
const report = (name, esc, calls) => {
  const an = hard.filter((e) => e.label !== "normal").length, no = hard.length - an;
  const rec = hard.reduce((a, e, i) => a + (e.label !== "normal" && esc[i] ? 1 : 0), 0) / an;
  const fe = hard.reduce((a, e, i) => a + (e.label === "normal" && esc[i] ? 1 : 0), 0) / no;
  const callsN = hard.reduce((a, e, i) => a + (e.label === "normal" && calls[i] ? 1 : 0), 0) / no;
  const callsA = hard.reduce((a, e, i) => a + (e.label !== "normal" && calls[i] ? 1 : 0), 0) / an;
  const p = (x) => (100 * x).toFixed(1);
  // at 1% anomaly prevalence: share forwarded to the expensive tier, and share sent to the reasoning model
  console.log(`${name.padEnd(46)} recall ${p(rec)}  false esc ${p(fe)}  escalated@1% ${p(0.01 * rec + 0.99 * fe)}  reasoning calls@1% ${p(0.01 * callsA + 0.99 * callsN)}`);
  const missed = {};
  hard.forEach((e, i) => { if (e.label !== "normal" && !esc[i]) missed[template(e)] = (missed[template(e)] || 0) + 1; });
  if (Object.keys(missed).length) console.log(`${"".padEnd(48)}missed: ${JSON.stringify(missed)}`);
};

const fp = hard.map(fingerprint);
report("Decisions S1-b (tau 0.3)", P.map((p) => !archived(p, 0.3)), hard.map(() => false));
report("fingerprints OR Decisions S1-b", P.map((p, i) => fp[i] || !archived(p, 0.3)), hard.map(() => false));
for (const g of ["gen_medium", "gen_sol"]) {
  const R = load(`${g}_hard`).map((r) => r.broad.probs);
  report(`${g} alone`, R.map((r) => argmax(r) !== 0), hard.map(() => true));
  for (const hi of [0.6, 0.7, 0.8, 0.9]) {
    // archive only if the head is very sure. Anything archived at 0.3 but below `hi` goes to the reasoning model
    const band = P.map((p, i) => !fp[i] && archived(p, 0.3) && !archived(p, hi));
    const esc = P.map((p, i) => fp[i] || !archived(p, 0.3) || (band[i] && argmax(R[i]) !== 0));
    report(`fp + Decisions + ${g} for P(normal) in [0.3, ${hi})`, esc, band);
  }
}
console.log(`\nreasoning-model price per decision: ${JSON.stringify(PRICE_PER_CALL)}`);

// Is the head less sure about the anomalies it archives than about benign traffic?
const q = (xs) => { const s = [...xs].sort((a, b) => a - b); const at = (f) => s[Math.min(s.length - 1, Math.floor(f * s.length))]; return `min ${s[0].toFixed(2)} median ${at(0.5).toFixed(2)} max ${s[s.length - 1].toFixed(2)} (n ${s.length})`; };
const pn = (sel) => hard.map((e, i) => [e, P[i]]).filter(([e, p], i) => sel(e, p, i)).map(([, p]) => p[0]);
console.log(`\nP(normal) after the head, D_hard`);
console.log(`  benign examples:                  ${q(pn((e) => e.label === "normal"))}`);
console.log(`  anomalies it archives (misses):   ${q(pn((e, p) => e.label !== "normal" && archived(p, 0.3)))}`);
console.log(`  misses that no fingerprint flags: ${q(pn((e, p, i) => e.label !== "normal" && archived(p, 0.3) && !fp[i]))}`);
