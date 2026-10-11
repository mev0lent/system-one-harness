// Post-hoc gate analysis on D_hard (PROTOCOL.md 18.7): which templates each gate escalates, and what two
// structural fingerprints add in front of the best gate. Reads results/ only. Prints a report.
import fs from "node:fs";
import { generate, generateHard, CLASSES } from "./dataset.mjs";
import { parseResponse } from "./preprocess.mjs";
import { heuristic } from "./heuristic.mjs";
import { trainLogReg, layaFeatures } from "./adapt.mjs";
import { K, argmax } from "./stats.mjs";

const load = (f) => JSON.parse(fs.readFileSync(`results/${f}.json`, "utf8"));
const dev = generate({ perClass: 100, seed: 2024 }), test = generate({ perClass: 100, seed: 1337 }), hard = generateHard({ perClass: 50, seed: 4242 });
const yDev = dev.map((e) => CLASSES.indexOf(e.label));
const TAU = 0.3; // the dev-selected threshold for every gate below (compare_summary.json)
const archived = (p) => argmax(p) === 0 && p[0] >= TAU;
const s1b = (m, split) => {
  const head = trainLogReg(load(`${m}_dev`).map((r) => layaFeatures(r).slice(0, 6)), yDev, K);
  return load(`${m}_${split}`).map((r) => head(layaFeatures(r).slice(0, 6)));
};

const template = (e) => {
  const c = e.state.candidate_response;
  switch (e.label) {
    case "normal": return /X-Cache/.test(c) ? "normal: cache headers" : /Set-Cookie/.test(c) ? "normal: new session cookie" : "normal: Vary + ETag";
    case "disclosure": return /X-Backend-Server/.test(c) ? "disclosure: internal hostname in header" : /java\.lang/.test(c) ? "disclosure: Java stack trace" : /\[core\]/.test(c) ? "disclosure: .git/config" : "disclosure: AWS key ID";
    case "parser": return /cached page/.test(c) ? "parser: second response in body" : / 501 /.test(c) ? "parser: 501" : / 505 /.test(c) ? "parser: 505" : "parser: two HTML documents";
    case "connection": return /prematurely/.test(c) ? "connection: upstream closed" : /dial tcp/.test(c) ? "connection: DNS timeout" : / 520 /.test(c) ? "connection: 520" : "connection: truncated body";
    default: return e.label;
  }
};

const gates = {
  heuristic: hard.map((e) => heuristic(e.state) !== "normal"),
  "Jev zero-shot": load("jev_hard").map((r) => !archived(r.broad.probs)),
  "Jev S1-b": s1b("jev", "hard").map((p) => !archived(p)),
  "OpenAI zero-shot": load("openai_hard").map((r) => !archived(r.broad.probs)),
  "OpenAI S1-b": s1b("openai", "hard").map((p) => !archived(p)),
};
const rows = {};
hard.forEach((e, i) => {
  const t = (rows[template(e)] ??= { n: 0, ...Object.fromEntries(Object.keys(gates).map((g) => [g, 0])) });
  t.n++;
  for (const [g, esc] of Object.entries(gates)) if (esc[i]) t[g]++;
});
console.log(`Escalated per template on D_hard (tau ${TAU})\n`);
console.log(["template", "n", ...Object.keys(gates)].join(" | "));
for (const [t, v] of Object.entries(rows).sort()) console.log([t, v.n, ...Object.keys(gates).map((g) => v[g])].join(" | "));

// Structural fingerprints of the kind HTTP Terminator hard-codes: a second HTTP status line inside the body,
// or more than one HTML document in one response.
const fingerprint = (e) => {
  const c = parseResponse(e.state.candidate_response);
  return !!c && (/HTTP\/1\.[01] \d{3}/.test(c.body) || (c.body.match(/<html/gi) || []).length > 1);
};
console.log("\nOpenAI S1-b, alone and OR fingerprints");
for (const [name, split] of [["test", test], ["hard", hard]]) {
  const P = s1b("openai", name);
  const c = { an: 0, no: 0, rec: 0, recOr: 0, fe: 0, feOr: 0, fpAlone: 0 };
  split.forEach((e, i) => {
    const model = !archived(P[i]), both = model || fingerprint(e);
    if (e.label === "normal") { c.no++; c.fe += model; c.feOr += both; }
    else { c.an++; c.rec += model; c.recOr += both; c.fpAlone += fingerprint(e); }
  });
  const p = (x, n) => (100 * x / n).toFixed(1);
  console.log(`  ${name}: recall ${p(c.rec, c.an)} -> ${p(c.recOr, c.an)}; false escalation ${p(c.fe, c.no)} -> ${p(c.feOr, c.no)}; fingerprints alone catch ${c.fpAlone}/${c.an} anomalies`);
}
