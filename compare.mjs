// Runs one decision model over D_dev, D_test and D_hard on the diff input, with the same questions as the
// Laya runs (one broad call and one call with the seven Noul questions per state).
// Usage: node --env-file=.env compare.mjs <jev|openai|gliner> [concurrency]
// Writes results/<model>_<split>.json in the same shape as results/diff.json.
import fs from "node:fs";
import { generate, generateHard, CLASSES } from "./dataset.mjs";
import { diffState } from "./preprocess.mjs";
import { BROAD, NARROW, ANOMALOUS } from "./questions.mjs";
import { jev, openaiDecisions, gliner, decideAll } from "./providers.mjs";

const which = process.argv[2];
const concurrency = Number(process.argv[3]) || (which === "gliner" ? 1 : 8);
const provider = which === "jev" ? jev() : which === "openai" ? openaiDecisions() : which === "gliner" ? await gliner() : null;
if (!provider) throw new Error("usage: compare.mjs <jev|openai|gliner> [concurrency]");

const splits = {
  test: generate({ perClass: 100, seed: 1337 }),
  hard: generateHard({ perClass: 50, seed: 4242 }),
  dev: generate({ perClass: 100, seed: 2024 }),
};
const uniform = Object.fromEntries(CLASSES.map((c) => [c, 1 / CLASSES.length]));
const NQ = { ...NARROW, ...ANOMALOUS };

for (const [split, ex] of Object.entries(splits)) {
  const states = ex.map((e) => diffState(e.state));
  const t = performance.now();
  const B = await decideAll(provider, states, BROAD, concurrency, (d, n) => console.log(`  ${which} ${split} broad ${d}/${n}`));
  const N = await decideAll(provider, states, NQ, concurrency, (d, n) => console.log(`  ${which} ${split} narrow ${d}/${n}`));
  // a refused or missing answer counts as uninformative (uniform / 0.5) and is flagged
  const rows = ex.map((e, i) => {
    const b = B[i].answers.cause, refusedB = !b || b.refusal || !b.probabilities;
    const probs = CLASSES.map((c) => (refusedB ? uniform : b.probabilities)[c] ?? 0);
    const nou = (k) => { const a = N[i].answers[k]; return a && !a.refusal && typeof a.noul === "number" ? a.noul : null; };
    const refusedN = [...CLASSES, "anomalous"].filter((k) => nou(k) === null);
    return {
      id: e.id, label: e.label, tier: e.tier, model: B[i].model,
      broad: { probs, tokens: B[i].usage?.input_tokens ?? null, ms: B[i].ms },
      narrow: { p: CLASSES.map((c) => nou(c) ?? 0.5), anomalous: nou("anomalous") ?? 0.5, tokens: N[i].usage?.input_tokens ?? null, ms: N[i].ms },
      refused: { broad: refusedB, narrow: refusedN },
    };
  });
  fs.writeFileSync(`results/${which}_${split}.json`, JSON.stringify(rows));
  console.log(`saved ${which}_${split} in ${((performance.now() - t) / 1000).toFixed(0)} s, refusals: broad ${rows.filter((r) => r.refused.broad).length}, narrow ${rows.filter((r) => r.refused.narrow.length).length}`);
}
provider.close?.();
