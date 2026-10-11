// Frozen broad-question run for PROTOCOL.md §18.11.
// Usage: node --env-file=.env hard2_run.mjs <openai|jev|medium> [concurrency]
import fs from "node:fs";
import { CLASSES, generateHard2 } from "./dataset.mjs";
import { diffState } from "./preprocess.mjs";
import { BROAD } from "./questions.mjs";
import { decideAll, jev, openaiDecisions, openaiGenerative } from "./providers.mjs";

const which = process.argv[2];
const concurrency = Number(process.argv[3]) || 8;
const provider = which === "openai" ? openaiDecisions()
  : which === "jev" ? jev()
  : which === "medium" ? openaiGenerative({ model: "gpt-6-luna", effort: "medium" })
  : null;
if (!provider) throw new Error("usage: hard2_run.mjs <openai|jev|medium> [concurrency]");

const examples = generateHard2();
const uniform = Object.fromEntries(CLASSES.map((c) => [c, 1 / CLASSES.length]));
const responses = await decideAll(provider, examples.map((e) => diffState(e.state)), BROAD, concurrency,
  (done, total) => console.log(`  ${which} hard2 ${done}/${total}`));
const rows = examples.map((e, i) => {
  const r = responses[i], b = r.answers?.cause;
  const refused = !b || b.refusal || !b.probabilities;
  return {
    id: e.id, label: e.label, tier: e.tier, template: e.template, model: r.model,
    broad: {
      probs: CLASSES.map((c) => (refused ? uniform : b.probabilities)[c] ?? 0),
      tokens: r.usage?.input_tokens ?? null,
      output_tokens: r.usage?.output_tokens ?? null,
      reasoning_tokens: r.usage?.output_tokens_details?.reasoning_tokens ?? null,
      ms: r.ms,
    },
    refused: { broad: refused },
  };
});
fs.mkdirSync("results", { recursive: true });
fs.writeFileSync(`results/hard2_${which}.json`, JSON.stringify(rows));
console.log(`saved hard2_${which}: ${rows.length} examples, ${rows.filter((r) => r.refused.broad).length} refusals`);
