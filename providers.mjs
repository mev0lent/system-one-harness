// One adapter per decision model, all returning the Laya/Jev shape:
//   { model, answers: { name: { type: "choice", probabilities: {label: p} } | { type: "noul", noul: p } }, usage, ms }
// Hosted responses are cached on disk (cache/<provider>/<sha256>.json), so reruns cost nothing and are exact.
// API keys come from the environment only (node --env-file=.env). They are never logged or written.
import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { spawn } from "node:child_process";
import readline from "node:readline";
import { pyJsonDumps } from "./node_modules/@receptron/laya/dist/sequence.js";

const sha = (s) => crypto.createHash("sha256").update(s).digest("hex");
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function cached(provider, version, state, questions, fn) {
  const key = sha(JSON.stringify([provider, version, state, questions]));
  const file = path.join("cache", provider, `${key}.json`);
  if (fs.existsSync(file)) return JSON.parse(fs.readFileSync(file, "utf8"));
  return fn().then((res) => {
    fs.mkdirSync(path.dirname(file), { recursive: true });
    fs.writeFileSync(file, JSON.stringify({ ...res, cached_at: new Date().toISOString() }));
    return res;
  });
}

async function postJson(url, key, body, tries = 6) {
  for (let i = 0; ; i++) {
    const t = performance.now();
    const r = await fetch(url, { method: "POST", headers: { Authorization: `Bearer ${key}`, "Content-Type": "application/json" }, body: JSON.stringify(body) });
    const ms = performance.now() - t;
    if (r.ok) return { json: await r.json(), ms };
    const text = (await r.text()).slice(0, 500);
    if ((r.status === 429 || r.status >= 500) && i < tries - 1) { await sleep(1000 * 2 ** i); continue; }
    throw new Error(`${url} -> HTTP ${r.status}: ${text}`);
  }
}

// ---- Jev (TypeSafe) ----
export function jev({ model = "jev-1.13.0" } = {}) {
  const key = process.env.TYPESAFE_API_KEY;
  if (!key) throw new Error("TYPESAFE_API_KEY is not set");
  return {
    name: "jev",
    async decide(state, questions) {
      const s = pyJsonDumps(state);
      return cached("jev", model, s, questions, async () => {
        const { json, ms } = await postJson("https://api.typesafe.ai/v1/systemone", key, { state: s, model, questions });
        return { model: json.model, answers: json.answers, usage: json.usage, ms };
      });
    },
  };
}

// ---- OpenAI Decisions (preview) ----
// choice -> choices [{value, description}], noul -> predicate. Answers come back as an array keyed by name.
export function openaiDecisions({ model = "gpt-6-luna" } = {}) {
  const key = process.env.OPENAI_API_KEY;
  if (!key) throw new Error("OPENAI_API_KEY is not set");
  const toQuestion = (name, q) => q.type === "choice"
    ? { type: "choice", name, instructions: q.instructions, choices: Object.entries(q.criteria).map(([value, description]) => ({ value, description })) }
    : { type: "predicate", name, instructions: q.instructions };
  return {
    name: "openai",
    async decide(state, questions) {
      const s = pyJsonDumps(state);
      return cached("openai", model, s, questions, async () => {
        const body = { model, input: s, questions: Object.entries(questions).map(([n, q]) => toQuestion(n, q)) };
        const { json, ms } = await postJson("https://api.openai.com/v1/decisions", key, body);
        const answers = {};
        for (const a of json.answers ?? []) {
          const q = questions[a.name];
          if (!q) continue;
          if (a.type === "refusal") answers[a.name] = { type: q.type, refusal: true };
          else if (q.type === "choice") answers[a.name] = { type: "choice", probabilities: Object.fromEntries((a.probabilities ?? []).map((p) => [p.value, p.probability])), confidence: a.confidence };
          else answers[a.name] = { type: "noul", noul: a.probability };
        }
        return { model: json.model ?? model, answers, usage: json.usage, ms, raw: json };
      });
    },
  };
}

// ---- Generative LLM via the Responses API (PROTOCOL.md 18.9) ----
// Same instructions and options as the decision models. The model returns one probability per option through
// structured outputs (self-reported, not logprobs). The adapter normalises them to sum to 1.
// Only Choice questions are supported.
export const GEN_SYSTEM = "You are a decision component inside an automated HTTP security testing pipeline. You receive a state as JSON and one question with a fixed set of options. For every option, give the probability that it is the correct answer. The probabilities must sum to 1. Answer only with the JSON object.";
export function openaiGenerative({ model = "gpt-6-luna", effort = "none" } = {}) {
  const key = process.env.OPENAI_API_KEY;
  if (!key) throw new Error("OPENAI_API_KEY is not set");
  return {
    name: `openai-gen-${effort}`,
    async decide(state, questions) {
      const s = pyJsonDumps(state);
      const [name, q] = Object.entries(questions)[0];
      if (q.type !== "choice") throw new Error("openaiGenerative supports choice questions only");
      const options = Object.keys(q.criteria);
      const user = `State:\n${s}\n\nQuestion: ${q.instructions}\n\nOptions:\n${Object.entries(q.criteria).map(([k, d]) => `- ${k}: ${d}`).join("\n")}`;
      const schema = { type: "object", properties: Object.fromEntries(options.map((o) => [o, { type: "number" }])), required: options, additionalProperties: false };
      return cached("openai-gen", `${model}:${effort}`, s, { questions, system: GEN_SYSTEM }, async () => {
        const body = { model, reasoning: { effort }, input: [{ role: "system", content: GEN_SYSTEM }, { role: "user", content: user }], text: { format: { type: "json_schema", name: "decision", strict: true, schema } } };
        const { json, ms } = await postJson("https://api.openai.com/v1/responses", key, body);
        const text = (json.output ?? []).flatMap((o) => o.content ?? []).find((c) => c.type === "output_text")?.text;
        let probabilities = null;
        try {
          const raw = JSON.parse(text);
          const sum = options.reduce((a, o) => a + Math.max(0, Number(raw[o]) || 0), 0);
          if (sum > 0) probabilities = Object.fromEntries(options.map((o) => [o, Math.max(0, Number(raw[o]) || 0) / sum]));
        } catch { /* recorded as a refusal below */ }
        const answers = { [name]: probabilities ? { type: "choice", probabilities } : { type: "choice", refusal: true } };
        return { model: json.model ?? model, answers, usage: json.usage, ms, raw_text: text ?? null };
      });
    },
  };
}

// ---- GLiNER2.5-Decide (local, Python sidecar) ----
export async function gliner() {
  const py = path.join(".venv-gliner", process.platform === "win32" ? "Scripts/python.exe" : "bin/python");
  const proc = spawn(py, ["-I", "gliner_sidecar.py"], { stdio: ["pipe", "pipe", "inherit"] });
  const rl = readline.createInterface({ input: proc.stdout });
  const pending = new Map();
  let ready, revision;
  const readyP = new Promise((r) => (ready = r));
  rl.on("line", (line) => {
    const m = JSON.parse(line);
    if (m.ready) { revision = m.revision; ready(); return; }
    pending.get(m.id)?.(m);
    pending.delete(m.id);
  });
  await readyP;
  let next = 0;
  return {
    name: "gliner",
    async decide(state, questions) {
      const s = pyJsonDumps(state);
      return cached("gliner", revision, s, questions, () => new Promise((resolve) => {
        const id = next++;
        pending.set(id, (m) => resolve({ model: `fastino/GLiNER2.5-Decide@${revision.slice(0, 8)}`, answers: m.answers, ms: m.ms }));
        proc.stdin.write(JSON.stringify({ id, state: s, questions }) + "\n");
      }));
    },
    close: () => proc.stdin.end(),
  };
}

// Run decide() over many states with bounded concurrency, preserving order.
export async function decideAll(provider, states, questions, concurrency = 1, onProgress = () => {}) {
  const out = new Array(states.length);
  let i = 0, done = 0;
  async function worker() {
    while (i < states.length) {
      const j = i++;
      out[j] = await provider.decide(states[j], questions);
      if (++done % 50 === 0) onProgress(done, states.length);
    }
  }
  await Promise.all(Array.from({ length: Math.max(1, concurrency) }, worker));
  return out;
}
