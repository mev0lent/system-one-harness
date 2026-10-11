# System One Harness

[![Reproduce](https://github.com/mev0lent/system-one-harness/actions/workflows/reproduce.yml/badge.svg)](https://github.com/mev0lent/system-one-harness/actions/workflows/reproduce.yml) [![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**Can a cheap decision model decide which HTTP responses deserve an expensive reasoning model, sparing that model most of its work, without throwing away the anomalies that matter?**

This repository is the complete, reproducible pilot behind the article [System One AI for Cybersecurity: Not Every Decision Needs a Generative Model](https://niklas-heringer.com/research/system-one-ai-cybersecurity/). It tests four decision models (Laya, GLiNER2.5-Decide, Jev 1.13.0 and OpenAI's Decisions API) and three generative LLM settings as a triage gate for a security research pipeline, on synthetic HTTP data.

Every model output is committed. You can recompute every number in the article with one command, without API keys and without network access.

> **Read this first.** This is a pilot on synthetic, author-designed templates. It is a method and a set of leads, not evidence of performance on real traffic. The limits are listed [below](#limits).

## The Results in One Table

| Finding | Number |
|---|---|
| Deterministic diff before the model | Laya +21.7 accuracy points, twice the throughput |
| Spread between models, hard tier, zero-shot | 37.7% (Laya) to 79.7% (OpenAI's Decisions API). Rules: 9.3% |
| Best gate, first hard tier | 0% of benign examples escalated, 92.8% of anomalies kept |
| Where that gate loses anomalies | 4 templates, including a second HTTP response in the body (request smuggling) |
| Generative LLM as the gate | 100% of anomalies kept, but two thirds of benign traffic escalated, at up to 58 times the cost |
| One planted sentence in the response body | GLiNER bypassed 98% of the time, Jev at most 8%, OpenAI at most 15% |
| Frozen cascade, second hard tier | 96.0% of anomalies kept, 5% benign escalated, reasoning on 8.9% of traffic at 1% prevalence |
| Hosted decision cost | about $23 (Jev) and $32 (OpenAI) per million decisions |

The first-tier cascade result (98% recall, reasoning on 2% of traffic) was tuned after seeing the data. The second tier ran with settings frozen in advance, but its templates were written by someone who had seen the first-tier results. Treat it as an exploratory check, not independent confirmation.

## What the Harness Does

```text
seeded templates ─► request, baseline and candidate response ─► deterministic diff
                                                                     │
          ┌──────────────────────────────────────────────────────────┘
          ▼
  decision model or LLM ─► six class probabilities ─► small trained head (optional)
          │
          ▼
  gate: archive if P(normal) ≥ τ, otherwise escalate
          │
          ▼
  metrics: accuracy, macro-F1, ECE, Brier, recall per anomaly type,
           false escalation, bootstrap intervals, cost and latency
```

Each example pairs a baseline response with a candidate response and has one of six labels:

| Class | Meaning |
|---|---|
| `normal` | Only dynamic values such as timestamps, IDs or tokens differ |
| `filtering` | Blocked by a firewall, WAF, rate limiter or access policy |
| `app_error` | The application rejected the input or failed with an ordinary error |
| `parser` | A server in the chain framed the request differently, or returned another request's content |
| `connection` | Timeout, reset, closed connection or unreachable upstream |
| `disclosure` | Exposes internal details such as stack traces, paths, internal addresses or versions |

There are five data splits, all generated deterministically from seeds:

| Split | Size | Purpose |
|---|---|---|
| Test | 600 (100 per class) | In-distribution evaluation |
| Dev | 600 | Training the small heads and choosing gate thresholds |
| Hard tier | 300 (50 per class) | Out of distribution, written after the rule baseline was frozen |
| Second hard tier | 250 (100 normal, 30 per anomaly class) | Frozen-setting check of the cascade |
| Adversarial | 50 anomalies | Planted-sentence injection test |

## Quick Start: Reproduce Every Number (No Keys Needed)

Requirements: Node.js 22 or newer.

```sh
npm ci
npm run reproduce
```

`npm run reproduce` runs the seven analysis scripts in order. They read the committed model outputs in `results/`, rewrite the summary files there, and print the tables from the article. On a clean checkout the summaries come out byte-identical. The "Reproduce" workflow checks exactly that on every push.

You can also run them one by one:

| Script | Reproduces |
|---|---|
| `node analyze.mjs` | The Laya pilot: preprocessing, context limit, narrow questions, TF-IDF baseline |
| `node compare_analyze.mjs` | The four-model comparison, calibration, trained heads, gate table |
| `node inject_analyze.mjs` | The planted-sentence injection test for every model |
| `node llm_analyze.mjs` | Generative LLMs as the gate, cost and latency |
| `node cascade.mjs` | The first-tier cascade (post hoc) |
| `node gate_templates.mjs` | Per-template losses and the two fingerprints |
| `node hard2_analyze.mjs` | The frozen cascade on the second hard tier |

## Re-running the Models (Optional)

This step calls the models again. It needs local compute or your own API keys, and hosted results can differ from the committed ones because hosted models change.

**Keys.** Hosted calls need `TYPESAFE_API_KEY` (Jev) and `OPENAI_API_KEY` (Decisions API and Responses API). Put them in a local `.env` file, which is gitignored, and pass it with `node --env-file=.env`. The harness never logs or writes keys. Never commit them.

**Cache.** Hosted responses are cached in `cache/` (gitignored), keyed by model, version, state and question. Re-running a script with unchanged inputs costs nothing.

### Laya (local, CPU)

The ONNX weights (about 1.7 GB) download on first use to `~/.cache/receptron-laya` and need about 2 GB of RAM.

```sh
node infer.mjs main         # 600 examples, broad vs. narrow questions on raw input (about 1.5 h on CPU)
node infer.mjs diff         # the same examples with the deterministic diff as input
node infer.mjs dev          # the training split for the small heads
node infer.mjs hard         # the first hard tier
node infer.mjs candonly     # candidate response only, no baseline
node infer.mjs ablation     # context-length probe
node infer.mjs adversarial  # planted-sentence test
node infer.mjs latency
node infer.mjs throughput   # batched vs. unbatched states per second
node analyze.mjs
```

On a CPU, batching makes Laya slower, so the harness uses batch size 1.

### Jev, OpenAI's Decisions API and GLiNER2.5-Decide

```sh
node --env-file=.env compare.mjs jev      # about 2 min
node --env-file=.env compare.mjs openai   # about 1 min
py -3.12 -m venv .venv-gliner
.venv-gliner/Scripts/python -m pip install "gliner2[local]"
node compare.mjs gliner                   # about 2 h on CPU, do not run other heavy jobs meanwhile
node compare_analyze.mjs
```

GLiNER runs in a Python 3.12 sidecar (`gliner_sidecar.py`) pinned to one Hugging Face revision. On Linux or macOS, use `python3.12 -m venv` and `.venv-gliner/bin/python`.

### Injection Test and Generative LLMs

```sh
node --env-file=.env inject.mjs <laya|gliner|jev|openai>
node inject_analyze.mjs
node --env-file=.env llm_compare.mjs <none|medium|sol>
node llm_analyze.mjs
```

Generative calls take 1.5 to 2.2 seconds each. One setting over all splits plus the injection test takes about 10 minutes at concurrency 8.

### Second Hard Tier

Read `PROTOCOL.md` section 18.11 first. It fixes the cascade settings, targets and stop rule. Each command makes 250 hosted calls.

```sh
node --env-file=.env hard2_run.mjs openai
node --env-file=.env hard2_run.mjs jev
node --env-file=.env hard2_run.mjs medium
node hard2_analyze.mjs
```

## Files

| File | Role |
|---|---|
| `dataset.mjs` | Seeded generators for the test, dev, hard and adversarial splits, labels assigned by template |
| `hard2_templates.mjs` | Templates and generator for the second hard tier |
| `preprocess.mjs` | Deterministic diff of baseline and candidate, with dynamic values masked |
| `questions.mjs` | The broad and narrow question sets, identical for every model |
| `heuristic.mjs` | The frozen rule baseline |
| `providers.mjs` | Adapters for Jev, OpenAI's Decisions API and GLiNER, with the on-disk cache |
| `gliner_sidecar.py` | GLiNER2.5-Decide over JSON lines |
| `infer.mjs`, `batch.mjs` | Laya inference and batching |
| `compare.mjs`, `inject.mjs`, `llm_compare.mjs`, `hard2_run.mjs` | Model runs that write raw outputs to `results/` |
| `adapt.mjs` | Small logistic-regression heads and the TF-IDF baseline |
| `stats.mjs`, `gatelib.mjs` | Shared metrics, bootstrap intervals and gate code |
| `*_analyze.mjs`, `cascade.mjs`, `gate_templates.mjs` | Analyses that turn raw outputs into the article's tables |
| `results/` | Every raw model output and every summary |
| `PROTOCOL.md` | The written protocol: every variant, hypothesis, endpoint, deviation and result |

## Method Notes

- **Preregistration.** Hypotheses and endpoints were written into `PROTOCOL.md` before each run. Deviations are listed there (section 17) with their reason. Post-hoc analyses are labelled as such.
- **Intervals.** Brackets are 95% bootstrap intervals over 2,000 resamples. Examples come from a limited set of templates, so these intervals are, if anything, too narrow.
- **Fairness.** All models get the same input and the same questions, word for word. The questions were written once and never tuned for any model.
- **Gate thresholds.** Each gate's threshold was chosen on the dev split as the most aggressive value that still kept 95% of anomalies, then applied unchanged to the hard tiers.
- **Prevalence.** The hard tiers are mostly anomalous. Figures "at 1% prevalence" reweight the measured class-conditional rates to a world where 1% of observations are anomalous. They are projections, not measurements.

## Limits

- The data is synthetic and templated, and some labels are ambiguous (is a `413` with an empty body a parser anomaly?).
- Both hard tiers were written within this project, so they measure distance from its own templates, not from real traffic.
- The first gate result rests on 50 normal examples, the second on 100.
- The LLMs' probabilities are self-reported in JSON, not read from token probabilities.
- Each configuration ran once. Hosted models can change without notice.
- The injection test used three fixed sentences. A real attacker would search for the one that works.
- The trained heads are a lower bound on adaptation, not a substitute for fine-tuning.

## Open Work

The most useful next steps, roughly in order:

1. Real labelled HTTP experiments with real benign variation and realistic prevalence.
2. A cascade whose settings are fixed in advance and tested on independently written data.
3. An adaptive attacker who searches for the planted sentence that works.
4. Repeating the hosted queries to measure nondeterminism.
5. A real fine-tune of the open models on a GPU.

If you have labelled traffic and run this harness on it, I would love to hear what transfers and what does not.

## Citation

GitHub shows a "Cite this repository" button generated from `CITATION.cff`. In plain text:

```text
Niklas Heringer (2026). System One Harness: a pilot on decision models as a triage gate
for security research. Version 1.0.1. https://github.com/mev0lent/system-one-harness
```

The article was published alongside version `v1.0.1`. Later changes are listed in the GitHub releases.

## License

The harness code is MIT licensed, see `LICENSE`. Model weights and hosted services have their own terms. The synthetic data and model outputs in `results/` are included for reproducibility.
