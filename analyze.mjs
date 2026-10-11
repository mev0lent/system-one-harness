// Computes metrics from results/*.json and writes results/summary.json + results/summary.md
import fs from "node:fs";
import { generate, generateHard, CLASSES } from "./dataset.mjs";
import { diffState } from "./preprocess.mjs";
import { trainLogReg, layaFeatures, fitTfidf } from "./adapt.mjs";
import { K, argmax, mean, pct, boot, accOn, f1On, confusion, macroF1, calibration } from "./stats.mjs";
import { heuristic } from "./heuristic.mjs";

const read = (n) => (fs.existsSync(`results/${n}.json`) ? JSON.parse(fs.readFileSync(`results/${n}.json`, "utf8")) : null);
const norm = (p) => { const s = p.reduce((a, b) => a + b, 0) || 1; return p.map((v) => v / s); };
function evalSet(rows, probs) {
  const pred = probs.map((p) => CLASSES[argmax(p)]);
  const m = confusion(rows, pred);
  const acc = mean(rows.map((r, i) => (pred[i] === r.label ? 1 : 0)));
  return { accuracy: acc, macroF1: macroF1(m), ...calibration(rows, probs), confusion: m, pred };
}
function byTier(rows, pred) {
  const o = {};
  for (const t of ["easy", "subtle"]) {
    const idx = rows.map((r, i) => i).filter((i) => rows[i].tier === t);
    o[t] = mean(idx.map((i) => (pred[i] === rows[i].label ? 1 : 0)));
  }
  return o;
}

const summary = {};
const md = [];

const main = read("main");
if (main) {
  const ex = generate({ perClass: 100, seed: 1337 });
  const hPred = ex.map((e) => heuristic(e.state));
  const hRows = main;
  const hAcc = mean(hRows.map((r, i) => (hPred[i] === r.label ? 1 : 0)));
  const hM = confusion(hRows, hPred);
  const broadP = main.map((r) => r.broad.probs);
  const narrowP = main.map((r) => norm(r.narrow.p));
  const B = evalSet(main, broadP), N = evalSet(main, narrowP);
  summary.main = {
    heuristic: { accuracy: hAcc, macroF1: macroF1(hM), tier: byTier(main, hPred) },
    laya_broad: { ...B, tier: byTier(main, B.pred) },
    laya_decomposed: { ...N, tier: byTier(main, N.pred) },
  };
  summary.ci = {
    heuristic: { accuracy: boot(main.length, accOn(main, hPred)), macroF1: boot(main.length, f1On(main, hPred)) },
    laya_broad: { accuracy: boot(main.length, accOn(main, B.pred)), macroF1: boot(main.length, f1On(main, B.pred)) },
    laya_decomposed: { accuracy: boot(main.length, accOn(main, N.pred)), macroF1: boot(main.length, f1On(main, N.pred)) },
    // paired: same resample for both methods, so the interval is on the difference itself
    decomposed_minus_broad_accuracy: boot(main.length, (idx) => accOn(main, N.pred)(idx) - accOn(main, B.pred)(idx)),
  };
  // per-class recall with intervals (n = 100 per class)
  summary.ci.recall = {};
  for (const [name, pred] of [["broad", B.pred], ["decomposed", N.pred]]) {
    summary.ci.recall[name] = {};
    for (const c of CLASSES) {
      const rows = main.map((r, i) => i).filter((i) => main[i].label === c);
      summary.ci.recall[name][c] = boot(rows.length, (idx) => mean(idx.map((j) => (pred[rows[j]] === c ? 1 : 0))));
    }
  }
  // binary anomaly detection
  const nonNormal = main.filter((r) => r.label !== "normal"), normal = main.filter((r) => r.label === "normal");
  const hAnom = (i) => hPred[i] !== "normal";
  const hIdx = main.map((r, i) => i);
  const binRow = (name, f) => {
    const tpr = mean(hIdx.filter((i) => main[i].label !== "normal").map((i) => (f(i) ? 1 : 0)));
    const fpr = mean(hIdx.filter((i) => main[i].label === "normal").map((i) => (f(i) ? 1 : 0)));
    return { name, recall_interesting: tpr, false_positive_rate: fpr };
  };
  summary.binary = [
    binRow("heuristic", hAnom),
    binRow("laya anomalous>0.5", (i) => main[i].narrow.anomalous > 0.5),
    binRow("laya broad (not normal)", (i) => argmax(main[i].broad.probs) !== 0),
    binRow("laya decomposed (not normal)", (i) => argmax(narrowP[i]) !== 0),
  ];
  // anomaly cascade: escalate if not confidently normal. recall at several taus
  summary.cascade = [0.5, 0.7, 0.9].map((tau) => {
    const escal = (i) => !(argmax(broadP[i]) === 0 && Math.max(...broadP[i]) >= tau);
    const esc = hIdx.filter(escal);
    return { tau, escalated_fraction: esc.length / main.length, recall_interesting: mean(hIdx.filter((i) => main[i].label !== "normal").map((i) => (escal(i) ? 1 : 0))) };
  });
}

const dif = read("diff");
if (dif) {
  const bP = dif.map((r) => r.broad.probs), nP = dif.map((r) => norm(r.narrow.p));
  const DB = evalSet(dif, bP), DN = evalSet(dif, nP);
  const recall = (pred) => Object.fromEntries(CLASSES.map((c) => {
    const rows = dif.map((r, i) => i).filter((i) => dif[i].label === c);
    return [c, boot(rows.length, (idx) => mean(idx.map((j) => (pred[rows[j]] === c ? 1 : 0))))];
  }));
  summary.diff = {
    tokens_mean: mean(dif.map((r) => r.broad.tokens)),
    broad: { accuracy: boot(dif.length, accOn(dif, DB.pred)), macroF1: DB.macroF1, ece: DB.ece, brier: DB.brier, riskCoverage: DB.riskCoverage, tier: byTier(dif, DB.pred), recall: recall(DB.pred), confusion: DB.confusion },
    decomposed: { accuracy: boot(dif.length, accOn(dif, DN.pred)), macroF1: DN.macroF1, ece: DN.ece, brier: DN.brier, tier: byTier(dif, DN.pred), recall: recall(DN.pred), confusion: DN.confusion },
    binary_anomalous: {
      recall_interesting: mean(dif.filter((r) => r.label !== "normal").map((r) => (r.narrow.anomalous > 0.5 ? 1 : 0))),
      false_positive_rate: mean(dif.filter((r) => r.label === "normal").map((r) => (r.narrow.anomalous > 0.5 ? 1 : 0))),
    },
  };
  if (main) {
    // paired against the raw-input run on the same examples
    const rawB = main.map((r) => CLASSES[argmax(r.broad.probs)]);
    summary.diff.broad_minus_raw_broad = boot(dif.length, (idx) => accOn(dif, DB.pred)(idx) - accOn(main, rawB)(idx));
  }
  summary.diff.cascade = [0.5, 0.7, 0.9].map((tau) => {
    const escal = (i) => !(argmax(bP[i]) === 0 && Math.max(...bP[i]) >= tau);
    const all = dif.map((r, i) => i);
    return { tau, escalated_fraction: all.filter(escal).length / dif.length, recall_interesting: mean(all.filter((i) => dif[i].label !== "normal").map((i) => (escal(i) ? 1 : 0))) };
  });
}

// ---- hard tier and the hybrid gate ----
// Gate question: which observations go to the expensive tier? "interesting" = any non-normal label.
function gateStats(rows, escalate) {
  const all = rows.map((r, i) => i);
  const anom = all.filter((i) => rows[i].label !== "normal"), norm_ = all.filter((i) => rows[i].label === "normal");
  return {
    escalated_fraction: boot(rows.length, (idx) => mean(idx.map((i) => (escalate(i) ? 1 : 0)))),
    recall_interesting: boot(anom.length, (idx) => mean(idx.map((j) => (escalate(anom[j]) ? 1 : 0)))),
    false_escalation_rate: boot(norm_.length, (idx) => mean(idx.map((j) => (escalate(norm_[j]) ? 1 : 0)))),
  };
}
// hybrid: the heuristic proposes, Laya (on the diff) may only archive what it is confident is normal
// archive only when Laya's top choice is "normal" with probability >= tau. Reported for a small tau sweep
function gatePolicies(rows, states) {
  const h = states.map(heuristic);
  const confNormal = (i, tau) => argmax(rows[i].broad.probs) === 0 && rows[i].broad.probs[0] >= tau;
  const out = { heuristic: gateStats(rows, (i) => h[i] !== "normal"), laya_diff_anomalous: gateStats(rows, (i) => rows[i].narrow.anomalous > 0.5) };
  for (const tau of [0.5, 0.7, 0.9]) {
    out[`laya_diff_broad_tau${tau}`] = gateStats(rows, (i) => !confNormal(i, tau));
    out[`hybrid_tau${tau}`] = gateStats(rows, (i) => h[i] !== "normal" && !confNormal(i, tau));
  }
  return out;
}

const hard = read("hard");
if (hard) {
  const ex = generateHard({ perClass: 50, seed: 4242 });
  const hPred = ex.map((e) => heuristic(e.state));
  const rawPred = hard.map((r) => CLASSES[argmax(r.raw_broad.probs)]);
  const bPred = hard.map((r) => CLASSES[argmax(r.broad.probs)]);
  const nPred = hard.map((r) => CLASSES[argmax(norm(r.narrow.p))]);
  const recall = (pred) => Object.fromEntries(CLASSES.map((c) => [c, mean(hard.filter((r, i) => r.label === c).map((r) => (pred[hard.indexOf(r)] === c ? 1 : 0)))]));
  summary.hard = {
    n: hard.length,
    heuristic: { accuracy: boot(hard.length, accOn(hard, hPred)), macroF1: macroF1(confusion(hard, hPred)), recall: recall(hPred) },
    laya_raw_broad: { accuracy: boot(hard.length, accOn(hard, rawPred)), macroF1: macroF1(confusion(hard, rawPred)), recall: recall(rawPred) },
    laya_diff_broad: { accuracy: boot(hard.length, accOn(hard, bPred)), macroF1: macroF1(confusion(hard, bPred)), recall: recall(bPred), confusion: confusion(hard, bPred) },
    laya_diff_decomposed: { accuracy: boot(hard.length, accOn(hard, nPred)), macroF1: macroF1(confusion(hard, nPred)), recall: recall(nPred) },
    gate: gatePolicies(hard, ex.map((e) => e.state)),
  };
  summary.hard.decomposed_minus_broad = boot(hard.length, (idx) => accOn(hard, nPred)(idx) - accOn(hard, bPred)(idx));
}
if (dif) {
  const ex = generate({ perClass: 100, seed: 1337 });
  summary.diff.gate = gatePolicies(dif, ex.map((e) => e.state));
}

// ---- adaptation: models trained on the dev split, tested on the test split and the hard tier ----
const dev = read("dev");
if (dev && dif) {
  const yDev = dev.map((r) => CLASSES.indexOf(r.label));
  const head = trainLogReg(dev.map(layaFeatures), yDev, K);
  const devEx = generate({ perClass: 100, seed: 2024 }), testEx = generate({ perClass: 100, seed: 1337 });
  const text = (e) => JSON.stringify(diffState(e.state));
  const vec = fitTfidf(devEx.map(text));
  const tfidf = trainLogReg(devEx.map((e) => vec(text(e))), yDev, K, { standardize: false });
  const report = (rows, probs) => {
    const pred = probs.map((p) => CLASSES[argmax(p)]);
    return {
      accuracy: boot(rows.length, accOn(rows, pred)),
      macroF1: macroF1(confusion(rows, pred)),
      recall: Object.fromEntries(CLASSES.map((c) => [c, mean(rows.map((r, i) => [r, i]).filter(([r]) => r.label === c).map(([, i]) => (pred[i] === c ? 1 : 0)))])),
      ...(() => { const c = calibration(rows, probs); return { ece: c.ece, brier: c.brier }; })(),
    };
  };
  summary.adapt = {
    train: "dev split, 600 examples, seed 2024",
    test: { laya_head: report(dif, dif.map((r) => head(layaFeatures(r)))), tfidf_logreg: report(dif, testEx.map((e) => tfidf(vec(text(e))))) },
  };
  if (hard) {
    const hardEx = generateHard({ perClass: 50, seed: 4242 });
    const hp = hard.map((r) => head(layaFeatures(r)));
    const tp = hardEx.map((e) => tfidf(vec(text(e))));
    summary.adapt.hard = { laya_head: report(hard, hp), tfidf_logreg: report(hard, tp) };
    const hpPred = hp.map((p) => CLASSES[argmax(p)]), tpPred = tp.map((p) => CLASSES[argmax(p)]);
    const zsPred = hard.map((r) => CLASSES[argmax(r.broad.probs)]);
    summary.adapt.hard.head_minus_tfidf = boot(hard.length, (idx) => accOn(hard, hpPred)(idx) - accOn(hard, tpPred)(idx));
    summary.adapt.hard.head_minus_zero_shot_broad = boot(hard.length, (idx) => accOn(hard, hpPred)(idx) - accOn(hard, zsPred)(idx));
    // gate with the adapted head deciding what may be archived
    const h = hardEx.map((e) => heuristic(e.state));
    summary.adapt.hard.gate = {
      laya_head: gateStats(hard, (i) => !(argmax(hp[i]) === 0 && hp[i][0] >= 0.9)),
      hybrid_head: gateStats(hard, (i) => h[i] !== "normal" && !(argmax(hp[i]) === 0 && hp[i][0] >= 0.9)),
    };
  }
}

const co = read("candonly");
if (co) {
  const E = evalSet(co, co.map((r) => r.probs));
  summary.candidate_only = { accuracy: E.accuracy, macroF1: E.macroF1, ece: E.ece, brier: E.brier, tier: byTier(co, E.pred), riskCoverage: E.riskCoverage, confusion: E.confusion };
}

const abl = read("ablation");
if (abl) {
  summary.ablation = {};
  for (const [w, rows] of Object.entries(abl)) {
    const B = evalSet(rows.map((r) => ({ ...r, tier: "?" })), rows.map((r) => r.broad.probs));
    summary.ablation[w] = { n: rows.length, tokens_mean: mean(rows.map((r) => r.broad.tokens)), broad_acc: boot(rows.length, accOn(rows, B.pred)), broad_ece: B.ece };
  }
}

const adv = read("adversarial");
if (adv) {
  summary.adversarial = {};
  const clean = adv.clean;
  for (const [name, rows] of Object.entries(adv)) {
    const out = {};
    for (const mode of ["broad", "decomposed"]) {
      const probs = (rs) => rs.map((r) => (mode === "broad" ? r.broad.probs : norm(r.narrow.p)));
      const cp = probs(clean), vp = probs(rows);
      const cPred = cp.map(argmax), vPred = vp.map(argmax);
      const acc = mean(rows.map((r, i) => (CLASSES[vPred[i]] === r.label ? 1 : 0)));
      const confidentCorrect = rows.map((r, i) => i).filter((i) => CLASSES[cPred[i]] === rows[i].label && Math.max(...cp[i]) >= 0.8);
      out[mode] = {
        accuracy: acc,
        flipped_prediction: boot(rows.length, (idx) => mean(idx.map((i) => (vPred[i] !== cPred[i] ? 1 : 0)))),
        flipped_to_normal: boot(rows.length, (idx) => mean(idx.map((i) => (vPred[i] === 0 && cPred[i] !== 0 ? 1 : 0)))),
        confident_correct_n: confidentCorrect.length,
        confident_dropped_below_0_8: confidentCorrect.length ? mean(confidentCorrect.map((i) => (Math.max(...vp[i]) < 0.8 ? 1 : 0))) : null,
      };
    }
    summary.adversarial[name] = out;
  }
}

const lat = read("latency");
if (lat) {
  summary.latency = {};
  for (const [k, v] of Object.entries(lat)) summary.latency[k] = { p50: pct(v, 0.5), p95: pct(v, 0.95), mean: mean(v) };
}

fs.writeFileSync("results/summary.json", JSON.stringify(summary, null, 2));
console.log(JSON.stringify(summary, (k, v) => (k === "confusion" || k === "pred" ? undefined : typeof v === "number" ? Math.round(v * 1000) / 1000 : v), 2));
