// Shared gate helpers, identical in behaviour to compare_analyze.mjs (PROTOCOL.md 9, 18.4, 18.7).
import { CLASSES } from "./dataset.mjs";
import { parseResponse } from "./preprocess.mjs";
import { trainLogReg, layaFeatures } from "./adapt.mjs";
import { K, argmax, mean } from "./stats.mjs";

// the six broad logits exactly as head S1-b sees them. The narrow placeholders are sliced off
export const broadFeat = (probs) => layaFeatures({ broad: { probs }, narrow: { p: CLASSES.map(() => 0.5), anomalous: 0.5 } }).slice(0, 6);
export const archived = (p, tau) => argmax(p) === 0 && p[0] >= tau;

export function gateStats(examples, escalate) {
  const idx = examples.map((_, i) => i);
  const an = idx.filter((i) => examples[i].label !== "normal"), no = idx.filter((i) => examples[i].label === "normal");
  return {
    escalated: mean(idx.map((i) => (escalate[i] ? 1 : 0))),
    recall: mean(an.map((i) => (escalate[i] ? 1 : 0))),
    false_escalation: mean(no.map((i) => (escalate[i] ? 1 : 0))),
  };
}
export const escAtPrev = (g, prev = 0.01) => prev * g.recall + (1 - prev) * g.false_escalation;

const TAUS = Array.from({ length: 70 }, (_, i) => +(0.3 + i * 0.01).toFixed(2));
// most aggressive tau with recall >= 0.95 on dev predictions. 1.01 means "never archive"
export function pickTau(devExamples, Pdev) {
  for (const tau of TAUS) if (gateStats(devExamples, Pdev.map((p) => !archived(p, tau))).recall >= 0.95) return tau;
  return 1.01;
}
// 5-fold cross-fitted dev predictions, folds by index mod 5
export function crossFit(X, y) {
  const out = new Array(X.length);
  for (let f = 0; f < 5; f++) {
    const tr = X.map((_, i) => i).filter((i) => i % 5 !== f);
    const head = trainLogReg(tr.map((i) => X[i]), tr.map((i) => y[i]), K);
    X.forEach((x, i) => { if (i % 5 === f) out[i] = head(x); });
  }
  return out;
}

export function template(e) {
  const c = e.state.candidate_response;
  switch (e.label) {
    case "normal": return /X-Cache/.test(c) ? "normal: cache headers" : /Set-Cookie/.test(c) ? "normal: new session cookie" : "normal: Vary + ETag";
    case "disclosure": return /X-Backend-Server/.test(c) ? "disclosure: internal hostname in header" : /java\.lang/.test(c) ? "disclosure: Java stack trace" : /\[core\]/.test(c) ? "disclosure: .git/config" : "disclosure: AWS key ID";
    case "parser": return /cached page/.test(c) ? "parser: second response in body" : / 501 /.test(c) ? "parser: 501" : / 505 /.test(c) ? "parser: 505" : "parser: two HTML documents";
    case "connection": return /prematurely/.test(c) ? "connection: upstream closed" : /dial tcp/.test(c) ? "connection: DNS timeout" : / 520 /.test(c) ? "connection: 520" : "connection: truncated body";
    default: return e.label;
  }
}
// structural fingerprints of the kind HTTP Terminator hard-codes (18.7, post hoc)
export function fingerprint(e) {
  const c = parseResponse(e.state.candidate_response);
  return !!c && (/HTTP\/1\.[01] \d{3}/.test(c.body) || (c.body.match(/<html/gi) || []).length > 1);
}
