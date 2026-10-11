// Cheap adaptation without touching the model weights: multinomial logistic regression, trained on the
// dev split. Used twice: as a head over Laya's output probabilities, and over TF-IDF features of the
// diff text as a conventional trained-classifier baseline.

// Features are z-scored with the training set's statistics unless standardize is false (TF-IDF vectors are
// already L2-normalised and sparse, so they are left as they are). Optimiser: full-batch gradient descent with
// backtracking, so a step is only taken if it lowers the regularised loss. (The original fixed step of 0.5
// diverged on saturated probabilities, where logits reach +-9.2. See PROTOCOL.md, deviation D1.)
export function trainLogReg(X, y, K, { epochs = 400, lr = 1, l2 = 1e-3, standardize = true, trace = null } = {}) {
  const D = X[0].length, N = X.length;
  const mu = Array.from({ length: D }, (_, d) => (standardize ? X.reduce((s, x) => s + x[d], 0) / N : 0));
  const sd = Array.from({ length: D }, (_, d) => (standardize ? Math.sqrt(X.reduce((s, x) => s + (x[d] - mu[d]) ** 2, 0) / N) || 1 : 1));
  const std = (x) => x.map((v, d) => (v - mu[d]) / sd[d]);
  const Xs = X.map(std);
  let W = Array.from({ length: K }, () => new Float64Array(D + 1)); // last column is the bias
  const lossGrad = (W) => {
    const G = Array.from({ length: K }, () => new Float64Array(D + 1));
    let L = 0;
    for (let n = 0; n < N; n++) {
      const p = softmaxRow(W, Xs[n]);
      L -= Math.log(Math.max(p[y[n]], 1e-300));
      for (let k = 0; k < K; k++) {
        const g = p[k] - (y[n] === k ? 1 : 0);
        for (let d = 0; d < D; d++) G[k][d] += g * Xs[n][d];
        G[k][D] += g;
      }
    }
    let reg = 0;
    for (let k = 0; k < K; k++) for (let d = 0; d <= D; d++) { G[k][d] = G[k][d] / N + (d < D ? l2 * W[k][d] : 0); if (d < D) reg += W[k][d] ** 2; }
    return { L: L / N + (l2 / 2) * reg, G };
  };
  let { L, G } = lossGrad(W);
  for (let e = 0; e < epochs; e++) {
    let step = lr, accepted = false;
    for (let tries = 0; tries < 40 && !accepted; tries++) {
      const Wn = W.map((w, k) => w.map((v, d) => v - step * G[k][d]));
      const r = lossGrad(Wn);
      if (r.L <= L) { W = Wn; L = r.L; G = r.G; accepted = true; } else step /= 2;
    }
    lr = accepted ? Math.min(step * 1.25, 10) : step;
    if (trace) trace.push(L);
    if (!accepted) break;
  }
  const predict = (x) => softmaxRow(W, std(x));
  return Object.assign(predict, { W, mu, sd, loss: L });
}

function softmaxRow(W, x) {
  const z = W.map((w) => { let s = w[x.length]; for (let d = 0; d < x.length; d++) s += w[d] * x[d]; return s; });
  const m = Math.max(...z);
  const e = z.map((v) => Math.exp(v - m));
  const t = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / t);
}

// Laya features: logits of the broad Choice probabilities and of the seven Noul answers
const logit = (p) => { const q = Math.min(1 - 1e-4, Math.max(1e-4, p)); return Math.log(q / (1 - q)); };
export const layaFeatures = (row) => [...row.broad.probs.map(logit), ...row.narrow.p.map(logit), logit(row.narrow.anomalous)];

// TF-IDF over the diff text. Numbers and hex are masked so the classifier cannot key on random IDs
const tokenize = (s) => s.toLowerCase().replace(/\b[0-9a-f]{6,}\b/g, " hex ").replace(/\d+/g, " num ").split(/[^a-z_]+/).filter((t) => t.length > 1);
export function fitTfidf(texts, minDf = 2) {
  const df = new Map();
  texts.forEach((t) => new Set(tokenize(t)).forEach((w) => df.set(w, (df.get(w) ?? 0) + 1)));
  const vocab = [...df.entries()].filter(([, c]) => c >= minDf).map(([w]) => w);
  const index = new Map(vocab.map((w, i) => [w, i]));
  const idf = vocab.map((w) => Math.log((1 + texts.length) / (1 + df.get(w))) + 1);
  return (text) => {
    const v = new Float64Array(vocab.length);
    for (const w of tokenize(text)) { const i = index.get(w); if (i !== undefined) v[i] += 1; }
    let norm = 0;
    for (let i = 0; i < v.length; i++) { v[i] *= idf[i]; norm += v[i] * v[i]; }
    norm = Math.sqrt(norm) || 1;
    return Array.from(v, (x) => x / norm);
  };
}
