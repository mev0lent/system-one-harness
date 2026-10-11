// Batched inference: many states x many questions in one forward pass.
// @receptron/laya only batches the questions of a single state, so this mirrors its systemOne()
// (same sequence layout, temperatures and padding) with the batch dimension extended across states.
import * as ort from "onnxruntime-node";
import { buildSequence, QTYPES, renderOptions, softmax, tempBucket, toInternal } from "./node_modules/@receptron/laya/dist/sequence.js";

const round4 = (x) => Math.round(x * 1e4) / 1e4;

export async function systemOneBatch(laya, states, questions) {
  const qids = Object.keys(questions);
  const qs = qids.map((qid) => toInternal(questions[qid]));
  const items = [];
  states.forEach((state, s) => {
    qs.forEach((q, j) => {
      const { ids, markers } = buildSequence(laya.encode, laya.ids, state, q, laya.config.max_len, laya.config.head_max_len);
      if (markers.length !== renderOptions(q).length) throw new Error(`question ${qids[j]}: options do not fit`);
      items.push({ s, qid: qids[j], q, ids, markers, qtype: QTYPES[q.t] });
    });
  });
  const n = items.length;
  const L = Math.max(...items.map((it) => it.ids.length));
  const K = Math.max(...items.map((it) => it.markers.length));
  const inputIds = new BigInt64Array(n * L).fill(BigInt(laya.ids.pad));
  const attention = new BigInt64Array(n * L);
  const markerPos = new BigInt64Array(n * K);
  const markerMask = new Uint8Array(n * K);
  const qtype = new BigInt64Array(n);
  items.forEach((it, i) => {
    it.ids.forEach((v, j) => { inputIds[i * L + j] = BigInt(v); attention[i * L + j] = 1n; });
    it.markers.forEach((m, j) => { markerPos[i * K + j] = BigInt(m); markerMask[i * K + j] = 1; });
    qtype[i] = BigInt(it.qtype);
  });
  const out = await laya.session.run({
    input_ids: new ort.Tensor("int64", inputIds, [n, L]),
    attention_mask: new ort.Tensor("int64", attention, [n, L]),
    marker_pos: new ort.Tensor("int64", markerPos, [n, K]),
    marker_mask: new ort.Tensor("bool", markerMask, [n, K]),
    qtype: new ort.Tensor("int64", qtype, [n]),
  });
  const logits = out.logits.data;
  const results = states.map(() => ({ answers: {}, usage: { input_tokens: 0 } }));
  items.forEach((it, r) => {
    const k = it.markers.length;
    const temp = laya.config.temperature_by_options[tempBucket(it.qtype, k)] ?? laya.config.temperature[it.qtype] ?? 1;
    const p = softmax(Array.from(logits.subarray(r * K, r * K + k), (v) => v / temp));
    const res = results[it.s];
    res.usage.input_tokens += it.ids.length;
    if (it.q.t === "choice") {
      const keys = Object.keys(it.q.crit);
      res.answers[it.qid] = { type: "choice", choice: keys[p.indexOf(Math.max(...p))], probabilities: Object.fromEntries(keys.map((kk, i) => [kk, round4(p[i] ?? 0)])) };
    } else if (it.q.t === "noul") {
      res.answers[it.qid] = { type: "noul", noul: round4(p[1] ?? 0) };
    } else {
      res.answers[it.qid] = { type: "score", probabilities: Object.fromEntries(p.map((v, i) => [String(i), round4(v)])) };
    }
  });
  return results;
}

// Run a whole list through systemOneBatch in chunks. Sorting by serialized length first keeps
// padding low. Results come back in the original order.
export async function runBatched(laya, states, questions, batchSize = 16) {
  const order = states.map((s, i) => [JSON.stringify(s).length, i]).sort((a, b) => a[0] - b[0]).map((x) => x[1]);
  const out = new Array(states.length);
  for (let i = 0; i < order.length; i += batchSize) {
    const idx = order.slice(i, i + batchSize);
    const res = await systemOneBatch(laya, idx.map((j) => states[j]), questions);
    idx.forEach((j, k) => { out[j] = res[k]; });
  }
  return out;
}
