// Deterministic preprocessing: turn (baseline, candidate) into a compact diff before the model sees it.
// Dynamic values (digits, hex runs) are masked before comparing, so a changed timestamp or token is
// not reported as a difference. The model then judges the difference instead of having to find it.

const VOLATILE = new Set(["Date", "X-Request-Id", "Content-Length"]);
const MAX_BODY = 600; // characters of candidate body kept when it differs

export function parseResponse(r) {
  if (!r.startsWith("HTTP/")) return null;
  const [head, ...rest] = r.split("\n\n");
  const lines = head.split("\n");
  const [, status, ...reason] = lines[0].split(" ");
  const headers = Object.fromEntries(lines.slice(1).map((l) => [l.split(": ")[0], l.split(": ").slice(1).join(": ")]));
  return { status: Number(status), reason: reason.join(" "), headers, body: rest.join("\n\n") };
}

const mask = (s) => s.replace(/\b[0-9a-f]{6,}\b/gi, "<hex>").replace(/\d+/g, "<n>");

export function diffState(state) {
  const b = parseResponse(state.baseline_response);
  const c = parseResponse(state.candidate_response);
  if (!c) return { request: state.request, status: `${b.status} ${b.reason} -> no HTTP response`, candidate: state.candidate_response };

  const out = { request: state.request };
  out.status = b.status === c.status ? `unchanged (${c.status} ${c.reason})` : `${b.status} ${b.reason} -> ${c.status} ${c.reason}`;

  const added = Object.keys(c.headers).filter((k) => !(k in b.headers));
  const removed = Object.keys(b.headers).filter((k) => !(k in c.headers));
  const changed = Object.keys(c.headers).filter((k) => k in b.headers && !VOLATILE.has(k) && mask(c.headers[k]) !== mask(b.headers[k]));
  out.headers = added.length || removed.length || changed.length ? {} : "unchanged apart from dynamic values";
  if (added.length) out.headers.added = Object.fromEntries(added.map((k) => [k, c.headers[k]]));
  if (removed.length) out.headers.removed = removed;
  if (changed.length) out.headers.changed = Object.fromEntries(changed.map((k) => [k, `${b.headers[k]} -> ${c.headers[k]}`]));

  const pctChange = Math.round(((c.body.length - b.body.length) / Math.max(1, b.body.length)) * 100);
  out.body_length = `${b.body.length} -> ${c.body.length} bytes (${pctChange >= 0 ? "+" : ""}${pctChange}%)`;
  if (mask(c.body) === mask(b.body)) {
    out.body = "unchanged apart from dynamic values";
  } else if (c.body.startsWith(b.body)) {
    out.body = { appended_to_baseline: c.body.slice(b.body.length, b.body.length + MAX_BODY) };
  } else {
    out.body = { candidate: c.body.length ? c.body.slice(0, MAX_BODY) : "(empty)" };
  }
  return out;
}
