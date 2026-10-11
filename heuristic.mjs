// The frozen deterministic baseline (unchanged since commit 1fd0247, see PROTOCOL.md section 6.3).
// ---- deterministic heuristic (what a hand-written rule set would do) ----
const LEAK = /Traceback|SQL syntax|SECRET_KEY|\b10\.\d+\.\d+\.\d+|\/srv\/|\/home\/|corp\.local|PHP\/|DEBUG=True/;
export function parseResp(r) {
  if (!r.startsWith("HTTP/")) return { status: 0, headers: {}, body: r };
  const [head, ...rest] = r.split("\n\n");
  const lines = head.split("\n");
  const status = Number(lines[0].split(" ")[1]);
  const headers = Object.fromEntries(lines.slice(1).map((l) => [l.split(": ")[0], l.split(": ").slice(1).join(": ")]));
  return { status, headers, body: rest.join("\n\n") };
}
export function heuristic(state) {
  const b = parseResp(state.baseline_response), c = parseResp(state.candidate_response);
  if (c.status === 0) return "connection";
  if (LEAK.test(c.body) || LEAK.test(c.headers.Server ?? "")) return "disclosure";
  const ignore = new Set(["Date", "X-Request-Id", "Content-Length"]);
  const hb = Object.keys(b.headers).filter((k) => !ignore.has(k)).sort().join();
  const hc = Object.keys(c.headers).filter((k) => !ignore.has(k)).sort().join();
  const ratio = Math.abs(c.body.length - b.body.length) / Math.max(1, b.body.length);
  if (c.status === b.status && hb === hc && ratio < 0.25) return "normal";
  if ([403, 406, 429].includes(c.status)) return "filtering";
  if ([408, 502, 503, 504].includes(c.status)) return "connection";
  if ([400, 413].includes(c.status)) return "parser";
  if (c.status >= 500 || [404, 409, 422].includes(c.status)) return "app_error";
  return "parser";
}

