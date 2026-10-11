// Synthetic HTTP-anomaly dataset with ground-truth labels by construction.
// Every example is a (request, baseline_response, candidate_response) triple.
// "easy" candidates carry explicit wording, "subtle" candidates avoid the obvious keywords.
import { hard2Candidate } from "./hard2_templates.mjs";

export const CLASSES = ["normal", "filtering", "app_error", "parser", "connection", "disclosure"];

function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function makeRng(seed) {
  const r = mulberry32(seed);
  const pick = (arr) => arr[Math.floor(r() * arr.length)];
  const int = (lo, hi) => lo + Math.floor(r() * (hi - lo + 1));
  const hex = (n) => Array.from({ length: n }, () => "0123456789abcdef"[int(0, 15)]).join("");
  return { r, pick, int, hex };
}

const SERVERS = ["nginx/1.24.0", "Apache/2.4.58 (Ubuntu)", "envoy", "gunicorn/21.2.0", "Caddy", "openresty/1.21.4.3"];
const PATHS = ["/api/users/42", "/api/orders/1187", "/login", "/search?q=shoes", "/api/v2/invoices", "/account/settings", "/products/88", "/api/cart"];
const MUTATIONS = [
  "",
  "Transfer-Encoding: chunked\nContent-Length: 4",
  "X-Forwarded-Host: internal.example\n",
  "Content-Length: 0\nContent-Length: 12",
  "Connection: keep-alive, Upgrade",
  "Host: example.com:80\nHost: example.com",
  "X-Original-URL: /admin",
];
const RESOURCES = [
  (g) => ({ ct: "application/json", body: `{"id": ${g.int(1, 999)}, "name": "${g.pick(["Alice", "Bruno", "Chen", "Dara"])}", "plan": "${g.pick(["free", "pro", "team"])}", "updated": "2026-09-${g.int(10, 28)}"}` }),
  (g) => ({ ct: "text/html", body: `<html><head><title>${g.pick(["Dashboard", "Orders", "Settings"])}</title></head><body><h1>${g.pick(["Welcome back", "Your orders", "Account"])}</h1><p>${g.pick(["3 items in your cart", "No new notifications", "Last login yesterday"])}</p><input type="hidden" name="csrf" value="${g.hex(16)}"></body></html>` }),
  (g) => ({ ct: "application/json", body: `{"items": [{"sku": "A-${g.int(100, 999)}", "qty": ${g.int(1, 5)}}, {"sku": "B-${g.int(100, 999)}", "qty": ${g.int(1, 5)}}], "total": ${g.int(20, 400)}.${g.int(10, 99)}}` }),
];

function render(status, reason, headers, body) {
  const head = Object.entries(headers).map(([k, v]) => `${k}: ${v}`).join("\n");
  return `HTTP/1.1 ${status} ${reason}\n${head}\n\n${body}`;
}

function baseHeaders(g, server, ct, body, extra = {}) {
  return {
    Date: `Tue, ${g.int(1, 28)} Sep 2026 ${String(g.int(0, 23)).padStart(2, "0")}:${String(g.int(0, 59)).padStart(2, "0")}:${String(g.int(0, 59)).padStart(2, "0")} GMT`,
    "Content-Type": ct,
    "Content-Length": String(body.length),
    Server: server,
    "X-Request-Id": g.hex(12),
    ...extra,
  };
}

function bodyGen(klass, tier, g, baseBody) {
  switch (klass) {
    case "filtering":
      return tier === "easy"
        ? { status: 403, reason: "Forbidden", body: g.pick(["Request blocked by security policy. Reference ID: " + g.hex(8), "Access denied by web application firewall. Incident " + g.hex(6), "Attention Required! Your request was blocked (rule 942100)."]) }
        : g.pick([
            { status: 403, reason: "Forbidden", body: "Forbidden" },
            { status: 406, reason: "Not Acceptable", body: "Not Acceptable" },
            { status: 200, reason: "OK", body: "<html><body><h1>One moment please</h1><p>Please complete the verification to continue.</p></body></html>" },
            { status: 429, reason: "Too Many Requests", body: "Slow down." },
          ]);
    case "app_error":
      return tier === "easy"
        ? { status: 500, reason: "Internal Server Error", body: g.pick(["Internal Server Error", "Something went wrong. Error code E" + g.int(1000, 9999), '{"error": "unexpected failure", "code": ' + g.int(1000, 9999) + "}"]) }
        : g.pick([
            { status: 422, reason: "Unprocessable Entity", body: `{"error": "field 'id' must be an integer"}` },
            { status: 404, reason: "Not Found", body: `{"error": "resource does not exist"}` },
            { status: 400, reason: "Bad Request", body: `{"error": "missing required parameter 'page'"}` },
            { status: 409, reason: "Conflict", body: `{"error": "version mismatch, reload and retry"}` },
          ]);
    case "parser":
      return tier === "easy"
        ? { status: 400, reason: "Bad Request", body: g.pick(["Requested protocol is unsupported by this upstream.", "Invalid chunk size in request body.", "Bad request: conflicting Content-Length headers received."]) }
        : g.pick([
            { status: 200, reason: "OK", body: `{"id": ${g.int(1000, 1999)}, "name": "someone else", "plan": "team", "updated": "2026-08-01", "note": "response to a different request"}` },
            { status: 200, reason: "OK", body: "<html><head><title>Admin console</title></head><body><h1>Admin</h1><p>User management</p></body></html>" },
            { status: 400, reason: "Bad Request", body: "Request header or cookie too large" },
            { status: 413, reason: "Payload Too Large", body: "" },
          ]);
    case "connection":
      return tier === "easy"
        ? g.pick([
            { status: 504, reason: "Gateway Timeout", body: "upstream request timeout" },
            { status: 502, reason: "Bad Gateway", body: "upstream connect error or disconnect/reset before headers. reset reason: connection termination" },
            { status: 503, reason: "Service Unavailable", body: "no healthy upstream" },
          ])
        : g.pick([
            { status: 408, reason: "Request Timeout", body: "" },
            { status: 502, reason: "Bad Gateway", body: "" },
            { status: 503, reason: "Service Unavailable", body: "Try again later." },
            { status: 0, reason: "", body: "" },
          ]);
    case "disclosure":
      return tier === "easy"
        ? g.pick([
            { status: 500, reason: "Internal Server Error", body: `Traceback (most recent call last):\n  File "/srv/app/views.py", line ${g.int(20, 400)}, in get\n    user = db.query(sql)\nKeyError: 'id'` },
            { status: 500, reason: "Internal Server Error", body: `You have an error in your SQL syntax near '"' at line 1 (mysql 8.0.${g.int(20, 36)}, host 10.0.${g.int(1, 9)}.${g.int(2, 200)})` },
            { status: 200, reason: "OK", body: `DEBUG=True\nSECRET_KEY=${g.hex(24)}\nDATABASE_URL=postgres://app:${g.hex(8)}@10.0.${g.int(1, 9)}.${g.int(2, 200)}/prod` },
          ])
        : g.pick([
            { status: 404, reason: "Not Found", body: `No route matches /internal/v${g.int(1, 3)} on node app-${g.int(1, 9)}.corp.local` },
            { status: 200, reason: "OK", body: baseBody + `<!-- build ${g.hex(7)} deployed from /home/deploy/releases/${g.int(100, 400)} -->` },
            { status: 400, reason: "Bad Request", body: `{"error": "invalid input", "upstream": "http://10.${g.int(0, 3)}.${g.int(0, 9)}.${g.int(2, 90)}:8080/validate"}` },
          ]);
    default:
      throw new Error(klass);
  }
}

// filler placed BEFORE the discriminating content, to probe context truncation
function filler(g, words) {
  const w = ["dashboard", "analytics", "report", "metric", "summary", "overview", "monthly", "customer", "region", "segment", "trend", "quarter"];
  return Array.from({ length: words }, () => g.pick(w)).join(" ");
}

export function generate({ perClass = 100, seed = 1337, tierMix = 0.5, fillerWords = 0 } = {}) {
  const g = makeRng(seed);
  const examples = [];
  for (const klass of CLASSES) {
    for (let i = 0; i < perClass; i++) {
      const server = g.pick(SERVERS);
      const path = g.pick(PATHS);
      const mutation = g.pick(MUTATIONS);
      const res = g.pick(RESOURCES)(g);
      const baseBody = res.body;
      const baseline = render(200, "OK", baseHeaders(g, server, res.ct, baseBody), baseBody);
      const tier = g.r() < tierMix ? "easy" : "subtle";
      let candidate;
      if (klass === "normal") {
        const varBody = g.pick(RESOURCES)(g).body;
        const nb = res.ct === "text/html" ? baseBody.replace(/value="[0-9a-f]+"/, `value="${g.hex(16)}"`) : baseBody.replace(/\d+/g, (m) => (g.r() < 0.3 ? String(Number(m) + g.int(1, 3)) : m));
        const body = g.r() < 0.2 ? varBody.length > 0 && res.ct === "application/json" ? baseBody : nb : nb;
        candidate = render(200, "OK", baseHeaders(g, server, res.ct, body), body);
      } else {
        const c = bodyGen(klass, tier, g, baseBody);
        if (c.status === 0) {
          candidate = "(connection closed before any response was received, 0 bytes)";
        } else {
          const body = fillerWords > 0 && c.body ? `<div>${filler(g, fillerWords)}</div>\n${c.body}` : c.body;
          const serverHdr = klass === "disclosure" && tier === "subtle" && g.r() < 0.4 ? "Apache/2.4.29 (Ubuntu) PHP/7.2.24" : server;
          candidate = render(c.status, c.reason, baseHeaders(g, serverHdr, c.status === 200 ? res.ct : "text/plain", body), body);
        }
      }
      if (klass === "normal" && fillerWords > 0) {
        // keep normal examples comparable in length
        candidate = candidate.replace("\n\n", `\n\n<div>${filler(g, fillerWords)}</div>\n`);
      }
      examples.push({
        id: `${klass}-${i}`,
        label: klass,
        tier: klass === "normal" ? "n/a" : tier,
        state: {
          request: `GET ${path} HTTP/1.1${mutation ? "\n" + mutation : ""}`,
          baseline_response: baseline,
          candidate_response: candidate,
        },
      });
    }
  }
  return examples;
}

// ---- hard tier ----
// Written after the heuristic in analyze.mjs was frozen. The wording and status codes deliberately sit
// outside its rules, the way a new anomaly sits outside hard-coded fingerprints. "normal" here carries
// real benign variation (reordered or different list items, cache and cookie headers) instead of only
// changed numbers.
function hardCandidate(klass, g, res, baseBody) {
  switch (klass) {
    case "normal": {
      let body = baseBody;
      if (res.ct === "application/json") {
        const o = JSON.parse(baseBody);
        const keys = Object.keys(o).reverse();
        const r = Object.fromEntries(keys.map((k) => [k, o[k]]));
        if (Array.isArray(r.items)) r.items = [...r.items, { sku: `C-${g.int(100, 999)}`, qty: g.int(1, 5) }, { sku: `D-${g.int(100, 999)}`, qty: g.int(1, 5) }];
        else r.recently_viewed = [g.int(1, 99), g.int(1, 99), g.int(1, 99)];
        body = JSON.stringify(r);
      } else {
        body = baseBody.replace("</h1>", `</h1><div class="banner">${g.pick(["Free shipping this week", "New: dark mode", "We updated our privacy policy"])}</div>`);
      }
      const extra = g.pick([{ Age: String(g.int(1, 300)), "X-Cache": "HIT" }, { "Set-Cookie": `session=${g.hex(20)}; Path=/; HttpOnly` }, { Vary: "Accept-Encoding", ETag: `"${g.hex(10)}"` }]);
      return { status: 200, reason: "OK", ct: res.ct, body, extra };
    }
    case "filtering":
      return g.pick([
        { status: 200, reason: "OK", ct: "text/html", body: "<html><body><h1>Your access to this site has been limited by the site owner</h1><p>If you think this is a mistake, contact the administrator.</p></body></html>" },
        { status: 200, reason: "OK", ct: "text/html", body: "<html><body><h2>Pardon our interruption</h2><p>Something about your browser made us think you were a bot.</p></body></html>" },
        { status: 401, reason: "Unauthorized", ct: "application/json", body: '{"message": "client not allowed from this network"}' },
        { status: 503, reason: "Service Unavailable", ct: "text/html", body: "<html><body>Checking your browser before accessing the site. This process is automatic.</body></html>" },
      ]);
    case "app_error":
      return g.pick([
        { status: 200, reason: "OK", ct: "application/json", body: `{"success": false, "error": "invalid date format for 'from', expected YYYY-MM-DD"}` },
        { status: 200, reason: "OK", ct: "application/json", body: `{"ok": false, "message": "order ${g.int(1000, 9999)} is already cancelled"}` },
        { status: 405, reason: "Method Not Allowed", ct: "application/json", body: '{"error": "use POST for this endpoint"}' },
        { status: 400, reason: "Bad Request", ct: "application/json", body: `{"errors": [{"field": "email", "message": "must be a valid address"}]}` },
      ]);
    case "parser":
      return g.pick([
        { status: 200, reason: "OK", ct: "text/html", body: baseBody + "HTTP/1.1 200 OK\nContent-Type: text/html\n\n<html><body>cached page</body></html>" },
        { status: 501, reason: "Not Implemented", ct: "text/plain", body: "Unsupported transfer coding" },
        { status: 505, reason: "HTTP Version Not Supported", ct: "text/plain", body: "" },
        { status: 200, reason: "OK", ct: "text/html", body: "<html><body><p>Order summary</p></body></html>\n<html><head><title>Account</title></head><body><h1>Welcome back</h1>" },
      ]);
    case "connection":
      return g.pick([
        { status: 200, reason: "OK", ct: res.ct, body: baseBody.slice(0, Math.floor(baseBody.length / 3)) },
        { status: 500, reason: "Internal Server Error", ct: "text/plain", body: "upstream prematurely closed connection while reading response header" },
        { status: 500, reason: "Internal Server Error", ct: "text/plain", body: "dial tcp: lookup orders-svc: i/o timeout" },
        { status: 520, reason: "Unknown Error", ct: "text/html", body: "<html><body>Web server is returning an unknown error</body></html>" },
      ]);
    case "disclosure":
      return g.pick([
        { status: 200, reason: "OK", ct: res.ct, body: baseBody, extra: { "X-Backend-Server": `ip-172-31-${g.int(1, 60)}-${g.int(2, 250)}.eu-west-1.compute.internal` } },
        { status: 500, reason: "Internal Server Error", ct: "text/plain", body: `java.lang.NullPointerException\n\tat com.acme.billing.InvoiceService.load(InvoiceService.java:${g.int(40, 300)})\n\tat com.acme.billing.InvoiceController.get(InvoiceController.java:${g.int(20, 90)})` },
        { status: 200, reason: "OK", ct: "text/plain", body: `[core]\n\trepositoryformatversion = 0\n[remote "origin"]\n\turl = git@git.acme-internal.net:billing/api.git` },
        { status: 400, reason: "Bad Request", ct: "application/json", body: `{"error": "signature mismatch", "debug": {"aws_access_key_id": "AKIA${g.hex(16).toUpperCase()}", "region": "eu-west-1"}}` },
      ]);
    default:
      throw new Error(klass);
  }
}

export function generateHard({ perClass = 50, seed = 4242 } = {}) {
  const g = makeRng(seed);
  const examples = [];
  for (const klass of CLASSES) {
    for (let i = 0; i < perClass; i++) {
      const server = g.pick(SERVERS);
      const path = g.pick(PATHS);
      const mutation = g.pick(MUTATIONS);
      const res = g.pick(RESOURCES)(g);
      const baseline = render(200, "OK", baseHeaders(g, server, res.ct, res.body), res.body);
      const c = hardCandidate(klass, g, res, res.body);
      const candidate = render(c.status, c.reason, baseHeaders(g, server, c.ct, c.body, c.extra ?? {}), c.body);
      examples.push({
        id: `hard-${klass}-${i}`,
        label: klass,
        tier: "hard",
        state: { request: `GET ${path} HTTP/1.1${mutation ? "\n" + mutation : ""}`, baseline_response: baseline, candidate_response: candidate },
      });
    }
  }
  return examples;
}

export function generateHard2({ perNormal = 100, perAnomaly = 30, seed = 9090 } = {}) {
  const g = makeRng(seed);
  const examples = [];
  for (const klass of CLASSES) {
    const count = klass === "normal" ? perNormal : perAnomaly;
    for (let i = 0; i < count; i++) {
      const server = g.pick(SERVERS);
      const path = g.pick(PATHS);
      const mutation = g.pick(MUTATIONS);
      const res = g.pick(RESOURCES)(g);
      const baseline = render(200, "OK", baseHeaders(g, server, res.ct, res.body), res.body);
      const c = hard2Candidate(klass, g, res, res.body);
      const candidate = c.status === 0
        ? "(connection closed before any response was received, 0 bytes)"
        : render(c.status, c.reason, baseHeaders(g, server, c.ct, c.body, c.extra ?? {}), c.body);
      examples.push({
        id: `hard2-${klass}-${i}`,
        label: klass,
        tier: "hard2",
        template: c.template,
        state: { request: `GET ${path} HTTP/1.1${mutation ? "\n" + mutation : ""}`, baseline_response: baseline, candidate_response: candidate },
      });
    }
  }
  return examples;
}
