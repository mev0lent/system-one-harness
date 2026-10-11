// A second synthetic HTTP tier. Template selection and variable data use only g.
export function hard2Candidate(klass, g, res, baseBody) {
  switch (klass) {
    case "normal": {
      let dynamicBody = baseBody;
      if (res.ct === "application/json") {
        const value = JSON.parse(baseBody);
        if (typeof value.updated === "string") {
          value.updated = `2026-09-${String(g.int(10, 28)).padStart(2, "0")}`;
        } else if (typeof value.id === "number") {
          value.id = g.int(1, 999);
        }
        dynamicBody = JSON.stringify(value);
      } else {
        dynamicBody = baseBody.replace(/name="csrf" value="[0-9a-f]+"/, `name="csrf" value="${g.hex(16)}"`);
      }
      return g.pick([
        { template: "normal_policy_allow", status: 200, reason: "OK", ct: res.ct, body: baseBody, extra: { "X-Policy-Evaluation": "allow" } },
        { template: "normal_origin_healthy", status: 200, reason: "OK", ct: res.ct, body: baseBody, extra: { "X-Origin-Health": "healthy" } },
        { template: "normal_public_route", status: 200, reason: "OK", ct: res.ct, body: dynamicBody, extra: { "X-Route-Tag": "public" } },
        { template: "normal_trace_marker", status: 200, reason: "OK", ct: res.ct, body: dynamicBody, extra: { "X-Trace-Marker": g.hex(12) } },
      ]);
    }
    case "filtering":
      return g.pick([
        { template: "filter_legal_policy", status: 451, reason: "Unavailable For Legal Reasons", ct: "text/plain", body: "This resource is unavailable under the site's access policy." },
        { template: "filter_edge_acl", status: 403, reason: "Forbidden", ct: "application/json", body: '{"decision":"deny","enforcer":"edge access-control list"}' },
        { template: "filter_rate_quota", status: 429, reason: "Too Many Requests", ct: "text/plain", body: "The rate limiter rejected this request because the client quota was exceeded." },
        { template: "filter_browser_challenge", status: 200, reason: "OK", ct: "text/html", body: "<html><body><h1>Security check</h1><p>The edge firewall requires a browser challenge before this request is allowed.</p></body></html>" },
      ]);
    case "app_error":
      return g.pick([
        { template: "app_schema_rejection", status: 422, reason: "Unprocessable Content", ct: "application/json", body: '{"error":"Application validation failed: quantity must be positive"}' },
        { template: "app_state_conflict", status: 409, reason: "Conflict", ct: "application/json", body: '{"error":"This item cannot move from archived to pending"}' },
        { template: "app_missing_field", status: 400, reason: "Bad Request", ct: "text/plain", body: "Application rejected input: required field 'sort_order' is absent." },
        { template: "app_transaction_failed", status: 500, reason: "Internal Server Error", ct: "application/json", body: '{"error":"The application could not complete the transaction"}' },
      ]);
    case "parser":
      return g.pick([
        { template: "parser_frame_disagreement", status: 400, reason: "Bad Request", ct: "text/plain", body: "Request framing mismatch: edge used a chunked body; origin used the declared length." },
        { template: "parser_boundary_shift", status: 400, reason: "Bad Request", ct: "text/plain", body: "Request boundary mismatch: origin interpreted trailing bytes as the start of another request." },
        { template: "parser_authority_split", status: 400, reason: "Bad Request", ct: "text/plain", body: "Authority parsing differed between gateway and origin: they selected different request targets." },
        { template: "parser_other_request", status: 200, reason: "OK", ct: "application/json", body: '{"response_for":"GET /synthetic/other-request","note":"The server returned the queued response for a different request"}' },
      ]);
    case "connection":
      return g.pick([
        { template: "connection_no_response", status: 0, reason: "", ct: res.ct, body: "" },
        { template: "connection_upstream_deadline", status: 504, reason: "Gateway Timeout", ct: "text/plain", body: "Upstream did not answer before the gateway deadline." },
        { template: "connection_peer_reset", status: 502, reason: "Bad Gateway", ct: "text/plain", body: "TCP connection to origin was reset before response headers arrived." },
        { template: "connection_dns_unreachable", status: 503, reason: "Service Unavailable", ct: "text/plain", body: "Origin is unreachable: DNS resolution for origin.lab.invalid failed." },
      ]);
    case "disclosure":
      return g.pick([
        { template: "disclosure_source_path", status: 500, reason: "Internal Server Error", ct: "text/plain", body: `Unhandled exception in /opt/synthetic-app/src/checkout/handler.py:${g.int(20, 180)}` },
        { template: "disclosure_private_address", status: 200, reason: "OK", ct: res.ct, body: baseBody, extra: { "X-Diagnostic-Origin": `10.77.${g.int(1, 20)}.${g.int(1, 200)}:8088` } },
        { template: "disclosure_runtime_version", status: 200, reason: "OK", ct: res.ct, body: baseBody, extra: { "X-Internal-Runtime": `synthetic-engine/${g.int(3, 7)}.${g.int(0, 9)}` } },
        { template: "disclosure_stack_frame", status: 500, reason: "Internal Server Error", ct: "text/plain", body: `SyntheticError: lookup failed\n    at CatalogStore.read (/opt/synthetic-app/lib/catalog.js:${g.int(10, 90)})` },
      ]);
    default:
      throw new Error(klass);
  }
}
