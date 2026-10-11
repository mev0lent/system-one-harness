// Question sets sent to Laya. One broad Choice question versus six narrow Noul questions.

export const BROAD = {
  cause: {
    type: "choice",
    instructions: "Compared with the baseline, which explanation best fits the candidate response?",
    criteria: {
      normal: "Same behaviour as the baseline, only dynamic values such as timestamps, IDs or tokens differ",
      filtering: "Blocked or filtered by a firewall, WAF, rate limiter or access policy",
      app_error: "The application itself rejected the input or failed with an ordinary error",
      parser: "A server in the chain parsed or framed the request differently, or returned content belonging to another request",
      connection: "Timeout, reset, closed connection or unreachable upstream",
      disclosure: "Exposes internal details such as stack traces, file paths, internal addresses, credentials or software versions",
    },
  },
};

// keys are aligned with the class labels in dataset.mjs
export const NARROW = {
  normal: { type: "noul", instructions: "Does the candidate response match the baseline apart from dynamic values such as timestamps, IDs or tokens?" },
  filtering: { type: "noul", instructions: "Does the candidate response look like it was blocked or filtered by a firewall, WAF, rate limiter or access policy?" },
  app_error: { type: "noul", instructions: "Does the candidate response show an ordinary application-level error or an application rejecting the input?" },
  parser: { type: "noul", instructions: "Does the candidate response suggest the request was parsed, framed or interpreted differently by a server in the chain, or that it contains content belonging to a different request?" },
  connection: { type: "noul", instructions: "Does the candidate response indicate a timeout, reset, closed connection or unreachable upstream?" },
  disclosure: { type: "noul", instructions: "Does the candidate response expose internal details such as stack traces, file paths, internal addresses, credentials or software versions?" },
};

export const ANOMALOUS = {
  anomalous: { type: "noul", instructions: "Is the candidate response meaningfully different from the baseline in a way that dynamic values such as timestamps, IDs or tokens do not explain?" },
};

// Variant for the candidate-only run: the baseline is dropped and "normal" means an ordinary success.
export const BROAD_CANDIDATE_ONLY = {
  cause: {
    type: "choice",
    instructions: "Which explanation best fits this HTTP response to the request?",
    criteria: {
      normal: "An ordinary successful response with no sign of a problem",
      filtering: "Blocked or filtered by a firewall, WAF, rate limiter or access policy",
      app_error: "The application itself rejected the input or failed with an ordinary error",
      parser: "A server in the chain parsed or framed the request differently, or returned content belonging to another request",
      connection: "Timeout, reset, closed connection or unreachable upstream",
      disclosure: "Exposes internal details such as stack traces, file paths, internal addresses, credentials or software versions",
    },
  },
};
