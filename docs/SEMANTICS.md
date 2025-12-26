\# IWC Semantics v0.1



\## Canonical Workload

A workload is a JSONL file. Each line is one request object.



`arrival\_time\_ms` is relative to the workload start (t=0). A runner MUST submit each request at its specified time offset.



\## Fields



\### Identity

\- request\_id (string, required): unique id for this request.

\- session\_id (string|null): groups multi-turn requests.

\- turn\_id (int|null): turn index within a session.



\### Prompt

\- prompt (string, required): raw prompt text passed to the backend.

\- prompt\_format (enum): raw | chatml | openai\_messages (advisory for adapters).



\### Generation

\- max\_output\_tokens (int, required): upper bound on generated tokens.

\- stop (string\[]): stop sequences (backend support may vary).



\### Sampling

\- temperature (float, default=0.0): default deterministic.

\- top\_p (float, default=1.0)

\- seed (int|null): if supported.



\### Timing \& control

\- arrival\_time\_ms (int, required): when this request becomes eligible for submission.

\- streaming (bool, default=false): whether client expects streaming tokens.

\- cancel\_after\_ms (int|null): cancel request after this time since submission (if supported).



\## Metric Semantics (v0.1)

\- TTFT: time from backend-receive to first generated token emission (server-side if possible).

\- E2E latency: time from request submission to final token (or stream end).

\- ITL: average inter-token gap during decode (streaming runs only).

If an adapter cannot produce timestamps required by a metric definition, it MUST mark the metric as unsupported rather than approximating silently.



