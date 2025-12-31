# IWC — Inference Workload Characterizer

IWC is a lightweight CLI tool to **analyze, compare, and guard LLM inference workloads**.  
It helps you answer a question that most benchmarks and load tests ignore:

> _“Did my workload shape change — and will that affect inference performance?”_

Instead of raw request counts or average latency, IWC fingerprints workloads using
token behavior, arrival patterns, and session structure — and lets you **fail CI**
when those properties drift.

---

## Why IWC exists

LLM inference performance is dominated by:
- **Prompt length vs output length (prefill vs decode)**
- **Burstiness vs smooth arrivals**
- **Session context growth and reuse**

Two workloads with the same RPS can behave *very* differently on GPUs.

IWC makes those differences explicit and testable.

---

## Key Features

- 🔍 **Analyze workloads** (tokens, arrivals, sessions)
- 🔁 **Diff two workloads** to understand how they changed
- 🧠 **Classify workload type** (bursty API, batch, interactive chat)
- 🚨 **CI regression gating** (`--fail-on-*`)
- 🧪 **Golden tests + CI** for stable metrics
- ⚙️ **Tokenizer-aware** (`simple` or `tiktoken`)

---

## Install

```bash
pip install -e .
pip install tiktoken

Analyze a workload
python -m iwc analyze examples/session_chat_5turns_cumulative.jsonl \
  --tokenizer tiktoken --tokenizer-model gpt-4o-mini
Example output:
WORKLOAD SUMMARY
----------------
Requests           : 5
Tokenizer          : tiktoken:gpt-4o-mini
WORKLOAD TYPE      : smooth, prefill-heavy, high-reuse

TOKENS
-----
Avg prompt tokens  : 51.40
P90 prompt tokens  : 76
Prefill dominance : high

ARRIVAL PROFILE
---------------
Mean RPS           : 2.50
Burstiness (CV)    : 0.00

SESSION ANALYSIS
---------------
Sessions detected  : 1
Avg turns/session  : 5
Prompt reuse ratio : 0.71

Compare two workloads (diff)
python -m iwc diff \
  examples/session_chat_5turns.jsonl \
  examples/session_chat_5turns_cumulative.jsonl \
  --tokenizer tiktoken --tokenizer-model gpt-4o-mini
This answers:

Did prompt length increase?

Did prefill dominate more?

Did the workload shift from batch → chat?

Did burstiness or reuse change?

CI-friendly diff (JSON)
python -m iwc diff A.jsonl B.jsonl \
  --format json \
  --tokenizer tiktoken --tokenizer-model gpt-4o-mini

Regression gating (fail CI if workload shape changes)
python -m iwc diff A.jsonl B.jsonl \
  --fail-on-prefill-delta 0.05 \
  --fail-on-reuse-delta 0.05 \
  --fail-on-burstiness-delta 0.5

If thresholds are exceeded:

Diff is printed

CI fails with exit code 2

This prevents accidental performance regressions due to workload drift.
What the metrics mean (brief)

Prefill dominance
Fraction of tokens spent in prompt processing vs output generation.
High values → memory-bandwidth heavy, KV-cache pressure.

Burstiness (CV)
Coefficient of variation of inter-arrival times.
High values → scheduler stress, latency spikes.

Prompt reuse ratio
How much of each prompt is repeated session context.
High values → chat-like workloads.

Primary workload class

bursty-api

batch/offline

interactive-chat (prefill-heavy)

When to use IWC

Before running expensive inference benchmarks

When changing datasets or prompt construction

In CI to block silent workload regressions

When comparing synthetic vs real traffic traces

Status

✅ Core analyze + diff complete

✅ Golden tests

✅ GitHub Actions CI

🚧 Visualizations and fingerprint export planned
