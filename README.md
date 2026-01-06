# IWC — Inference Workload Compiler & Characterizer

IWC is a focused CLI tool that decouples LLM datasets from inference engines by compiling them into a canonical workload format, and then analyzing and comparing workload behavior to ensure benchmarking is reproducible, comparable, and meaningful.

At its core, IWC answers two critical questions in LLM inference benchmarking:

1. **Can I reproduce this workload exactly?**
2. **Is this workload behaviorally comparable to another one?**

Most tools answer the first only. IWC does both.

---

## Why IWC Exists

While benchmarking LLM inference, the same issues kept appearing:

- Every dataset uses a different structure
- Inference tools assume their own request formats
- Arrival patterns (bursty vs steady vs Poisson) are implicit or undocumented
- Small prompt formatting changes silently alter workload behavior
- Re-running the "same" benchmark weeks later is rarely identical
- Two workloads with the same RPS can stress hardware very differently

IWC was built to make inference workloads **explicit**, **auditable**, and **comparable** — before you ever run a benchmark.

---

## What IWC Does (High Level)

IWC has four tightly connected layers:

### 1. Compile Workloads (Reproducibility)

Convert datasets into a single, schema-validated canonical workload JSONL, plus a manifest that records exactly how it was generated.

### 2. Characterize Workloads (Comparability)

Analyze workload behavior (tokens, arrivals, sessions) and diff two workloads to detect semantic drift — with optional CI gating.

### 3. Calibrate & Predict (Performance Modeling)

Calibrate against real inference servers to build timing models, then predict latency distributions for new workloads without running them.

### 4. Evaluate (Validation)

Compare predictions against actual measurements with statistical rigor (bootstrap CIs, MAPE, RMSE, R²).

---

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install tiktoken
```

---

## Part 1 — Workload Compilation

### Canonical Workload Format (JSONL)

Each line represents one inference request:

```json
{
  "request_id": "req-000001",
  "prompt": "Explain KV cache in one sentence.",
  "max_output_tokens": 128,
  "arrival_time_ms": 0,
  "temperature": 0.0,
  "top_p": 1.0,
  "streaming": false
}
```

The format is validated against: `schema/workload.schema.json`

### Manifest (Reproducibility Metadata)

For every workload, IWC also produces a `.manifest.yaml` file that records:

- SHA256 hashes of inputs, outputs, and schema
- Compiler type and parameters
- Arrival model and seed
- Summary statistics (request count, arrival span, etc.)
- IWC version

This makes benchmarking auditable and reproducible.

### Quickstart (2 Minutes)

```bash
iwc compile simple-json --input data.json --output workload.jsonl
iwc validate workload.jsonl
```

Outputs:
- `workload.jsonl`
- `workload.jsonl.manifest.yaml`

You can now feed `workload.jsonl` into any inference engine.

### Supported Compilers

#### 1. Simple JSON

Accepted input forms:

```json
["prompt1", "prompt2"]
```

```json
[{ "prompt": "..." }]
```

Command:

```bash
iwc compile simple-json --input data.json --output out.jsonl
```

#### 2. ShareGPT

Supports common ShareGPT-style formats:
- `conversations` with `human/gpt` or `user/assistant`
- `messages` arrays with `role/content`

**Single-turn mode:**

```bash
iwc compile sharegpt \
  --input sharegpt.json \
  --output sh_single.jsonl \
  --mode single-turn
```

**Session mode:**

```bash
iwc compile sharegpt \
  --input sharegpt.json \
  --output sh_session.jsonl \
  --mode session \
  --user-tag "User" \
  --assistant-tag "Assistant" \
  --separator "\n"
```

### Arrival Models

IWC explicitly models request arrival patterns.

**Fixed step (default):**

```bash
--arrival fixed-step --arrival-step-ms 100
```

**Poisson arrivals (realistic traffic):**

```bash
--arrival poisson --rate-rps 5 --seed 123
```

Arrivals are seeded for reproducibility.

---

## Part 2 — Workload Analysis

### Analyze a Workload

```bash
iwc analyze workload.jsonl \
  --tokenizer tiktoken --tokenizer-model gpt-4o-mini
```

Example output:

```
WORKLOAD SUMMARY
----------------
Requests  : 5
Tokenizer : tiktoken:gpt-4o-mini

WORKLOAD TYPE : smooth, prefill-heavy, high-reuse
```

This surfaces properties that dominate inference performance:
- Prefill vs decode dominance
- Arrival variability
- Session reuse and context growth

### What the Metrics Mean

| Metric | Description |
|--------|-------------|
| **Prefill dominance** | Fraction of tokens spent in prompt processing vs output generation. High → memory bandwidth and KV-cache pressure. |
| **Burstiness (CV)** | Variability of inter-arrival times. High → scheduler stress and latency spikes. |
| **Prompt reuse ratio** | Fraction of prompt tokens reused across turns. High → chat-like workloads. |

### Primary Workload Classes

- `bursty-api`
- `batch/offline`
- `interactive-chat` (prefill-heavy)

---

## Part 3 — Workload Diffing

### Compare Two Workloads

```bash
iwc diff A.jsonl B.jsonl \
  --tokenizer tiktoken --tokenizer-model gpt-4o-mini
```

### JSON Output (CI / Dashboards)

```bash
iwc diff A.jsonl B.jsonl \
  --format json \
  --tokenizer tiktoken --tokenizer-model gpt-4o-mini
```

### Regression Gating (CI)

```bash
iwc diff A.jsonl B.jsonl \
  --fail-on-prefill-delta 0.05 \
  --fail-on-reuse-delta 0.05 \
  --fail-on-burstiness-delta 0.5
```

If thresholds are exceeded:
- Diff is printed
- Exit code = 2
- CI fails

This prevents silent workload drift that invalidates benchmarks.

---

## Part 4 — Calibration

Calibrate IWC against a live inference server to build a timing model.

### Run Calibration

```bash
iwc calibrate \
  --base-url http://localhost:8000 \
  --model meta-llama/Llama-3-8B \
  --output cal.json \
  --verbose
```

### Calibration Features

- **Robust regression**: Theil-Sen estimator for outlier resistance + OLS for confidence intervals
- **Confidence intervals** on all calibration parameters
- **KV cache pressure** measurement for large contexts
- **Batch scheduling effects** detection
- **Decode variance** characterization (CV for tail latency prediction)
- **Multi-phase warmup** with JIT detection
- **IQR filtering** for outlier removal

### Calibration Output

The `cal.json` file contains:

```json
{
  "cal_version": "2.0",
  "engine": "vllm",
  "model": "meta-llama/Llama-3-8B",
  "prefill_fixed_overhead_ms": 12.5,
  "prefill_ms_per_token": 0.15,
  "decode_fixed_overhead_ms": 8.2,
  "decode_ms_per_token": 22.5,
  "prefill_r_squared": 0.97,
  "decode_r_squared": 0.95,
  "warnings": []
}
```

### Check Calibration Health

```bash
iwc calibrate-check cal.json
```

Reports fit quality, confidence intervals, and warnings.

---

## Part 5 — Prediction

Predict latency distributions for workloads without running them.

### Basic Prediction

```bash
iwc predict \
  --workload workload.jsonl \
  --calibration cal.json \
  --output predictions.json
```

### Prediction with Concurrency

```bash
iwc predict \
  --workload workload.jsonl \
  --calibration cal.json \
  --concurrency 8 \
  --output predictions.json
```

### Prediction Features

- **M/M/c queueing model** with Erlang C formula
- **Kingman approximation** for G/G/c (realistic for LLM inference)
- **Concurrency-aware predictions** with proper queueing delay
- **Log-normal tail latency** modeling (p50/p90/p95/p99)
- **Per-request latency breakdown**: overhead, prefill, decode, KV pressure, batch, queue
- **SLA compliance probability** estimation using CDF
- **Confidence interval propagation** from calibration

### Prediction Output

```
PREDICTION SUMMARY
------------------
Requests     : 100
Concurrency  : 8

Latency Distribution:
  p50  : 245.3 ms
  p90  : 412.7 ms
  p95  : 523.1 ms
  p99  : 891.4 ms

SLA Compliance (500ms): 92.3%
```

---

## Part 6 — Evaluation

Compare predictions against actual measurements.

### Run Evaluation

```bash
iwc eval \
  --workload workload.jsonl \
  --calibration cal.json \
  --base-url http://localhost:8000 \
  --output eval_results.json
```

### Evaluation Features

- **Bootstrap confidence intervals** (1000 samples)
- **MAPE, RMSE, R²** for prediction accuracy
- **Warmup detection** and automatic exclusion
- **Comprehensive tail latency** analysis (p50/p90/p95/p99)
- **Distribution fitting** test (log-normal)
- **Per-request tracking** with detailed results

### Evaluation Output

```
EVALUATION RESULTS
------------------
Requests evaluated: 100

Prediction Accuracy:
  MAPE     : 8.2%
  RMSE     : 45.3 ms
  R²       : 0.94

Latency Comparison:
           Predicted    Actual    Delta
  p50      245.3 ms    251.2 ms   +2.4%
  p90      412.7 ms    398.5 ms   -3.4%
  p95      523.1 ms    545.8 ms   +4.3%
  p99      891.4 ms    923.1 ms   +3.6%
```

---

## Part 7 — Fingerprinting

Generate compact, comparable fingerprints for workloads.

### Generate Fingerprint

```bash
iwc fingerprint workload.jsonl \
  --tokenizer tiktoken --tokenizer-model gpt-4o-mini \
  --output fingerprint.json
```

### Extended Fingerprint

```bash
iwc fingerprint workload.jsonl \
  --tokenizer tiktoken --tokenizer-model gpt-4o-mini \
  --extended \
  --output fingerprint_extended.json
```

### Fingerprint Features

- **Distribution descriptors**: skewness, kurtosis, Gini coefficient
- **Burstiness metrics**: CV of inter-arrival times, burstiness index
- **Token distribution histograms** and multimodal detection
- **Shannon entropy** for diversity metrics
- **Correlation** between prompt/output lengths
- **Complexity classification**: size, variability, arrival pattern
- **Fingerprint comparison** with similarity scoring

### Compare Fingerprints

```bash
iwc fingerprint-compare fp1.json fp2.json
```

---

## Validation

```bash
iwc validate workload.jsonl
iwc validate ./folder_with_jsonl_files/
```

---

## Status & Roadmap

### Completed

- ✅ Canonical workload compilation
- ✅ Schema validation + manifests
- ✅ Workload analysis and classification
- ✅ Workload diffing
- ✅ CI regression gating
- ✅ Golden tests + GitHub Actions CI
- ✅ Server calibration with robust regression
- ✅ Latency prediction with queueing models
- ✅ Evaluation with bootstrap CIs
- ✅ Workload fingerprinting

### Planned

- ⬚ Additional dataset adapters (Alpaca, MT-Bench, OpenAI logs)
- ⬚ Runner integrations (vLLM, TGI)
- ⬚ Optional visualization exports
- ⬚ Cost prediction ($/1k tokens)

---

## About

**IWC (Inference Workload Compiler & Characterizer)** converts LLM datasets into a canonical, reproducible inference workload format with explicit arrival models and manifest-based provenance. It also provides calibration, prediction, and evaluation capabilities for performance modeling.

## License

MIT License