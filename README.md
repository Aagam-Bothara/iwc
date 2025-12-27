# IWC — Inference Workload Compiler

IWC compiles different dataset formats into a **canonical JSONL workload** for LLM inference benchmarking.
It also emits a **manifest** (hashes + config) so runs are reproducible.

## What IWC Produces

### Workload JSONL (one JSON object per line)
Each line matches the schema in `schema/workload.schema.json`:

- `request_id` (string)
- `prompt` (string)
- `max_output_tokens` (int)
- `arrival_time_ms` (int)
- `temperature` (float)
- `top_p` (float)
- `streaming` (bool)

### Manifest YAML
`<output>.manifest.yaml` includes:
- input/output/schema hashes (sha256)
- compiler config
- summary stats (num_requests, arrival_span_ms, skipped_records, etc.)

---

## Install (editable)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
