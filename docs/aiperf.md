# AIPerf export

IWC compiles datasets into a canonical, engine-agnostic workload JSONL. This repo also provides a thin adapter to export workloads into NVIDIA AIPerf custom dataset formats.

## Supported AIPerf format

Currently supported:

- `CustomDatasetType.SINGLE_TURN` (AIPerf `single_turn`)

Each JSONL line becomes one AIPerf `SingleTurn` record.

AIPerf `SingleTurn` requires at least one modality field. For text-only traces, we emit `text`.

## Field mapping (IWC -> AIPerf SingleTurn)

Given an IWC canonical request with (at minimum):

- `prompt` (string)
- `arrival_time_ms` (int, >= 0)

We emit:

- `type`: `"single_turn"`
- `text`: from `prompt`
- `timestamp` **or** `delay`: derived from `arrival_time_ms` depending on `--time-mode`
- `role`: `"user"`

Notes:
- `--time-mode timestamp` emits absolute `timestamp` in milliseconds.
- `--time-mode delay` emits inter-arrival `delay` in milliseconds (first request delay = 0).
- Extra canonical metadata (e.g., session/turn ids) is not represented in AIPerf `SingleTurn` schema.

## Usage

### Export an AIPerf trace

```bash
python -m iwc.cli export aiperf \
  --input examples/single_request.jsonl \
  --output artifacts/aiperf_single_request.jsonl \
  --time-mode timestamp
