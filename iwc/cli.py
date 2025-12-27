import argparse
import json
from pathlib import Path

import jsonschema

from iwc.compile import (
    SimpleJsonConfig,
    ShareGPTConfig,
    compile_simple_json,
    compile_sharegpt,
)


# -------------------------
# Validation logic
# -------------------------

def load_schema(schema_path: Path) -> dict:
    return json.loads(schema_path.read_text(encoding="utf-8"))


def validate_jsonl(jsonl_path: Path, schema: dict) -> None:
    validator = jsonschema.Draft202012Validator(schema)
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise SystemExit(f"{jsonl_path}:{line_no}: invalid JSON: {e}") from e

            errors = sorted(validator.iter_errors(obj), key=lambda e: list(e.path))
            if errors:
                msg = "\n".join(
                    f"  - {list(err.path)}: {err.message}"
                    for err in errors
                )
                raise SystemExit(f"{jsonl_path}:{line_no}: schema validation failed:\n{msg}")


def cmd_validate(args: argparse.Namespace) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    schema = load_schema(repo_root / "schema" / "workload.schema.json")

    for p in args.paths:
        path = Path(p)
        if path.is_dir():
            for jsonl in sorted(path.glob("*.jsonl")):
                validate_jsonl(jsonl, schema)
                print(f"OK  {jsonl}")
        else:
            validate_jsonl(path, schema)
            print(f"OK  {path}")


# -------------------------
# Compile handlers
# -------------------------

def _default_manifest_path(output: str, manifest: str | None) -> Path:
    out_path = Path(output)
    return Path(manifest) if manifest else Path(str(out_path) + ".manifest.yaml")


def cmd_compile_simple_json(args: argparse.Namespace) -> None:
    out_path = Path(args.output)
    manifest_path = _default_manifest_path(args.output, args.manifest)

    cfg = SimpleJsonConfig(
        max_output_tokens=args.max_output_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        streaming=args.streaming,
        arrival=args.arrival,
        arrival_step_ms=args.arrival_step_ms,
        rate_rps=args.rate_rps,
        seed=args.seed,
    )

    compile_simple_json(Path(args.input), out_path, manifest_path, cfg)
    print(f"Wrote workload: {out_path}")
    print(f"Wrote manifest: {manifest_path}")


def cmd_compile_sharegpt(args: argparse.Namespace) -> None:
    out_path = Path(args.output)
    manifest_path = _default_manifest_path(args.output, args.manifest)

    cfg = ShareGPTConfig(
        mode=args.mode,
        user_tag=args.user_tag,
        assistant_tag=args.assistant_tag,
        separator=args.separator,
        max_output_tokens=args.max_output_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        streaming=args.streaming,
        arrival=args.arrival,
        arrival_step_ms=args.arrival_step_ms,
        rate_rps=args.rate_rps,
        seed=args.seed,
    )

    compile_sharegpt(Path(args.input), out_path, manifest_path, cfg)
    print(f"Wrote workload: {out_path}")
    print(f"Wrote manifest: {manifest_path}")


# -------------------------
# CLI entrypoint
# -------------------------

def main() -> None:
    parser = argparse.ArgumentParser(prog="iwc")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # ---- compile ----
    p_comp = sub.add_parser("compile", help="Compile a dataset into canonical workload JSONL.")
    comp_sub = p_comp.add_subparsers(dest="compiler", required=True)

    # ===== simple-json =====
    p_sj = comp_sub.add_parser("simple-json", help="Compile a simple JSON list into workload JSONL.")
    p_sj.add_argument("--input", required=True, help="Path to input JSON.")
    p_sj.add_argument("--output", required=True, help="Path to output workload JSONL.")
    p_sj.add_argument("--manifest", default=None, help="Manifest YAML path. Default: <output>.manifest.yaml")

    p_sj.add_argument("--max-output-tokens", type=int, default=128)
    p_sj.add_argument("--temperature", type=float, default=0.0)
    p_sj.add_argument("--top-p", type=float, default=1.0)
    p_sj.add_argument("--streaming", action="store_true")

    p_sj.add_argument("--arrival", choices=["fixed-step", "poisson"], default="fixed-step")
    p_sj.add_argument("--arrival-step-ms", type=int, default=100, help="Used when --arrival fixed-step")
    p_sj.add_argument("--rate-rps", type=float, default=None, help="Required when --arrival poisson")
    p_sj.add_argument("--seed", type=int, default=None, help="Optional seed for arrival randomness")

    p_sj.set_defaults(func=cmd_compile_simple_json)

    # ===== sharegpt =====
    p_sh = comp_sub.add_parser("sharegpt", help="Compile ShareGPT-style JSON into workload JSONL.")
    p_sh.add_argument("--input", required=True, help="Path to ShareGPT JSON (list).")
    p_sh.add_argument("--output", required=True, help="Path to output workload JSONL.")
    p_sh.add_argument("--manifest", default=None, help="Manifest YAML path. Default: <output>.manifest.yaml")

    p_sh.add_argument("--mode", choices=["single-turn", "session"], default="single-turn")
    p_sh.add_argument("--user-tag", default="User", help="Used in session mode transcript formatting")
    p_sh.add_argument("--assistant-tag", default="Assistant", help="Used in session mode transcript formatting")
    p_sh.add_argument("--separator", default="\n", help="Separator between transcript lines in session mode")

    p_sh.add_argument("--max-output-tokens", type=int, default=128)
    p_sh.add_argument("--temperature", type=float, default=0.0)
    p_sh.add_argument("--top-p", type=float, default=1.0)
    p_sh.add_argument("--streaming", action="store_true")

    p_sh.add_argument("--arrival", choices=["fixed-step", "poisson"], default="fixed-step")
    p_sh.add_argument("--arrival-step-ms", type=int, default=100, help="Used when --arrival fixed-step")
    p_sh.add_argument("--rate-rps", type=float, default=None, help="Required when --arrival poisson")
    p_sh.add_argument("--seed", type=int, default=None, help="Optional seed for arrival randomness")

    p_sh.set_defaults(func=cmd_compile_sharegpt)

    # ---- validate ----
    p_val = sub.add_parser("validate", help="Validate workload JSONL against schema.")
    p_val.add_argument("paths", nargs="+", help="JSONL files or directories containing JSONL files.")
    p_val.set_defaults(func=cmd_validate)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
