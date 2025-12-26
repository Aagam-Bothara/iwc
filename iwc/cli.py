import argparse
import json
from pathlib import Path

import jsonschema


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


def main() -> None:
    parser = argparse.ArgumentParser(prog="iwc")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_val = sub.add_parser("validate", help="Validate workload JSONL against schema.")
    p_val.add_argument("paths", nargs="+", help="JSONL files or directories containing JSONL files.")
    p_val.set_defaults(func=cmd_validate)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
