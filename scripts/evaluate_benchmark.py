#!/usr/bin/env python3
"""Evaluate labelled Best-Image orderings from a JSON Lines benchmark file."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.evaluation import evaluate_rankings  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="JSONL file containing ranking and preferred_index per line")
    args = parser.parse_args()
    if not args.input.is_file():
        parser.error(f"Input file does not exist: {args.input}")

    cases = []
    try:
        with args.input.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip() or line.lstrip().startswith("#"):
                    continue
                try:
                    cases.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Line {line_number}: invalid JSON ({exc.msg})") from exc
        report = evaluate_rankings(cases)
    except (OSError, ValueError) as exc:
        print(f"Benchmark failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
