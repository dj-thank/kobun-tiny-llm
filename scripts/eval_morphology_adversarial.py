from __future__ import annotations

import argparse
import json
from pathlib import Path

from kobun_llm.morphology import annotate, morphology_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate adversarial morphology cases for substring false positives.")
    parser.add_argument("--cases", type=Path, default=Path("data/eval/morphology_adversarial_cases.jsonl"))
    parser.add_argument("--min-cases", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    total = 0
    passed = 0
    for line in args.cases.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        text = str(row["text"])
        tokens = annotate(text)
        surfaces = [token.surface for token in tokens]
        score = morphology_score(text)
        required = [str(item) for item in row.get("required_surfaces", [])]
        forbidden = [str(item) for item in row.get("forbidden_surfaces", [])]
        required_ok = all(surface in surfaces for surface in required)
        forbidden_ok = all(surface not in surfaces for surface in forbidden)
        min_score = row.get("min_score")
        max_score = row.get("max_score")
        min_ok = min_score is None or score >= int(min_score)
        max_ok = max_score is None or score <= int(max_score)
        ok = required_ok and forbidden_ok and min_ok and max_ok
        total += 1
        passed += int(ok)
        print(f"ok={ok} score={score} text={text!r} surfaces={surfaces}")
    print(f"morphology_adversarial_accuracy={passed}/{total}={passed / max(1, total):.3f}")
    if total < args.min_cases:
        raise SystemExit(f"only {total} cases found, below --min-cases {args.min_cases}")
    if passed != total:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
