from __future__ import annotations

import argparse
import json
from pathlib import Path

from kobun_llm.genre_rules import waka_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate waka genre rules.")
    parser.add_argument("--cases", type=Path, default=Path("data/eval/waka_rule_cases.jsonl"))
    parser.add_argument("--min-cases", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    total = 0
    passed = 0
    for line in args.cases.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        result = waka_score(str(row["text"]), reading=str(row["reading"]) if "reading" in row else None)
        expected_meter = tuple(int(value) for value in row.get("expected_meter", ()))
        meter_ok = not expected_meter or result.meter == expected_meter
        min_score_ok = "min_score" not in row or result.score >= int(row["min_score"])
        max_score_ok = "max_score" not in row or result.score <= int(row["max_score"])
        required_reasons = tuple(str(reason) for reason in row.get("required_reasons", ()))
        forbidden_reasons = tuple(str(reason) for reason in row.get("forbidden_reasons", ()))
        reasons_ok = all(reason in result.reasons for reason in required_reasons)
        forbidden_ok = all(reason not in result.reasons for reason in forbidden_reasons)
        ok = meter_ok and min_score_ok and max_score_ok and reasons_ok and forbidden_ok
        total += 1
        passed += int(ok)
        print(
            f"{row['rule_ids']} ok={ok} score={result.score} meter={result.meter} "
            f"reasons={','.join(result.reasons)}"
        )
    print(f"waka_rule_accuracy={passed}/{total}={passed / max(1, total):.3f}")
    if total < args.min_cases:
        raise SystemExit(f"only {total} cases found, below --min-cases {args.min_cases}")
    if passed != total:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
