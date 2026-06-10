from __future__ import annotations

import argparse
import json
from pathlib import Path

from kobun_llm.grammar_constraints import next_char_decision


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate deterministic Kobun grammar next-character constraints.")
    parser.add_argument("--cases", type=Path, default=Path("data/eval/grammar_constraint_cases.jsonl"))
    parser.add_argument("--min-cases", type=int, default=28)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    total = 0
    passed = 0
    for line in args.cases.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        context = str(row["context"])
        decision = next_char_decision(context)
        expected_allowed = tuple(row.get("allowed_next", ()))
        expected_banned = tuple(row.get("banned_next", ()))
        no_decision = bool(row.get("no_decision", False))
        expected_force = row.get("force_allowed")
        allow_extra = bool(row.get("allow_extra_decisions", False))
        if allow_extra:
            allowed_ok = not expected_allowed or all(ch in decision.allowed for ch in expected_allowed)
            banned_ok = not expected_banned or all(ch in decision.banned for ch in expected_banned)
        else:
            allowed_ok = tuple(decision.allowed) == expected_allowed
            banned_ok = tuple(decision.banned) == expected_banned
        no_decision_ok = not no_decision or (not decision.allowed and not decision.banned)
        force_ok = expected_force is None or bool(decision.force_allowed) == bool(expected_force)
        ok = allowed_ok and banned_ok and no_decision_ok and force_ok
        total += 1
        passed += int(ok)
        print(
            f"{row['rule_id']} ok={ok} context={context!r} "
            f"allowed={decision.allowed} banned={decision.banned} force={decision.force_allowed} reason={decision.reason}"
        )
    print(f"grammar_constraint_accuracy={passed}/{total}={passed / max(1, total):.3f}")
    if total < args.min_cases:
        raise SystemExit(f"only {total} cases found, below --min-cases {args.min_cases}")
    if passed != total:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
