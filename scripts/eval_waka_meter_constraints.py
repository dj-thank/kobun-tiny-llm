from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from kobun_llm.genre_rules import waka_meter
from kobun_llm.tokenizer import CharTokenizer
from kobun_llm.waka_meter_constraints import (
    BOUNDARY,
    FINAL_STOP,
    HIRAGANA,
    WakaMeterLogitsProcessor,
    exact_waka_meter_ok,
    parse_meter_pattern,
    validate_waka_prefix,
)


TARGET = (5, 7, 5, 7, 7)
FULL_WAKA = "あしひきの/ちはやぶるかみ/たらちねの/こころもしらで/ひとをこひけり"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate static waka meter constraint invariants.")
    parser.add_argument("--cases", type=Path, default=Path("data/eval/waka_meter_constraint_cases.jsonl"))
    parser.add_argument("--min-cases", type=int, default=19)
    return parser.parse_args()


def legal_next_chars(tokenizer: CharTokenizer, processor: WakaMeterLogitsProcessor, prefix: str) -> set[str]:
    idx = torch.tensor([tokenizer.encode(prefix)], dtype=torch.long)
    logits = torch.zeros((1, tokenizer.vocab_size), dtype=torch.float32)
    masked = processor(idx, logits)
    legal_ids = torch.isfinite(masked[0]).nonzero(as_tuple=False).flatten().tolist()
    return {tokenizer.itos[int(token_id)] for token_id in legal_ids}


def expect_raises(fn, label: str) -> bool:
    try:
        fn()
    except Exception:
        return True
    print(f"expected_exception_missing={label}")
    return False


def read_cases(path: Path) -> list[dict[str, object]]:
    cases: list[dict[str, object]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        row["_line_no"] = line_no
        cases.append(row)
    if not cases:
        raise SystemExit(f"waka meter constraint cases are empty: {path}")
    return cases


def check_case(
    row: dict[str, object],
    tokenizer: CharTokenizer,
    processor: WakaMeterLogitsProcessor,
) -> tuple[str, bool]:
    case_id = str(row.get("id") or f"line_{row.get('_line_no')}")
    if row.get("llm_generated_eval_answer_text") is not False:
        raise SystemExit(f"waka meter constraint case must attest llm_generated_eval_answer_text=false: {case_id}")
    check = str(row.get("check") or "")
    if check == "parse_meter_pattern":
        expected = tuple(int(item) for item in row.get("expected", []))
        return case_id, parse_meter_pattern(str(row.get("pattern") or "")) == expected
    if check == "waka_meter":
        expected = tuple(int(item) for item in row.get("expected", []))
        return case_id, waka_meter(str(row.get("text") or "")) == expected
    if check == "exact_waka_meter_ok":
        return case_id, exact_waka_meter_ok(str(row.get("text") or "")) is bool(row.get("expected"))
    if check == "valid_prefixes":
        for prefix in row.get("prefixes", []):
            validate_waka_prefix(str(prefix), TARGET, kana_only=True)
        return case_id, True
    if check == "validate_raises":
        return case_id, expect_raises(lambda: validate_waka_prefix(str(row.get("prefix") or ""), TARGET), case_id)
    if check == "legal_chars":
        legal = legal_next_chars(tokenizer, processor, str(row.get("prefix") or ""))
        exact = row.get("exact")
        if isinstance(exact, list):
            return case_id, legal == {str(item) for item in exact}
        contains = row.get("contains") or []
        excludes = row.get("excludes") or []
        return case_id, all(str(item) in legal for item in contains) and all(str(item) not in legal for item in excludes)
    raise SystemExit(f"unsupported waka meter constraint check={check!r} case={case_id}")


def main() -> None:
    args = parse_args()
    tokenizer = CharTokenizer.from_text("".join(sorted(HIRAGANA)) + BOUNDARY + FINAL_STOP + "漢カ")
    processor = WakaMeterLogitsProcessor(tokenizer=tokenizer, target=TARGET, kana_only=True)
    checks = [check_case(row, tokenizer, processor) for row in read_cases(args.cases)]

    passed = sum(1 for _, ok in checks if ok)
    total = len(checks)
    for name, ok in checks:
        print(f"ok={ok} case={name}")
    print(f"waka_meter_constraint_static_accuracy={passed}/{total}={passed / max(1, total):.3f}")
    if total < args.min_cases:
        raise SystemExit(f"only {total} cases found, below --min-cases {args.min_cases}")
    if passed != total:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
