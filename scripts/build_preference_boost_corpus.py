from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a grammar-preference boosted corpus from minimal pairs.")
    parser.add_argument("--base", type=Path, default=Path("data/kobun_labeled_grammar_train.txt"))
    parser.add_argument("--pairs", type=Path, default=Path("data/grammar/train_preference_pairs.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("data/kobun_labeled_grammar_boost_train.txt"))
    parser.add_argument("--repeat", type=int, default=80)
    parser.add_argument("--allow-eval-pairs", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pair_path = args.pairs.resolve()
    eval_dir = Path("data/eval").resolve()
    if eval_dir in pair_path.parents and not args.allow_eval_pairs:
        raise SystemExit(
            f"Refusing to train from evaluation pairs: {args.pairs}. "
            "Move trainable synthetic pairs under data/grammar or pass --allow-eval-pairs only for a deliberate ablation."
        )
    base = args.base.read_text(encoding="utf-8").strip()
    preferred_lines: list[str] = []
    for line in args.pairs.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        prompt = str(row["prompt"])
        good = str(row["good"])
        preferred = prompt + good
        preferred_lines.append(preferred)
    block = "\n".join(preferred_lines)
    parts = [base]
    for idx in range(max(0, args.repeat)):
        parts.append(f"選好注入 {idx + 1}\n{block}")
    args.out.write_text("\n\n".join(parts) + "\n", encoding="utf-8", newline="\n")
    print(
        f"wrote {args.out} base_chars={len(base)} "
        f"pairs={len(preferred_lines)} repeat={args.repeat} bytes={args.out.stat().st_size}"
    )


if __name__ == "__main__":
    main()
