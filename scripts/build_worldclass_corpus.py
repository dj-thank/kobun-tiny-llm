from __future__ import annotations

import argparse
from pathlib import Path

from validate_corpus import validate_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a broader Kobun corpus with prose, grammar, and train-split-safe rules.")
    parser.add_argument("--base", type=Path, default=Path("data/kobun_labeled_grammar_boost_train.txt"))
    parser.add_argument(
        "--waka",
        type=Path,
        default=None,
        help="Optional extra waka corpus. Refused unless --allow-whole-waka is set because full waka files can leak validation data.",
    )
    parser.add_argument("--waka-meter", type=Path, default=Path("data/waka/waka_meter_corpus.txt"))
    parser.add_argument("--allow-whole-waka", action="store_true")
    parser.add_argument("--aux-rules", type=Path, default=Path("data/grammar/auxiliary_rules.jsonl"))
    parser.add_argument("--genre-rules", type=Path, default=Path("data/grammar/genre_rules.jsonl"))
    parser.add_argument("--external-surfaces", type=Path, default=Path("data/external_knowledge/classical_surface_patterns.txt"))
    parser.add_argument("--out", type=Path, default=Path("data/kobun_worldclass_corpus.txt"))
    parser.add_argument("--waka-repeat", type=int, default=0)
    parser.add_argument("--waka-meter-repeat", type=int, default=8)
    parser.add_argument(
        "--allow-missing-waka-meter",
        action="store_true",
        help="Allow building without data/waka/waka_meter_corpus.txt. Off by default because this removes meter supervision.",
    )
    parser.add_argument("--rule-repeat", type=int, default=6)
    parser.add_argument("--external-surface-repeat", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    parts = [args.base.read_text(encoding="utf-8").strip()]
    if args.waka is not None:
        if not args.allow_whole_waka:
            raise SystemExit(
                "Refusing to append a standalone waka corpus without --allow-whole-waka. "
                "Use the manifest-split training corpus to avoid validation leakage."
            )
        if args.waka.exists():
            waka = args.waka.read_text(encoding="utf-8").strip()
            for index in range(max(0, args.waka_repeat)):
                parts.append(f"時代 中古。ジャンル 和歌。文体 waka。資料 追加和歌。反復 {index + 1}。\n{waka}")
    if args.waka_meter_repeat > 0 and not args.waka_meter.exists():
        if not args.allow_missing_waka_meter:
            raise SystemExit(
                f"Missing required waka meter corpus: {args.waka_meter}. "
                "Run scripts/build_waka_meter_corpus.py first, or pass --allow-missing-waka-meter for an explicit ablation."
            )
    if args.waka_meter_repeat > 0 and args.waka_meter.exists():
        waka_meter = args.waka_meter.read_text(encoding="utf-8").strip()
        for index in range(max(0, args.waka_meter_repeat)):
            parts.append(f"和歌音数制御 五七五七七 反復 {index + 1}\n{waka_meter}")
    aux_rules = args.aux_rules.read_text(encoding="utf-8").strip()
    genre_rules = args.genre_rules.read_text(encoding="utf-8").strip()
    for index in range(max(0, args.rule_repeat)):
        parts.append(f"助動詞接続活用表 {index + 1}\n{aux_rules}")
        parts.append(f"ジャンル別規則表 {index + 1}\n{genre_rules}")
    if args.external_surface_repeat > 0 and args.external_surfaces.exists():
        external_surfaces = args.external_surfaces.read_text(encoding="utf-8").strip()
        for index in range(max(0, args.external_surface_repeat)):
            parts.append(f"古語表面形 {index + 1}\n{external_surfaces}")
    args.out.write_text("\n\n".join(parts) + "\n", encoding="utf-8")
    validate_text(args.out, "training")
    print(f"wrote {args.out} bytes={args.out.stat().st_size}")


if __name__ == "__main__":
    main()
