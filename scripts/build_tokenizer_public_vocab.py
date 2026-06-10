from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from split_policy import SPLIT_POLICY, split_name


TOKENIZER_POLICY = "train_split_plus_core_japanese_inventory_plus_utf8_byte_fallback_v1"

CORE_JAPANESE_RANGES: tuple[tuple[str, int, int], ...] = (
    ("Basic Latin source markup", 0x0020, 0x007E),
    ("General Punctuation source marks", 0x2000, 0x206F),
    ("Arrows source marks", 0x2190, 0x21FF),
    ("Geometric Shapes source marks", 0x25A0, 0x25FF),
    ("CJK Symbols and Punctuation", 0x3000, 0x303F),
    ("Hiragana", 0x3040, 0x309F),
    ("Katakana", 0x30A0, 0x30FF),
    ("Kanbun", 0x3190, 0x319F),
    ("Katakana Phonetic Extensions", 0x31F0, 0x31FF),
    ("Halfwidth and Fullwidth Forms", 0xFF00, 0xFFEF),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a tokenizer character vocabulary without heldout text leakage.")
    parser.add_argument("--manifest", type=Path, default=Path("data/corpus_manifest.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("data/tokenizer_public_char_vocab.txt"))
    parser.add_argument("--meta-out", type=Path, default=Path("data/tokenizer_public_char_vocab.meta.json"))
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.05)
    parser.add_argument("--include-core-japanese-inventory", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def core_japanese_inventory() -> set[str]:
    chars: set[str] = set()
    for _name, start, end in CORE_JAPANESE_RANGES:
        chars.update(chr(codepoint) for codepoint in range(start, end + 1))
    return chars


def manifest_path(value: object) -> Path:
    return Path(str(value).replace("\\", "/"))


def main() -> None:
    args = parse_args()
    chars: set[str] = set()
    train_source_count = 0
    core_chars = core_japanese_inventory() if args.include_core_japanese_inventory else set()
    chars.update(core_chars)
    rows = []
    for line in args.manifest.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    for row in rows:
        if not row.get("include_in_training", True):
            continue
        if split_name(row, args.val_ratio, args.test_ratio) != "train":
            continue
        clean_file = manifest_path(row.get("clean_file", ""))
        if not clean_file.exists():
            raise SystemExit(f"manifest clean_file does not exist: {clean_file}")
        chars.update(clean_file.read_text(encoding="utf-8"))
        train_source_count += 1
    output = "".join(sorted(chars))
    output_bytes = output.encode("utf-8")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(output_bytes)
    meta = {
        "policy": TOKENIZER_POLICY,
        "split_policy": SPLIT_POLICY,
        "byte_fallback": True,
        "byte_fallback_encoding": "utf-8",
        "byte_fallback_tokens": 256,
        "manifest": args.manifest.as_posix(),
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
        "train_source_count": train_source_count,
        "included_manifest_rows": sum(1 for row in rows if row.get("include_in_training", True)),
        "core_inventory_enabled": args.include_core_japanese_inventory,
        "core_inventory_ranges": [
            {"name": name, "start": f"U+{start:04X}", "end": f"U+{end:04X}"}
            for name, start, end in CORE_JAPANESE_RANGES
        ],
        "core_inventory_chars": len(core_chars),
        "direct_vocab_chars": len(chars),
        "estimated_total_vocab_with_byte_fallback_and_unk": len(chars) + 256 + 1,
        "vocab_sha256": hashlib.sha256(output_bytes).hexdigest(),
        "core_inventory_sha256": hashlib.sha256("".join(sorted(core_chars)).encode("utf-8")).hexdigest(),
        "vocab_hash_policy": "sha256_raw_utf8_bytes",
    }
    args.meta_out.parent.mkdir(parents=True, exist_ok=True)
    args.meta_out.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        f"wrote {args.out} chars={len(chars)} train_sources={train_source_count} "
        f"core_inventory={len(core_chars)} byte_fallback_tokens=256 meta={args.meta_out}"
    )


if __name__ == "__main__":
    main()
