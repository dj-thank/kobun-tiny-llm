from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from split_policy import SPLIT_POLICY, is_model_split, split_group_key, split_name
from validate_corpus import validate_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a grammar-aware Kobun training corpus.")
    parser.add_argument("--base", type=Path, default=None, help="Unsafe legacy base corpus. Requires --allow-unsafe-base.")
    parser.add_argument("--manifest", type=Path, default=Path("data/corpus_manifest.jsonl"))
    parser.add_argument("--allow-unsafe-base", action="store_true")
    parser.add_argument("--grammar", type=Path, default=Path("data/grammar/kobun_grammar_rules.txt"))
    parser.add_argument("--morph-examples", type=Path, default=Path("data/grammar/morph_examples.txt"))
    parser.add_argument("--out", type=Path, default=Path("data/kobun_labeled_grammar_train.txt"))
    parser.add_argument("--val-out", type=Path, default=Path("data/kobun_labeled_grammar_val.txt"))
    parser.add_argument("--test-out", type=Path, default=Path("data/kobun_labeled_grammar_test.txt"))
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.05)
    parser.add_argument("--grammar-repeat", type=int, default=8)
    parser.add_argument("--morph-repeat", type=int, default=12)
    return parser.parse_args()


def read_manifest_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def split_manifest_rows_three(
    rows: list[dict[str, object]],
    val_ratio: float,
    test_ratio: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    if not 0.0 < val_ratio < 0.5:
        raise SystemExit("--val-ratio must be greater than 0 and less than 0.5.")
    if not 0.0 < test_ratio < 0.5:
        raise SystemExit("--test-ratio must be greater than 0 and less than 0.5.")
    if val_ratio + test_ratio >= 0.8:
        raise SystemExit("--val-ratio + --test-ratio must be less than 0.8.")
    included = [
        row
        for row in rows
        if row.get("include_in_training", True) and is_model_split(split_name(row, val_ratio, test_ratio))
    ]
    if not included:
        raise SystemExit("Manifest has no include_in_training=true rows.")
    train_rows: list[dict[str, object]] = []
    val_rows: list[dict[str, object]] = []
    test_rows: list[dict[str, object]] = []
    for row in included:
        role = split_name(row, val_ratio, test_ratio)
        if role == "test":
            test_rows.append(row)
        elif role == "validation":
            val_rows.append(row)
        elif role == "train":
            train_rows.append(row)
        else:
            raise SystemExit(f"Unsupported model split role for source {row.get('source_id')}: {role}")
    if not train_rows or not val_rows or not test_rows:
        raise SystemExit("Manifest split produced an empty train, validation, or test set; adjust split ratios.")
    groups = {
        "train": {split_group_key(row) for row in train_rows},
        "validation": {split_group_key(row) for row in val_rows},
        "test": {split_group_key(row) for row in test_rows},
    }
    if groups["train"] & groups["validation"] or groups["train"] & groups["test"] or groups["validation"] & groups["test"]:
        raise SystemExit(f"Manifest split is not work/group disjoint under {SPLIT_POLICY}: {groups}")
    return train_rows, val_rows, test_rows


def split_manifest_rows(rows: list[dict[str, object]], val_ratio: float) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    train_rows, val_rows, test_rows = split_manifest_rows_three(rows, val_ratio, test_ratio=0.05)
    return train_rows, val_rows + test_rows


def manifest_text(rows: list[dict[str, object]]) -> str:
    parts = []
    for row in rows:
        clean_file = Path(row["clean_file"])
        text = clean_training_text(clean_file.read_text(encoding="utf-8"))
        header = (
            f"時代 {row['period']}。ジャンル {row['genre']}。"
            f"文体 {row['style']}。作品 {training_header_title(row)}。"
        )
        parts.append(header + "\n" + text)
    return "\n\n".join(parts)


def training_header_title(row: dict[str, object]) -> str:
    work_id = str(row.get("work_id") or "")
    if work_id.startswith("work:"):
        return work_id.removeprefix("work:")
    title = str(row.get("title") or "")
    title = re.sub(r"\s*\([^)]*\)", "", title)
    title = title.replace("/", " ")
    title = re.sub(r"[A-Za-z0-9_+=<>\\[\\]().-]+", "", title)
    title = re.sub(r"\s+", " ", title).strip()
    return title or "古典本文"


def clean_training_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"-{20,}\n.*?(-{20,}\n)", "", text, flags=re.S)
    text = re.sub(r"【テキスト中に現れる記号について】.*?(?:-------------------------------------------------------|\Z)", "", text, flags=re.S)
    text = re.sub(r"底本：.*\Z", "", text, flags=re.S)
    text = re.sub(r"入力：.*\Z", "", text, flags=re.S)
    text = re.sub(r"校正：.*\Z", "", text, flags=re.S)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def main() -> None:
    args = parse_args()
    if args.base is not None:
        if not args.allow_unsafe_base:
            raise SystemExit("Refusing --base unless --allow-unsafe-base is explicit. Use --manifest for training.")
        base = args.base.read_text(encoding="utf-8").strip()
        train_base = base
        val_base = base
        test_base = base
    else:
        rows = read_manifest_rows(args.manifest)
        train_rows, val_rows, test_rows = split_manifest_rows_three(rows, args.val_ratio, args.test_ratio)
        train_base = manifest_text(train_rows).strip()
        val_base = manifest_text(val_rows).strip()
        test_base = manifest_text(test_rows).strip()
    grammar = args.grammar.read_text(encoding="utf-8").strip()
    parts = [train_base]
    for index in range(args.grammar_repeat):
        parts.append(f"文法注入 {index + 1}\n{grammar}")
    if args.morph_examples.exists():
        morph = args.morph_examples.read_text(encoding="utf-8").strip()
        for index in range(args.morph_repeat):
            parts.append(f"形態素注入 {index + 1}\n{morph}")
    write_text(args.out, "\n\n".join(parts) + "\n")
    write_text(args.val_out, val_base + "\n")
    write_text(args.test_out, test_base + "\n")
    validate_text(args.out, "training")
    validate_text(args.val_out, "training")
    validate_text(args.test_out, "training")
    print(f"wrote {args.out} ({args.out.stat().st_size} bytes)")
    print(f"wrote {args.val_out} ({args.val_out.stat().st_size} bytes)")
    print(f"wrote {args.test_out} ({args.test_out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
