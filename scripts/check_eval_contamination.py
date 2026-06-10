from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check whether eval examples appear verbatim in train corpora.")
    parser.add_argument("--train", nargs="+", type=Path, required=True)
    parser.add_argument(
        "--template-train",
        nargs="+",
        type=Path,
        default=[],
        help="Train-side template/preference files checked for short good/bad continuation reuse.",
    )
    parser.add_argument("--eval", nargs="+", type=Path, required=True)
    parser.add_argument("--min-chars", type=int, default=8)
    parser.add_argument(
        "--strict-prompts",
        action="store_true",
        help="Check prompt-only strings even when shorter than --min-chars.",
    )
    parser.add_argument("--allow-hits", action="store_true")
    parser.add_argument("--write-clean-dir", type=Path, default=None)
    return parser.parse_args()


def normalize(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\s+", "", text)
    return text


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        row["_line_no"] = line_no
        rows.append(row)
    return rows


def candidate_texts(row: dict[str, Any]) -> list[tuple[str, str]]:
    values: list[tuple[str, str]] = []
    prompt = str(row.get("prompt", ""))
    if prompt:
        values.append(("prompt", prompt))
    if "good" in row:
        values.append(("good", str(row["good"])))
        values.append(("prompt_good", prompt + str(row["good"])))
    if "bad" in row:
        values.append(("bad", str(row["bad"])))
        values.append(("prompt_bad", prompt + str(row["bad"])))
    for key in ("text", "context", "prefix"):
        if key in row:
            values.append((key, str(row[key])))
    prefixes = row.get("prefixes")
    if isinstance(prefixes, list):
        for index, prefix in enumerate(prefixes):
            values.append((f"prefixes[{index}]", str(prefix)))
    return values


def min_chars_for_label(
    label: str,
    default_min_chars: int,
    strict_fragments: bool,
    strict_prompts: bool,
    row: dict[str, Any],
) -> int:
    is_prompt_only_eval = "good" not in row and "bad" not in row and "text" not in row and "context" not in row
    if strict_prompts and label == "prompt" and is_prompt_only_eval:
        return 1
    if strict_fragments and label in {"good", "bad"}:
        return 1
    if label in {"prompt_good", "prompt_bad", "good", "bad", "text", "context"}:
        return 1 if label not in {"good", "bad"} else default_min_chars
    return default_min_chars


def contaminated_labels(
    row: dict[str, Any],
    train_blobs: list[tuple[Path, str, bool]],
    min_chars: int,
    strict_prompts: bool,
) -> list[str]:
    labels: list[str] = []
    for label, text in candidate_texts(row):
        normalized = normalize(text)
        for train_path, train_text, strict_fragments in train_blobs:
            if len(normalized) < min_chars_for_label(label, min_chars, strict_fragments, strict_prompts, row):
                continue
            if normalized in train_text:
                labels.append(f"{label} in {train_path}")
                break
    return labels


def main() -> None:
    args = parse_args()
    train_blobs = [(path, normalize(path.read_text(encoding="utf-8")), False) for path in args.train]
    train_blobs.extend((path, normalize(path.read_text(encoding="utf-8")), True) for path in args.template_train)
    hits: list[str] = []
    checked = 0

    clean_rows_by_path: dict[Path, list[dict[str, Any]]] = {}
    removed_by_path: dict[Path, int] = {}

    for eval_path in args.eval:
        clean_rows_by_path[eval_path] = []
        removed_by_path[eval_path] = 0
        for row in read_jsonl(eval_path):
            line_no = int(row["_line_no"])
            row_hits = contaminated_labels(row, train_blobs, args.min_chars, args.strict_prompts)
            if row_hits:
                removed_by_path[eval_path] += 1
            else:
                clean_row = {key: value for key, value in row.items() if key != "_line_no"}
                clean_rows_by_path[eval_path].append(clean_row)
            for label, text in candidate_texts(row):
                normalized = normalize(text)
                checked += 1
                for train_path, train_text, strict_fragments in train_blobs:
                    if len(normalized) < min_chars_for_label(label, args.min_chars, strict_fragments, args.strict_prompts, row):
                        continue
                    if normalized in train_text:
                        hits.append(
                            f"{eval_path}:{line_no}:{label} appears in {train_path} "
                            f"snippet={normalized[:40]!r}"
                        )
                        break

    print("eval_contamination_train=" + json.dumps([str(path) for path in args.train], ensure_ascii=False))
    print("eval_contamination_eval=" + json.dumps([str(path) for path in args.eval], ensure_ascii=False))
    print(f"eval_contamination_checked={checked} hits={len(hits)}")
    for hit in hits:
        print("EVAL_CONTAMINATION " + hit)
    if args.write_clean_dir is not None:
        args.write_clean_dir.mkdir(parents=True, exist_ok=True)
        for eval_path, rows in clean_rows_by_path.items():
            out_path = args.write_clean_dir / eval_path.name
            text = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
            out_path.write_text(text + ("\n" if rows else ""), encoding="utf-8", newline="\n")
            print(
                f"wrote_clean_eval={out_path} kept={len(rows)} "
                f"removed={removed_by_path[eval_path]} source={eval_path}"
            )
    if hits and not args.allow_hits:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
