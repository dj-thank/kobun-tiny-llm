from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from build_training_corpus import clean_training_text, read_manifest_rows, split_manifest_rows_three
from split_policy import SPLIT_POLICY
from waka_variant_dedup import WakaVariantIndex, normalize_waka


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check that validation source blocks are absent from a training corpus.")
    parser.add_argument("--manifest", type=Path, default=Path("data/corpus_manifest.jsonl"))
    parser.add_argument("--train", type=Path, default=Path("data/kobun_worldclass_corpus.txt"))
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.05)
    parser.add_argument("--min-snippet", type=int, default=40)
    parser.add_argument("--waka-fuzzy-threshold", type=float, default=0.86)
    return parser.parse_args()


def normalize(text: str) -> str:
    return re.sub(r"\s+", "", text)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_hash(path: Path, expected_hash: str, label: str) -> None:
    if not expected_hash:
        raise SystemExit(f"{label} is missing sha256 in manifest: {path}")
    if not path.exists():
        raise SystemExit(f"{label} is missing: {path}")
    actual_hash = sha256_file(path)
    if actual_hash != expected_hash:
        raise SystemExit(f"{label} hash mismatch: path={path} expected={expected_hash} actual={actual_hash}")


def windows(text: str, min_len: int, width: int = 160, stride: int = 80) -> list[str]:
    values: set[str] = set()
    for line in text.splitlines():
        line = normalize(line)
        if len(line) >= min_len:
            values.add(line)
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = normalize(paragraph)
        if len(paragraph) >= min_len:
            values.add(paragraph)
    normalized = normalize(text)
    if len(normalized) >= min_len:
        if len(normalized) <= width:
            values.add(normalized)
        else:
            for start in range(0, max(1, len(normalized) - width + 1), stride):
                values.add(normalized[start : start + width])
            values.add(normalized[-width:])
    return sorted(values)


def waka_items(row: dict[str, object]) -> list[tuple[str, str]]:
    items: list[tuple[str, str]] = []
    records_raw = str(row.get("records_file") or "")
    readings_raw = str(row.get("readings_file") or "")
    if not records_raw:
        raise SystemExit(f"waka validation source missing records_file: {row.get('source_id')}")
    if not readings_raw:
        raise SystemExit(f"waka validation source missing readings_file: {row.get('source_id')}")
    records_file = Path(records_raw)
    require_hash(records_file, str(row.get("records_sha256") or ""), "validation waka records_file")
    for index, raw_line in enumerate(records_file.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw_line.strip():
            continue
        record = json.loads(raw_line)
        for field in ("poem", "reading"):
            value = normalize_waka(str(record.get(field, "")))
            if value:
                label = f"{row.get('source_id')} records:{records_file.name}:{index}:{field}"
                items.append((label, value))
    readings_file = Path(readings_raw)
    require_hash(readings_file, str(row.get("readings_sha256") or ""), "validation waka readings_file")
    for index, raw_line in enumerate(readings_file.read_text(encoding="utf-8").splitlines(), start=1):
        value = normalize_waka(raw_line)
        if value:
            label = f"{row.get('source_id')} readings:{readings_file.name}:{index}"
            items.append((label, value))
    return items


def role_text(rows: list[dict[str, object]]) -> str:
    parts = []
    for row in rows:
        clean_file = Path(str(row["clean_file"]))
        require_hash(clean_file, str(row.get("clean_sha256") or ""), f"{row.get('source_id')} clean_file")
        parts.append(clean_training_text(clean_file.read_text(encoding="utf-8")))
    return normalize("\n\n".join(parts))


def role_windows(rows: list[dict[str, object]], min_snippet: int) -> list[tuple[str, str]]:
    values: list[tuple[str, str]] = []
    for row in rows:
        clean_file = Path(str(row["clean_file"]))
        require_hash(clean_file, str(row.get("clean_sha256") or ""), f"{row.get('source_id')} clean_file")
        source_text = clean_training_text(clean_file.read_text(encoding="utf-8"))
        for window in windows(source_text, min_snippet):
            values.append((f"{row.get('source_id')} {row.get('title')}", window))
    return values


def role_waka_items(rows: list[dict[str, object]]) -> list[tuple[str, str]]:
    values: list[tuple[str, str]] = []
    for row in rows:
        if row.get("style") == "waka":
            values.extend(waka_items(row))
    return values


def main() -> None:
    args = parse_args()
    rows = read_manifest_rows(args.manifest)
    train_rows, validation_rows, test_rows = split_manifest_rows_three(rows, args.val_ratio, test_ratio=args.test_ratio)
    heldout_rows = validation_rows + test_rows
    train_text = normalize(args.train.read_text(encoding="utf-8"))
    train_text_waka = normalize_waka(args.train.read_text(encoding="utf-8"))
    validation_text = role_text(validation_rows)
    test_text = role_text(test_rows)
    manifest_hash = sha256_file(args.manifest)
    leaks: list[str] = []
    checked = 0
    checked_windows = 0
    checked_waka_items = 0
    waka_leaks = 0
    for row in heldout_rows:
        row_checked = False
        clean_file = Path(str(row["clean_file"]))
        require_hash(clean_file, str(row.get("clean_sha256") or ""), "validation clean_file")
        source_text = clean_training_text(clean_file.read_text(encoding="utf-8"))
        source_windows = windows(source_text, args.min_snippet)
        if not source_windows:
            if row.get("style") != "waka":
                continue
        else:
            row_checked = True
            checked_windows += len(source_windows)
            for window in source_windows:
                if window in train_text:
                    leaks.append(f"{row['source_id']} {row['title']} window_sha256={hashlib.sha256(window.encode('utf-8')).hexdigest()[:16]}")
                    break
        if row.get("style") == "waka":
            items = waka_items(row)
            if not items:
                raise SystemExit(f"waka validation source has no poem/reading items: {row.get('source_id')}")
            row_checked = True
            checked_waka_items += len(items)
            for label, item in items:
                if item in train_text_waka:
                    waka_leaks += 1
                    leaks.append(f"{label} waka_sha256={hashlib.sha256(item.encode('utf-8')).hexdigest()[:16]}")
        if row_checked:
            checked += 1
    role_pair_leaks = 0
    for label, window in role_windows(validation_rows, args.min_snippet):
        if window in test_text:
            role_pair_leaks += 1
            leaks.append(f"validation_vs_test {label} window_sha256={hashlib.sha256(window.encode('utf-8')).hexdigest()[:16]}")
            break
    for label, window in role_windows(test_rows, args.min_snippet):
        if window in validation_text:
            role_pair_leaks += 1
            leaks.append(f"test_vs_validation {label} window_sha256={hashlib.sha256(window.encode('utf-8')).hexdigest()[:16]}")
            break
    role_waka_leaks = 0
    fuzzy_role_waka_leaks = 0
    variant_index = WakaVariantIndex(threshold=args.waka_fuzzy_threshold)
    for role, role_rows in (("train", train_rows), ("validation", validation_rows), ("test", test_rows)):
        for label, item in role_waka_items(role_rows):
            match = variant_index.find_cross_role(role, item)
            if match:
                if match.kind == "exact":
                    role_waka_leaks += 1
                else:
                    fuzzy_role_waka_leaks += 1
                waka_leaks += 1
                leaks.append(
                    f"{match.role}_vs_{role} {label} waka_{match.kind}_sha256={hashlib.sha256(item.encode('utf-8')).hexdigest()[:16]} "
                    f"matched_sha256={hashlib.sha256(match.value.encode('utf-8')).hexdigest()[:16]} similarity={match.ratio:.3f}"
                )
            else:
                variant_index.add(role, label, item)
    print(
        f"split_leakage_checked_sources={checked} expected_sources={len(heldout_rows)} "
        f"split_policy={SPLIT_POLICY} "
        f"val_ratio={args.val_ratio} test_ratio={args.test_ratio} "
        f"checked_windows={checked_windows} checked_waka_items={checked_waka_items} "
        f"role_pair_leaks={role_pair_leaks} role_waka_leaks={role_waka_leaks} "
        f"fuzzy_role_waka_leaks={fuzzy_role_waka_leaks} "
        f"waka_fuzzy_threshold={args.waka_fuzzy_threshold} "
        f"waka_leaks={waka_leaks} leaks={len(leaks)} manifest={args.manifest} "
        f"manifest_sha256={manifest_hash} train={args.train}"
    )
    for leak in leaks:
        print("LEAK " + leak)
    if checked != len(heldout_rows):
        raise SystemExit(f"checked {checked} heldout sources, expected {len(heldout_rows)}")
    if leaks:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
