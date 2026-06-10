from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from kobun_llm.genre_rules import waka_meter
from split_policy import split_name


TARGET = (5, 7, 5, 7, 7)


def normalize_reading(value: str) -> str:
    return value.strip().replace("－", "/").replace("|", "/").replace("　", "/")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build slash-separated kana waka readings with exact 5-7-5-7-7 meter.")
    parser.add_argument("--records", type=Path, default=Path("data/waka/waka_records_all.jsonl"))
    parser.add_argument("--manifest", type=Path, default=Path("data/corpus_manifest.jsonl"))
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.05)
    parser.add_argument("--include-validation", action="store_true")
    parser.add_argument("--allow-missing-records", action="store_true")
    parser.add_argument(
        "--allow-records-fallback",
        action="store_true",
        help="Use --records when the manifest is missing. Off by default to avoid split leakage.",
    )
    parser.add_argument("--out", type=Path, default=Path("data/waka/waka_meter_corpus.txt"))
    return parser.parse_args()


def record_lines_from_manifest(
    path: Path,
    val_ratio: float,
    test_ratio: float,
    include_validation: bool,
    allow_missing_records: bool,
) -> list[str]:
    lines: list[str] = []
    missing_records = 0
    waka_rows = 0
    included_rows = 0
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        row = json.loads(raw_line)
        if row.get("style") != "waka" or not row.get("include_in_training", True):
            continue
        waka_rows += 1
        role = split_name(row, val_ratio, test_ratio)
        in_heldout = role in {"validation", "test"}
        if in_heldout and not include_validation:
            continue
        included_rows += 1
        records_file = Path(str(row.get("records_file", "")))
        if records_file.exists():
            lines.extend(records_file.read_text(encoding="utf-8").splitlines())
        else:
            missing_records += 1
    if missing_records and not allow_missing_records:
        raise SystemExit(f"Missing waka records files for {missing_records} manifest rows.")
    if missing_records:
        print(f"warning=missing_waka_records_files count={missing_records}")
    print(f"manifest_waka_rows={waka_rows} train_waka_rows={included_rows} source_record_lines={len(lines)}")
    return lines


def main() -> None:
    args = parse_args()
    if args.manifest.exists():
        source_lines = record_lines_from_manifest(
            args.manifest,
            args.val_ratio,
            args.test_ratio,
            args.include_validation,
            args.allow_missing_records,
        )
    else:
        if not args.allow_records_fallback:
            raise SystemExit(
                f"Manifest is required for split-safe waka meter corpus: {args.manifest}. "
                "Pass --allow-records-fallback only for an explicit leakage-unsafe experiment."
            )
        source_lines = args.records.read_text(encoding="utf-8").splitlines()
    rows = []
    for line in source_lines:
        if not line.strip():
            continue
        row = json.loads(line)
        reading = normalize_reading(str(row.get("reading", "")))
        if reading and waka_meter(reading) == TARGET:
            rows.append(reading)
    if not rows:
        raise SystemExit("No exact-meter readings found.")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"wrote {args.out} rows={len(rows)} bytes={args.out.stat().st_size}")


if __name__ == "__main__":
    main()
