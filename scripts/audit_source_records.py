from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


HASH_FIELDS = (
    ("clean_file", "clean_sha256"),
    ("records_file", "records_sha256"),
    ("training_file", "training_sha256"),
    ("readings_file", "readings_sha256"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit source JSON file hashes against the files on disk.")
    parser.add_argument("sources", nargs="+", type=Path)
    parser.add_argument("--fix-clean-sha256", action="store_true", help="Refresh file hash fields in-place.")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_path(value: object) -> Path:
    return Path(str(value).replace("\\", "/"))


def audit_source_file(path: Path, *, fix: bool) -> tuple[int, int, int]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise SystemExit(f"{path} must contain a JSON list")
    checked = 0
    mismatches = 0
    missing = 0
    changed = False
    root = Path.cwd()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise SystemExit(f"{path}: row {index} is not an object")
        for file_key, hash_key in HASH_FIELDS:
            raw_file = str(row.get(file_key) or "")
            recorded_hash = str(row.get(hash_key) or "")
            if not raw_file and not recorded_hash:
                continue
            if not raw_file or not recorded_hash:
                missing += 1
                if not fix:
                    print(f"ISSUE {path}: row={index} missing {file_key} or {hash_key}")
                continue
            file_path = manifest_path(raw_file)
            if not file_path.is_absolute():
                file_path = root / file_path
            if not file_path.exists():
                missing += 1
                print(f"ISSUE {path}: row={index} missing file for {file_key}: {raw_file}")
                continue
            checked += 1
            actual_hash = sha256_file(file_path)
            if actual_hash != recorded_hash:
                mismatches += 1
                if fix:
                    row[hash_key] = actual_hash
                    if file_key == "clean_file":
                        row["characters"] = len(file_path.read_text(encoding="utf-8"))
                    changed = True
                else:
                    print(
                        f"ISSUE {path}: row={index} {hash_key} mismatch "
                        f"recorded={recorded_hash} actual={actual_hash} file={raw_file}"
                    )
    if fix and changed:
        path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return checked, mismatches, missing


def main() -> None:
    args = parse_args()
    total_checked = 0
    total_mismatches = 0
    total_missing = 0
    for source_path in args.sources:
        checked, mismatches, missing = audit_source_file(source_path, fix=args.fix_clean_sha256)
        total_checked += checked
        total_mismatches += mismatches
        total_missing += missing
        print(
            f"source_record_audit path={source_path} checked={checked} "
            f"mismatches={mismatches} missing={missing} fixed={args.fix_clean_sha256}"
        )
    if not args.fix_clean_sha256 and (total_mismatches or total_missing):
        raise SystemExit(
            f"source record audit failed: checked={total_checked} "
            f"mismatches={total_mismatches} missing={total_missing}"
        )


if __name__ == "__main__":
    main()
