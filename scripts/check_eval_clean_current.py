from __future__ import annotations

import argparse
import json
from pathlib import Path

from audit_eval_provenance_manifest import REQUIRED_ROLES
from snapshot_eval_files import (
    audited_source_for,
    content_hash,
    load_eval_provenance_manifest,
    read_jsonl,
    sha256_file,
    source_content_hashes,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify data/eval/clean_current is complete and provenance-bound.")
    parser.add_argument("--clean-dir", type=Path, default=Path("data/eval/clean_current"))
    parser.add_argument("--eval-provenance-manifest", type=Path, default=Path("data/eval/eval_provenance_manifest.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.clean_dir.exists():
        raise SystemExit(f"missing_clean_current_dir={args.clean_dir}")
    manifest_sha256, manifest_entries = load_eval_provenance_manifest(args.eval_provenance_manifest)
    entries_by_role = {
        str(entry.get("role")): entry
        for entry in json.loads(args.eval_provenance_manifest.read_text(encoding="utf-8-sig")).get("entries") or []
    }
    missing_roles = sorted(REQUIRED_ROLES - set(entries_by_role))
    if missing_roles:
        raise SystemExit(f"eval_provenance_manifest_missing_roles={missing_roles}")
    checked: list[str] = []
    for role in sorted(REQUIRED_ROLES):
        entry = entries_by_role[role]
        source = Path(str(entry.get("path") or ""))
        clean_path = args.clean_dir / source.name
        if not clean_path.exists():
            raise SystemExit(f"clean_current_missing_role={role} path={clean_path}")
        audited_source, audited_entry = audited_source_for(clean_path, manifest_entries)
        audited_hashes = source_content_hashes(audited_source, audited_entry)
        rows = read_jsonl(clean_path)
        row_hashes = {content_hash(row) for row in rows}
        unknown_hashes = sorted(row_hashes - audited_hashes)
        if unknown_hashes:
            raise SystemExit(f"clean_current_unknown_rows role={role} hashes={unknown_hashes[:3]}")
        if len(row_hashes) != len(rows):
            raise SystemExit(f"clean_current_duplicate_rows role={role} path={clean_path}")
        checked.append(f"{role}:{clean_path.name}:{len(rows)}:{sha256_file(clean_path)}")
    print(
        "eval_clean_current_ok=true "
        f"roles={len(checked)} "
        f"eval_provenance_manifest_sha256={manifest_sha256} "
        f"files={json.dumps(checked, ensure_ascii=False, separators=(',', ':'))}"
    )


if __name__ == "__main__":
    main()
