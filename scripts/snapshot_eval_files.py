from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Copy eval files into a quality-run snapshot and print hash evidence.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--eval-provenance-manifest",
        type=Path,
        default=Path("data/eval/eval_provenance_manifest.json"),
        help="Audited eval provenance manifest used to bind snapshot rows to source files.",
    )
    parser.add_argument(
        "--named",
        action="append",
        required=True,
        help="Role/path pair in the form role=path/to/file.jsonl.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_row(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key != "_line_no"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        row["_line_no"] = line_no
        rows.append(row)
    return rows


def case_id(row: dict[str, Any]) -> str:
    if row.get("id"):
        return str(row["id"])
    if row.get("case_id"):
        return str(row["case_id"])
    rule_ids = row.get("rule_ids") or row.get("labels") or []
    if isinstance(rule_ids, list) and rule_ids:
        prefix = "+".join(str(item) for item in rule_ids)
    else:
        prefix = "row"
    prompt = str(row.get("prompt") or row.get("text") or row.get("context") or "")
    digest = hashlib.blake2b(
        json.dumps(canonical_row(row), ensure_ascii=False, sort_keys=True).encode("utf-8"),
        digest_size=8,
    ).hexdigest()
    return f"{prefix}:{row['_line_no']}:{hashlib.blake2b(prompt.encode('utf-8'), digest_size=4).hexdigest()}:{digest}"


def content_hash(row: dict[str, Any]) -> str:
    return hashlib.blake2b(
        json.dumps(canonical_row(row), ensure_ascii=False, sort_keys=True).encode("utf-8"),
        digest_size=16,
    ).hexdigest()


def safe_role(role: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in role)
    if not cleaned:
        raise SystemExit(f"unsafe empty eval role from {role!r}")
    return cleaned


def load_eval_provenance_manifest(path: Path) -> tuple[str, dict[str, dict[str, Any]]]:
    if not path.exists():
        raise SystemExit(f"eval provenance manifest does not exist: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8-sig"))
    if manifest.get("schema") != "old_japanese_eval_provenance_manifest_v1":
        raise SystemExit(f"unexpected eval provenance manifest schema: {path}")
    if manifest.get("llm_generated_eval_answer_text") is not False:
        raise SystemExit("eval provenance manifest must attest llm_generated_eval_answer_text=false")
    by_path: dict[str, dict[str, Any]] = {}
    for entry in manifest.get("entries") or []:
        if not isinstance(entry, dict):
            raise SystemExit("eval provenance manifest contains a non-object entry")
        entry_path = str(entry.get("path") or "").replace("\\", "/")
        if not entry_path:
            raise SystemExit("eval provenance manifest entry missing path")
        by_path[entry_path] = entry
    return sha256_file(path), by_path


def audited_source_for(source: Path, manifest_entries: dict[str, dict[str, Any]]) -> tuple[Path, dict[str, Any]]:
    rel = source.as_posix()
    if rel in manifest_entries:
        return source, manifest_entries[rel]
    parts = source.parts
    if len(parts) >= 4 and parts[0] == "data" and parts[1] == "eval" and parts[2] in {"clean_current", "clean_current_strict"}:
        candidate = Path("data") / "eval" / Path(*parts[3:])
        candidate_rel = candidate.as_posix()
        if candidate_rel in manifest_entries:
            return candidate, manifest_entries[candidate_rel]
    raise SystemExit(f"eval source is not bound to audited provenance manifest: {source}")


def source_content_hashes(source: Path, expected_entry: dict[str, Any]) -> set[str]:
    if not source.exists():
        raise SystemExit(f"audited eval source does not exist: {source}")
    source_hash = sha256_file(source)
    expected_hash = str(expected_entry.get("file_sha256") or "")
    if source_hash != expected_hash:
        raise SystemExit(f"audited eval source hash mismatch: {source}")
    rows = read_jsonl(source)
    expected_rows = int(expected_entry.get("row_count", -1))
    if len(rows) != expected_rows:
        raise SystemExit(f"audited eval source row_count mismatch: {source}")
    hashes = {content_hash(row) for row in rows}
    if len(hashes) != len(rows):
        raise SystemExit(f"audited eval source has duplicate row content: {source}")
    return hashes


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_sha256, manifest_entries = load_eval_provenance_manifest(args.eval_provenance_manifest)
    for item in args.named:
        if "=" not in item:
            raise SystemExit(f"--named must be role=path, got {item!r}")
        role, raw_path = item.split("=", 1)
        role = safe_role(role)
        source = Path(raw_path)
        if not source.exists():
            raise SystemExit(f"eval source does not exist: {source}")
        destination = args.out_dir / f"{role}{source.suffix}"
        shutil.copy2(source, destination)
        rows = read_jsonl(destination)
        ids = [case_id(row) for row in rows]
        content_hashes = [content_hash(row) for row in rows]
        if len(ids) != len(set(ids)):
            raise SystemExit(f"eval snapshot role={role} has duplicate case ids")
        if len(content_hashes) != len(set(content_hashes)):
            raise SystemExit(f"eval snapshot role={role} has duplicate content hashes")
        audited_source, audited_entry = audited_source_for(source, manifest_entries)
        audited_hashes = source_content_hashes(audited_source, audited_entry)
        unknown_hashes = sorted(set(content_hashes) - audited_hashes)
        if unknown_hashes:
            raise SystemExit(
                f"eval snapshot role={role} contains rows not present in audited source: {unknown_hashes[:3]}"
            )
        removed_from_source = len(audited_hashes) - len(set(content_hashes))
        print(
            "eval_snapshot_file "
            f"role={role} "
            f"path={destination} "
            f"sha256={sha256_file(destination)} "
            f"rows={len(rows)} "
            f"case_ids={json.dumps(ids, ensure_ascii=False, separators=(',', ':'))} "
            f"content_hashes={json.dumps(content_hashes, ensure_ascii=False, separators=(',', ':'))} "
            f"source_sha256={sha256_file(source)} "
            f"audited_source={audited_source} "
            f"audited_source_sha256={audited_entry['file_sha256']} "
            f"eval_provenance_manifest_sha256={manifest_sha256} "
            f"removed_from_source={removed_from_source} "
            f"source={source}"
        )


if __name__ == "__main__":
    main()
