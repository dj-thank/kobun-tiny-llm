from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


REQUIRED_ROLES = {
    "grammar_primary",
    "grammar_heldout",
    "morphology_adversarial",
    "grammar_constraints",
    "waka_rules",
    "waka_meter_constraints",
    "waka_generation_prompts",
}
REQUIRED_RELEASE_EVIDENCE_ROLES = {"smoke_regression_gate", "decoder_path_smoke"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit eval/smoke file provenance without reading it into release artifacts.")
    parser.add_argument("--manifest", type=Path, default=Path("data/eval/eval_provenance_manifest.json"))
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        row["_line_no"] = line_no
        rows.append(row)
    return rows


def row_id(row: dict[str, Any]) -> str:
    for key in ("id", "case_id"):
        if row.get(key):
            return str(row[key])
    digest = hashlib.blake2b(
        json.dumps({key: value for key, value in row.items() if key != "_line_no"}, ensure_ascii=False, sort_keys=True).encode("utf-8"),
        digest_size=8,
    ).hexdigest()
    if row.get("rule_id"):
        return f"{row['rule_id']}:{row.get('_line_no', 'unknown')}:{digest}"
    rules = row.get("rule_ids")
    if isinstance(rules, list) and rules:
        return f"{'+'.join(str(item) for item in rules)}:{row.get('_line_no', 'unknown')}:{digest}"
    return f"line_{row.get('_line_no', 'unknown')}:{digest}"


def audit_entry(index: int, entry: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in (
        "path",
        "role",
        "file_sha256",
        "row_count",
        "source_type",
        "authoring_method",
        "copyability_status",
        "release_evidence_role",
        "llm_generated_eval_answer_text",
    ):
        if field not in entry:
            errors.append(f"entry {index} missing {field}")
    role = str(entry.get("role") or "")
    if role not in REQUIRED_ROLES:
        errors.append(f"entry {index} unsupported role={role!r}")
    if str(entry.get("release_evidence_role") or "") not in REQUIRED_RELEASE_EVIDENCE_ROLES:
        errors.append(f"entry {index} unsupported release_evidence_role={entry.get('release_evidence_role')!r}")
    if entry.get("llm_generated_eval_answer_text") is not False:
        errors.append(f"entry {index} must attest llm_generated_eval_answer_text=false")
    path_text = str(entry.get("path") or "")
    path = Path(path_text)
    if path.is_absolute():
        errors.append(f"entry {index} path must be repository-relative: {path_text}")
    if not path.exists():
        errors.append(f"entry {index} missing file: {path_text}")
        return errors
    actual_hash = sha256_file(path)
    if str(entry.get("file_sha256") or "") != actual_hash:
        errors.append(f"entry {index} hash mismatch for {path_text}")
    rows = read_jsonl(path)
    if int(entry.get("row_count", -1)) != len(rows):
        errors.append(f"entry {index} row_count mismatch for {path_text}")
    ids = [row_id(row) for row in rows]
    if len(ids) != len(set(ids)):
        errors.append(f"entry {index} duplicate case ids in {path_text}")
    for line_index, row in enumerate(rows, start=1):
        if row.get("llm_generated_eval_answer_text") is True:
            errors.append(f"{path_text}:{line_index} row claims llm_generated_eval_answer_text=true")
    return errors


def main() -> None:
    args = parse_args()
    payload = read_json(args.manifest)
    errors: list[str] = []
    if not isinstance(payload, dict):
        raise SystemExit("eval provenance manifest must be a JSON object")
    if payload.get("schema") != "old_japanese_eval_provenance_manifest_v1":
        errors.append("manifest schema must be old_japanese_eval_provenance_manifest_v1")
    if payload.get("llm_generated_eval_answer_text") is not False:
        errors.append("manifest must attest llm_generated_eval_answer_text=false")
    entries = payload.get("entries")
    if not isinstance(entries, list) or not entries:
        errors.append("manifest entries must be a non-empty list")
        entries = []
    seen_roles = set()
    manifest_paths = set()
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            errors.append(f"entry {index} is not an object")
            continue
        manifest_paths.add(str(entry.get("path") or "").replace("\\", "/"))
        seen_roles.add(str(entry.get("role") or ""))
        errors.extend(audit_entry(index, entry))
    missing_roles = sorted(REQUIRED_ROLES - seen_roles)
    if missing_roles:
        errors.append(f"missing required eval roles: {missing_roles}")
    eval_root = args.manifest.parent
    if eval_root.exists():
        for direct_jsonl in sorted(eval_root.glob("*.jsonl")):
            rel = direct_jsonl.as_posix()
            if rel not in manifest_paths:
                errors.append(
                    "unmanifested direct eval jsonl must be moved out of data/eval "
                    f"or added to eval provenance manifest: {rel}"
                )
    manifest_sha = sha256_file(args.manifest)
    for error in errors:
        print(f"EVAL_PROVENANCE_ISSUE {error}")
    print(
        "eval_provenance_manifest "
        f"path={args.manifest} "
        f"manifest_sha256={manifest_sha} "
        f"entries={len(entries)} "
        f"errors={len(errors)} "
        "llm_generated_eval_answer_text=false"
    )
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
