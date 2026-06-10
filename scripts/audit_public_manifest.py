from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

from split_policy import grammar_scope as expected_grammar_scope
from split_policy import split_group_key, split_role_for_work


REQUIRED_FIELDS = (
    "source_id",
    "split_key",
    "split_policy",
    "work_id",
    "split_group_key",
    "split_role",
    "grammar_scope",
    "title",
    "source_kind",
    "source_url",
    "license_name",
    "license_note",
    "redistribution_policy",
    "clean_sha256",
    "retrieved_at_utc",
)
ALLOWED_SOURCE_KINDS = {"aozora", "wikisource"}
MODEL_SPLIT_ROLES = {"train", "validation", "test"}
NON_MODEL_SPLIT_ROLES = {"reference", "excluded"}
NON_MODEL_GRAMMAR_SCOPES = {
    "reference_only_outside_genji_era_scope",
    "unregistered_outside_genji_era_scope",
}
ALLOWED_DOMAINS = {
    "aozora": {"www.aozora.gr.jp", "aozora.gr.jp"},
    "wikisource": {"ja.wikisource.org"},
}
PLACEHOLDER_RE = re.compile(r"\b(verify|check|todo|tbd|unknown)\b", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit manifest rows for public release metadata completeness.")
    parser.add_argument("--manifest", type=Path, default=Path("data/corpus_manifest.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("logs/public_manifest_summary.json"))
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Audit without writing the summary artifact. Use -RefreshEvidence to refresh artifacts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for line_number, line in enumerate(args.manifest.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        row["_line"] = line_number
        rows.append(row)

    errors: list[str] = []
    included = [row for row in rows if row.get("include_in_training", True)]
    seen_source_ids: set[str] = set()
    for row in rows:
        source_id = str(row.get("source_id", ""))
        if source_id in seen_source_ids:
            errors.append(f"line={row['_line']} source_id={source_id} duplicate_source_id")
        seen_source_ids.add(source_id)
        missing = [field for field in REQUIRED_FIELDS if not str(row.get(field, "")).strip()]
        if missing:
            errors.append(f"line={row['_line']} source_id={row.get('source_id')} missing={','.join(missing)}")
        source_kind = str(row.get("source_kind", ""))
        if source_kind not in ALLOWED_SOURCE_KINDS:
            errors.append(
                f"line={row['_line']} source_id={row.get('source_id')} unsupported_source_kind={source_kind}"
            )
        parsed = urlparse(str(row.get("source_url", "")))
        if source_kind in ALLOWED_DOMAINS and parsed.netloc not in ALLOWED_DOMAINS[source_kind]:
            errors.append(f"line={row['_line']} source_id={source_id} unexpected_source_domain={parsed.netloc}")
        for field in ("license_name", "license_note"):
            value = str(row.get(field, ""))
            if PLACEHOLDER_RE.search(value):
                errors.append(f"line={row['_line']} source_id={source_id} placeholder_{field}={value!r}")
        if str(row.get("redistribution_policy", "")) != "corpus_text_not_distributed":
            errors.append(
                f"line={row['_line']} source_id={source_id} redistribution_policy={row.get('redistribution_policy')!r}"
            )
        if str(row.get("split_policy", "")) != "work_group_genji_reference_v1":
            errors.append(f"line={row['_line']} source_id={source_id} split_policy={row.get('split_policy')!r}")
        include_in_training = bool(row.get("include_in_training", True))
        work_id = str(row.get("work_id") or "")
        group_key = str(row.get("split_group_key") or "")
        if group_key != split_group_key(row) or group_key != work_id:
            errors.append(
                f"line={row['_line']} source_id={source_id} split_group_key_mismatch "
                f"work_id={work_id!r} split_group_key={group_key!r}"
            )
        split_role = str(row.get("split_role", ""))
        expected_role = split_role_for_work(work_id, include_in_training)
        if split_role != expected_role:
            errors.append(
                f"line={row['_line']} source_id={source_id} split_role_mismatch "
                f"expected={expected_role!r} actual={split_role!r}"
            )
        grammar_scope = str(row.get("grammar_scope", ""))
        expected_scope = expected_grammar_scope(work_id)
        if grammar_scope != expected_scope:
            errors.append(
                f"line={row['_line']} source_id={source_id} grammar_scope_mismatch "
                f"expected={expected_scope!r} actual={grammar_scope!r}"
            )
        if include_in_training:
            if split_role not in MODEL_SPLIT_ROLES:
                errors.append(f"line={row['_line']} source_id={source_id} included_invalid_split_role={split_role!r}")
            if grammar_scope != "genji-era-reference":
                errors.append(
                    f"line={row['_line']} source_id={source_id} included_invalid_grammar_scope={grammar_scope!r}"
                )
        else:
            if split_role not in NON_MODEL_SPLIT_ROLES:
                errors.append(f"line={row['_line']} source_id={source_id} excluded_invalid_split_role={split_role!r}")
            if grammar_scope not in NON_MODEL_GRAMMAR_SCOPES:
                errors.append(
                    f"line={row['_line']} source_id={source_id} excluded_invalid_grammar_scope={grammar_scope!r}"
                )
        if source_kind == "wikisource":
            for field in ("source_revision", "source_revision_timestamp"):
                if not str(row.get(field, "")).strip():
                    errors.append(f"line={row['_line']} source_id={source_id} missing_{field}")
            if not (
                str(row.get("source_payload_sha256", "")).strip()
                or str(row.get("api_payload_sha256", "")).strip()
            ):
                errors.append(f"line={row['_line']} source_id={source_id} missing_wikisource_payload_hash")
        if source_kind == "aozora":
            for field in ("source_payload_sha256", "download_payload_sha256"):
                if not str(row.get(field, "")).strip():
                    errors.append(f"line={row['_line']} source_id={source_id} missing_{field}")
        if str(row.get("style", "")) == "waka":
            for field in ("records_sha256", "readings_sha256", "training_sha256"):
                if not str(row.get(field, "")).strip():
                    errors.append(f"line={row['_line']} source_id={source_id} missing_{field}")

    by_kind = Counter(str(row.get("source_kind", "")) for row in rows)
    included_by_kind = Counter(str(row.get("source_kind", "")) for row in included)
    public_sources = [
        {
            "source_id": row.get("source_id", ""),
            "split_key_sha256": hashlib.sha256(str(row.get("split_key", "")).encode("utf-8")).hexdigest(),
            "split_policy": row.get("split_policy", ""),
            "split_group_sha256": hashlib.sha256(str(row.get("split_group_key", "")).encode("utf-8")).hexdigest(),
            "split_role": row.get("split_role", ""),
            "grammar_scope": row.get("grammar_scope", ""),
            "title": row.get("title", ""),
            "source_kind": row.get("source_kind", ""),
            "source_url": row.get("source_url", ""),
            "license_name": row.get("license_name", ""),
            "license_note": row.get("license_note", ""),
            "redistribution_policy": row.get("redistribution_policy", ""),
            "characters": row.get("characters", 0),
            "source_revision": row.get("source_revision", ""),
            "source_revision_timestamp": row.get("source_revision_timestamp", ""),
            "retrieved_at_utc": row.get("retrieved_at_utc", ""),
            "source_payload_sha256": row.get("source_payload_sha256", ""),
            "download_payload_sha256": row.get("download_payload_sha256", ""),
            "api_payload_sha256": row.get("api_payload_sha256", ""),
            "clean_sha256": row.get("clean_sha256", ""),
            "records_sha256": row.get("records_sha256", ""),
            "readings_sha256": row.get("readings_sha256", ""),
            "training_sha256": row.get("training_sha256", ""),
        }
        for row in rows
    ]
    summary = {
        "manifest": args.manifest.name,
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "total_rows": len(rows),
        "audited_rows": len(rows),
        "included_rows": len(included),
        "excluded_rows": len(rows) - len(included),
        "audited_by_source_kind": dict(sorted(by_kind.items())),
        "included_by_source_kind": dict(sorted(included_by_kind.items())),
        "errors": errors,
        "sources": public_sources,
    }
    out_text = "(not written; check-only)"
    if not args.check_only:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        out_text = str(args.out)
    print(f"manifest_rows={len(rows)} included_rows={len(included)} errors={len(errors)} out={out_text}")
    if errors:
        for error in errors[:20]:
            print(error)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
