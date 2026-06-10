from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


HASH_FIELDS = (
    "clean_sha256",
    "source_payload_sha256",
    "download_payload_sha256",
    "api_payload_sha256",
    "records_sha256",
    "readings_sha256",
    "training_sha256",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score public-source metadata quality without emitting corpus text.")
    parser.add_argument("--manifest", type=Path, default=Path("data/corpus_manifest.jsonl"))
    parser.add_argument("--out-json", type=Path, default=Path("logs/source_quality_board.json"))
    parser.add_argument("--out-md", type=Path, default=Path("logs/source_quality_boards_md/SOURCE_QUALITY_BOARD.md"))
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Compute and validate source quality without writing board artifacts.",
    )
    return parser.parse_args()


def load_manifest(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def has_any(row: dict[str, Any], keys: tuple[str, ...] | list[str]) -> bool:
    return any(str(row.get(key, "")).strip() for key in keys)


def score_row(row: dict[str, Any]) -> dict[str, Any]:
    score = 100
    blockers: list[str] = []
    warnings: list[str] = []
    if not row.get("include_in_training", True):
        score = 0
        warnings.append("excluded_from_training")
    required = {
        "source_id": "missing_source_id",
        "work_id": "missing_work_id",
        "split_role": "missing_split_role",
        "split_group_key": "missing_split_group_key",
        "source_url": "missing_source_url",
        "license_name": "missing_license_name",
        "redistribution_policy": "missing_redistribution_policy",
        "retrieved_at_utc": "missing_retrieved_at_utc",
        "clean_sha256": "missing_clean_sha256",
    }
    for key, label in required.items():
        if not str(row.get(key, "")).strip():
            blockers.append(label)
            score -= 15
    if row.get("redistribution_policy") != "corpus_text_not_distributed":
        blockers.append("unsafe_redistribution_policy")
        score -= 30
    if row.get("split_policy") != "work_group_genji_reference_v1":
        blockers.append("unexpected_split_policy")
        score -= 20
    if row.get("split_role") not in {"train", "validation", "test"}:
        blockers.append("invalid_split_role")
        score -= 25
    if row.get("grammar_scope") != "genji-era-reference":
        blockers.append("invalid_grammar_scope")
        score -= 15
    source_kind = str(row.get("source_kind", ""))
    if source_kind == "wikisource":
        if not str(row.get("source_revision", "")).strip():
            blockers.append("missing_source_revision")
            score -= 20
        if not str(row.get("source_revision_timestamp", "")).strip():
            blockers.append("missing_source_revision_timestamp")
            score -= 20
        if not has_any(row, ["source_payload_sha256", "api_payload_sha256"]):
            blockers.append("missing_wikisource_payload_hash")
            score -= 20
    elif source_kind == "aozora":
        if not has_any(row, ["source_payload_sha256", "download_payload_sha256"]):
            blockers.append("missing_aozora_payload_hash")
            score -= 20
    else:
        blockers.append(f"unknown_source_kind:{source_kind}")
        score -= 25
    if str(row.get("style", "")) == "waka":
        for key in ("records_sha256", "readings_sha256", "training_sha256"):
            if not str(row.get(key, "")).strip():
                blockers.append(f"missing_waka_{key}")
                score -= 10
    if not any(str(row.get(key, "")).strip() for key in HASH_FIELDS):
        blockers.append("no_hash_evidence")
        score -= 30
    characters = int(row.get("characters") or 0)
    if characters and characters < 200:
        warnings.append("very_short_source")
        score -= 3
    return {
        "source_id": row.get("source_id", ""),
        "work_id": row.get("work_id", ""),
        "split_role": row.get("split_role", ""),
        "split_group_sha256_prefix": str(row.get("split_group_key", ""))[:12],
        "source_kind": row.get("source_kind", ""),
        "style": row.get("style", ""),
        "grammar_scope": row.get("grammar_scope", ""),
        "characters": row.get("characters", 0),
        "score": max(0, min(100, score)),
        "blockers": blockers,
        "warnings": warnings,
    }


def render_md(payload: dict[str, Any]) -> str:
    lines = [
        "# Source Quality Board",
        "",
        "This board scores manifest metadata and hash/provenance coverage only.",
        "It intentionally does not include raw, clean, train, validation, or test text.",
        "",
        f"Generated UTC: `{payload['generated_at_utc']}`",
        f"Manifest: `{payload['manifest']}`",
        f"Included rows: `{payload['included_rows']}`",
        f"Average included score: `{payload['average_included_score']:.2f}`",
        f"Hard blocker rows: `{payload['hard_blocker_rows']}`",
        "",
        "## By Split Role",
        "",
        "| split_role | rows | average_score |",
        "| --- | ---: | ---: |",
    ]
    for role, stats in sorted(payload["by_split_role"].items()):
        lines.append(f"| {role} | {stats['rows']} | {stats['average_score']:.2f} |")
    lines.extend(
        [
            "",
            "## Lowest Scoring Included Sources",
            "",
            "| source_id | work_id | role | kind | chars | score | blockers | warnings |",
            "| --- | --- | --- | --- | ---: | ---: | --- | --- |",
        ]
    )
    for row in payload["lowest_scoring_included_sources"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["source_id"]),
                    str(row["work_id"]),
                    str(row["split_role"]),
                    str(row["source_kind"]),
                    str(row["characters"]),
                    str(row["score"]),
                    ", ".join(row["blockers"]),
                    ", ".join(row["warnings"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Policy",
            "",
            "- Hard blockers here are data-governance blockers, not model-quality metrics.",
            "- Later-medieval or diachronic evidence must remain reference-only for Genji-era release grammar.",
            "- HF export/package/upload remain manual-only and are not performed by this board.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    rows = load_manifest(args.manifest)
    scored = [score_row(row) for row in rows]
    included = [row for row, original in zip(scored, rows, strict=True) if original.get("include_in_training", True)]
    by_role_raw: dict[str, list[int]] = defaultdict(list)
    for row in included:
        by_role_raw[str(row["split_role"])].append(int(row["score"]))
    by_role = {
        role: {"rows": len(scores), "average_score": sum(scores) / len(scores) if scores else 0.0}
        for role, scores in by_role_raw.items()
    }
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": str(args.manifest),
        "total_rows": len(rows),
        "included_rows": len(included),
        "included_by_source_kind": dict(Counter(str(row["source_kind"]) for row in included)),
        "included_by_split_role": dict(Counter(str(row["split_role"]) for row in included)),
        "by_split_role": by_role,
        "average_included_score": sum(int(row["score"]) for row in included) / len(included) if included else 0.0,
        "hard_blocker_rows": sum(1 for row in included if row["blockers"]),
        "lowest_scoring_included_sources": sorted(included, key=lambda row: (row["score"], row["source_id"]))[:20],
        "artifact_policy": "metadata_and_hashes_only_no_raw_clean_train_validation_test_text",
    }
    if args.check_only:
        print("source_quality_board_json=(not written; check-only)")
        print("source_quality_board_md=(not written; check-only)")
    else:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(render_md(payload), encoding="utf-8")
        print(f"source_quality_board_json={args.out_json}")
        print(f"source_quality_board_md={args.out_md}")
    print(
        "source_quality_summary="
        + json.dumps(
            {
                "included_rows": payload["included_rows"],
                "average_included_score": round(payload["average_included_score"], 2),
                "hard_blocker_rows": payload["hard_blocker_rows"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    if payload["hard_blocker_rows"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
