from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from copy import deepcopy
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from parse_quality_log import parse_log, read_log_text

from kobun_llm.checkpoint_io import load_trusted_checkpoint
from kobun_autonomy.augmentation_audit import REQUIRED_AUGMENTATION_ROLES
from kobun_llm.model import GPT, GPTConfig
from kobun_autonomy.release_policy import require_release_candidate_run
from split_policy import SPLIT_POLICY, split_group_key, split_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a sanitized Hugging Face model package.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("release/hf_model"))
    parser.add_argument("--model-id", default="old-japanese-0.1B")
    parser.add_argument("--eval-results", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=Path("data/corpus_manifest.jsonl"))
    parser.add_argument("--waka-sources", type=Path, default=Path("data/waka/sources.json"))
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--confirm-explicit-user-request",
        action="store_true",
        help="Required acknowledgement that the user explicitly asked for HF package creation in this turn.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def read_json(path: Path) -> Any:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sanitize_path_text(value: str) -> str:
    if not value:
        return value
    return Path(value).name


PATHISH_RE = re.compile(
    r"(?i)([A-Z]:\\[^\s\"'`,;]+|(?:^|[\s=:\"'`])((?:logs|checkpoints|release|data[\\/](?:run_snapshots|raw|clean|aozora|waka))[\\/][^\s\"'`,;]+))"
)

PUBLIC_EVAL_TOP_LEVEL_KEYS = {
    "checkpoint",
    "checkpoint_best_val",
    "checkpoint_from_log",
    "checkpoint_sha256",
    "checkpoint_step",
    "checkpoint_tokenizer_vocab_scope",
    "corpus_checks",
    "duplicate_metrics",
    "eval_contamination_checks",
    "eval_source_overlap",
    "eval_source_overlap_checks",
    "eval_files",
    "eval_provenance_audit",
    "leakage",
    "model_metrics",
    "public_manifest_audit",
    "smoke_metrics",
    "source_record_audits",
    "split_consistency",
    "status",
    "test_lm",
    "tokenizer_vocab_scope",
}
PUBLIC_METRIC_FIELDS = {
    "count",
    "denominator",
    "kind",
    "n",
    "numerator",
    "passed",
    "role",
    "threshold",
    "total",
    "value",
}
REQUIRED_SMOKE_METRICS = {
    "primary_contrastive_preference_accuracy": {"min_value": 1.0, "min_total": 8},
    "heldout_contrastive_preference_accuracy": {"min_value": 1.0, "min_total": 12},
    "grammar_constraint_accuracy": {"min_value": 1.0, "min_total": 28},
    "waka_rule_accuracy": {"min_value": 1.0, "min_total": 20},
    "waka_meter_constraint_static_accuracy": {"min_value": 1.0, "min_total": 19},
    "waka_meter_constrained_generation_accuracy": {"min_value": 1.0, "min_total": 4},
    "morphology_adversarial_accuracy": {"min_value": 1.0, "min_total": 4},
}
PUBLIC_TEST_LM_KEYS = {
    "chars",
    "loss",
    "nll",
    "num_tokens",
    "perplexity",
    "test_data",
    "test_sha256",
    "token_nll",
    "tokens",
    "total_nll",
    "value",
}
PUBLIC_TOKENIZER_SCOPE_KEYS = {
    "byte_fallback",
    "byte_fallback_tokens",
    "core_inventory_sha256",
    "checkpoint_bound",
    "direct_vocab_chars",
    "forbidden_heldout_tokenizer_leakage",
    "heldout_covered_by_byte_fallback",
    "heldout_missing_from_tokenizer",
    "policy",
    "tokenizer_chars",
    "tokenizer_meta_sha256",
    "tokenizer_meta_verified",
    "vocab_sha256",
}
PUBLIC_SPLIT_CONSISTENCY_KEYS = {
    "group_disjoint",
    "manifest",
    "manifest_sha256",
    "split_policy",
    "test",
    "test_groups",
    "test_match",
    "test_sha256",
    "test_source_chars",
    "test_sources",
    "test_text_chars",
    "train",
    "train_groups",
    "train_match",
    "train_reconstruction_sha256",
    "augmentation_manifest",
    "augmentation_manifest_sha256",
    "validation",
    "validation_groups",
    "validation_match",
    "validation_sha256",
}
PUBLIC_LEAKAGE_KEYS = {
    "checked_sources",
    "checked_waka_items",
    "checked_windows",
    "expected_sources",
    "leaks",
    "manifest",
    "manifest_sha256",
    "role_pair_leaks",
    "role_waka_leaks",
    "split_policy",
    "waka_leaks",
}
PUBLIC_CONTAMINATION_CHECK_KEYS = {"checked", "eval_paths", "hits", "train_paths"}
PUBLIC_EVAL_SOURCE_OVERLAP_KEYS = {
    "checked",
    "eval_paths",
    "hits",
    "manifest",
    "manifest_sha256",
    "prose_hits",
    "source_items",
    "source_items_by_role",
    "source_roles_checked",
    "split_policy",
    "test_ratio",
    "val_ratio",
    "waka_exact_hits",
    "waka_variant_hits",
}
PUBLIC_EVAL_FILE_KEYS = {
    "audited_source",
    "audited_source_sha256",
    "case_ids",
    "content_hashes",
    "eval_provenance_manifest_sha256",
    "path",
    "removed_from_source",
    "role",
    "rows",
    "sha256",
    "source",
    "source_sha256",
}
PUBLIC_SOURCE_RECORD_AUDIT_KEYS = {"path", "checked", "mismatches", "missing", "fixed"}
PUBLIC_MANIFEST_AUDIT_KEYS = {"manifest_rows", "included_rows", "errors", "out"}
PUBLIC_EVAL_PROVENANCE_AUDIT_KEYS = {
    "entries",
    "errors",
    "llm_generated_eval_answer_text",
    "manifest_sha256",
    "path",
}
PUBLIC_TRAINING_METADATA_SCALAR_KEYS = {
    "backend",
    "batch_size",
    "checkpoint",
    "checkpoint_best_val",
    "checkpoint_sha256",
    "checkpoint_step",
    "data_chars",
    "data_path",
    "data_sha256",
    "determinism_note",
    "device_description",
    "effective_batch_size",
    "from_scratch",
    "grad_accum_steps",
    "init_from",
    "license_policy",
    "optimizer",
    "optimizer_step_count",
    "param_count",
    "param_count_b",
    "release_name",
    "resume",
    "run_id",
    "seed",
    "test_data_chars",
    "test_data_path",
    "test_data_sha256",
    "tokenizer_source",
    "val_data_chars",
    "val_data_path",
    "val_data_sha256",
}


def sanitize_freeform_text(value: str) -> str:
    if not value:
        return value

    def repl(match: re.Match[str]) -> str:
        token = match.group(1)
        prefix = ""
        if token and token[0].isspace():
            prefix, token = token[0], token[1:]
        if not token and match.group(2):
            token = match.group(2)
        return prefix + sanitize_path_text(token)

    value = PATHISH_RE.sub(repl, value)
    value = re.sub(r"(?i)(api[_-]?key|token|secret|password)\s*[:=]\s*[^\s,;]+", r"\1=<redacted>", value)
    return value


def sanitize_nested_paths(value: Any) -> Any:
    if isinstance(value, str):
        return sanitize_freeform_text(value)
    if isinstance(value, list):
        return [sanitize_nested_paths(item) for item in value]
    if isinstance(value, dict):
        return {key: sanitize_nested_paths(item) for key, item in value.items()}
    return value


def sanitize_eval_payload(eval_payload: dict[str, Any]) -> dict[str, Any]:
    payload = deepcopy(eval_payload)
    for key in ("quality_log", "checkpoint", "checkpoint_from_log"):
        if isinstance(payload.get(key), str):
            payload[key] = sanitize_path_text(payload[key])
    environment = payload.get("environment")
    if isinstance(environment, dict) and isinstance(environment.get("python"), str):
        environment["python"] = Path(environment["python"]).name
    leakage = payload.get("leakage")
    if isinstance(leakage, dict):
        for key in ("manifest", "train"):
            if isinstance(leakage.get(key), str):
                leakage[key] = sanitize_path_text(leakage[key])
    split_consistency = payload.get("split_consistency")
    if isinstance(split_consistency, dict):
        for key in ("manifest", "train", "validation", "test"):
            if isinstance(split_consistency.get(key), str):
                split_consistency[key] = sanitize_path_text(split_consistency[key])
    eval_contamination = payload.get("eval_contamination")
    if isinstance(eval_contamination, dict):
        for key in ("train_paths", "eval_paths"):
            values = eval_contamination.get(key)
            if isinstance(values, list):
                eval_contamination[key] = [sanitize_path_text(str(value)) for value in values]
    for check in payload.get("eval_contamination_checks", []) or []:
        if isinstance(check, dict):
            for key in ("train_paths", "eval_paths"):
                values = check.get(key)
                if isinstance(values, list):
                    check[key] = [sanitize_path_text(str(value)) for value in values]
    eval_source_overlap = payload.get("eval_source_overlap")
    if isinstance(eval_source_overlap, dict):
        for key in ("manifest",):
            if isinstance(eval_source_overlap.get(key), str):
                eval_source_overlap[key] = sanitize_path_text(eval_source_overlap[key])
        values = eval_source_overlap.get("eval_paths")
        if isinstance(values, list):
            eval_source_overlap["eval_paths"] = [sanitize_path_text(str(value)) for value in values]
    for check in payload.get("eval_source_overlap_checks", []) or []:
        if isinstance(check, dict):
            if isinstance(check.get("manifest"), str):
                check["manifest"] = sanitize_path_text(check["manifest"])
            values = check.get("eval_paths")
            if isinstance(values, list):
                check["eval_paths"] = [sanitize_path_text(str(value)) for value in values]
    for item in payload.get("eval_files", []) or []:
        if isinstance(item, dict):
            for key in ("path", "source"):
                if isinstance(item.get(key), str):
                    item[key] = sanitize_path_text(item[key])
    eval_files = payload.get("eval_files")
    if isinstance(eval_files, list):
        for item in eval_files:
            if isinstance(item, dict):
                for key in ("path", "source", "audited_source"):
                    if isinstance(item.get(key), str):
                        item[key] = sanitize_path_text(item[key])
    eval_provenance_audit = payload.get("eval_provenance_audit")
    if isinstance(eval_provenance_audit, dict) and isinstance(eval_provenance_audit.get("path"), str):
        eval_provenance_audit["path"] = sanitize_path_text(Path(eval_provenance_audit["path"]).name)
    for key in ("heldout_lm", "validation_lm", "test_lm"):
        record = payload.get(key)
        if isinstance(record, dict):
            for nested_key, value in list(record.items()):
                if nested_key.endswith("_data") and isinstance(value, str):
                    record[nested_key] = sanitize_path_text(value)
    corpus_checks = payload.get("corpus_checks")
    if isinstance(corpus_checks, list):
        payload["corpus_checks"] = [sanitize_path_text(str(value)) for value in corpus_checks]
    if isinstance(payload.get("failure_reasons"), list):
        payload["failure_reasons"] = [sanitize_freeform_text(str(value)) for value in payload["failure_reasons"]]
    return sanitize_nested_paths(payload)


def public_metric_map(metrics: Any) -> dict[str, Any]:
    if not isinstance(metrics, dict):
        return {}
    public: dict[str, Any] = {}
    for name, record in metrics.items():
        if isinstance(record, dict):
            public[str(name)] = {
                key: sanitize_nested_paths(value)
                for key, value in record.items()
                if key in PUBLIC_METRIC_FIELDS
            }
        elif isinstance(record, (int, float, bool)):
            public[str(name)] = {"value": record}
    return public


def public_eval_payload(eval_payload: dict[str, Any]) -> dict[str, Any]:
    """Return only the release-safe evaluation evidence schema.

    The full internal eval JSON may contain free-form diagnostics. Release
    packages get this allowlisted projection so a hand-authored or future field
    cannot smuggle paths, logs, text excerpts, or secrets into public artifacts.
    """

    sanitized = sanitize_eval_payload(eval_payload)
    public: dict[str, Any] = {}
    for key in (
        "status",
        "checkpoint",
        "checkpoint_from_log",
        "checkpoint_sha256",
        "checkpoint_step",
        "checkpoint_best_val",
        "duplicate_metrics",
    ):
        if key in sanitized:
            public[key] = sanitize_nested_paths(sanitized[key])

    public["model_metrics"] = public_metric_map(sanitized.get("model_metrics"))
    public["smoke_metrics"] = public_metric_map(sanitized.get("smoke_metrics"))

    for key, allowed in (
        ("test_lm", PUBLIC_TEST_LM_KEYS),
        ("tokenizer_vocab_scope", PUBLIC_TOKENIZER_SCOPE_KEYS),
        ("checkpoint_tokenizer_vocab_scope", PUBLIC_TOKENIZER_SCOPE_KEYS),
        ("split_consistency", PUBLIC_SPLIT_CONSISTENCY_KEYS),
        ("leakage", PUBLIC_LEAKAGE_KEYS),
    ):
        record = sanitized.get(key)
        if isinstance(record, dict):
            public[key] = {field: sanitize_nested_paths(record[field]) for field in allowed if field in record}

    corpus_checks = sanitized.get("corpus_checks")
    if isinstance(corpus_checks, list):
        public["corpus_checks"] = [sanitize_path_text(str(value)) for value in corpus_checks]

    checks = sanitized.get("eval_contamination_checks")
    if isinstance(checks, list):
        public["eval_contamination_checks"] = [
            {field: sanitize_nested_paths(item[field]) for field in PUBLIC_CONTAMINATION_CHECK_KEYS if field in item}
            for item in checks
            if isinstance(item, dict)
        ]

    source_overlap_checks = sanitized.get("eval_source_overlap_checks")
    if isinstance(source_overlap_checks, list):
        public["eval_source_overlap_checks"] = [
            {field: sanitize_nested_paths(item[field]) for field in PUBLIC_EVAL_SOURCE_OVERLAP_KEYS if field in item}
            for item in source_overlap_checks
            if isinstance(item, dict)
        ]

    eval_files = sanitized.get("eval_files")
    if isinstance(eval_files, list):
        public["eval_files"] = [
            {field: sanitize_nested_paths(item[field]) for field in PUBLIC_EVAL_FILE_KEYS if field in item}
            for item in eval_files
            if isinstance(item, dict)
        ]

    source_record_audits = sanitized.get("source_record_audits")
    if isinstance(source_record_audits, list):
        public["source_record_audits"] = [
            {field: sanitize_nested_paths(item[field]) for field in PUBLIC_SOURCE_RECORD_AUDIT_KEYS if field in item}
            for item in source_record_audits
            if isinstance(item, dict)
        ]

    public_manifest_audit = sanitized.get("public_manifest_audit")
    if isinstance(public_manifest_audit, dict):
        public["public_manifest_audit"] = {
            field: sanitize_nested_paths(public_manifest_audit[field])
            for field in PUBLIC_MANIFEST_AUDIT_KEYS
            if field in public_manifest_audit
        }

    eval_provenance_audit = sanitized.get("eval_provenance_audit")
    if isinstance(eval_provenance_audit, dict):
        public["eval_provenance_audit"] = {
            field: sanitize_nested_paths(eval_provenance_audit[field])
            for field in PUBLIC_EVAL_PROVENANCE_AUDIT_KEYS
            if field in eval_provenance_audit
        }

    return {key: public[key] for key in PUBLIC_EVAL_TOP_LEVEL_KEYS if key in public}


def sanitize_file_records(records: Any) -> Any:
    if not isinstance(records, list):
        return records
    sanitized = []
    for record in records:
        if not isinstance(record, dict):
            sanitized.append(record)
            continue
        item = {
            key: record[key]
            for key in ("path", "sha256", "role", "label", "kind")
            if key in record
        }
        if isinstance(item.get("path"), str):
            item["path"] = sanitize_path_text(item["path"])
        sanitized.append(item)
    return sanitized


def release_augmentation_summary(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    entries = []
    for entry in payload.get("entries", []) if isinstance(payload, dict) else []:
        if not isinstance(entry, dict):
            continue
        entries.append(
            {
                "role": entry.get("role", ""),
                "source_type": entry.get("source_type", ""),
                "path": sanitize_path_text(str(entry.get("path", ""))),
                "sha256": entry.get("sha256", ""),
                "bytes": entry.get("bytes", 0),
                "lines": entry.get("lines", 0),
                "repeat_count": entry.get("repeat_count", 0),
                "copyability_status": entry.get("copyability_status", ""),
                "llm_generated_corpus_text": bool(entry.get("llm_generated_corpus_text", True)),
                "public_release_policy": entry.get("public_release_policy", ""),
            }
        )
    return {
        "schema": payload.get("schema", ""),
        "attestation": payload.get("attestation", ""),
        "llm_generated_corpus_text": bool(payload.get("llm_generated_corpus_text", True)),
        "entries": entries,
    }


def sanitize_training_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key in PUBLIC_TRAINING_METADATA_SCALAR_KEYS:
        if key not in metadata:
            continue
        payload[key] = sanitize_nested_paths(metadata[key])
    for key in ("checkpoint", "data_path", "val_data_path", "test_data_path", "init_from", "resume"):
        if isinstance(payload.get(key), str):
            payload[key] = sanitize_path_text(payload[key])
    for key in ("val_oov_chars", "test_oov_chars"):
        value = metadata.get(key)
        if isinstance(value, str):
            payload[f"{key}_count"] = len(value)
    payload["tokenizer_extra_data"] = sanitize_file_records(metadata.get("tokenizer_extra_data"))
    payload["provenance_files"] = sanitize_file_records(metadata.get("provenance_files"))
    payload["release_metadata_policy"] = (
        "Allowlisted metadata only; paths are basename-only and hashes bind public evidence to local release gates."
    )
    return payload


def summarize_sources(manifest_path: Path, waka_sources_path: Path) -> dict[str, Any]:
    rows = read_jsonl(manifest_path)
    included = [row for row in rows if row.get("include_in_training")]
    excluded = [row for row in rows if not row.get("include_in_training")]
    kind_counts = Counter(str(row.get("source_kind", "unknown")) for row in included)
    license_counts = Counter(str(row.get("license_name", "unknown")) for row in included)
    char_count = sum(int(row.get("characters") or 0) for row in included)
    waka_sources = read_json(waka_sources_path) or []
    return {
        "manifest": sanitize_path_text(str(manifest_path)),
        "included_sources": len(included),
        "excluded_sources": len(excluded),
        "included_characters": char_count,
        "included_by_kind": dict(sorted(kind_counts.items())),
        "included_by_license": dict(sorted(license_counts.items())),
        "excluded": [
            {
                "source_id": row.get("source_id"),
                "title": row.get("title"),
                "reason": row.get("exclude_reason"),
            }
            for row in excluded
        ],
        "waka_source_file": sanitize_path_text(str(waka_sources_path)) if waka_sources_path.exists() else "",
        "waka_source_count": len(waka_sources) if isinstance(waka_sources, list) else 0,
        "release_policy": "Do not include raw, clean, training, validation, or test text in this package.",
    }


def release_source_manifest(manifest_path: Path) -> dict[str, Any]:
    rows = read_jsonl(manifest_path)
    safe_rows = []
    for row in rows:
        include = bool(row.get("include_in_training", True))
        split_key = str(row.get("split_key") or "")
        group_key = split_group_key(row)
        role = split_name(row)
        split = "training" if role == "train" else role
        clean_sha256 = str(row.get("clean_sha256") or "")
        safe_rows.append(
            {
                "source_id": row.get("source_id"),
                "split_key_sha256": hashlib.sha256(split_key.encode("utf-8")).hexdigest(),
                "split_policy": row.get("split_policy", SPLIT_POLICY),
                "split_group_sha256": hashlib.sha256(group_key.encode("utf-8")).hexdigest(),
                "split": split,
                "title": row.get("title"),
                "period": row.get("period"),
                "genre": row.get("genre"),
                "style": row.get("style"),
                "include_in_training": include,
                "exclude_reason": row.get("exclude_reason", ""),
                "source_kind": row.get("source_kind", ""),
                "license_name": row.get("license_name", ""),
                "license_note": row.get("license_note", ""),
                "redistribution_policy": row.get("redistribution_policy", ""),
                "source_url": row.get("source_url", ""),
                "download_url": row.get("download_url", ""),
                "characters": row.get("characters", 0),
                "source_revision": row.get("source_revision", ""),
                "source_revision_timestamp": row.get("source_revision_timestamp", ""),
                "retrieved_at_utc": row.get("retrieved_at_utc", ""),
                "source_payload_sha256": row.get("source_payload_sha256", ""),
                "download_payload_sha256": row.get("download_payload_sha256", ""),
                "api_payload_sha256": row.get("api_payload_sha256", ""),
                "clean_sha256": clean_sha256,
                "records_sha256": row.get("records_sha256", ""),
                "readings_sha256": row.get("readings_sha256", ""),
                "training_sha256": row.get("training_sha256", ""),
            }
        )
    return {
        "manifest": sanitize_path_text(str(manifest_path)),
        "split_policy": SPLIT_POLICY,
        "release_policy": "Sanitized provenance only; no raw, clean, training, validation, or test text.",
        "sources": safe_rows,
    }


def source_manifest_issues(source_manifest: dict[str, Any]) -> list[str]:
    rows = source_manifest.get("sources")
    if not isinstance(rows, list) or not rows:
        return ["source_manifest has no sources"]
    issues: list[str] = []
    augmentations = source_manifest.get("training_augmentations")
    if not isinstance(augmentations, dict):
        issues.append("source_manifest missing training_augmentations")
    else:
        entries = augmentations.get("entries")
        if not isinstance(entries, list) or not entries:
            issues.append("source_manifest training_augmentations has no entries")
        if augmentations.get("llm_generated_corpus_text") is not False:
            issues.append("source_manifest training_augmentations must attest llm_generated_corpus_text=false")
        for index, entry in enumerate(entries or []):
            if not isinstance(entry, dict):
                issues.append(f"training augmentation index {index} is not an object")
                continue
            if entry.get("llm_generated_corpus_text") is not False:
                issues.append(f"training augmentation index {index} must have llm_generated_corpus_text=false")
            for field in ("role", "source_type", "sha256", "copyability_status", "repeat_count"):
                if not str(entry.get(field, "")).strip():
                    issues.append(f"training augmentation index {index} missing {field}")
        roles = {
            str(entry.get("role") or "")
            for entry in entries or []
            if isinstance(entry, dict)
        }
        missing_roles = sorted(REQUIRED_AUGMENTATION_ROLES - roles)
        if missing_roles:
            issues.append(f"source_manifest training_augmentations missing roles: {missing_roles}")
    for index, row in enumerate(rows):
        for field in (
            "source_id",
            "split_key_sha256",
            "split_policy",
            "split_group_sha256",
            "split",
            "title",
            "source_kind",
            "license_name",
            "license_note",
            "redistribution_policy",
            "source_url",
            "retrieved_at_utc",
            "clean_sha256",
        ):
            if not str(row.get(field, "")).strip():
                issues.append(f"source index {index} missing {field}")
        if str(row.get("split_policy", "")) != SPLIT_POLICY:
            issues.append(f"source index {index} split_policy={row.get('split_policy')!r}")
        if str(row.get("redistribution_policy", "")) != "corpus_text_not_distributed":
            issues.append(f"source index {index} redistribution_policy={row.get('redistribution_policy')!r}")
        source_kind = str(row.get("source_kind", ""))
        if not str(row.get("retrieved_at_utc", "")).strip():
            issues.append(f"source index {index} missing retrieved_at_utc")
        if source_kind == "wikisource":
            if not str(row.get("source_revision", "")).strip():
                issues.append(f"wikisource source index {index} missing source_revision")
            if not str(row.get("source_revision_timestamp", "")).strip():
                issues.append(f"wikisource source index {index} missing source_revision_timestamp")
            if not (
                str(row.get("source_payload_sha256", "")).strip()
                or str(row.get("api_payload_sha256", "")).strip()
            ):
                issues.append(f"wikisource source index {index} missing API payload hash")
        elif source_kind == "aozora":
            if not str(row.get("source_payload_sha256", "")).strip():
                issues.append(f"aozora source index {index} missing card payload hash")
            if not str(row.get("download_payload_sha256", "")).strip():
                issues.append(f"aozora source index {index} missing downloaded text payload hash")
        else:
            issues.append(f"source index {index} unsupported source_kind={source_kind!r}")
        if not str(row.get("clean_sha256", "")).strip():
            issues.append(f"source index {index} missing clean_sha256")
        if row.get("style") == "waka":
            if not str(row.get("records_sha256", "")).strip():
                issues.append(f"waka source index {index} missing records_sha256")
            if not str(row.get("readings_sha256", "")).strip():
                issues.append(f"waka source index {index} missing readings_sha256")
    return issues


def checkpoint_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve_repo_file(path_text: str) -> Path:
    if not path_text:
        raise SystemExit("empty release evidence path")
    path = Path(path_text)
    if not path.is_absolute():
        path = Path.cwd() / path
    resolved = path.resolve(strict=False)
    root = Path.cwd().resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise SystemExit(f"release evidence path escapes repository: {path_text}") from exc
    return resolved


def canonical_quality_evidence(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "model_metrics": payload.get("model_metrics") or {},
        "smoke_metrics": payload.get("smoke_metrics") or {},
        "duplicate_metrics": payload.get("duplicate_metrics") or [],
        "corpus_checks": payload.get("corpus_checks") or [],
        "leakage": payload.get("leakage"),
        "eval_contamination_checks": payload.get("eval_contamination_checks") or [],
        "eval_files": payload.get("eval_files") or [],
        "eval_provenance_audit": payload.get("eval_provenance_audit"),
        "source_record_audits": payload.get("source_record_audits") or [],
        "public_manifest_audit": payload.get("public_manifest_audit"),
        "tokenizer_vocab_scope": payload.get("tokenizer_vocab_scope"),
        "checkpoint_tokenizer_vocab_scope": payload.get("checkpoint_tokenizer_vocab_scope"),
        "split_consistency": payload.get("split_consistency"),
        "test_lm": payload.get("test_lm"),
    }


def verify_eval_quality_log(eval_payload: dict[str, Any], expected_runner: str = "scripts/run_quality_checks_dml.ps1") -> None:
    if eval_payload.get("hf_export") is not False:
        raise SystemExit("eval results must attest hf_export=false.")
    quality_log_value = str(eval_payload.get("quality_log") or "")
    quality_log = resolve_repo_file(quality_log_value)
    root = Path.cwd().resolve(strict=False)
    try:
        relative_quality_log = quality_log.relative_to(root)
    except ValueError as exc:
        raise SystemExit("eval quality_log must stay inside the repository.") from exc
    if not relative_quality_log.parts or relative_quality_log.parts[0] != "logs":
        raise SystemExit("eval quality_log must point to a measured log under logs/.")
    if not quality_log.exists():
        raise SystemExit(f"eval quality_log is missing: {quality_log_value}")
    expected_log_sha = str(eval_payload.get("quality_log_sha256") or "")
    if not expected_log_sha or file_sha256(quality_log) != expected_log_sha:
        raise SystemExit("eval quality_log_sha256 does not match measured quality log.")
    parser_value = str(eval_payload.get("quality_parser") or "")
    if parser_value != "scripts/parse_quality_log.py":
        raise SystemExit("eval quality_parser must be scripts/parse_quality_log.py.")
    parser_path = resolve_repo_file(parser_value)
    if str(eval_payload.get("quality_parser_sha256") or "") != file_sha256(parser_path):
        raise SystemExit("eval quality_parser_sha256 does not match current parser.")
    runner_value = str(eval_payload.get("quality_runner") or "")
    if runner_value != expected_runner:
        raise SystemExit(f"eval quality_runner must be {expected_runner}.")
    runner_path = resolve_repo_file(runner_value)
    if str(eval_payload.get("quality_runner_sha256") or "") != file_sha256(runner_path):
        raise SystemExit("eval quality_runner_sha256 does not match current quality runner.")
    raw, text, encoding = read_log_text(quality_log)
    if hashlib.sha256(raw).hexdigest() != expected_log_sha:
        raise SystemExit("eval quality log hash changed during verification.")
    if str(eval_payload.get("quality_log_encoding") or "") != encoding:
        raise SystemExit("eval quality_log_encoding does not match measured log decoding.")
    reparsed = parse_log(text)
    if canonical_quality_evidence(eval_payload) != canonical_quality_evidence(reparsed):
        raise SystemExit("eval results do not match canonical evidence reparsed from quality log.")


def checkpoint_provenance_record(metadata: dict[str, Any], filename: str) -> dict[str, Any]:
    for record in metadata.get("provenance_files", []) or []:
        if not isinstance(record, dict):
            continue
        raw_path = str(record.get("path") or "")
        if not raw_path or Path(raw_path).name != filename:
            continue
        return record
    raise SystemExit(f"checkpoint metadata missing required provenance file: {filename}")


def checkpoint_provenance_path(metadata: dict[str, Any], filename: str) -> Path:
    record = checkpoint_provenance_record(metadata, filename)
    raw_path = str(record.get("path") or "")
    expected_hash = str(record.get("sha256") or "")
    if not raw_path or not expected_hash:
        raise SystemExit(f"checkpoint provenance record for {filename} is missing path or sha256")
    path = Path(raw_path)
    if not path.is_absolute():
        path = Path.cwd() / path
    if not path.exists():
        raise SystemExit(f"checkpoint provenance file is missing: {raw_path}")
    if checkpoint_sha256(path) != expected_hash:
        raise SystemExit(f"checkpoint provenance file hash mismatch: {raw_path}")
    return path


def require_checkpoint_provenance(metadata: dict[str, Any]) -> None:
    for filename in (
        "aozora_sources.json",
        "corpus_manifest.jsonl",
        "public_manifest_summary.json",
        "snapshot_manifest.json",
        "tokenizer_public_char_vocab.meta.json",
        "training_augmentation_manifest.json",
        "waka_sources.json",
    ):
        checkpoint_provenance_path(metadata, filename)


def run_gate(args: list[str]) -> None:
    command = [sys.executable, *args]
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        raise SystemExit(f"release gate failed: {' '.join(command)} exit={result.returncode}")


def run_checkpoint_gates(checkpoint: Path, eval_results: Path, model_id: str) -> None:
    scripts_dir = Path(__file__).parent
    run_id = checkpoint.stem.removesuffix("_best")
    require_release_candidate_run(run_id, context="HF export release gate")
    run_gate(
        [
            str(scripts_dir / "check_release_gate.py"),
            "--run-id",
            run_id,
            "--checkpoint",
            str(checkpoint),
            "--eval-results",
            str(eval_results),
        ]
    )


def paths_equivalent(left: str, right: Path) -> bool:
    if not left:
        return False
    try:
        left_resolved = Path(left).resolve(strict=False)
        right_resolved = right.resolve(strict=False)
    except OSError:
        return Path(left).as_posix().casefold() == right.as_posix().casefold()
    return str(left_resolved).casefold() == str(right_resolved).casefold()


def path_in_values(target: str, values: list[str]) -> bool:
    if not target:
        return False
    return any(paths_equivalent(value, Path(target)) for value in values)


def validate_eval_payload(eval_payload: dict[str, Any], checkpoint: Path, payload: dict[str, Any]) -> None:
    if eval_payload.get("status") != "passed":
        raise SystemExit("eval results must contain status='passed'.")
    metadata = dict(payload.get("metadata", {}) or {})
    backend = str(metadata.get("backend") or "")
    expected_runner_by_backend = {
        "dml": "scripts/run_quality_checks_dml.ps1",
        "cuda": "scripts/run_quality_checks_cuda.py",
    }
    expected_runner = expected_runner_by_backend.get(backend)
    if not expected_runner:
        raise SystemExit(f"unsupported release backend for eval quality runner binding: {backend!r}")
    verify_eval_quality_log(eval_payload, expected_runner=expected_runner)
    if backend == "cuda" and str(eval_payload.get("device") or "") != "cuda":
        raise SystemExit("CUDA checkpoint eval evidence must be produced with device=cuda.")
    if backend == "cuda":
        environment = eval_payload.get("environment") or {}
        if not isinstance(environment, dict):
            raise SystemExit("CUDA eval evidence must include measured runtime environment.")
        if str(environment.get("real_cuda_runtime") or "").lower() != "true":
            raise SystemExit("CUDA eval evidence must attest real_cuda_runtime=true.")
        if str(environment.get("cuda_runtime_kind") or "") != "cuda":
            raise SystemExit("CUDA eval evidence must come from cuda_runtime_kind=cuda.")
        hip_version = str(environment.get("torch_hip_version") or "")
        if hip_version:
            raise SystemExit("CUDA eval evidence must not be produced by a ROCm/HIP runtime.")
    if backend == "dml" and str(eval_payload.get("device") or "") != "dml":
        raise SystemExit("DML checkpoint eval evidence must be produced with device=dml.")
    if eval_payload.get("duplicate_metrics"):
        raise SystemExit("eval results contain duplicate metric keys.")
    model_metrics = eval_payload.get("model_metrics") or {}
    required_model_metrics = {"test_lm_token_nll"}
    missing_model_metrics = sorted(required_model_metrics - set(model_metrics))
    if missing_model_metrics:
        raise SystemExit(f"eval results missing model-facing metrics: {missing_model_metrics}")
    test_metric = model_metrics.get("test_lm_token_nll") or {}
    test_loss = float(test_metric.get("value", float("inf")))
    if not 0.0 <= test_loss <= 8.0:
        raise SystemExit(f"test_lm_token_nll outside release threshold: {test_loss}")
    smoke_metrics = eval_payload.get("smoke_metrics") or {}
    for metric_name, requirements in REQUIRED_SMOKE_METRICS.items():
        metric = smoke_metrics.get(metric_name) or eval_payload.get("metrics", {}).get(metric_name) or {}
        value = float(metric.get("value", float("-inf")))
        min_value = float(requirements["min_value"])
        if value < min_value:
            raise SystemExit(f"eval smoke/static metric {metric_name}={value} is below required {min_value}")
        total = int(metric.get("total", metric.get("denominator", metric.get("count", 0))) or 0)
        min_total = int(requirements["min_total"])
        if total < min_total:
            raise SystemExit(f"eval smoke/static metric {metric_name} has only {total} cases, below required {min_total}")
    checkpoint_text = str(checkpoint)
    if eval_payload.get("checkpoint") not in {checkpoint_text, checkpoint.as_posix()}:
        raise SystemExit(
            f"eval checkpoint mismatch: eval={eval_payload.get('checkpoint')!r} checkpoint={checkpoint_text!r}"
        )
    if not paths_equivalent(str(eval_payload.get("checkpoint_from_log") or ""), checkpoint):
        raise SystemExit("eval checkpoint_from_log is missing or does not match exported checkpoint.")
    expected_sha = checkpoint_sha256(checkpoint)
    if eval_payload.get("checkpoint_sha256") != expected_sha:
        raise SystemExit("eval checkpoint_sha256 does not match exported checkpoint.")
    if eval_payload.get("checkpoint_step") != payload.get("step"):
        raise SystemExit("eval checkpoint_step does not match exported checkpoint.")
    eval_best = eval_payload.get("checkpoint_best_val")
    ckpt_best = payload.get("best_val")
    if eval_best is not None and ckpt_best is not None and abs(float(eval_best) - float(ckpt_best)) > 1e-8:
        raise SystemExit("eval checkpoint_best_val does not match exported checkpoint.")
    leakage = eval_payload.get("leakage")
    if not isinstance(leakage, dict):
        raise SystemExit("eval leakage check is missing.")
    if int(leakage.get("checked_sources", 0)) <= 0 or int(leakage.get("leaks", -1)) != 0:
        raise SystemExit("eval leakage check is missing or not clean.")
    if int(leakage.get("checked_windows", 0) or 0) <= 0:
        raise SystemExit("eval leakage check did not report any checked windows.")
    expected_sources = int(leakage.get("expected_sources", 0) or 0)
    if expected_sources > 0 and int(leakage.get("checked_sources", 0)) != expected_sources:
        raise SystemExit("eval leakage check did not cover every expected validation source.")
    if int(leakage.get("checked_waka_items", 0) or 0) <= 0:
        raise SystemExit("eval leakage check did not report waka poem/reading items.")
    if "waka_leaks" not in leakage or int(leakage["waka_leaks"]) != 0:
        raise SystemExit("eval waka leakage check is missing or not clean.")
    if int(leakage.get("role_pair_leaks", -1)) != 0:
        raise SystemExit("eval validation/test prose split leakage check is missing or not clean.")
    if int(leakage.get("role_waka_leaks", -1)) != 0:
        raise SystemExit("eval validation/test waka split leakage check is missing or not clean.")
    test_lm = eval_payload.get("test_lm")
    if not isinstance(test_lm, dict):
        raise SystemExit("eval test_lm record is missing.")
    if not paths_equivalent(str(test_lm.get("test_data") or ""), Path(str(metadata.get("test_data_path") or ""))):
        raise SystemExit("eval test_lm test_data is not bound to checkpoint test_data_path.")
    if str(test_lm.get("test_sha256") or "") != str(metadata.get("test_data_sha256") or ""):
        raise SystemExit("eval test_lm test_sha256 does not match checkpoint test_data_sha256.")
    corpus_manifest_record = checkpoint_provenance_record(metadata, "corpus_manifest.jsonl")
    tokenizer_scope = eval_payload.get("tokenizer_vocab_scope")
    if not isinstance(tokenizer_scope, dict):
        raise SystemExit("eval tokenizer_vocab_scope evidence is missing.")
    if str(tokenizer_scope.get("policy") or "") != "train_split_plus_core_japanese_inventory_plus_utf8_byte_fallback_v1":
        raise SystemExit("eval tokenizer_vocab_scope policy is not release-safe.")
    if tokenizer_scope.get("byte_fallback") is not True or int(tokenizer_scope.get("byte_fallback_tokens", -1)) != 256:
        raise SystemExit("eval tokenizer_vocab_scope does not prove UTF-8 byte fallback coverage.")
    if int(tokenizer_scope.get("tokenizer_chars", 1_000_000) or 1_000_000) >= 10_000:
        raise SystemExit("eval tokenizer_vocab_scope vocab is too large for the DirectML release policy.")
    if bool(tokenizer_scope.get("tokenizer_meta_verified")) is not True:
        raise SystemExit("eval tokenizer_vocab_scope did not verify checkpoint-bound tokenizer metadata.")
    if int(tokenizer_scope.get("forbidden_heldout_tokenizer_leakage", -1)) != 0:
        raise SystemExit("eval tokenizer_vocab_scope found heldout-derived tokenizer leakage.")
    if int(tokenizer_scope.get("heldout_missing_from_tokenizer", -1)) != 0:
        raise SystemExit("eval tokenizer_vocab_scope found heldout tokenizer OOV.")
    tokenizer_meta_record = checkpoint_provenance_record(metadata, "tokenizer_public_char_vocab.meta.json")
    if str(tokenizer_scope.get("tokenizer_meta_sha256") or "") != str(tokenizer_meta_record.get("sha256") or ""):
        raise SystemExit("eval tokenizer_vocab_scope metadata hash is not bound to checkpoint tokenizer metadata.")
    tokenizer_extra_records = metadata.get("tokenizer_extra_data") or []
    if not tokenizer_extra_records:
        raise SystemExit("checkpoint metadata missing tokenizer_extra_data records.")
    if str(tokenizer_scope.get("vocab_sha256") or "") != str(tokenizer_extra_records[0].get("sha256") or ""):
        raise SystemExit("eval tokenizer_vocab_scope vocab hash is not bound to checkpoint tokenizer extra data.")
    checkpoint_tokenizer_scope = eval_payload.get("checkpoint_tokenizer_vocab_scope")
    if not isinstance(checkpoint_tokenizer_scope, dict):
        raise SystemExit("eval checkpoint_tokenizer_vocab_scope evidence is missing.")
    if checkpoint_tokenizer_scope.get("checkpoint_bound") is not True:
        raise SystemExit("eval checkpoint_tokenizer_vocab_scope is not checkpoint-bound.")
    for key in (
        "policy",
        "byte_fallback",
        "byte_fallback_tokens",
        "tokenizer_chars",
        "direct_vocab_chars",
        "forbidden_heldout_tokenizer_leakage",
        "heldout_missing_from_tokenizer",
        "tokenizer_meta_verified",
        "vocab_sha256",
        "tokenizer_meta_sha256",
        "manifest_sha256",
    ):
        if checkpoint_tokenizer_scope.get(key) != tokenizer_scope.get(key):
            raise SystemExit(f"checkpoint tokenizer scope disagrees with tokenizer scope for {key}.")
    if str(checkpoint_tokenizer_scope.get("manifest_sha256") or "") != str(corpus_manifest_record.get("sha256") or ""):
        raise SystemExit("eval checkpoint_tokenizer_vocab_scope manifest hash is not bound to checkpoint corpus manifest.")
    split_consistency = eval_payload.get("split_consistency")
    if not isinstance(split_consistency, dict):
        raise SystemExit("eval split_consistency evidence is missing.")
    if not split_consistency.get("train_match"):
        raise SystemExit("eval split_consistency did not verify full train corpus reconstruction.")
    augmentation_record = checkpoint_provenance_record(metadata, "training_augmentation_manifest.json")
    if str(split_consistency.get("augmentation_manifest_sha256") or "") != str(augmentation_record.get("sha256") or ""):
        raise SystemExit("eval split_consistency augmentation manifest hash is not bound to checkpoint provenance.")
    if not split_consistency.get("validation_match"):
        raise SystemExit("eval split_consistency did not verify validation split.")
    if not split_consistency.get("test_match"):
        raise SystemExit("eval split_consistency did not verify test split.")
    if str(split_consistency.get("split_policy") or "") != SPLIT_POLICY:
        raise SystemExit("eval split_consistency did not use the release split policy.")
    if split_consistency.get("group_disjoint") is not True:
        raise SystemExit("eval split_consistency did not verify work/group-disjoint train/validation/test splits.")
    if int(split_consistency.get("test_sources", 0) or 0) < 3:
        raise SystemExit("eval split_consistency has too few test sources.")
    if int(split_consistency.get("test_source_chars", 0) or 0) < 30_000:
        raise SystemExit("eval split_consistency has too few test source characters.")
    if int(split_consistency.get("test_text_chars", 0) or 0) < 20_000:
        raise SystemExit("eval split_consistency has too few generated test text characters.")
    if str(split_consistency.get("manifest_sha256") or "") != str(corpus_manifest_record.get("sha256") or ""):
        raise SystemExit("eval split_consistency manifest hash is not bound to checkpoint corpus_manifest.jsonl.")
    if str(split_consistency.get("test_sha256") or "") != str(metadata.get("test_data_sha256") or ""):
        raise SystemExit("eval split_consistency test hash does not match checkpoint test_data_sha256.")
    if str(split_consistency.get("validation_sha256") or "") != str(metadata.get("val_data_sha256") or ""):
        raise SystemExit("eval split_consistency validation hash does not match checkpoint val_data_sha256.")
    corpus_checks = [str(value) for value in eval_payload.get("corpus_checks", [])]
    for label, metadata_key in (
        ("train", "data_path"),
        ("validation", "val_data_path"),
        ("test", "test_data_path"),
    ):
        expected_path = str(metadata.get(metadata_key) or "")
        if not expected_path:
            raise SystemExit(f"checkpoint metadata missing {metadata_key}")
        if not any(paths_equivalent(value, Path(expected_path)) for value in corpus_checks):
            raise SystemExit(f"eval corpus validation evidence is not bound to checkpoint {label} snapshot.")
    if str(leakage.get("manifest_sha256") or "") != str(corpus_manifest_record.get("sha256") or ""):
        raise SystemExit("eval leakage manifest_sha256 is not bound to checkpoint corpus_manifest.jsonl.")
    if str(leakage.get("split_policy") or "") != SPLIT_POLICY:
        raise SystemExit("eval leakage evidence did not use the release split policy.")
    if not paths_equivalent(str(leakage.get("manifest") or ""), Path(str(corpus_manifest_record.get("path") or ""))):
        raise SystemExit("eval leakage manifest path is not bound to checkpoint corpus_manifest.jsonl.")
    contamination_checks = eval_payload.get("eval_contamination_checks") or []
    if not contamination_checks:
        raise SystemExit("eval contamination check is missing or not clean.")
    for check in contamination_checks:
        if int(check.get("checked", 0)) <= 0 or int(check.get("hits", -1)) != 0:
            raise SystemExit("eval contamination check is missing or not clean.")
    train_path = str(metadata.get("data_path") or "")
    if train_path:
        train_paths = [
            str(path)
            for check in contamination_checks
            for path in check.get("train_paths", [])
        ]
        if not path_in_values(train_path, train_paths):
            raise SystemExit("eval contamination train paths do not include checkpoint metadata data_path.")
    source_overlap_checks = eval_payload.get("eval_source_overlap_checks") or []
    if not source_overlap_checks:
        raise SystemExit("eval source-overlap check is missing or not clean.")
    expected_source_overlap_roles = {"train", "validation", "test", "reference", "excluded"}
    for check in source_overlap_checks:
        if int(check.get("checked", 0)) <= 0 or int(check.get("source_items", 0)) <= 0:
            raise SystemExit("eval source-overlap check did not inspect eval rows and source items.")
        if int(check.get("hits", -1)) != 0:
            raise SystemExit("eval source-overlap check found copied source items.")
        roles_checked = (
            {str(role) for role in check.get("source_roles_checked", [])}
            if isinstance(check.get("source_roles_checked"), list)
            else set()
        )
        if roles_checked != expected_source_overlap_roles:
            raise SystemExit(
                "eval source-overlap check did not cover train/validation/test/reference/excluded source roles."
            )
        items_by_role = check.get("source_items_by_role")
        if not isinstance(items_by_role, dict):
            raise SystemExit("eval source-overlap check is missing source_items_by_role.")
        for role in sorted(expected_source_overlap_roles):
            if int(items_by_role.get(role, 0) or 0) <= 0:
                raise SystemExit(f"eval source-overlap check did not inspect source role: {role}")
        if str(check.get("split_policy") or "") != SPLIT_POLICY:
            raise SystemExit("eval source-overlap check did not use the release split policy.")
        if str(check.get("manifest_sha256") or "") != str(corpus_manifest_record.get("sha256") or ""):
            raise SystemExit("eval source-overlap manifest_sha256 is not bound to checkpoint corpus_manifest.jsonl.")
        if not paths_equivalent(str(check.get("manifest") or ""), Path(str(corpus_manifest_record.get("path") or ""))):
            raise SystemExit("eval source-overlap manifest path is not bound to checkpoint corpus_manifest.jsonl.")
    eval_files = eval_payload.get("eval_files") or []
    required_eval_roles = {
        "primary",
        "heldout",
        "morphology",
        "grammar_constraints",
        "waka_rules",
        "waka_meter_constraints",
        "waka_generation_prompts",
    }
    roles = {str(item.get("role")) for item in eval_files if isinstance(item, dict)}
    missing_roles = sorted(required_eval_roles - roles)
    if missing_roles:
        raise SystemExit(f"eval evidence snapshots missing roles: {missing_roles}")
    for item in eval_files:
        if not isinstance(item, dict):
            raise SystemExit("eval_files entries must be objects.")
        path = Path(str(item.get("path") or ""))
        if not path.exists():
            raise SystemExit(f"eval evidence snapshot is missing locally: {path}")
        if file_sha256(path) != str(item.get("sha256") or ""):
            raise SystemExit(f"eval evidence snapshot hash mismatch: {path}")
        if int(item.get("rows", 0) or 0) <= 0:
            raise SystemExit(f"eval evidence snapshot has no rows: {path}")
        case_ids = item.get("case_ids") or []
        if not isinstance(case_ids, list) or len(case_ids) != int(item.get("rows", 0)):
            raise SystemExit(f"eval evidence snapshot case_ids do not match row count: {path}")
        content_hashes = item.get("content_hashes") or []
        if not isinstance(content_hashes, list) or len(content_hashes) != int(item.get("rows", 0)):
            raise SystemExit(f"eval evidence snapshot content_hashes do not match row count: {path}")
        if len(set(str(value) for value in content_hashes)) != len(content_hashes):
            raise SystemExit(f"eval evidence snapshot has duplicate content hashes: {path}")
        for hash_key in (
            "source_sha256",
            "audited_source_sha256",
            "eval_provenance_manifest_sha256",
        ):
            if not re.fullmatch(r"[0-9a-f]{64}", str(item.get(hash_key) or "")):
                raise SystemExit(f"eval evidence snapshot is missing {hash_key}: {path}")
        if int(item.get("removed_from_source", -1)) < 0:
            raise SystemExit(f"eval evidence snapshot has invalid removed_from_source: {path}")

    eval_provenance_audit = eval_payload.get("eval_provenance_audit")
    if not isinstance(eval_provenance_audit, dict):
        raise SystemExit("eval provenance manifest audit evidence is missing.")
    if int(eval_provenance_audit.get("entries", 0) or 0) < len(required_eval_roles):
        raise SystemExit("eval provenance manifest audit has too few entries.")
    if int(eval_provenance_audit.get("errors", -1)) != 0:
        raise SystemExit("eval provenance manifest audit is not clean.")
    if bool(eval_provenance_audit.get("llm_generated_eval_answer_text")) is not False:
        raise SystemExit("eval provenance manifest must attest llm_generated_eval_answer_text=false.")
    if not re.fullmatch(r"[0-9a-f]{64}", str(eval_provenance_audit.get("manifest_sha256") or "")):
        raise SystemExit("eval provenance manifest audit is missing a manifest hash.")
    eval_manifest_sha = str(eval_provenance_audit.get("manifest_sha256") or "")
    for item in eval_files:
        if str(item.get("eval_provenance_manifest_sha256") or "") != eval_manifest_sha:
            raise SystemExit("eval snapshot is not bound to the audited eval provenance manifest.")

    source_record_audits = eval_payload.get("source_record_audits")
    if not isinstance(source_record_audits, list) or len(source_record_audits) < 2:
        raise SystemExit("eval source_record_audits evidence is missing.")
    audited_names = set()
    for audit in source_record_audits:
        if not isinstance(audit, dict):
            raise SystemExit("source_record_audits entries must be objects.")
        if int(audit.get("checked", 0) or 0) <= 0:
            raise SystemExit("source_record_audits entry did not check any files.")
        if int(audit.get("mismatches", -1)) != 0 or int(audit.get("missing", -1)) != 0:
            raise SystemExit("source_record_audits entry is not clean.")
        if bool(audit.get("fixed")):
            raise SystemExit("source_record_audits release evidence must be non-mutating, fixed=false.")
        audited_names.add(Path(str(audit.get("path") or "")).name)
    if not {"aozora_sources.json", "waka_sources.json"}.issubset(audited_names):
        raise SystemExit("source_record_audits must include checkpoint-bound aozora_sources.json and waka_sources.json.")

    public_manifest_audit = eval_payload.get("public_manifest_audit")
    if not isinstance(public_manifest_audit, dict):
        raise SystemExit("eval public_manifest_audit evidence is missing.")
    if int(public_manifest_audit.get("manifest_rows", 0) or 0) <= 0:
        raise SystemExit("public_manifest_audit did not inspect manifest rows.")
    if int(public_manifest_audit.get("included_rows", 0) or 0) <= 0:
        raise SystemExit("public_manifest_audit found no included rows.")
    if int(public_manifest_audit.get("errors", -1)) != 0:
        raise SystemExit("public_manifest_audit is not clean.")


def model_card(model_id: str, metadata: dict[str, Any], source_summary: dict[str, Any]) -> str:
    params = metadata.get("param_count")
    params_b = metadata.get("param_count_b")
    params_line = "unknown"
    if isinstance(params, int):
        params_line = f"{params:,} parameters"
        if isinstance(params_b, float):
            params_line += f" ({params_b:.3f}B)"
    return f"""---
license: cc-by-sa-4.0
language:
- ja
tags:
- classical-japanese
- old-japanese
- kobun
- gpt
- pytorch
- safetensors
---

# {model_id}

This is a from-scratch character-level GPT-style model for classical Japanese experiments.
It does not use OpenAI APIs or pretrained model weights.

## Model

- Parameters: {params_line}
- Architecture: Qwen3-style local GPT block when trained with the 0.1B script
- Intended use: classical Japanese generation and research experiments
- Not intended for: general chat, factual QA, modern Japanese assistant behavior

## Training Data Policy

The release package intentionally does not include raw text, cleaned text, training text,
validation text, test text, logs, optimizer state, or full training checkpoints.

Source summary:

- Included source rows: {source_summary.get("included_sources")}
- Excluded source rows: {source_summary.get("excluded_sources")}
- Included source characters in manifest: {source_summary.get("included_characters")}

Public-source provenance is provided in `source_manifest_summary.json`.
Reuse conditions depend on the upstream source, especially Aozora Bunko handling rules
and Japanese Wikisource CC BY-SA/GFDL page notices.

## Evaluation

See `eval_results.json`. Publish this model only after rerunning the quality gates against
the exact exported checkpoint. The model-facing release metric is test language-model
loss over the checkpoint-bound test snapshot that was not used for training or checkpoint selection. Minimal pairs, morphology adversarial
cases, rule tables, and constrained waka meter generation are smoke/static checks unless
separate source-bound heldout suites are added.

## Limitations

This is a small specialized model. Grammar and waka meter behavior should be treated as
research output, not guaranteed linguistic authority. Exact 5-7-5-7-7 meter is enforced
by decoding constraints for kana readings. The constrained-generation metric tests that
decoding path, not unconstrained model generalization.
"""


def main() -> None:
    args = parse_args()
    if not args.confirm_explicit_user_request:
        raise SystemExit("HF package creation is manual-only; pass --confirm-explicit-user-request after an explicit user request.")
    release_root = (Path.cwd() / "release").resolve(strict=False)
    target_dir = args.out_dir
    target_resolved = target_dir.resolve(strict=False)
    temp_dir = target_dir.with_name(target_dir.name + ".tmp")
    temp_resolved = temp_dir.resolve(strict=False)
    if target_resolved == release_root or release_root not in target_resolved.parents:
        raise SystemExit(f"refusing to export outside a release subdirectory: {target_dir}")
    if temp_resolved == release_root or release_root not in temp_resolved.parents:
        raise SystemExit(f"refusing to create temp package outside a release subdirectory: {temp_dir}")
    if args.eval_results is None:
        raise SystemExit("--eval-results is required for every release package.")
    eval_payload = read_json(args.eval_results)
    if not isinstance(eval_payload, dict):
        raise SystemExit(f"{args.eval_results} must contain a JSON object.")

    target_has_contents = target_dir.exists() and any(target_dir.iterdir())
    if target_has_contents:
        if not args.force:
            raise SystemExit(f"{target_dir} is not empty. Pass --force to overwrite package files.")

    payload = load_trusted_checkpoint(args.checkpoint, map_location="cpu")
    metadata = dict(payload.get("metadata", {}))
    require_release_candidate_run(str(metadata.get("run_id") or args.checkpoint.stem.removesuffix("_best")), context="HF export")
    run_checkpoint_gates(args.checkpoint, args.eval_results, args.model_id)
    validate_eval_payload(eval_payload, args.checkpoint, payload)
    require_checkpoint_provenance(metadata)
    manifest_path = checkpoint_provenance_path(metadata, "corpus_manifest.jsonl")
    augmentation_manifest_path = checkpoint_provenance_path(metadata, "training_augmentation_manifest.json")
    waka_sources_path = checkpoint_provenance_path(metadata, "waka_sources.json")
    source_manifest = release_source_manifest(manifest_path)
    source_manifest["training_augmentations"] = release_augmentation_summary(augmentation_manifest_path)
    source_issues = source_manifest_issues(source_manifest)
    if source_issues:
        preview = "\n".join(source_issues[:20])
        raise SystemExit(f"source manifest is not release-ready:\n{preview}")

    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    package_dir = temp_dir
    try:
        package_dir.mkdir(parents=True, exist_ok=True)

        state_dict = payload["model"]
        config = GPTConfig(**payload["config"])
        model = GPT(config)
        model.load_state_dict(state_dict)
        if config.tie_word_embeddings:
            model.tie_weights()
        model.eval()
        metadata["checkpoint"] = str(args.checkpoint)
        metadata["checkpoint_sha256"] = checkpoint_sha256(args.checkpoint)
        metadata["checkpoint_step"] = payload.get("step")
        metadata["checkpoint_best_val"] = payload.get("best_val")
        package_metadata = sanitize_training_metadata(metadata)

        try:
            from safetensors.torch import save_model
        except ImportError as exc:
            raise SystemExit(
                "safetensors export unavailable; install release extras with "
                f"`python -m pip install -e .[release]`. reason={exc}"
            ) from exc
        save_model(model, str(package_dir / "model.safetensors"), metadata={"format": "pt"})

        write_json(package_dir / "config.json", payload["config"])
        write_json(package_dir / "tokenizer.json", payload["tokenizer"])
        write_json(package_dir / "training_metadata.json", package_metadata)
        write_json(
            package_dir / "generation_config.json",
            {
                "temperature": 0.7,
                "top_k": 20,
                "top_p": 0.8,
                "presence_penalty": 0.2,
                "max_new_tokens": 260,
                "soft_grammar_constraints": True,
                "grammar_rerank": True,
                "candidates": 5,
            },
        )
        write_json(package_dir / "eval_results.json", public_eval_payload(eval_payload))
        source_summary = summarize_sources(manifest_path, waka_sources_path)
        write_json(package_dir / "source_manifest_summary.json", source_summary)
        write_json(package_dir / "source_manifest.json", source_manifest)
        (package_dir / "README.md").write_text(model_card(args.model_id, package_metadata, source_summary), encoding="utf-8")

        scan_args = [
            sys.executable,
            str(Path(__file__).with_name("check_release_package.py")),
            "--release-dir",
            str(package_dir),
            "--require-passed-eval",
            "--require-safetensors",
        ]
        scan = subprocess.run(scan_args, check=False)
        if scan.returncode != 0:
            raise SystemExit(f"release package self-check failed before publishing temp dir: exit={scan.returncode}")

        if target_dir.exists():
            shutil.rmtree(target_dir)
        temp_dir.replace(target_dir)
    except BaseException:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        raise

    print(f"exported release package: {target_dir}")
    print("does_not_include=raw_text,clean_text,training_text,validation_text,test_text,logs,optimizer_state,full_checkpoint")


if __name__ == "__main__":
    main()
