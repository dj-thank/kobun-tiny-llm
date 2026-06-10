from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

REQUIRED_AUGMENTATION_ROLES = {
    "auxiliary_rule_table",
    "genre_rule_table",
    "grammar_rule_text",
    "external_knowledge_surface_patterns",
    "morphology_examples",
    "train_preference_pairs",
    "waka_meter_training_text",
}

ALLOWED_AUGMENTATION_SOURCE_TYPES = {
    "derived_from_train_split_public_waka_records",
    "project_authored_preference_pairs",
    "project_authored_rule_table",
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_augmentation_manifest(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read augmentation manifest {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("augmentation manifest must be a JSON object")
    return payload


def audit_augmentation_manifest(path: Path, *, require_local_files: bool = True) -> list[str]:
    payload = load_augmentation_manifest(path)
    errors: list[str] = []
    if payload.get("schema") != "old_japanese_training_augmentation_manifest_v1":
        errors.append("schema mismatch")
    if payload.get("llm_generated_corpus_text") is not False:
        errors.append("top-level llm_generated_corpus_text must be false")
    if not str(payload.get("attestation") or "").strip():
        errors.append("missing attestation")
    entries = payload.get("entries")
    if not isinstance(entries, list) or not entries:
        errors.append("entries missing")
        return errors
    seen_roles: set[str] = set()
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            errors.append(f"entry {index} is not an object")
            continue
        role = str(entry.get("role") or "")
        source_type = str(entry.get("source_type") or "")
        seen_roles.add(role)
        if role not in REQUIRED_AUGMENTATION_ROLES:
            errors.append(f"entry {index} unsupported role={role!r}")
        if source_type not in ALLOWED_AUGMENTATION_SOURCE_TYPES:
            errors.append(f"entry {index} unsupported source_type={source_type!r}")
        if entry.get("llm_generated_corpus_text") is not False:
            errors.append(f"entry {index} llm_generated_corpus_text must be false")
        for key in ("path", "sha256", "copyability_status", "public_release_policy", "used_by"):
            if not str(entry.get(key) or "").strip():
                errors.append(f"entry {index} missing {key}")
        repeat_count = entry.get("repeat_count")
        if not isinstance(repeat_count, int) or repeat_count <= 0:
            errors.append(f"entry {index} repeat_count must be a positive integer")
        byte_count = entry.get("bytes")
        if not isinstance(byte_count, int) or byte_count <= 0:
            errors.append(f"entry {index} bytes must be a positive integer")
        line_count = entry.get("lines")
        if not isinstance(line_count, int) or line_count <= 0:
            errors.append(f"entry {index} lines must be a positive integer")
        if require_local_files:
            source_path = Path(str(entry.get("path") or ""))
            if not source_path.exists():
                errors.append(f"entry {index} missing local file: {source_path}")
            else:
                expected_hash = str(entry.get("sha256") or "")
                actual_hash = sha256_file(source_path)
                if actual_hash != expected_hash:
                    errors.append(f"entry {index} hash mismatch: {source_path}")
                if source_path.stat().st_size != byte_count:
                    errors.append(f"entry {index} byte size mismatch: {source_path}")
    missing_roles = sorted(REQUIRED_AUGMENTATION_ROLES - seen_roles)
    if missing_roles:
        errors.append(f"missing augmentation roles: {missing_roles}")
    transforms = payload.get("transform_scripts")
    if not isinstance(transforms, list) or not transforms:
        errors.append("transform_scripts missing")
    else:
        for index, transform in enumerate(transforms):
            if not isinstance(transform, dict):
                errors.append(f"transform script {index} is not an object")
                continue
            if not str(transform.get("path") or "").strip():
                errors.append(f"transform script {index} missing path")
            if not str(transform.get("sha256") or "").strip():
                errors.append(f"transform script {index} missing sha256")
            if require_local_files and str(transform.get("path") or "").strip():
                transform_path = Path(str(transform.get("path") or ""))
                if not transform_path.exists():
                    errors.append(f"transform script {index} missing local file: {transform_path}")
                else:
                    expected_hash = str(transform.get("sha256") or "")
                    actual_hash = sha256_file(transform_path)
                    if expected_hash and actual_hash != expected_hash:
                        errors.append(f"transform script {index} hash mismatch: {transform_path}")
    return errors


def require_clean_augmentation_manifest(path: Path, *, require_local_files: bool = True) -> None:
    errors = audit_augmentation_manifest(path, require_local_files=require_local_files)
    if errors:
        preview = "\n".join(errors[:20])
        raise SystemExit(f"training augmentation manifest is not release-ready: {path}\n{preview}")
