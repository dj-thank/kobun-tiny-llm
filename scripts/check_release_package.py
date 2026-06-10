from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from split_policy import SPLIT_POLICY


EXPECTED_EVAL_SOURCE_OVERLAP_ROLES = {"train", "validation", "test", "reference", "excluded"}


REQUIRED_FILES = {
    "README.md",
    "config.json",
    "tokenizer.json",
    "training_metadata.json",
    "generation_config.json",
    "eval_results.json",
    "source_manifest_summary.json",
    "source_manifest.json",
}
ALLOWED_FILES = REQUIRED_FILES | {"model.safetensors"}
FORBIDDEN_SUFFIXES = {
    ".txt",
    ".log",
    ".jsonl",
    ".ckpt",
    ".bin",
    ".pt",
    ".pth",
    ".tmp",
    ".cache",
    ".pkl",
    ".pickle",
    ".npy",
    ".npz",
    ".sqlite",
    ".db",
    ".parquet",
    ".arrow",
    ".csv",
    ".tsv",
}
FORBIDDEN_PATH_PARTS = {
    "raw",
    "clean",
    "training",
    "train",
    "validation",
    "val",
    "logs",
    "checkpoints",
    "codex_context",
    "__pycache__",
}
SECRET_PATTERNS = (
    re.compile(r"api[_-]?key\s*[:=]", re.IGNORECASE),
    re.compile(r"secret[_-]?(?:key|token)\s*[:=]", re.IGNORECASE),
    re.compile(r"password\s*[:=]", re.IGNORECASE),
    re.compile(r"access[_-]?token\s*[:=]", re.IGNORECASE),
    re.compile(r"auth[_-]?token\s*[:=]", re.IGNORECASE),
    re.compile(r"bearer\s+[A-Za-z0-9._-]+", re.IGNORECASE),
    re.compile(r"hf_[A-Za-z0-9_]{20,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY"),
)
CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")
ABSOLUTE_LOCAL_PATH_PATTERNS = (
    re.compile(r"\b[A-Za-z]:\\"),
    re.compile(r"/(?:Users|home|mnt|private|var|tmp)/"),
    re.compile(r"(?:^|[\"'\\s])(?:logs|checkpoints|data[/\\\\]run_snapshots)[/\\\\]", re.IGNORECASE),
    re.compile(r"data[/\\\\](?:aozora[/\\\\](?:raw|clean)|waka[/\\\\](?:records|training|readings))[/\\\\]", re.IGNORECASE),
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
PUBLIC_TRAINING_METADATA_KEYS = {
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
    "provenance_files",
    "release_metadata_policy",
    "release_name",
    "resume",
    "run_id",
    "seed",
    "test_data_chars",
    "test_data_path",
    "test_data_sha256",
    "test_oov_chars_count",
    "tokenizer_extra_data",
    "tokenizer_source",
    "val_data_chars",
    "val_data_path",
    "val_data_sha256",
    "val_oov_chars_count",
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fail if a release package contains unsafe artifacts.")
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--require-safetensors", action="store_true")
    parser.add_argument("--require-passed-eval", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scan_text_file(path: Path) -> list[str]:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return []
    hits = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        if CONTROL_CHARS_RE.search(line):
            hits.append(f"{path}:{line_no}: unexpected control character")
        for pattern in SECRET_PATTERNS:
            if pattern.search(line):
                hits.append(f"{path}:{line_no}: secret-like pattern {pattern.pattern!r}")
        for pattern in ABSOLUTE_LOCAL_PATH_PATTERNS:
            if pattern.search(line):
                hits.append(f"{path}:{line_no}: absolute local path pattern {pattern.pattern!r}")
    return hits


def scan_json_values_for_controls(value: Any, owner: Path, json_path: str = "$") -> list[str]:
    if isinstance(value, str):
        if CONTROL_CHARS_RE.search(value):
            return [f"{owner}: {json_path} contains unexpected control character"]
        return []
    if isinstance(value, list):
        issues: list[str] = []
        for index, item in enumerate(value):
            issues.extend(scan_json_values_for_controls(item, owner, f"{json_path}[{index}]"))
        return issues
    if isinstance(value, dict):
        issues = []
        for key, item in value.items():
            issues.extend(scan_json_values_for_controls(item, owner, f"{json_path}.{key}"))
        return issues
    return []


def paths_equivalent(left: str, right: str) -> bool:
    if not left or not right:
        return False
    try:
        left_resolved = Path(left).resolve(strict=False)
        right_resolved = Path(right).resolve(strict=False)
    except OSError:
        return Path(left).as_posix().casefold() == Path(right).as_posix().casefold()
    return str(left_resolved).casefold() == str(right_resolved).casefold()


def provenance_record(metadata: dict[str, Any], filename: str) -> dict[str, Any] | None:
    for record in metadata.get("provenance_files", []) or []:
        if not isinstance(record, dict):
            continue
        raw_path = str(record.get("path") or "")
        if raw_path and Path(raw_path).name == filename:
            return record
    return None


def basename_equivalent(left: str, right: str) -> bool:
    if not left or not right:
        return False
    return Path(left).name.casefold() == Path(right).name.casefold()


def is_sanitized_basename(value: str) -> bool:
    if not value:
        return True
    if Path(value).is_absolute() or re.search(r"\b[A-Za-z]:\\", value):
        return False
    if "\\" in value or "/" in value:
        return False
    return value == PureWindowsPath(value).name == PurePosixPath(value).name


def require_sanitized_field(issues: list[str], owner: Path, field: str, value: Any) -> None:
    if isinstance(value, str) and not is_sanitized_basename(value):
        issues.append(f"{owner}: {field} is not sanitized basename metadata: {value}")


def require_sanitized_list(issues: list[str], owner: Path, field: str, values: Any) -> None:
    if not isinstance(values, list):
        return
    for index, value in enumerate(values):
        if isinstance(value, str) and not is_sanitized_basename(value):
            issues.append(f"{owner}: {field}[{index}] is not sanitized basename metadata: {value}")


def check_required_provenance(metadata: dict[str, Any], eval_results: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    required = {
        "aozora_sources.json",
        "corpus_manifest.jsonl",
        "public_manifest_summary.json",
        "snapshot_manifest.json",
        "tokenizer_public_char_vocab.meta.json",
        "training_augmentation_manifest.json",
        "waka_sources.json",
    }
    for filename in sorted(required):
        record = provenance_record(metadata, filename)
        if record is None:
            issues.append(f"training_metadata.json: missing checkpoint-bound provenance {filename}")
            continue
        raw_path = str(record.get("path") or "")
        expected_hash = str(record.get("sha256") or "")
        if not raw_path or not expected_hash:
            issues.append(f"training_metadata.json: provenance {filename} missing path or sha256")
            continue
        if not is_sanitized_basename(raw_path):
            issues.append(f"training_metadata.json: provenance {filename} path is not sanitized: {raw_path}")
    corpus_manifest = provenance_record(metadata, "corpus_manifest.jsonl")
    leakage = eval_results.get("leakage") if isinstance(eval_results, dict) else None
    if corpus_manifest is not None and isinstance(leakage, dict):
        if str(leakage.get("manifest_sha256") or "") != str(corpus_manifest.get("sha256") or ""):
            issues.append("eval_results.json: leakage manifest_sha256 is not bound to checkpoint corpus_manifest.jsonl")
        if not basename_equivalent(str(leakage.get("manifest") or ""), str(corpus_manifest.get("path") or "")):
            issues.append("eval_results.json: leakage manifest path/name is not bound to checkpoint corpus_manifest.jsonl")
    return issues


def check_eval_binding(release_dir: Path) -> list[str]:
    issues: list[str] = []
    eval_path = release_dir / "eval_results.json"
    metadata_path = release_dir / "training_metadata.json"
    if not eval_path.exists() or not metadata_path.exists():
        return issues
    eval_results = read_json(eval_path)
    metadata = read_json(metadata_path)
    eval_unknown = sorted(set(eval_results) - PUBLIC_EVAL_TOP_LEVEL_KEYS)
    if eval_unknown:
        issues.append(f"{eval_path}: unexpected top-level public eval keys {eval_unknown}")
    metadata_unknown = sorted(set(metadata) - PUBLIC_TRAINING_METADATA_KEYS)
    if metadata_unknown:
        issues.append(f"{metadata_path}: unexpected public metadata keys {metadata_unknown}")
    checkpoint = metadata.get("checkpoint")
    if not checkpoint or checkpoint == "static-only":
        issues.append(f"{metadata_path}: checkpoint is missing or static-only")
    if not is_sanitized_basename(str(checkpoint or "")):
        issues.append(f"{metadata_path}: checkpoint path is not sanitized")
    metadata_checkpoint_sha = str(metadata.get("checkpoint_sha256") or "")
    if not re.fullmatch(r"[0-9a-f]{64}", metadata_checkpoint_sha):
        issues.append(f"{metadata_path}: checkpoint_sha256 is missing or malformed")
    for key in ("data_path", "val_data_path", "test_data_path", "init_from", "resume"):
        value = str(metadata.get(key) or "")
        if value and not is_sanitized_basename(value):
            issues.append(f"{metadata_path}: {key} path is not sanitized")
    for key in ("tokenizer_extra_data", "provenance_files"):
        records = metadata.get(key) or []
        if isinstance(records, list):
            for index, record in enumerate(records):
                if isinstance(record, dict):
                    record_path = str(record.get("path") or "")
                    if record_path and not is_sanitized_basename(record_path):
                        issues.append(f"{metadata_path}: {key}[{index}] path is not sanitized")
    if not basename_equivalent(str(eval_results.get("checkpoint") or ""), str(checkpoint or "")):
        issues.append(f"{eval_path}: checkpoint does not match training_metadata checkpoint")
    if not basename_equivalent(str(eval_results.get("checkpoint_from_log") or ""), str(checkpoint or "")):
        issues.append(f"{eval_path}: checkpoint_from_log does not match training_metadata checkpoint")
    eval_checkpoint_sha = str(eval_results.get("checkpoint_sha256") or "")
    if not re.fullmatch(r"[0-9a-f]{64}", eval_checkpoint_sha):
        issues.append(f"{eval_path}: missing checkpoint_sha256")
    elif metadata_checkpoint_sha and eval_checkpoint_sha != metadata_checkpoint_sha:
        issues.append(f"{eval_path}: checkpoint_sha256 does not match training_metadata checkpoint_sha256")
    if eval_results.get("checkpoint_step") != metadata.get("checkpoint_step"):
        issues.append(f"{eval_path}: checkpoint_step does not match training metadata")
    if eval_results.get("duplicate_metrics"):
        issues.append(f"{eval_path}: duplicate metric keys present")
    model_metrics = eval_results.get("model_metrics") or {}
    required_model_metrics = {"test_lm_token_nll"}
    missing_model_metrics = sorted(required_model_metrics - set(model_metrics))
    if missing_model_metrics:
        issues.append(f"{eval_path}: missing model-facing metrics {missing_model_metrics}")
    test_metric = model_metrics.get("test_lm_token_nll") or {}
    test_loss = float(test_metric.get("value", float("inf")))
    if not 0.0 <= test_loss <= 8.0:
        issues.append(f"{eval_path}: test_lm_token_nll outside release threshold")
    smoke_metrics = eval_results.get("smoke_metrics") or {}
    for metric_name, requirements in REQUIRED_SMOKE_METRICS.items():
        metric = smoke_metrics.get(metric_name) or (eval_results.get("metrics") or {}).get(metric_name) or {}
        value = float(metric.get("value", float("-inf")))
        min_value = float(requirements["min_value"])
        if value < min_value:
            issues.append(f"{eval_path}: smoke/static metric {metric_name}={value} below required {min_value}")
        total = int(metric.get("total", metric.get("denominator", metric.get("count", 0))) or 0)
        min_total = int(requirements["min_total"])
        if total < min_total:
            issues.append(f"{eval_path}: smoke/static metric {metric_name} has only {total} cases, below required {min_total}")
    test_lm = eval_results.get("test_lm")
    if not isinstance(test_lm, dict):
        issues.append(f"{eval_path}: missing test_lm record")
    else:
        if not basename_equivalent(str(test_lm.get("test_data") or ""), str(metadata.get("test_data_path") or "")):
            issues.append(f"{eval_path}: test_lm test_data does not match checkpoint test_data_path")
        if str(test_lm.get("test_sha256") or "") != str(metadata.get("test_data_sha256") or ""):
            issues.append(f"{eval_path}: test_lm test_sha256 does not match checkpoint test_data_sha256")
    tokenizer_scope = eval_results.get("tokenizer_vocab_scope")
    if not isinstance(tokenizer_scope, dict):
        issues.append(f"{eval_path}: missing tokenizer_vocab_scope evidence")
    else:
        if str(tokenizer_scope.get("policy") or "") != "train_split_plus_core_japanese_inventory_plus_utf8_byte_fallback_v1":
            issues.append(f"{eval_path}: tokenizer_vocab_scope policy is not release-safe")
        if tokenizer_scope.get("byte_fallback") is not True or int(tokenizer_scope.get("byte_fallback_tokens", -1)) != 256:
            issues.append(f"{eval_path}: tokenizer_vocab_scope does not prove UTF-8 byte fallback coverage")
        if int(tokenizer_scope.get("tokenizer_chars", 1_000_000) or 1_000_000) >= 10_000:
            issues.append(f"{eval_path}: tokenizer vocab is too large for the DirectML release policy")
        if bool(tokenizer_scope.get("tokenizer_meta_verified")) is not True:
            issues.append(f"{eval_path}: tokenizer_vocab_scope did not verify tokenizer metadata")
        if int(tokenizer_scope.get("forbidden_heldout_tokenizer_leakage", -1)) != 0:
            issues.append(f"{eval_path}: tokenizer vocab scope found heldout-derived leakage")
        if int(tokenizer_scope.get("heldout_missing_from_tokenizer", -1)) != 0:
            issues.append(f"{eval_path}: tokenizer vocab scope found heldout OOV")
        tokenizer_meta_record = provenance_record(metadata, "tokenizer_public_char_vocab.meta.json")
        if tokenizer_meta_record is None:
            issues.append(f"{metadata_path}: missing tokenizer_public_char_vocab.meta.json provenance")
        elif str(tokenizer_scope.get("tokenizer_meta_sha256") or "") != str(tokenizer_meta_record.get("sha256") or ""):
            issues.append(f"{eval_path}: tokenizer metadata hash is not bound to checkpoint provenance")
        tokenizer_records = metadata.get("tokenizer_extra_data") or []
        if not isinstance(tokenizer_records, list) or not tokenizer_records:
            issues.append(f"{metadata_path}: missing tokenizer_extra_data record")
        elif str(tokenizer_scope.get("vocab_sha256") or "") != str(tokenizer_records[0].get("sha256") or ""):
            issues.append(f"{eval_path}: tokenizer vocab hash is not bound to checkpoint tokenizer extra data")
        checkpoint_tokenizer_scope = eval_results.get("checkpoint_tokenizer_vocab_scope")
        if not isinstance(checkpoint_tokenizer_scope, dict):
            issues.append(f"{eval_path}: missing checkpoint_tokenizer_vocab_scope evidence")
        else:
            if checkpoint_tokenizer_scope.get("checkpoint_bound") is not True:
                issues.append(f"{eval_path}: checkpoint_tokenizer_vocab_scope is not checkpoint-bound")
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
                    issues.append(f"{eval_path}: checkpoint tokenizer scope disagrees on {key}")
    split_consistency = eval_results.get("split_consistency")
    if not isinstance(split_consistency, dict):
        issues.append(f"{eval_path}: missing split_consistency evidence")
    else:
        for field in ("manifest", "train", "validation", "test"):
            require_sanitized_field(issues, eval_path, f"split_consistency.{field}", split_consistency.get(field))
        if not split_consistency.get("train_match"):
            issues.append(f"{eval_path}: split_consistency did not verify full train corpus reconstruction")
        if not split_consistency.get("validation_match"):
            issues.append(f"{eval_path}: split_consistency did not verify validation split")
        if not split_consistency.get("test_match"):
            issues.append(f"{eval_path}: split_consistency did not verify test split")
        if str(split_consistency.get("split_policy") or "") != SPLIT_POLICY:
            issues.append(f"{eval_path}: split_consistency did not use release split policy")
        if split_consistency.get("group_disjoint") is not True:
            issues.append(f"{eval_path}: split_consistency did not verify group-disjoint splits")
        if int(split_consistency.get("test_sources", 0) or 0) < 3:
            issues.append(f"{eval_path}: split_consistency has too few test sources")
        if int(split_consistency.get("test_source_chars", 0) or 0) < 30_000:
            issues.append(f"{eval_path}: split_consistency has too few test source characters")
        if int(split_consistency.get("test_text_chars", 0) or 0) < 20_000:
            issues.append(f"{eval_path}: split_consistency has too few generated test text characters")
        corpus_manifest = provenance_record(metadata, "corpus_manifest.jsonl")
        if corpus_manifest is not None and str(split_consistency.get("manifest_sha256") or "") != str(corpus_manifest.get("sha256") or ""):
            issues.append(f"{eval_path}: split_consistency manifest hash is not bound to checkpoint provenance")
        augmentation_manifest = provenance_record(metadata, "training_augmentation_manifest.json")
        if augmentation_manifest is not None and str(split_consistency.get("augmentation_manifest_sha256") or "") != str(augmentation_manifest.get("sha256") or ""):
            issues.append(f"{eval_path}: split_consistency augmentation manifest hash is not bound to checkpoint provenance")
        if str(split_consistency.get("test_sha256") or "") != str(metadata.get("test_data_sha256") or ""):
            issues.append(f"{eval_path}: split_consistency test hash does not match checkpoint metadata")
        if str(split_consistency.get("validation_sha256") or "") != str(metadata.get("val_data_sha256") or ""):
            issues.append(f"{eval_path}: split_consistency validation hash does not match checkpoint metadata")
    corpus_checks = [str(value) for value in eval_results.get("corpus_checks", [])]
    for label, metadata_key in (
        ("train", "data_path"),
        ("validation", "val_data_path"),
        ("test", "test_data_path"),
    ):
        expected_path = str(metadata.get(metadata_key) or "")
        if not expected_path:
            issues.append(f"{metadata_path}: missing {metadata_key}")
        elif not any(basename_equivalent(value, expected_path) for value in corpus_checks):
            issues.append(f"{eval_path}: corpus validation evidence is not bound to checkpoint {label} snapshot")
    leakage = eval_results.get("leakage")
    if not isinstance(leakage, dict):
        issues.append(f"{eval_path}: missing leakage check")
    elif int(leakage.get("checked_sources", 0)) <= 0 or int(leakage.get("leaks", -1)) != 0:
        issues.append(f"{eval_path}: leakage check is missing or not clean")
    elif int(leakage.get("checked_windows", 0) or 0) <= 0:
        issues.append(f"{eval_path}: leakage check did not report checked windows")
    elif str(leakage.get("split_policy") or "") != SPLIT_POLICY:
        issues.append(f"{eval_path}: leakage check did not use release split policy")
    elif int(leakage.get("checked_waka_items", 0) or 0) <= 0:
        issues.append(f"{eval_path}: leakage check did not report checked waka items")
    elif "waka_leaks" not in leakage or int(leakage["waka_leaks"]) != 0:
        issues.append(f"{eval_path}: waka leakage check is missing or not clean")
    elif int(leakage.get("expected_sources", 0) or 0) > 0 and int(leakage.get("checked_sources", 0)) != int(leakage.get("expected_sources", 0)):
        issues.append(f"{eval_path}: leakage check did not cover every expected validation source")
    if isinstance(leakage, dict):
        if int(leakage.get("role_pair_leaks", -1)) != 0:
            issues.append(f"{eval_path}: validation/test prose split leakage check is missing or not clean")
        if int(leakage.get("role_waka_leaks", -1)) != 0:
            issues.append(f"{eval_path}: validation/test waka split leakage check is missing or not clean")
    contamination_checks = eval_results.get("eval_contamination_checks") or []
    top_contamination = eval_results.get("eval_contamination")
    if isinstance(top_contamination, dict):
        require_sanitized_list(issues, eval_path, "eval_contamination.train_paths", top_contamination.get("train_paths"))
        require_sanitized_list(issues, eval_path, "eval_contamination.eval_paths", top_contamination.get("eval_paths"))
    if not contamination_checks:
        issues.append(f"{eval_path}: missing clean final eval contamination check")
    else:
        for index, check in enumerate(contamination_checks):
            require_sanitized_list(issues, eval_path, f"eval_contamination_checks[{index}].train_paths", check.get("train_paths"))
            require_sanitized_list(issues, eval_path, f"eval_contamination_checks[{index}].eval_paths", check.get("eval_paths"))
            if int(check.get("checked", 0)) <= 0 or int(check.get("hits", -1)) != 0:
                issues.append(f"{eval_path}: contamination check {index} is missing or not clean")
        data_path = str(metadata.get("data_path") or "")
        if data_path:
            train_paths = [str(path) for check in contamination_checks for path in check.get("train_paths", [])]
            if not any(basename_equivalent(path, data_path) for path in train_paths):
                issues.append(f"{eval_path}: contamination train paths do not include checkpoint training data_path")
    source_overlap_checks = eval_results.get("eval_source_overlap_checks") or []
    if not source_overlap_checks:
        issues.append(f"{eval_path}: missing clean eval source-overlap check")
    else:
        corpus_manifest = provenance_record(metadata, "corpus_manifest.jsonl")
        for index, check in enumerate(source_overlap_checks):
            require_sanitized_list(issues, eval_path, f"eval_source_overlap_checks[{index}].eval_paths", check.get("eval_paths"))
            require_sanitized_field(issues, eval_path, f"eval_source_overlap_checks[{index}].manifest", check.get("manifest"))
            if int(check.get("checked", 0)) <= 0 or int(check.get("source_items", 0)) <= 0:
                issues.append(f"{eval_path}: source-overlap check {index} did not inspect eval rows and source items")
            if int(check.get("hits", -1)) != 0:
                issues.append(f"{eval_path}: source-overlap check {index} found copied source items")
            roles_checked = {str(role) for role in check.get("source_roles_checked", [])} if isinstance(check.get("source_roles_checked"), list) else set()
            if roles_checked != EXPECTED_EVAL_SOURCE_OVERLAP_ROLES:
                issues.append(
                    f"{eval_path}: source-overlap check {index} did not cover all manifest source roles: "
                    f"expected={sorted(EXPECTED_EVAL_SOURCE_OVERLAP_ROLES)} actual={sorted(roles_checked)}"
                )
            items_by_role = check.get("source_items_by_role")
            if not isinstance(items_by_role, dict):
                issues.append(f"{eval_path}: source-overlap check {index} missing source_items_by_role")
            else:
                for role in sorted(EXPECTED_EVAL_SOURCE_OVERLAP_ROLES):
                    if int(items_by_role.get(role, 0) or 0) <= 0:
                        issues.append(f"{eval_path}: source-overlap check {index} did not inspect role {role}")
            if str(check.get("split_policy") or "") != SPLIT_POLICY:
                issues.append(f"{eval_path}: source-overlap check {index} did not use release split policy")
            if corpus_manifest is not None:
                if str(check.get("manifest_sha256") or "") != str(corpus_manifest.get("sha256") or ""):
                    issues.append(f"{eval_path}: source-overlap manifest hash is not bound to checkpoint provenance")
                if not basename_equivalent(str(check.get("manifest") or ""), str(corpus_manifest.get("path") or "")):
                    issues.append(f"{eval_path}: source-overlap manifest path/name is not bound to checkpoint provenance")
    eval_files = eval_results.get("eval_files") or []
    required_roles = {
        "primary",
        "heldout",
        "morphology",
        "grammar_constraints",
        "waka_rules",
        "waka_meter_constraints",
        "waka_generation_prompts",
    }
    roles = {str(item.get("role")) for item in eval_files if isinstance(item, dict)}
    missing_roles = sorted(required_roles - roles)
    if missing_roles:
        issues.append(f"{eval_path}: missing eval evidence snapshot roles {missing_roles}")
    for item in eval_files:
        if not isinstance(item, dict):
            issues.append(f"{eval_path}: eval_files contains a non-object entry")
            continue
        if Path(str(item.get("path") or "")).is_absolute():
            issues.append(f"{eval_path}: eval evidence snapshot path is not sanitized: {item.get('path')}")
        require_sanitized_field(issues, eval_path, "eval_files.path", item.get("path"))
        require_sanitized_field(issues, eval_path, "eval_files.source", item.get("source"))
        require_sanitized_field(issues, eval_path, "eval_files.audited_source", item.get("audited_source"))
        rows = int(item.get("rows", 0) or 0)
        case_ids = item.get("case_ids") or []
        content_hashes = item.get("content_hashes") or []
        if rows <= 0:
            issues.append(f"{eval_path}: eval evidence snapshot has no rows: {item.get('path')}")
        if not isinstance(case_ids, list) or len(case_ids) != rows:
            issues.append(f"{eval_path}: eval evidence snapshot case_ids do not match row count: {item.get('path')}")
        if not isinstance(content_hashes, list) or len(content_hashes) != rows:
            issues.append(f"{eval_path}: eval evidence snapshot content_hashes do not match row count: {item.get('path')}")
        if isinstance(content_hashes, list) and len(set(str(value) for value in content_hashes)) != len(content_hashes):
            issues.append(f"{eval_path}: eval evidence snapshot has duplicate content hashes: {item.get('path')}")
        if not re.fullmatch(r"[0-9a-f]{64}", str(item.get("sha256") or "")):
            issues.append(f"{eval_path}: eval_files entry missing or malformed sha256: {item.get('path')}")
        for hash_key in ("source_sha256", "audited_source_sha256", "eval_provenance_manifest_sha256"):
            if not re.fullmatch(r"[0-9a-f]{64}", str(item.get(hash_key) or "")):
                issues.append(f"{eval_path}: eval_files entry missing or malformed {hash_key}: {item.get('path')}")
        if int(item.get("removed_from_source", -1)) < 0:
            issues.append(f"{eval_path}: eval_files entry has invalid removed_from_source: {item.get('path')}")
    eval_provenance_audit = eval_results.get("eval_provenance_audit")
    if not isinstance(eval_provenance_audit, dict):
        issues.append(f"{eval_path}: missing eval_provenance_audit evidence")
    else:
        require_sanitized_field(issues, eval_path, "eval_provenance_audit.path", eval_provenance_audit.get("path"))
        if int(eval_provenance_audit.get("entries", 0) or 0) < len(required_roles):
            issues.append(f"{eval_path}: eval_provenance_audit has too few entries")
        if int(eval_provenance_audit.get("errors", -1)) != 0:
            issues.append(f"{eval_path}: eval_provenance_audit is not clean")
        if bool(eval_provenance_audit.get("llm_generated_eval_answer_text")) is not False:
            issues.append(f"{eval_path}: eval provenance must attest llm_generated_eval_answer_text=false")
        if not re.fullmatch(r"[0-9a-f]{64}", str(eval_provenance_audit.get("manifest_sha256") or "")):
            issues.append(f"{eval_path}: eval_provenance_audit missing manifest_sha256")
        manifest_sha = str(eval_provenance_audit.get("manifest_sha256") or "")
        for item in eval_files:
            if str(item.get("eval_provenance_manifest_sha256") or "") != manifest_sha:
                issues.append(f"{eval_path}: eval snapshot is not bound to eval provenance manifest: {item.get('path')}")
    source_record_audits = eval_results.get("source_record_audits")
    if not isinstance(source_record_audits, list) or len(source_record_audits) < 2:
        issues.append(f"{eval_path}: missing source_record_audits evidence")
    else:
        audited_names = set()
        for index, audit in enumerate(source_record_audits):
            if not isinstance(audit, dict):
                issues.append(f"{eval_path}: source_record_audits[{index}] is not an object")
                continue
            require_sanitized_field(issues, eval_path, f"source_record_audits[{index}].path", audit.get("path"))
            audited_names.add(Path(str(audit.get("path") or "")).name)
            if int(audit.get("checked", 0) or 0) <= 0:
                issues.append(f"{eval_path}: source_record_audits[{index}] did not check any files")
            if int(audit.get("mismatches", -1)) != 0 or int(audit.get("missing", -1)) != 0:
                issues.append(f"{eval_path}: source_record_audits[{index}] is not clean")
            if bool(audit.get("fixed")):
                issues.append(f"{eval_path}: source_record_audits[{index}] must be non-mutating fixed=false")
        if not {"aozora_sources.json", "waka_sources.json"}.issubset(audited_names):
            issues.append(f"{eval_path}: source_record_audits must include aozora_sources.json and waka_sources.json")
    public_manifest_audit = eval_results.get("public_manifest_audit")
    if not isinstance(public_manifest_audit, dict):
        issues.append(f"{eval_path}: missing public_manifest_audit evidence")
    else:
        require_sanitized_field(issues, eval_path, "public_manifest_audit.out", public_manifest_audit.get("out"))
        if int(public_manifest_audit.get("manifest_rows", 0) or 0) <= 0:
            issues.append(f"{eval_path}: public_manifest_audit did not inspect manifest rows")
        if int(public_manifest_audit.get("included_rows", 0) or 0) <= 0:
            issues.append(f"{eval_path}: public_manifest_audit found no included rows")
        if int(public_manifest_audit.get("errors", -1)) != 0:
            issues.append(f"{eval_path}: public_manifest_audit is not clean")
    issues.extend(check_required_provenance(metadata, eval_results))
    return issues


def check_source_manifest(release_dir: Path) -> list[str]:
    path = release_dir / "source_manifest.json"
    if not path.exists():
        return []
    payload = read_json(path)
    rows = payload.get("sources")
    if not isinstance(rows, list) or not rows:
        return [f"{path}: missing sources"]
    augmentations = payload.get("training_augmentations")
    if not isinstance(augmentations, dict):
        return [f"{path}: missing training_augmentations"]
    required = {
        "source_id",
        "split_key_sha256",
        "split_policy",
        "split_group_sha256",
        "title",
        "include_in_training",
        "source_kind",
        "license_name",
        "license_note",
        "redistribution_policy",
        "source_url",
        "characters",
        "split",
        "retrieved_at_utc",
        "source_revision",
        "source_revision_timestamp",
        "source_payload_sha256",
        "download_payload_sha256",
        "api_payload_sha256",
        "clean_sha256",
        "records_sha256",
        "readings_sha256",
        "training_sha256",
    }
    issues = []
    augmentation_entries = augmentations.get("entries")
    if not isinstance(augmentation_entries, list) or not augmentation_entries:
        issues.append(f"{path}: training_augmentations has no entries")
    if augmentations.get("llm_generated_corpus_text") is not False:
        issues.append(f"{path}: training_augmentations must attest llm_generated_corpus_text=false")
    for index, entry in enumerate(augmentation_entries or []):
        if not isinstance(entry, dict):
            issues.append(f"{path}: training augmentation index {index} is not an object")
            continue
        if entry.get("llm_generated_corpus_text") is not False:
            issues.append(f"{path}: training augmentation index {index} must have llm_generated_corpus_text=false")
        for field in ("role", "source_type", "sha256", "copyability_status", "repeat_count"):
            if not str(entry.get(field, "")).strip():
                issues.append(f"{path}: training augmentation index {index} missing {field}")
    for index, row in enumerate(rows):
        missing = sorted(key for key in required if key not in row)
        if missing:
            issues.append(f"{path}: source index {index} missing {missing}")
            break
        source_kind = str(row.get("source_kind", ""))
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
        ):
            if not str(row.get(field, "")).strip():
                issues.append(f"{path}: source index {index} missing {field}")
        if str(row.get("split_policy", "")) != SPLIT_POLICY:
            issues.append(f"{path}: source index {index} split_policy={row.get('split_policy')!r}")
        if str(row.get("redistribution_policy", "")) != "corpus_text_not_distributed":
            issues.append(f"{path}: source index {index} redistribution_policy={row.get('redistribution_policy')!r}")
        if not str(row.get("retrieved_at_utc", "")).strip():
            issues.append(f"{path}: source index {index} missing retrieved_at_utc")
        if not str(row.get("clean_sha256", "")).strip():
            issues.append(f"{path}: source index {index} missing clean_sha256")
        if source_kind == "wikisource":
            if not str(row.get("source_revision", "")).strip():
                issues.append(f"{path}: wikisource source index {index} missing source_revision")
            if not str(row.get("source_revision_timestamp", "")).strip():
                issues.append(f"{path}: wikisource source index {index} missing source_revision_timestamp")
            if not (
                str(row.get("source_payload_sha256", "")).strip()
                or str(row.get("api_payload_sha256", "")).strip()
            ):
                issues.append(f"{path}: wikisource source index {index} missing API payload hash")
        elif source_kind == "aozora":
            if not str(row.get("source_payload_sha256", "")).strip():
                issues.append(f"{path}: aozora source index {index} missing card payload hash")
            if not str(row.get("download_payload_sha256", "")).strip():
                issues.append(f"{path}: aozora source index {index} missing downloaded text payload hash")
        if str(row.get("style", "")) == "waka":
            if not str(row.get("records_sha256", "")).strip():
                issues.append(f"{path}: waka source index {index} missing records_sha256")
            if not str(row.get("readings_sha256", "")).strip():
                issues.append(f"{path}: waka source index {index} missing readings_sha256")
    return issues


def main() -> None:
    args = parse_args()
    release_dir = args.release_dir
    if not release_dir.is_dir():
        raise SystemExit(f"missing_release_dir={release_dir}")

    issues: list[str] = []
    names = {path.name for path in release_dir.iterdir() if path.is_file()}
    missing = sorted(REQUIRED_FILES - names)
    if missing:
        issues.append(f"missing_required_files={missing}")
    if args.require_safetensors and "model.safetensors" not in names:
        issues.append("missing_required_model.safetensors")
    if "model.safetensors" not in names:
        issues.append("missing model weights: need model.safetensors")
    if any(name.endswith(".pt") for name in names):
        issues.append("pickle-based .pt weights are forbidden in release packages")

    for path in release_dir.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(release_dir)
        if len(relative.parts) != 1:
            issues.append(f"{path}: nested release files are not allowed")
        if relative.as_posix() not in ALLOWED_FILES:
            issues.append(f"{path}: unexpected release file")
        relative_parts = {part.lower() for part in path.relative_to(release_dir).parts[:-1]}
        bad_parts = sorted(relative_parts.intersection(FORBIDDEN_PATH_PARTS))
        if bad_parts:
            issues.append(f"{path}: forbidden path parts={bad_parts}")
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            issues.append(f"{path}: forbidden suffix {path.suffix}")
        if path.suffix.lower() in {".json", ".md"}:
            issues.extend(scan_text_file(path))
        if path.suffix.lower() == ".json":
            try:
                issues.extend(scan_json_values_for_controls(read_json(path), path))
            except json.JSONDecodeError as exc:
                issues.append(f"{path}: invalid JSON: {exc}")
        if path.suffix.lower() == ".pt":
            issues.append(f"{path}: .pt files are not allowed in release packages")

    eval_path = release_dir / "eval_results.json"
    if eval_path.exists() and args.require_passed_eval:
        eval_results = read_json(eval_path)
        if eval_results.get("status") != "passed":
            issues.append(f"{eval_path}: eval status is not passed")

    issues.extend(check_eval_binding(release_dir))
    issues.extend(check_source_manifest(release_dir))

    if issues:
        for issue in issues:
            print(f"ISSUE {issue}")
        raise SystemExit(1)
    print(f"release_package_ok={release_dir}")


if __name__ == "__main__":
    main()
