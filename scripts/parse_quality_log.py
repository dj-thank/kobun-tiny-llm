from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from kobun_llm.checkpoint_io import load_trusted_checkpoint


RATIO_RE = re.compile(
    r"(?P<name>[A-Za-z][A-Za-z0-9_]*)="
    r"(?P<passed>\d+)/(?P<total>\d+)"
    r"(?:=(?P<value>\d+(?:\.\d+)?))?"
)
KEY_VALUE_RE = re.compile(r"^(?P<key>[A-Za-z][A-Za-z0-9_]*)=(?P<value>.*)$")
LEAKAGE_RE = re.compile(
    r"split_leakage_checked_sources=(?P<checked>\d+)\s+"
    r"(?:expected_sources=(?P<expected>\d+)\s+)?"
    r"(?:split_policy=(?P<split_policy>\S+)\s+)?"
    r"(?:checked_windows=(?P<windows>\d+)\s+)?"
    r"(?:checked_waka_items=(?P<waka_items>\d+)\s+)?"
    r"(?:role_pair_leaks=(?P<role_pair_leaks>\d+)\s+)?"
    r"(?:role_waka_leaks=(?P<role_waka_leaks>\d+)\s+)?"
    r"(?:waka_leaks=(?P<waka_leaks>\d+)\s+)?"
    r"leaks=(?P<leaks>\d+)\s+"
    r"(?:manifest=(?P<manifest>.+?)\s+manifest_sha256=(?P<manifest_sha256>[0-9a-f]{64})\s+)?"
    r"train=(?P<train>.+)$"
)
EVAL_CONTAMINATION_RE = re.compile(r"eval_contamination_checked=(?P<checked>\d+)\s+hits=(?P<hits>\d+)")
EVAL_CONTAMINATION_TRAIN_RE = re.compile(r"eval_contamination_train=(?P<paths>\[.*\])$")
EVAL_CONTAMINATION_EVAL_RE = re.compile(r"eval_contamination_eval=(?P<paths>\[.*\])$")
EVAL_SOURCE_OVERLAP_RE = re.compile(
    r"eval_source_overlap_checked=(?P<checked>\d+)\s+"
    r"source_items=(?P<source_items>\d+)\s+"
    r"(?:source_roles_checked=(?P<source_roles_checked>\[.*?\])\s+)?"
    r"(?:source_items_by_role=(?P<source_items_by_role>\{.*?\})\s+)?"
    r"split_policy=(?P<split_policy>\S+)\s+"
    r"val_ratio=(?P<val_ratio>\d+(?:\.\d+)?)\s+"
    r"test_ratio=(?P<test_ratio>\d+(?:\.\d+)?)\s+"
    r"prose_hits=(?P<prose_hits>\d+)\s+"
    r"waka_exact_hits=(?P<waka_exact_hits>\d+)\s+"
    r"waka_variant_hits=(?P<waka_variant_hits>\d+)\s+"
    r"hits=(?P<hits>\d+)\s+"
    r"manifest=(?P<manifest>.+?)\s+manifest_sha256=(?P<manifest_sha256>[0-9a-f]{64})$"
)
EVAL_SOURCE_OVERLAP_EVAL_RE = re.compile(r"eval_source_overlap_eval=(?P<paths>\[.*\])$")
SOURCE_RECORD_AUDIT_RE = re.compile(
    r"source_record_audit\s+path=(?P<path>.+?)\s+checked=(?P<checked>\d+)\s+"
    r"mismatches=(?P<mismatches>\d+)\s+missing=(?P<missing>\d+)\s+fixed=(?P<fixed>true|false|True|False)"
)
PUBLIC_MANIFEST_AUDIT_RE = re.compile(
    r"manifest_rows=(?P<manifest_rows>\d+)\s+included_rows=(?P<included_rows>\d+)\s+"
    r"errors=(?P<errors>\d+)\s+out=(?P<out>.+)$"
)
EVAL_PROVENANCE_AUDIT_RE = re.compile(
    r"eval_provenance_manifest\s+path=(?P<path>.+?)\s+"
    r"manifest_sha256=(?P<manifest_sha256>[0-9a-f]{64})\s+"
    r"entries=(?P<entries>\d+)\s+errors=(?P<errors>\d+)\s+"
    r"llm_generated_eval_answer_text=(?P<llm_generated>false|true)"
)
EVAL_SNAPSHOT_RE = re.compile(
    r"eval_snapshot_file\s+role=(?P<role>[A-Za-z0-9_-]+)\s+"
    r"path=(?P<path>.+?)\s+sha256=(?P<sha256>[0-9a-f]{64})\s+"
    r"rows=(?P<rows>\d+)\s+case_ids=(?P<case_ids>\[.*?\])\s+"
    r"(?:content_hashes=(?P<content_hashes>\[.*?\])\s+)?"
    r"(?:source_sha256=(?P<source_sha256>[0-9a-f]{64})\s+)?"
    r"(?:audited_source=(?P<audited_source>.+?)\s+)?"
    r"(?:audited_source_sha256=(?P<audited_source_sha256>[0-9a-f]{64})\s+)?"
    r"(?:eval_provenance_manifest_sha256=(?P<eval_provenance_manifest_sha256>[0-9a-f]{64})\s+)?"
    r"(?:removed_from_source=(?P<removed_from_source>\d+)\s+)?"
    r"source=(?P<source>.+)$"
)
TOKENIZER_SCOPE_RE = re.compile(
    r"tokenizer_vocab_scope\s+"
    r"policy=(?P<policy>\S+)\s+"
    r"tokenizer_chars=(?P<tokenizer_chars>\d+)\s+"
    r"(?:direct_vocab_chars=(?P<direct_vocab_chars>\d+)\s+)?"
    r"(?:byte_fallback=(?P<byte_fallback>true|false)\s+)?"
    r"(?:byte_fallback_tokens=(?P<byte_fallback_tokens>\d+)\s+)?"
    r"train_chars=(?P<train_chars>\d+)\s+"
    r"heldout_exclusive_chars=(?P<heldout_exclusive_chars>\d+)\s+"
    r"covered_by_static_inventory=(?P<covered_by_static_inventory>\d+)\s+"
    r"(?:heldout_covered_by_byte_fallback=(?P<heldout_covered_by_byte_fallback>\d+)\s+)?"
    r"forbidden_heldout_tokenizer_leakage=(?P<forbidden_leakage>\d+)\s+"
    r"heldout_missing_from_tokenizer=(?P<missing>\d+)\s+"
    r"meta_verified=(?P<meta_verified>true|false)\s+"
    r"manifest_sha256=(?P<manifest_sha256>[0-9a-f]{64})\s+"
    r"vocab_sha256=(?P<vocab_sha256>[0-9a-f]{64})\s+"
    r"tokenizer_meta_sha256=(?P<meta_sha256>[0-9a-f]{64})"
)
CHECKPOINT_TOKENIZER_SCOPE_RE = re.compile(
    r"checkpoint_tokenizer_vocab_scope\s+"
    r"policy=(?P<policy>\S+)\s+"
    r"tokenizer_chars=(?P<tokenizer_chars>\d+)\s+"
    r"direct_vocab_chars=(?P<direct_vocab_chars>\d+)\s+"
    r"byte_fallback=(?P<byte_fallback>true|false)\s+"
    r"byte_fallback_tokens=(?P<byte_fallback_tokens>\d+)\s+"
    r"train_chars=(?P<train_chars>\d+)\s+"
    r"heldout_exclusive_chars=(?P<heldout_exclusive_chars>\d+)\s+"
    r"covered_by_static_inventory=(?P<covered_by_static_inventory>\d+)\s+"
    r"heldout_covered_by_byte_fallback=(?P<heldout_covered_by_byte_fallback>\d+)\s+"
    r"forbidden_heldout_tokenizer_leakage=(?P<forbidden_leakage>\d+)\s+"
    r"heldout_missing_from_tokenizer=(?P<missing>\d+)\s+"
    r"meta_verified=(?P<meta_verified>true|false)\s+"
    r"manifest_sha256=(?P<manifest_sha256>[0-9a-f]{64})\s+"
    r"vocab_sha256=(?P<vocab_sha256>[0-9a-f]{64})\s+"
    r"tokenizer_meta_sha256=(?P<meta_sha256>[0-9a-f]{64})\s+"
    r"core_inventory_sha256=(?P<core_inventory_sha256>[0-9a-f]{64})\s+"
    r"checkpoint_bound=(?P<checkpoint_bound>true|false)"
)
SPLIT_CONSISTENCY_RE = re.compile(
    r"split_consistency\s+"
    r"(?:split_policy=(?P<split_policy>\S+)\s+)?"
    r"manifest=(?P<manifest>.+?)\s+"
    r"manifest_sha256=(?P<manifest_sha256>[0-9a-f]{64})\s+"
    r"train=(?P<train>.+?)\s+"
    r"validation=(?P<validation>.+?)\s+"
    r"test=(?P<test>.+?)\s+"
    r"train_manifest_sources=(?P<train_sources>\d+)\s+"
    r"validation_sources=(?P<validation_sources>\d+)\s+"
    r"test_sources=(?P<test_sources>\d+)\s+"
    r"(?:train_groups=(?P<train_groups>\d+)\s+)?"
    r"(?:validation_groups=(?P<validation_groups>\d+)\s+)?"
    r"(?:test_groups=(?P<test_groups>\d+)\s+)?"
    r"(?:group_disjoint=(?P<group_disjoint>true|false)\s+)?"
    r"test_source_chars=(?P<test_source_chars>\d+)\s+"
    r"test_text_chars=(?P<test_text_chars>\d+)\s+"
    r"validation_text_chars=(?P<validation_text_chars>\d+)\s+"
    r"train_match=(?P<train_match>true|false)\s+"
    r"train_reconstruction_sha256=(?P<train_reconstruction_sha256>[0-9a-f]{64})\s+"
    r"augmentation_manifest=(?P<augmentation_manifest>.+?)\s+"
    r"augmentation_manifest_sha256=(?P<augmentation_manifest_sha256>[0-9a-f]{64})\s+"
    r"validation_match=(?P<validation_match>true|false)\s+"
    r"test_match=(?P<test_match>true|false)\s+"
    r"validation_sha256=(?P<validation_sha256>[0-9a-f]{64})\s+"
    r"test_sha256=(?P<test_sha256>[0-9a-f]{64})"
)
LM_SPLIT_RE = re.compile(
    r"(?P<split>heldout|validation|test)_lm_token_nll=(?P<loss>\d+(?:\.\d+)?)\s+"
    r"(?P=split)_lm_perplexity=(?P<perplexity>\d+(?:\.\d+)?)\s+"
    r"(?P=split)_lm_tokens=(?P<tokens>\d+)\s+"
    r"(?P=split)_data=(?P<data>.+?)\s+(?P=split)_sha256=(?P<sha256>[0-9a-f]{64})$"
)
FAILURE_HINTS = (
    "Command failed with exit code",
    "checkpoint metadata ",
    "Checkpoint training input validation failed",
    "Release package check failed",
    "HF export failed",
    "Traceback (most recent call last)",
)


def clean_text(value: str) -> str:
    return "".join(ch for ch in value if ch == "\t" or ord(ch) >= 32)


def paths_equivalent(left: str, right: str) -> bool:
    if not left or not right:
        return False
    left_path = Path(left)
    right_path = Path(right)
    try:
        left_resolved = left_path.resolve(strict=False)
        right_resolved = right_path.resolve(strict=False)
    except OSError:
        return left_path.as_posix().casefold() == right_path.as_posix().casefold()
    return str(left_resolved).casefold() == str(right_resolved).casefold()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert a quality-check log into machine-readable eval JSON.")
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--status", choices=["passed", "failed"], default="passed")
    parser.add_argument(
        "--runner",
        type=str,
        default="",
        help="Quality-check runner script whose output produced this log.",
    )
    parser.add_argument("--hf-export", action="store_true")
    return parser.parse_args()


def metric_from_match(match: re.Match[str]) -> dict[str, Any]:
    passed = int(match.group("passed"))
    total = int(match.group("total"))
    value_text = match.group("value")
    value = float(value_text) if value_text is not None else (passed / max(1, total))
    return {"passed": passed, "total": total, "value": value}


def parse_log(text: str) -> dict[str, Any]:
    metrics: dict[str, dict[str, Any]] = {}
    duplicate_metrics: list[str] = []
    environment: dict[str, str] = {}
    corpus_checks: list[str] = []
    leakage: dict[str, Any] | None = None
    eval_contamination: dict[str, Any] | None = None
    eval_contamination_checks: list[dict[str, Any]] = []
    eval_source_overlap: dict[str, Any] | None = None
    eval_source_overlap_checks: list[dict[str, Any]] = []
    eval_files: list[dict[str, Any]] = []
    source_record_audits: list[dict[str, Any]] = []
    public_manifest_audit: dict[str, Any] | None = None
    eval_provenance_audit: dict[str, Any] | None = None
    tokenizer_vocab_scope: dict[str, Any] | None = None
    checkpoint_tokenizer_vocab_scope: dict[str, Any] | None = None
    split_consistency: dict[str, Any] | None = None
    lm_splits: dict[str, dict[str, Any]] = {}
    current_contamination_train_paths: list[str] = []
    current_contamination_eval_paths: list[str] = []
    current_source_overlap_eval_paths: list[str] = []
    checkpoint = ""
    warnings: list[str] = []
    failure_reasons: list[str] = []

    for raw_line in text.splitlines():
        line = clean_text(raw_line).strip()
        if not line:
            continue
        if any(hint in line for hint in FAILURE_HINTS):
            failure_reasons.append(line)
        for match in RATIO_RE.finditer(line):
            metric_name = match.group("name")
            if metric_name in metrics:
                duplicate_metrics.append(metric_name)
            metrics[metric_name] = metric_from_match(match)

        leakage_match = LEAKAGE_RE.match(line)
        if leakage_match:
            leakage = {
                "checked_sources": int(leakage_match.group("checked")),
                "expected_sources": int(leakage_match.group("expected") or 0),
                "split_policy": leakage_match.group("split_policy") or "",
                "checked_windows": int(leakage_match.group("windows") or 0),
                "checked_waka_items": int(leakage_match.group("waka_items") or 0),
                "role_pair_leaks": int(leakage_match.group("role_pair_leaks") or 0),
                "role_waka_leaks": int(leakage_match.group("role_waka_leaks") or 0),
                "waka_leaks": int(leakage_match.group("waka_leaks") or 0),
                "leaks": int(leakage_match.group("leaks")),
                "manifest": leakage_match.group("manifest") or "",
                "manifest_sha256": leakage_match.group("manifest_sha256") or "",
                "train": leakage_match.group("train"),
            }
            continue

        contamination_match = EVAL_CONTAMINATION_RE.match(line)
        if contamination_match:
            eval_contamination = {
                "checked": int(contamination_match.group("checked")),
                "hits": int(contamination_match.group("hits")),
                "train_paths": list(current_contamination_train_paths),
                "eval_paths": list(current_contamination_eval_paths),
            }
            eval_contamination_checks.append(eval_contamination)
            continue

        contamination_train_match = EVAL_CONTAMINATION_TRAIN_RE.match(line)
        if contamination_train_match:
            current_contamination_train_paths = [
                str(path) for path in json.loads(contamination_train_match.group("paths"))
            ]
            continue

        contamination_eval_match = EVAL_CONTAMINATION_EVAL_RE.match(line)
        if contamination_eval_match:
            current_contamination_eval_paths = [
                str(path) for path in json.loads(contamination_eval_match.group("paths"))
            ]
            continue

        source_overlap_eval_match = EVAL_SOURCE_OVERLAP_EVAL_RE.match(line)
        if source_overlap_eval_match:
            current_source_overlap_eval_paths = [
                str(path) for path in json.loads(source_overlap_eval_match.group("paths"))
            ]
            continue

        source_overlap_match = EVAL_SOURCE_OVERLAP_RE.match(line)
        if source_overlap_match:
            source_roles_checked = (
                json.loads(source_overlap_match.group("source_roles_checked"))
                if source_overlap_match.group("source_roles_checked")
                else []
            )
            source_items_by_role = (
                json.loads(source_overlap_match.group("source_items_by_role"))
                if source_overlap_match.group("source_items_by_role")
                else {}
            )
            eval_source_overlap = {
                "checked": int(source_overlap_match.group("checked")),
                "source_items": int(source_overlap_match.group("source_items")),
                "source_roles_checked": source_roles_checked,
                "source_items_by_role": source_items_by_role,
                "split_policy": source_overlap_match.group("split_policy"),
                "val_ratio": float(source_overlap_match.group("val_ratio")),
                "test_ratio": float(source_overlap_match.group("test_ratio")),
                "prose_hits": int(source_overlap_match.group("prose_hits")),
                "waka_exact_hits": int(source_overlap_match.group("waka_exact_hits")),
                "waka_variant_hits": int(source_overlap_match.group("waka_variant_hits")),
                "hits": int(source_overlap_match.group("hits")),
                "manifest": source_overlap_match.group("manifest"),
                "manifest_sha256": source_overlap_match.group("manifest_sha256"),
                "eval_paths": list(current_source_overlap_eval_paths),
            }
            eval_source_overlap_checks.append(eval_source_overlap)
            continue

        eval_snapshot_match = EVAL_SNAPSHOT_RE.match(line)
        if eval_snapshot_match:
            eval_files.append(
                {
                    "role": eval_snapshot_match.group("role"),
                    "path": eval_snapshot_match.group("path"),
                    "sha256": eval_snapshot_match.group("sha256"),
                    "rows": int(eval_snapshot_match.group("rows")),
                    "case_ids": json.loads(eval_snapshot_match.group("case_ids")),
                    "content_hashes": json.loads(eval_snapshot_match.group("content_hashes") or "[]"),
                    "source_sha256": eval_snapshot_match.group("source_sha256") or "",
                    "audited_source": eval_snapshot_match.group("audited_source") or "",
                    "audited_source_sha256": eval_snapshot_match.group("audited_source_sha256") or "",
                    "eval_provenance_manifest_sha256": eval_snapshot_match.group("eval_provenance_manifest_sha256") or "",
                    "removed_from_source": int(eval_snapshot_match.group("removed_from_source") or 0),
                    "source": eval_snapshot_match.group("source"),
                }
            )
            continue

        source_record_audit_match = SOURCE_RECORD_AUDIT_RE.match(line)
        if source_record_audit_match:
            source_record_audits.append(
                {
                    "path": source_record_audit_match.group("path"),
                    "checked": int(source_record_audit_match.group("checked")),
                    "mismatches": int(source_record_audit_match.group("mismatches")),
                    "missing": int(source_record_audit_match.group("missing")),
                    "fixed": source_record_audit_match.group("fixed").lower() == "true",
                }
            )
            continue

        public_manifest_audit_match = PUBLIC_MANIFEST_AUDIT_RE.match(line)
        if public_manifest_audit_match:
            public_manifest_audit = {
                "manifest_rows": int(public_manifest_audit_match.group("manifest_rows")),
                "included_rows": int(public_manifest_audit_match.group("included_rows")),
                "errors": int(public_manifest_audit_match.group("errors")),
                "out": public_manifest_audit_match.group("out"),
            }
            continue

        eval_provenance_audit_match = EVAL_PROVENANCE_AUDIT_RE.match(line)
        if eval_provenance_audit_match:
            eval_provenance_audit = {
                "path": eval_provenance_audit_match.group("path"),
                "manifest_sha256": eval_provenance_audit_match.group("manifest_sha256"),
                "entries": int(eval_provenance_audit_match.group("entries")),
                "errors": int(eval_provenance_audit_match.group("errors")),
                "llm_generated_eval_answer_text": eval_provenance_audit_match.group("llm_generated") == "true",
            }
            continue

        tokenizer_scope_match = TOKENIZER_SCOPE_RE.match(line)
        if tokenizer_scope_match:
            tokenizer_vocab_scope = {
                "policy": tokenizer_scope_match.group("policy"),
                "tokenizer_chars": int(tokenizer_scope_match.group("tokenizer_chars")),
                "direct_vocab_chars": int(
                    tokenizer_scope_match.group("direct_vocab_chars")
                    or tokenizer_scope_match.group("tokenizer_chars")
                ),
                "byte_fallback": tokenizer_scope_match.group("byte_fallback") == "true",
                "byte_fallback_tokens": int(tokenizer_scope_match.group("byte_fallback_tokens") or 0),
                "train_chars": int(tokenizer_scope_match.group("train_chars")),
                "heldout_exclusive_chars": int(tokenizer_scope_match.group("heldout_exclusive_chars")),
                "heldout_exclusive_covered_by_static_inventory": int(tokenizer_scope_match.group("covered_by_static_inventory")),
                "heldout_covered_by_byte_fallback": int(
                    tokenizer_scope_match.group("heldout_covered_by_byte_fallback") or 0
                ),
                "forbidden_heldout_tokenizer_leakage": int(tokenizer_scope_match.group("forbidden_leakage")),
                "heldout_missing_from_tokenizer": int(tokenizer_scope_match.group("missing")),
                "tokenizer_meta_verified": tokenizer_scope_match.group("meta_verified") == "true",
                "manifest_sha256": tokenizer_scope_match.group("manifest_sha256"),
                "vocab_sha256": tokenizer_scope_match.group("vocab_sha256"),
                "tokenizer_meta_sha256": tokenizer_scope_match.group("meta_sha256"),
            }
            continue

        checkpoint_tokenizer_scope_match = CHECKPOINT_TOKENIZER_SCOPE_RE.match(line)
        if checkpoint_tokenizer_scope_match:
            checkpoint_tokenizer_vocab_scope = {
                "policy": checkpoint_tokenizer_scope_match.group("policy"),
                "tokenizer_chars": int(checkpoint_tokenizer_scope_match.group("tokenizer_chars")),
                "direct_vocab_chars": int(checkpoint_tokenizer_scope_match.group("direct_vocab_chars")),
                "byte_fallback": checkpoint_tokenizer_scope_match.group("byte_fallback") == "true",
                "byte_fallback_tokens": int(checkpoint_tokenizer_scope_match.group("byte_fallback_tokens")),
                "train_chars": int(checkpoint_tokenizer_scope_match.group("train_chars")),
                "heldout_exclusive_chars": int(checkpoint_tokenizer_scope_match.group("heldout_exclusive_chars")),
                "heldout_exclusive_covered_by_static_inventory": int(
                    checkpoint_tokenizer_scope_match.group("covered_by_static_inventory")
                ),
                "heldout_covered_by_byte_fallback": int(
                    checkpoint_tokenizer_scope_match.group("heldout_covered_by_byte_fallback")
                ),
                "forbidden_heldout_tokenizer_leakage": int(checkpoint_tokenizer_scope_match.group("forbidden_leakage")),
                "heldout_missing_from_tokenizer": int(checkpoint_tokenizer_scope_match.group("missing")),
                "tokenizer_meta_verified": checkpoint_tokenizer_scope_match.group("meta_verified") == "true",
                "manifest_sha256": checkpoint_tokenizer_scope_match.group("manifest_sha256"),
                "vocab_sha256": checkpoint_tokenizer_scope_match.group("vocab_sha256"),
                "tokenizer_meta_sha256": checkpoint_tokenizer_scope_match.group("meta_sha256"),
                "core_inventory_sha256": checkpoint_tokenizer_scope_match.group("core_inventory_sha256"),
                "checkpoint_bound": checkpoint_tokenizer_scope_match.group("checkpoint_bound") == "true",
            }
            continue

        split_consistency_match = SPLIT_CONSISTENCY_RE.match(line)
        if split_consistency_match:
            split_consistency = {
                "split_policy": split_consistency_match.group("split_policy") or "",
                "manifest": split_consistency_match.group("manifest"),
                "manifest_sha256": split_consistency_match.group("manifest_sha256"),
                "train": split_consistency_match.group("train"),
                "validation": split_consistency_match.group("validation"),
                "test": split_consistency_match.group("test"),
                "train_manifest_sources": int(split_consistency_match.group("train_sources")),
                "validation_sources": int(split_consistency_match.group("validation_sources")),
                "test_sources": int(split_consistency_match.group("test_sources")),
                "train_groups": int(split_consistency_match.group("train_groups") or 0),
                "validation_groups": int(split_consistency_match.group("validation_groups") or 0),
                "test_groups": int(split_consistency_match.group("test_groups") or 0),
                "group_disjoint": split_consistency_match.group("group_disjoint") == "true",
                "test_source_chars": int(split_consistency_match.group("test_source_chars")),
                "test_text_chars": int(split_consistency_match.group("test_text_chars")),
                "validation_text_chars": int(split_consistency_match.group("validation_text_chars")),
                "train_match": split_consistency_match.group("train_match") == "true",
                "train_reconstruction_sha256": split_consistency_match.group("train_reconstruction_sha256"),
                "augmentation_manifest": split_consistency_match.group("augmentation_manifest"),
                "augmentation_manifest_sha256": split_consistency_match.group("augmentation_manifest_sha256"),
                "validation_match": split_consistency_match.group("validation_match") == "true",
                "test_match": split_consistency_match.group("test_match") == "true",
                "validation_sha256": split_consistency_match.group("validation_sha256"),
                "test_sha256": split_consistency_match.group("test_sha256"),
            }
            continue

        lm_split_match = LM_SPLIT_RE.match(line)
        if lm_split_match:
            split = lm_split_match.group("split")
            metric_name = f"{split}_lm_token_nll"
            lm_splits[split] = {
                "value": float(lm_split_match.group("loss")),
                "perplexity": float(lm_split_match.group("perplexity")),
                "tokens": int(lm_split_match.group("tokens")),
                f"{split}_data": lm_split_match.group("data"),
                f"{split}_sha256": lm_split_match.group("sha256"),
            }
            metrics[metric_name] = {
                "value": lm_splits[split]["value"],
                "tokens": lm_splits[split]["tokens"],
                "perplexity": lm_splits[split]["perplexity"],
            }
            continue

        if line.startswith("validated "):
            corpus_checks.append(line.removeprefix("validated ").strip())
            continue

        key_match = KEY_VALUE_RE.match(line)
        if key_match:
            key = key_match.group("key")
            value = clean_text(key_match.group("value")).strip()
            if key in {
                "python",
                "torch",
                "torch_cuda_version",
                "torch_hip_version",
                "cuda_runtime_kind",
                "real_cuda_runtime",
                "cuda_available",
                "cuda_device",
                "directml_device",
                "device",
            }:
                environment[key] = value
            elif key == "checkpoint":
                checkpoint = value

        if "Traceback (most recent call last)" in line or "accuracy " in line and " below required " in line:
            warnings.append(line)

    return {
        "metrics": metrics,
        "model_metrics": {
            key: value
            for key, value in metrics.items()
            if key in {"test_lm_token_nll"}
        },
        "smoke_metrics": {
            key: value
            for key, value in metrics.items()
            if key not in {"test_lm_token_nll"}
        },
        "duplicate_metrics": duplicate_metrics,
        "environment": environment,
        "corpus_checks": corpus_checks,
        "leakage": leakage,
        "eval_contamination": eval_contamination,
        "eval_contamination_checks": eval_contamination_checks,
        "eval_source_overlap": eval_source_overlap,
        "eval_source_overlap_checks": eval_source_overlap_checks,
        "eval_files": eval_files,
        "source_record_audits": source_record_audits,
        "public_manifest_audit": public_manifest_audit,
        "eval_provenance_audit": eval_provenance_audit,
        "tokenizer_vocab_scope": tokenizer_vocab_scope,
        "checkpoint_tokenizer_vocab_scope": checkpoint_tokenizer_vocab_scope,
        "split_consistency": split_consistency,
        "heldout_lm": lm_splits.get("heldout"),
        "validation_lm": lm_splits.get("validation"),
        "test_lm": lm_splits.get("test"),
        "checkpoint_from_log": checkpoint,
        "warnings": warnings,
        "failure_reasons": failure_reasons[:10],
    }


def read_log_text(path: Path) -> tuple[bytes, str, str]:
    data = path.read_bytes()
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return data, data.decode("utf-16", errors="replace"), "utf-16"
    if b"\x00" in data[:200]:
        return data, data.decode("utf-16-le", errors="replace"), "utf-16-le"
    for encoding in ("utf-8-sig", "cp932"):
        try:
            return data, data.decode(encoding), encoding
        except UnicodeDecodeError:
            pass
    return data, data.decode("utf-8", errors="replace"), "utf-8-replace"


def repo_relative_script_path() -> str:
    path = Path(__file__).resolve()
    root = path.parents[1]
    return path.relative_to(root).as_posix()


def sha256_file_if_exists(path_text: str) -> str:
    if not path_text:
        return ""
    path = Path(path_text)
    if not path.is_absolute():
        path = Path.cwd() / path
    if not path.exists():
        return ""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    args = parse_args()
    raw, text, log_encoding = read_log_text(args.log)
    payload = parse_log(text)
    if args.status == "passed" and payload.get("duplicate_metrics"):
        raise SystemExit(f"passed quality log has duplicate metric keys: {payload['duplicate_metrics']}")
    log_sha256 = hashlib.sha256(raw).hexdigest()
    checkpoint_metadata: dict[str, Any] = {}
    checkpoint_sha256 = ""
    if args.checkpoint and args.checkpoint != "static-only":
        if args.status == "passed":
            checkpoint_from_log = str(payload.get("checkpoint_from_log") or "")
            if not checkpoint_from_log:
                raise SystemExit("passed quality logs must contain a checkpoint=... line.")
            if not paths_equivalent(checkpoint_from_log, args.checkpoint):
                raise SystemExit(
                    f"checkpoint mismatch between log and parser argument: "
                    f"log={checkpoint_from_log!r} arg={args.checkpoint!r}"
                )
        checkpoint_path = Path(args.checkpoint)
        if checkpoint_path.exists():
            checkpoint_sha256 = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
            try:
                checkpoint_payload = load_trusted_checkpoint(checkpoint_path, map_location="cpu")
            except Exception as exc:
                payload.setdefault("warnings", []).append(f"checkpoint_metadata_load_failed={type(exc).__name__}:{exc}")
            else:
                checkpoint_metadata = {
                    "checkpoint_step": checkpoint_payload.get("step"),
                    "checkpoint_best_val": checkpoint_payload.get("best_val"),
                    "metadata_param_count": checkpoint_payload.get("metadata", {}).get("param_count"),
                    "metadata_run_id": checkpoint_payload.get("metadata", {}).get("run_id"),
                }

    payload.update(
        {
            "status": args.status,
            "checkpoint": args.checkpoint or payload.get("checkpoint_from_log", ""),
            "checkpoint_sha256": checkpoint_sha256,
            **checkpoint_metadata,
            "quality_log": str(args.log),
            "quality_log_encoding": log_encoding,
            "quality_log_sha256": log_sha256,
            "quality_parser": repo_relative_script_path(),
            "quality_parser_sha256": sha256_file_if_exists(repo_relative_script_path()),
            "quality_runner": args.runner,
            "quality_runner_sha256": sha256_file_if_exists(args.runner),
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "device": args.device or payload.get("environment", {}).get("device", ""),
            "hf_export": bool(args.hf_export),
            "release_policy": (
                "No raw/clean/training/validation/test text, logs, optimizer state, "
                "caches, or secrets in release package."
            ),
        }
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote_eval_json={args.out}")


if __name__ == "__main__":
    main()
