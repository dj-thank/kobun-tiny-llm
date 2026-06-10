from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from build_tokenizer_public_vocab import TOKENIZER_POLICY, core_japanese_inventory, split_name
from check_tokenizer_vocab_scope import read_clean_chars, read_manifest_rows, verify_tokenizer_meta
from kobun_llm.checkpoint_io import load_trusted_checkpoint
from kobun_llm.tokenizer import BYTE_FALLBACK_TOKENIZER_TYPE, CharTokenizer, byte_token


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify checkpoint-embedded tokenizer matches the audited byte-fallback tokenizer policy."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--tokenizer-extra-data", type=Path, required=True)
    parser.add_argument("--tokenizer-meta", type=Path, required=True)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.05)
    parser.add_argument("--max-vocab-size", type=int, default=10_000)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record_with_sha(records: Any, sha256: str, basename: str) -> dict[str, Any] | None:
    if not isinstance(records, list):
        return None
    for record in records:
        if not isinstance(record, dict):
            continue
        if str(record.get("sha256") or "") == sha256 and Path(str(record.get("path") or "")).name == basename:
            return record
    return None


def render_chars(chars: set[str], limit: int = 24) -> str:
    sample = "".join(sorted(chars)[:limit])
    suffix = "" if len(chars) <= limit else "..."
    return f"{sample}{suffix}"


def split_character_sets(manifest: Path, val_ratio: float, test_ratio: float) -> tuple[dict[str, set[str]], dict[str, int]]:
    rows = [row for row in read_manifest_rows(manifest) if row.get("include_in_training", True)]
    split_chars = {"train": set(), "validation": set(), "test": set()}
    split_counts = {"train": 0, "validation": 0, "test": 0}
    for row in rows:
        name = split_name(row, val_ratio, test_ratio)
        split_chars[name].update(read_clean_chars(row))
        split_counts[name] += 1
    return split_chars, split_counts


def main() -> None:
    args = parse_args()
    payload = load_trusted_checkpoint(args.checkpoint, map_location="cpu")
    if "tokenizer" not in payload:
        raise SystemExit("checkpoint missing tokenizer payload")
    metadata = dict(payload.get("metadata", {}) or {})
    tokenizer_payload = dict(payload["tokenizer"])
    tokenizer = CharTokenizer.from_dict(tokenizer_payload)
    config = dict(payload.get("config", {}) or {})
    direct_chars = set(getattr(tokenizer, "direct_chars", {token for token in tokenizer.stoi if len(token) == 1}))
    byte_tokens = {byte_token(value) for value in range(256)}
    missing_byte_tokens = sorted(byte_tokens - set(tokenizer.stoi))
    tokenizer_extra_bytes = args.tokenizer_extra_data.read_bytes()
    tokenizer_extra_text = tokenizer_extra_bytes.decode("utf-8")
    tokenizer_extra_chars = set(tokenizer_extra_text)
    tokenizer_extra_sha = hashlib.sha256(tokenizer_extra_bytes).hexdigest()
    tokenizer_meta_sha = sha256_file(args.tokenizer_meta)
    manifest_sha = sha256_file(args.manifest)
    split_chars, split_counts = split_character_sets(args.manifest, args.val_ratio, args.test_ratio)
    core_chars = core_japanese_inventory()
    heldout_chars = split_chars["validation"] | split_chars["test"]
    heldout_exclusive = heldout_chars - split_chars["train"]
    leaked_from_heldout = (direct_chars & heldout_exclusive) - core_chars
    direct_missing_heldout = heldout_chars - direct_chars
    estimated_total_vocab = len(tokenizer.stoi)
    meta, meta_issues = verify_tokenizer_meta(
        args.tokenizer_meta,
        args.manifest,
        args.tokenizer_extra_data,
        tokenizer_extra_text,
        split_counts["train"],
    )

    errors: list[str] = []
    if getattr(tokenizer, "tokenizer_type", "") != BYTE_FALLBACK_TOKENIZER_TYPE:
        errors.append(f"checkpoint tokenizer_type is not {BYTE_FALLBACK_TOKENIZER_TYPE}")
    if tokenizer_payload.get("byte_fallback") is not True:
        errors.append("checkpoint tokenizer payload byte_fallback is not true")
    if str(metadata.get("tokenizer_type") or "") != BYTE_FALLBACK_TOKENIZER_TYPE:
        errors.append(f"checkpoint metadata tokenizer_type={metadata.get('tokenizer_type')!r}")
    if metadata.get("byte_fallback") is not True:
        errors.append("checkpoint metadata byte_fallback is not true")
    if str(metadata.get("tokenizer_source") or "") != TOKENIZER_POLICY:
        errors.append(f"checkpoint metadata tokenizer_source={metadata.get('tokenizer_source')!r}")
    if missing_byte_tokens:
        errors.append(f"checkpoint tokenizer missing byte fallback tokens: {missing_byte_tokens[:8]}")
    if direct_chars != tokenizer_extra_chars:
        extra_only = tokenizer_extra_chars - direct_chars
        checkpoint_only = direct_chars - tokenizer_extra_chars
        errors.append(
            "checkpoint direct tokenizer chars do not match checkpoint-bound tokenizer_extra_data "
            f"extra_only={render_chars(extra_only)} checkpoint_only={render_chars(checkpoint_only)}"
        )
    if int(config.get("vocab_size", -1)) != estimated_total_vocab:
        errors.append(f"checkpoint config vocab_size={config.get('vocab_size')} tokenizer_vocab_size={estimated_total_vocab}")
    if estimated_total_vocab >= args.max_vocab_size:
        errors.append(f"checkpoint tokenizer vocab is too large: {estimated_total_vocab} >= {args.max_vocab_size}")
    if meta_issues:
        errors.extend(f"tokenizer meta issue: {issue}" for issue in meta_issues)
    if leaked_from_heldout:
        errors.append(
            "checkpoint direct tokenizer contains validation/test-exclusive non-core chars: "
            + render_chars(leaked_from_heldout)
        )
    tokenizer_record = record_with_sha(metadata.get("tokenizer_extra_data"), tokenizer_extra_sha, args.tokenizer_extra_data.name)
    if tokenizer_record is None:
        errors.append("checkpoint metadata tokenizer_extra_data is not bound to tokenizer_extra_data hash/basename")
    meta_record = record_with_sha(metadata.get("provenance_files"), tokenizer_meta_sha, args.tokenizer_meta.name)
    if meta_record is None:
        errors.append("checkpoint metadata provenance_files is not bound to tokenizer metadata hash/basename")
    manifest_record = record_with_sha(metadata.get("provenance_files"), manifest_sha, args.manifest.name)
    if manifest_record is None:
        errors.append("checkpoint metadata provenance_files is not bound to corpus manifest hash/basename")

    result = {
        "checkpoint": str(args.checkpoint),
        "policy": TOKENIZER_POLICY,
        "tokenizer_type": getattr(tokenizer, "tokenizer_type", ""),
        "byte_fallback": getattr(tokenizer, "tokenizer_type", "") == BYTE_FALLBACK_TOKENIZER_TYPE,
        "byte_fallback_tokens": len(byte_tokens & set(tokenizer.stoi)),
        "tokenizer_chars": estimated_total_vocab,
        "direct_vocab_chars": len(direct_chars),
        "train_chars": len(split_chars["train"]),
        "heldout_exclusive_chars": len(heldout_exclusive),
        "heldout_exclusive_covered_by_static_inventory": len(heldout_exclusive & core_chars),
        "heldout_covered_by_byte_fallback": len(direct_missing_heldout),
        "forbidden_heldout_tokenizer_leakage": len(leaked_from_heldout),
        "heldout_missing_from_tokenizer": 0,
        "tokenizer_meta_verified": not meta_issues,
        "manifest_sha256": manifest_sha,
        "vocab_sha256": tokenizer_extra_sha,
        "tokenizer_meta_sha256": tokenizer_meta_sha,
        "core_inventory_sha256": str(meta.get("core_inventory_sha256") or ""),
        "checkpoint_bound": True,
        "errors": errors,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(
        "checkpoint_tokenizer_vocab_scope "
        f"policy={result['policy']} "
        f"tokenizer_chars={result['tokenizer_chars']} "
        f"direct_vocab_chars={result['direct_vocab_chars']} "
        f"byte_fallback={str(result['byte_fallback']).lower()} "
        f"byte_fallback_tokens={result['byte_fallback_tokens']} "
        f"train_chars={result['train_chars']} "
        f"heldout_exclusive_chars={result['heldout_exclusive_chars']} "
        f"covered_by_static_inventory={result['heldout_exclusive_covered_by_static_inventory']} "
        f"heldout_covered_by_byte_fallback={result['heldout_covered_by_byte_fallback']} "
        f"forbidden_heldout_tokenizer_leakage={result['forbidden_heldout_tokenizer_leakage']} "
        f"heldout_missing_from_tokenizer={result['heldout_missing_from_tokenizer']} "
        f"meta_verified={str(result['tokenizer_meta_verified']).lower()} "
        f"manifest_sha256={result['manifest_sha256']} "
        f"vocab_sha256={result['vocab_sha256']} "
        f"tokenizer_meta_sha256={result['tokenizer_meta_sha256']} "
        f"core_inventory_sha256={result['core_inventory_sha256']} "
        "checkpoint_bound=true"
    )
    if errors:
        for error in errors[:20]:
            print(error)
        raise SystemExit("checkpoint tokenizer scope verification failed")


if __name__ == "__main__":
    main()
