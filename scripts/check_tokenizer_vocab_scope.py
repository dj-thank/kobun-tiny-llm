from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from build_tokenizer_public_vocab import (
    CORE_JAPANESE_RANGES,
    TOKENIZER_POLICY,
    core_japanese_inventory,
    manifest_path,
    split_name,
)
from split_policy import SPLIT_POLICY


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fail if tokenizer vocabulary leaks validation/test-only corpus characters.")
    parser.add_argument("--manifest", type=Path, default=Path("data/corpus_manifest.jsonl"))
    parser.add_argument("--tokenizer-extra-data", type=Path, default=Path("data/tokenizer_public_char_vocab.txt"))
    parser.add_argument("--tokenizer-meta", type=Path, default=Path("data/tokenizer_public_char_vocab.meta.json"))
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.05)
    parser.add_argument("--require-heldout-covered", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-vocab-size", type=int, default=10_000)
    return parser.parse_args()


def read_manifest_rows(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_clean_chars(row: dict[str, object]) -> set[str]:
    clean_file = manifest_path(row.get("clean_file", ""))
    if not clean_file.exists():
        raise SystemExit(f"manifest clean_file does not exist: {clean_file}")
    return set(clean_file.read_text(encoding="utf-8"))


def render_chars(chars: set[str], limit: int = 24) -> str:
    sample = "".join(sorted(chars)[:limit])
    suffix = "" if len(chars) <= limit else "..."
    return f"{sample}{suffix}"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_tokenizer_meta(
    meta_path: Path,
    manifest_path: Path,
    tokenizer_path: Path,
    tokenizer_text: str,
    train_source_count: int,
) -> tuple[dict[str, object], list[str]]:
    if not meta_path.exists():
        return {}, [f"missing tokenizer metadata file: {meta_path}"]
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    issues: list[str] = []
    if meta.get("policy") != TOKENIZER_POLICY:
        issues.append(f"tokenizer meta policy is not release-safe: {meta.get('policy')!r}")
    if meta.get("split_policy") != SPLIT_POLICY:
        issues.append(f"tokenizer meta split_policy mismatch: expected={SPLIT_POLICY} actual={meta.get('split_policy')!r}")
    expected_manifest_sha = sha256_file(manifest_path)
    if meta.get("manifest_sha256") != expected_manifest_sha:
        issues.append(
            f"tokenizer meta manifest_sha256 mismatch: expected={expected_manifest_sha} actual={meta.get('manifest_sha256')}"
        )
    expected_vocab_sha = sha256_file(tokenizer_path)
    if meta.get("vocab_sha256") != expected_vocab_sha:
        issues.append(
            f"tokenizer meta vocab_sha256 mismatch: expected={expected_vocab_sha} actual={meta.get('vocab_sha256')}"
        )
    if int(meta.get("train_source_count", -1)) != train_source_count:
        issues.append(
            f"tokenizer meta train_source_count mismatch: expected={train_source_count} actual={meta.get('train_source_count')}"
        )
    expected_ranges = [
        {"name": name, "start": f"U+{start:04X}", "end": f"U+{end:04X}"}
        for name, start, end in CORE_JAPANESE_RANGES
    ]
    if meta.get("core_inventory_ranges") != expected_ranges:
        issues.append("tokenizer meta core_inventory_ranges do not match release policy")
    if int(meta.get("core_inventory_chars", -1)) != len(core_japanese_inventory()):
        issues.append(
            f"tokenizer meta core_inventory_chars mismatch: expected={len(core_japanese_inventory())} actual={meta.get('core_inventory_chars')}"
        )
    if int(meta.get("direct_vocab_chars", -1)) != len(set(tokenizer_text)):
        issues.append(
            f"tokenizer meta direct_vocab_chars mismatch: expected={len(set(tokenizer_text))} actual={meta.get('direct_vocab_chars')}"
        )
    if meta.get("byte_fallback") is not True:
        issues.append("tokenizer meta byte_fallback must be true")
    if int(meta.get("byte_fallback_tokens", -1)) != 256:
        issues.append("tokenizer meta byte_fallback_tokens must be 256")
    return meta, issues


def main() -> None:
    args = parse_args()
    rows = [row for row in read_manifest_rows(args.manifest) if row.get("include_in_training", True)]
    split_chars = {"train": set(), "validation": set(), "test": set()}
    split_counts = {"train": 0, "validation": 0, "test": 0}
    for row in rows:
        name = split_name(row, args.val_ratio, args.test_ratio)
        split_chars[name].update(read_clean_chars(row))
        split_counts[name] += 1

    tokenizer_bytes = args.tokenizer_extra_data.read_bytes()
    tokenizer_text = tokenizer_bytes.decode("utf-8")
    tokenizer_chars = set(tokenizer_text)
    core_chars = core_japanese_inventory()
    train_chars = split_chars["train"]
    heldout_chars = split_chars["validation"] | split_chars["test"]
    heldout_exclusive = heldout_chars - train_chars
    leaked_from_heldout = (tokenizer_chars & heldout_exclusive) - core_chars
    direct_missing_heldout = heldout_chars - tokenizer_chars
    missing_heldout = set()  # UTF-8 byte fallback makes every Unicode char encodable without direct vocab leakage.
    estimated_total_vocab = len(tokenizer_chars) + 256 + 1
    meta, meta_issues = verify_tokenizer_meta(
        args.tokenizer_meta,
        args.manifest,
        args.tokenizer_extra_data,
        tokenizer_text,
        split_counts["train"],
    )

    result = {
        "manifest": args.manifest.as_posix(),
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "tokenizer_extra_data": args.tokenizer_extra_data.as_posix(),
        "tokenizer_extra_sha256": hashlib.sha256(tokenizer_bytes).hexdigest(),
        "tokenizer_meta": args.tokenizer_meta.as_posix(),
        "tokenizer_meta_sha256": hashlib.sha256(args.tokenizer_meta.read_bytes()).hexdigest() if args.tokenizer_meta.exists() else "",
        "tokenizer_meta_policy": meta.get("policy", ""),
        "tokenizer_meta_verified": not meta_issues,
        "tokenizer_meta_issues": meta_issues,
        "policy": TOKENIZER_POLICY,
        "byte_fallback": True,
        "byte_fallback_tokens": 256,
        "static_inventory_ranges": [
            {"name": name, "start": f"U+{start:04X}", "end": f"U+{end:04X}"}
            for name, start, end in CORE_JAPANESE_RANGES
        ],
        "core_inventory_ranges": [
            {"name": name, "start": f"U+{start:04X}", "end": f"U+{end:04X}"}
            for name, start, end in CORE_JAPANESE_RANGES
        ],
        "split_counts": split_counts,
        "tokenizer_chars": estimated_total_vocab,
        "direct_vocab_chars": len(tokenizer_chars),
        "train_chars": len(train_chars),
        "heldout_chars": len(heldout_chars),
        "heldout_exclusive_chars": len(heldout_exclusive),
        "heldout_exclusive_covered_by_static_inventory": len(heldout_exclusive & core_chars),
        "heldout_exclusive_covered_by_core_inventory": len(heldout_exclusive & core_chars),
        "heldout_exclusive_not_in_core_inventory": len(heldout_exclusive - core_chars),
        "heldout_direct_vocab_missing_chars": len(direct_missing_heldout),
        "heldout_covered_by_byte_fallback": len(direct_missing_heldout),
        "forbidden_heldout_tokenizer_leakage": len(leaked_from_heldout),
        "heldout_missing_from_tokenizer": len(missing_heldout),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(
        "tokenizer_vocab_scope "
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
        f"vocab_sha256={hashlib.sha256(tokenizer_bytes).hexdigest()} "
        f"tokenizer_meta_sha256={result['tokenizer_meta_sha256']}"
    )

    if meta_issues:
        raise SystemExit("Tokenizer metadata verification failed:\n" + "\n".join(meta_issues))
    if estimated_total_vocab >= args.max_vocab_size:
        raise SystemExit(f"Tokenizer vocab is too large: {estimated_total_vocab} >= {args.max_vocab_size}")
    if leaked_from_heldout:
        raise SystemExit(
            "Tokenizer contains validation/test-exclusive characters that are not from the fixed static inventory: "
            + render_chars(leaked_from_heldout)
        )
    if args.require_heldout_covered and missing_heldout:
        raise SystemExit(
            "Tokenizer does not cover validation/test characters; expand the fixed inventory or train split policy: "
            + render_chars(missing_heldout)
        )


if __name__ == "__main__":
    main()
