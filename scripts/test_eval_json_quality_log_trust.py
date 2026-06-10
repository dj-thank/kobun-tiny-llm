from __future__ import annotations

import hashlib
import json
from pathlib import Path

from export_hf_release import verify_eval_quality_log
from parse_quality_log import parse_log, repo_relative_script_path, sha256_file_if_exists


ROOT = Path(__file__).resolve().parents[1]
LOG_PATH = ROOT / "logs" / "quality_eval_json_trust_test.log"
RUNNER = "scripts/run_quality_checks_dml.ps1"


QUALITY_LOG = """\
grammar_constraint_accuracy=28/28=1.000
test_lm_token_nll=2.500 test_lm_perplexity=12.182 test_lm_tokens=128 test_data=data/run_snapshots/example/test.txt test_sha256=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
split_leakage_checked_sources=8 expected_sources=8 split_policy=work_group_genji_reference_v1 checked_windows=100 checked_waka_items=10 role_pair_leaks=0 role_waka_leaks=0 waka_leaks=0 leaks=0 manifest=data/run_snapshots/example/corpus_manifest.jsonl manifest_sha256=bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb train=data/run_snapshots/example/train.txt
eval_contamination_train=["data/run_snapshots/example/train.txt"]
eval_contamination_eval=["data/eval/grammar_minimal_pairs.jsonl"]
eval_contamination_checked=105 hits=0
eval_source_overlap_eval=["data/eval/grammar_minimal_pairs.jsonl"]
eval_source_overlap_checked=105 source_items=600 split_policy=work_group_genji_reference_v1 val_ratio=0.1 test_ratio=0.05 prose_hits=0 waka_exact_hits=0 waka_variant_hits=0 hits=0 manifest=data/run_snapshots/example/corpus_manifest.jsonl manifest_sha256=bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
tokenizer_vocab_scope policy=train_split_plus_core_japanese_inventory_plus_utf8_byte_fallback_v1 tokenizer_chars=3317 direct_vocab_chars=3057 byte_fallback=true byte_fallback_tokens=256 train_chars=3000 heldout_exclusive_chars=10 covered_by_static_inventory=0 heldout_covered_by_byte_fallback=10 forbidden_heldout_tokenizer_leakage=0 heldout_missing_from_tokenizer=0 meta_verified=true manifest_sha256=bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb vocab_sha256=cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc tokenizer_meta_sha256=dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd
split_consistency split_policy=work_group_genji_reference_v1 manifest=data/run_snapshots/example/corpus_manifest.jsonl manifest_sha256=bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb train=data/run_snapshots/example/train.txt validation=data/run_snapshots/example/validation.txt test=data/run_snapshots/example/test.txt train_manifest_sources=8 validation_sources=2 test_sources=3 train_groups=8 validation_groups=2 test_groups=3 group_disjoint=true test_source_chars=30000 test_text_chars=20000 validation_text_chars=10000 train_match=true train_reconstruction_sha256=eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee augmentation_manifest=data/run_snapshots/example/training_augmentation_manifest.json augmentation_manifest_sha256=ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff validation_match=true test_match=true validation_sha256=1111111111111111111111111111111111111111111111111111111111111111 test_sha256=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
"""


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_payload() -> dict:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text(QUALITY_LOG, encoding="utf-8")
    payload = parse_log(QUALITY_LOG)
    payload.update(
        {
            "hf_export": False,
            "quality_log": str(LOG_PATH.relative_to(ROOT)),
            "quality_log_encoding": "utf-8-sig",
            "quality_log_sha256": sha256_file(LOG_PATH),
            "quality_parser": repo_relative_script_path(),
            "quality_parser_sha256": sha256_file_if_exists(repo_relative_script_path()),
            "quality_runner": RUNNER,
            "quality_runner_sha256": sha256_file_if_exists(RUNNER),
        }
    )
    return payload


def assert_rejected(payload: dict, expected: str) -> None:
    try:
        verify_eval_quality_log(payload)
    except SystemExit as exc:
        if expected not in str(exc):
            raise SystemExit(f"unexpected_eval_trust_failure expected={expected!r} got={exc!s}") from exc
        return
    raise SystemExit(f"tampered eval payload was accepted: expected={expected!r}")


def main() -> None:
    try:
        payload = build_payload()
        verify_eval_quality_log(payload)

        cuda_payload = build_payload()
        cuda_payload["quality_runner"] = "scripts/run_quality_checks_cuda.py"
        cuda_payload["quality_runner_sha256"] = sha256_file_if_exists("scripts/run_quality_checks_cuda.py")
        verify_eval_quality_log(cuda_payload, expected_runner="scripts/run_quality_checks_cuda.py")
        assert_rejected(cuda_payload, "quality_runner")

        tampered = json.loads(json.dumps(payload))
        tampered["model_metrics"]["test_lm_token_nll"]["value"] = 1.0
        assert_rejected(tampered, "canonical evidence")

        tampered = json.loads(json.dumps(payload))
        tampered["quality_log_sha256"] = "0" * 64
        assert_rejected(tampered, "quality_log_sha256")

        tampered = json.loads(json.dumps(payload))
        tampered["quality_runner"] = "scripts/run_static_quality_checks.ps1"
        assert_rejected(tampered, "quality_runner")
    finally:
        LOG_PATH.unlink(missing_ok=True)
    print("eval_json_quality_log_trust_ok=true")


if __name__ == "__main__":
    main()
