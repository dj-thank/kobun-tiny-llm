from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    root = Path.cwd()
    with tempfile.TemporaryDirectory() as tmp:
        release_dir = Path(tmp) / "hf_model_bad_paths"
        release_dir.mkdir()
        for filename in (
            "README.md",
            "config.json",
            "tokenizer.json",
            "generation_config.json",
            "source_manifest_summary.json",
            "source_manifest.json",
            "pytorch_model.pt",
        ):
            if filename.endswith(".json"):
                write_json(release_dir / filename, {})
            elif filename.endswith(".pt"):
                (release_dir / filename).write_bytes(b"not-a-real-model")
            else:
                (release_dir / filename).write_text("test\n", encoding="utf-8")
        metadata = {
            "checkpoint": "checkpoints\\model.pt",
            "checkpoint_sha256": "a" * 64,
            "checkpoint_step": 1,
            "device_description": "directml:AMD Radeon RX 7600 XT\u0000",
            "data_path": "data\\run_snapshots\\run\\train.txt",
            "val_data_path": "data\\run_snapshots\\run\\validation.txt",
            "test_data_path": "data\\run_snapshots\\run\\test.txt",
            "init_from": "",
            "resume": "",
            "test_data_sha256": "b" * 64,
            "val_data_sha256": "c" * 64,
            "tokenizer_extra_data": [{"path": "data\\run_snapshots\\run\\tokenizer_public_char_vocab.txt", "sha256": "d" * 64}],
            "provenance_files": [
                {"path": "data\\run_snapshots\\run\\provenance\\corpus_manifest.jsonl", "sha256": "e" * 64},
                {"path": "data\\aozora\\raw\\x.txt", "sha256": "f" * 64},
                {"path": "logs\\quality.log", "sha256": "0" * 64},
            ],
        }
        eval_results = {
            "status": "passed",
            "checkpoint": "checkpoints\\model.pt",
            "checkpoint_from_log": "checkpoints\\model.pt",
            "checkpoint_sha256": "a" * 64,
            "checkpoint_step": 1,
            "model_metrics": {"test_lm_token_nll": {"value": 1.0}},
            "smoke_metrics": {
                "primary_contrastive_preference_accuracy": {"value": 1.0, "total": 8},
                "heldout_contrastive_preference_accuracy": {"value": 1.0, "total": 12},
                "grammar_constraint_accuracy": {"value": 1.0, "total": 28},
                "waka_rule_accuracy": {"value": 1.0, "total": 20},
                "waka_meter_constraint_static_accuracy": {"value": 1.0, "total": 19},
                "waka_meter_constrained_generation_accuracy": {"value": 1.0, "total": 4},
                "morphology_adversarial_accuracy": {"value": 1.0, "total": 5},
            },
            "test_lm": {"test_data": "test.txt", "test_sha256": "b" * 64},
            "tokenizer_vocab_scope": {
                "policy": "train_split_plus_core_japanese_inventory_plus_utf8_byte_fallback_v1",
                "byte_fallback": True,
                "byte_fallback_tokens": 256,
                "tokenizer_chars": 3000,
                "direct_vocab_chars": 2743,
                "tokenizer_meta_verified": True,
                "forbidden_heldout_tokenizer_leakage": 0,
                "heldout_missing_from_tokenizer": 0,
                "tokenizer_meta_sha256": "0" * 64,
                "vocab_sha256": "d" * 64,
            },
            "split_consistency": {
                "split_policy": "work_group_genji_reference_v1",
                "manifest": "corpus_manifest.jsonl",
                "train": "train.txt",
                "validation": "validation.txt",
                "test": "test.txt",
                "train_match": True,
                "train_reconstruction_sha256": "4" * 64,
                "augmentation_manifest": "training_augmentation_manifest.json",
                "augmentation_manifest_sha256": "5" * 64,
                "validation_match": True,
                "test_match": True,
                "group_disjoint": True,
                "train_groups": 1,
                "validation_groups": 1,
                "test_groups": 1,
                "test_sources": 3,
                "test_source_chars": 30000,
                "test_text_chars": 20000,
                "manifest_sha256": "e" * 64,
                "test_sha256": "b" * 64,
                "validation_sha256": "c" * 64,
            },
            "corpus_checks": ["train.txt", "validation.txt", "test.txt"],
            "leakage": {
                "checked_sources": 3,
                "expected_sources": 3,
                "split_policy": "work_group_genji_reference_v1",
                "checked_windows": 1,
                "checked_waka_items": 1,
                "role_pair_leaks": 0,
                "role_waka_leaks": 0,
                "waka_leaks": 0,
                "leaks": 0,
                "manifest": "corpus_manifest.jsonl",
                "manifest_sha256": "e" * 64,
            },
            "eval_contamination_checks": [{"checked": 1, "hits": 0, "train_paths": ["train.txt"], "eval_paths": ["primary.jsonl"]}],
            "source_record_audits": [
                {"path": "aozora_sources.json", "checked": 1, "mismatches": 0, "missing": 0, "fixed": False},
                {"path": "waka_sources.json", "checked": 1, "mismatches": 0, "missing": 0, "fixed": False},
            ],
            "public_manifest_audit": {
                "manifest_rows": 2,
                "included_rows": 2,
                "errors": 0,
                "out": "public_manifest_summary.json",
            },
            "raw_excerpt": "this key must never be public release evidence",
            "eval_files": [
                {"role": "primary", "path": "primary.jsonl", "source": "primary.jsonl", "rows": 1, "case_ids": ["a"], "sha256": "1" * 64},
                {"role": "heldout", "path": "heldout.jsonl", "source": "heldout.jsonl", "rows": 1, "case_ids": ["b"], "sha256": "2" * 64},
                {"role": "morphology", "path": "morphology.jsonl", "source": "morphology.jsonl", "rows": 1, "case_ids": ["c"], "sha256": "3" * 64},
            ],
        }
        write_json(release_dir / "training_metadata.json", metadata)
        write_json(release_dir / "eval_results.json", eval_results)
        result = subprocess.run(
            [sys.executable, str(root / "scripts" / "check_release_package.py"), "--release-dir", str(release_dir)],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            raise SystemExit("check_release_package.py accepted relative local path leakage")
        required_fragments = (
            "data_path path is not sanitized",
            "checkpoint path is not sanitized",
            "provenance_files",
            "tokenizer_extra_data",
            "unexpected top-level public eval keys",
            "unexpected control character",
        )
        output = result.stdout + result.stderr
        missing = [fragment for fragment in required_fragments if fragment not in output]
        if missing:
            raise SystemExit(f"release path scan rejected package, but missing expected issues: {missing}\n{output}")
    print("release_package_path_scan_ok=true")


if __name__ == "__main__":
    main()
