from __future__ import annotations

import json
import sys
from pathlib import Path

from check_release_package import is_sanitized_basename
from export_hf_release import public_eval_payload, sanitize_eval_payload, sanitize_training_metadata

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from kobun_llm.device import clean_device_description


def main() -> None:
    if clean_device_description("AMD Radeon RX 7600 XT\x00\x00") != "AMD Radeon RX 7600 XT":
        raise SystemExit("DirectML device description sanitizer did not remove NUL padding")
    if clean_device_description("\x00\x1f") != "unknown-device":
        raise SystemExit("DirectML device description sanitizer did not fail closed for all-control text")

    dirty_payload = {
        "checkpoint": r"X:\PrivateHome\project\checkpoints\model_best.pt",
        "checkpoint_from_log": r"checkpoints\model_best.pt",
        "quality_log": r"logs\quality_model.log",
        "leakage": {
            "manifest": r"data\run_snapshots\run\corpus_manifest.jsonl",
            "train": r"data\run_snapshots\run\kobun_worldclass_corpus.txt",
        },
        "split_consistency": {
            "manifest": r"data\run_snapshots\run\corpus_manifest.jsonl",
            "train": r"data\run_snapshots\run\kobun_worldclass_corpus.txt",
            "validation": r"data\run_snapshots\run\kobun_labeled_grammar_val.txt",
            "test": r"data\run_snapshots\run\kobun_labeled_grammar_test.txt",
        },
        "eval_contamination": {
            "train_paths": [r"data\run_snapshots\run\kobun_worldclass_corpus.txt"],
            "eval_paths": [r"data\eval\grammar_minimal_pairs.jsonl"],
        },
        "eval_contamination_checks": [
            {
                "train_paths": [r"data\run_snapshots\run\kobun_worldclass_corpus.txt"],
                "eval_paths": [r"data\eval\morphology_adversarial_cases.jsonl"],
            }
        ],
        "eval_files": [
            {
                "path": r"data\eval\grammar_minimal_pairs.jsonl",
                "source": r"data\eval\grammar_minimal_pairs_heldout.jsonl",
            }
        ],
        "test_lm": {
            "test_data": r"data\run_snapshots\run\kobun_labeled_grammar_test.txt",
        },
        "corpus_checks": [
            r"data\run_snapshots\run\kobun_worldclass_corpus.txt",
            r"data\run_snapshots\run\kobun_labeled_grammar_val.txt",
            r"data\run_snapshots\run\kobun_labeled_grammar_test.txt",
        ],
        "failure_reasons": [
            r"X:\PrivateHome\ProjectRoot\logs\quality.log: checkpoint failed",
            r"token=secret-value path=data\run_snapshots\run\validation.txt",
        ],
        "raw_excerpt": "これは公開してはいけない本文断片",
        "internal_log_tail": r"logs\quality_model.log",
    }
    clean_payload = sanitize_eval_payload(dirty_payload)
    public_payload = public_eval_payload(dirty_payload)
    clean_text = json.dumps(clean_payload, ensure_ascii=False)
    public_text = json.dumps(public_payload, ensure_ascii=False)
    forbidden_fragments = (
        r"X:\PrivateHome",
        r"data\run_snapshots",
        r"data\\run_snapshots",
        r"logs\\",
        r"logs\quality",
        r"checkpoints\\",
        "secret-value",
    )
    hits = [fragment for fragment in forbidden_fragments if fragment in clean_text]
    if hits:
        raise SystemExit(f"sanitized eval payload still has local path fragments: {hits}")
    public_hits = [fragment for fragment in forbidden_fragments if fragment in public_text]
    if public_hits:
        raise SystemExit(f"public eval payload still has local path fragments: {public_hits}")
    if "raw_excerpt" in public_payload or "internal_log_tail" in public_payload:
        raise SystemExit("public eval payload accepted a non-allowlisted top-level key")

    for value in (
        clean_payload["split_consistency"]["manifest"],
        clean_payload["split_consistency"]["train"],
        clean_payload["split_consistency"]["validation"],
        clean_payload["split_consistency"]["test"],
        clean_payload["eval_contamination"]["train_paths"][0],
        clean_payload["eval_contamination"]["eval_paths"][0],
    ):
        if not is_sanitized_basename(value):
            raise SystemExit(f"sanitized value is not basename-only: {value!r}")

    for value in (
        r"data\run_snapshots\run\kobun_worldclass_corpus.txt",
        r"logs\quality.log",
        r"checkpoints\model.pt",
        r"X:\PrivateHome\model.pt",
    ):
        if is_sanitized_basename(value):
            raise SystemExit(f"unsafe path value was accepted as sanitized: {value!r}")

    dirty_metadata = {
        "run_id": "old_japanese_0_1b_dml_test",
        "checkpoint": r"X:\PrivateHome\project\checkpoints\model_best.pt",
        "data_path": r"data\run_snapshots\run\train.txt",
        "raw_text_excerpt": "これも公開不可",
        "optimizer_state": {"step": 10},
        "tokenizer_extra_data": [
            {"path": r"data\run_snapshots\run\tokenizer_public_char_vocab.txt", "sha256": "a" * 64, "raw": "bad"}
        ],
        "provenance_files": [
            {"path": r"data\run_snapshots\run\provenance\corpus_manifest.jsonl", "sha256": "b" * 64}
        ],
    }
    public_metadata = sanitize_training_metadata(dirty_metadata)
    if "raw_text_excerpt" in public_metadata or "optimizer_state" in public_metadata:
        raise SystemExit("public training metadata accepted a non-allowlisted key")
    metadata_text = json.dumps(public_metadata, ensure_ascii=False)
    metadata_hits = [fragment for fragment in forbidden_fragments if fragment in metadata_text]
    if metadata_hits:
        raise SystemExit(f"public training metadata still has local path fragments: {metadata_hits}")

    print("release_sanitization_ok=true")


if __name__ == "__main__":
    main()
