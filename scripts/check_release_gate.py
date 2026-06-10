from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from export_hf_release import checkpoint_provenance_path, validate_eval_payload
from kobun_autonomy.augmentation_audit import require_clean_augmentation_manifest
from kobun_llm.checkpoint_io import load_trusted_checkpoint
from kobun_autonomy.release_policy import require_release_candidate_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run non-mutating release-readiness gates for one exact best checkpoint.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--eval-results", type=Path, default=None)
    return parser.parse_args()


def infer_backend(run_id: str) -> str:
    if run_id.startswith("old_japanese_0_1b_dml_"):
        return "dml"
    if run_id.startswith("old_japanese_0_1b_cuda_"):
        return "cuda"
    raise SystemExit(
        "invalid release run id: expected old_japanese_0_1b_dml_* or "
        f"old_japanese_0_1b_cuda_*, got {run_id!r}"
    )


def run_checked(args: list[str]) -> None:
    result = subprocess.run(args, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        details = (result.stderr.strip() or result.stdout.strip() or f"exit={result.returncode}").splitlines()
        tail = details[-1] if details else f"exit={result.returncode}"
        raise SystemExit(f"release gate command failed: {' '.join(args)} :: {tail}")


def run_checked_output(args: list[str]) -> str:
    result = subprocess.run(args, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        details = (result.stderr.strip() or result.stdout.strip() or f"exit={result.returncode}").splitlines()
        tail = details[-1] if details else f"exit={result.returncode}"
        raise SystemExit(f"release gate command failed: {' '.join(args)} :: {tail}")
    return result.stdout


def main() -> None:
    args = parse_args()
    require_release_candidate_run(args.run_id, context="release gate")
    backend = infer_backend(args.run_id)
    checkpoint = args.checkpoint
    expected_checkpoint = Path("checkpoints") / f"{args.run_id}_best.pt"
    if checkpoint.as_posix() != expected_checkpoint.as_posix() and checkpoint.resolve(strict=False) != expected_checkpoint.resolve(strict=False):
        raise SystemExit(f"checkpoint must be the exact best checkpoint: expected={expected_checkpoint} actual={checkpoint}")
    if not checkpoint.exists():
        raise SystemExit(f"missing exact best checkpoint: {checkpoint}")

    eval_path = args.eval_results or Path("logs") / f"eval_results_{args.run_id}.json"
    if not eval_path.exists():
        raise SystemExit(f"missing eval results for release gate: {eval_path}")

    py = sys.executable
    run_checked(
        [
            py,
            "scripts/check_run_completion.py",
            "--run-id",
            args.run_id,
            "--checkpoint",
            str(checkpoint),
            "--backend",
            backend,
            "--require-no-active-process",
            "--active-process-scope",
            "supervision",
            "--require-no-active-lock",
        ]
    )
    run_checked(
        [
            py,
            "scripts/check_checkpoint_model_size.py",
            "--checkpoint",
            str(checkpoint),
            "--strict-config",
            "--require-release-prefix",
            "old-japanese-0.1B",
            "--fail-on-val-oov",
            "--require-from-scratch",
            "--require-seed",
            "--require-optimizer",
            "simple-adamw",
            "--require-backend",
            backend,
        ]
    )
    run_checked(
        [
            py,
            "scripts/check_checkpoint_training_inputs.py",
            "--checkpoint",
            str(checkpoint),
            "--require-val-data",
            "--require-test-data",
            "--require-from-scratch",
            "--require-run-snapshot",
            "--allow-same-run-resume",
        ]
    )

    eval_payload = json.loads(eval_path.read_text(encoding="utf-8-sig"))
    payload = load_trusted_checkpoint(checkpoint, map_location="cpu")
    metadata = dict(payload.get("metadata", {}) or {})
    aozora_sources = checkpoint_provenance_path(metadata, "aozora_sources.json")
    waka_sources = checkpoint_provenance_path(metadata, "waka_sources.json")
    corpus_manifest = checkpoint_provenance_path(metadata, "corpus_manifest.jsonl")
    training_augmentation_manifest = checkpoint_provenance_path(metadata, "training_augmentation_manifest.json")
    tokenizer_meta = checkpoint_provenance_path(metadata, "tokenizer_public_char_vocab.meta.json")
    tokenizer_extra_records = metadata.get("tokenizer_extra_data") or []
    if not tokenizer_extra_records or not isinstance(tokenizer_extra_records[0], dict):
        raise SystemExit("checkpoint metadata missing tokenizer_extra_data record for release gate.")
    tokenizer_extra = Path(str(tokenizer_extra_records[0].get("path") or ""))
    if not tokenizer_extra.is_absolute():
        tokenizer_extra = Path.cwd() / tokenizer_extra
    if not tokenizer_extra.exists():
        raise SystemExit(f"checkpoint tokenizer extra data path does not exist: {tokenizer_extra}")
    require_clean_augmentation_manifest(training_augmentation_manifest, require_local_files=False)
    source_audit_output = run_checked_output(
        [
            py,
            "scripts/audit_source_records.py",
            str(aozora_sources),
            str(waka_sources),
        ]
    )
    for required in ("mismatches=0", "missing=0", "fixed=False"):
        if required not in source_audit_output:
            raise SystemExit(f"checkpoint-bound source audit is not clean: missing {required}")
    with tempfile.TemporaryDirectory() as tmp:
        public_summary = Path(tmp) / "public_manifest_summary.json"
        public_manifest_output = run_checked_output(
            [
                py,
                "scripts/audit_public_manifest.py",
                "--manifest",
                str(corpus_manifest),
                "--out",
                str(public_summary),
            ]
        )
    if "errors=0" not in public_manifest_output:
        raise SystemExit("checkpoint-bound public manifest audit is not clean.")
    run_checked(
        [
            py,
            "scripts/check_checkpoint_tokenizer_scope.py",
            "--checkpoint",
            str(checkpoint),
            "--manifest",
            str(corpus_manifest),
            "--tokenizer-extra-data",
            str(tokenizer_extra),
            "--tokenizer-meta",
            str(tokenizer_meta),
        ]
    )
    validate_eval_payload(eval_payload, checkpoint, payload)
    print(f"release_gate_ok=true run_id={args.run_id} backend={backend} checkpoint={checkpoint}")


if __name__ == "__main__":
    main()
