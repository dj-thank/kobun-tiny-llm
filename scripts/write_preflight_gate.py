from __future__ import annotations

import hashlib
import json
import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "logs" / "preflight_gate_old_japanese_0_1b.json"
STATIC_MANIFEST = ROOT / "logs" / "static_quality_manifest_old_japanese_0_1b.json"


HASHED_INPUTS = [
    "scripts/run_static_quality_checks.ps1",
    "scripts/autonomous_old_japanese_0_1b_loop.ps1",
    "scripts/start_old_japanese_0_1b_dml_and_watch.ps1",
    "scripts/start_old_japanese_0_1b_cuda_colab_and_watch.py",
    "scripts/check_colab_cuda_environment.py",
    "scripts/run_quality_checks_cuda.py",
    "scripts/train_old_japanese_0_1b_dml.ps1",
    "scripts/watch_and_finalize_old_japanese_0_1b_dml.ps1",
    "scripts/finalize_old_japanese_0_1b_dml.ps1",
    "scripts/write_preflight_gate.py",
    "scripts/verify_preflight_gate.py",
    "scripts/write_zero_base_review_gate.py",
    "scripts/verify_zero_base_review_gate.py",
    "scripts/assert_run_id_unused.py",
    "scripts/run_command_capture.py",
    "scripts/check_run_completion.py",
    "scripts/check_checkpoint_model_size.py",
    "scripts/check_checkpoint_training_inputs.py",
    "scripts/check_release_gate.py",
    "scripts/old_japanese_run_intel.py",
    "scripts/build_waka_training_corpus.py",
    "scripts/audit_source_records.py",
    "scripts/build_manifest.py",
    "scripts/audit_public_manifest.py",
    "scripts/build_waka_meter_corpus.py",
    "scripts/build_tokenizer_public_vocab.py",
    "scripts/build_training_corpus.py",
    "scripts/build_preference_boost_corpus.py",
    "scripts/build_worldclass_corpus.py",
    "scripts/build_training_augmentation_manifest.py",
    "scripts/snapshot_training_inputs.py",
    "src/kobun_llm/train.py",
    "src/kobun_llm/device.py",
    "src/kobun_llm/tokenizer.py",
    "src/kobun_llm/model.py",
    "src/kobun_llm/checkpoint_io.py",
    "src/kobun_llm/optimizer_state.py",
    "src/kobun_autonomy/non_release_registry.py",
    "scripts/check_release_workspace_clean.py",
    "scripts/check_tokenizer_vocab_scope.py",
    "scripts/check_split_consistency.py",
    "scripts/check_split_leakage.py",
    "scripts/check_eval_contamination.py",
    "scripts/audit_eval_provenance_manifest.py",
    "scripts/eval_grammar_constraints.py",
    "scripts/eval_waka_rules.py",
    "scripts/eval_waka_meter_generation.py",
    "scripts/eval_waka_meter_constraints.py",
    "scripts/eval_morphology_adversarial.py",
    "scripts/eval_minimal_pairs.py",
    "scripts/snapshot_eval_files.py",
    "scripts/export_hf_release.py",
    "scripts/check_release_package.py",
    "src/kobun_autonomy/release_policy.py",
    "src/kobun_autonomy/augmentation_audit.py",
    "src/kobun_autonomy/types.py",
    "src/kobun_llm/grammar_constraints.py",
    "src/kobun_llm/grammar.py",
    "src/kobun_llm/genre_rules.py",
    "src/kobun_llm/waka_meter_constraints.py",
    "data/rules/period_scope_policy.json",
    "data/rules/generation_diagnostic_policy.json",
    "data/rules/genji_era_auxiliaries.json",
    "data/rules/genji_era_kakari_musubi.json",
    "data/rules/genji_era_honorifics.json",
    "data/rules/waka_meter_rules.json",
    "data/corpus_manifest.jsonl",
    "data/tokenizer_public_char_vocab.txt",
    "data/tokenizer_public_char_vocab.meta.json",
    "data/training_augmentation_manifest.json",
    "data/kobun_worldclass_corpus.txt",
    "data/kobun_labeled_grammar_val.txt",
    "data/kobun_labeled_grammar_test.txt",
    "data/eval/grammar_minimal_pairs.jsonl",
    "data/eval/grammar_minimal_pairs_heldout.jsonl",
    "data/eval/morphology_adversarial_cases.jsonl",
    "data/eval/grammar_constraint_cases.jsonl",
    "data/eval/waka_rule_cases.jsonl",
    "data/eval/waka_meter_constraint_cases.jsonl",
    "data/eval/waka_generation_prompts.jsonl",
    "data/eval/eval_provenance_manifest.json",
    "data/grammar/train_preference_pairs.jsonl",
    "logs/public_manifest_summary.json",
    "notebooks/old_japanese_0_1b_colab_cuda.ipynb",
]
HASHED_GLOBS = [
    "scripts/*.py",
    "scripts/*.ps1",
    "src/kobun_llm/*.py",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write the DML release-candidate preflight gate.")
    parser.add_argument("--max-age-minutes", type=float, default=120.0)
    return parser.parse_args()


def parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("naive datetime")
    return parsed.astimezone(timezone.utc)


def collect_hashes() -> dict[str, str]:
    hashes: dict[str, str] = {}
    missing: list[str] = []
    inputs = set(HASHED_INPUTS)
    for pattern in HASHED_GLOBS:
        for path in ROOT.glob(pattern):
            if path.is_file():
                inputs.add(path.relative_to(ROOT).as_posix())
    for rel in sorted(inputs):
        path = ROOT / rel
        if not path.exists():
            missing.append(rel)
            continue
        hashes[rel.replace("\\", "/")] = sha256_file(path)
    if missing:
        raise SystemExit(f"preflight_gate_missing_inputs={missing}")
    return hashes


def release_workspace_state() -> dict[str, Any]:
    release_dir = ROOT / "release"
    files = []
    if release_dir.exists():
        for path in sorted(p for p in release_dir.rglob("*") if p.is_file()):
            files.append(path.relative_to(ROOT).as_posix())
    return {"clean": len(files) == 0, "files": files}


def require_static_quality_manifest(max_age_minutes: float) -> dict[str, Any]:
    if not STATIC_MANIFEST.exists():
        raise SystemExit(f"missing_static_quality_manifest={STATIC_MANIFEST.relative_to(ROOT)}")
    manifest = read_json(STATIC_MANIFEST)
    if manifest.get("schema") != "old_japanese_0_1b_static_quality_manifest_v1":
        raise SystemExit("static_quality_manifest_schema_mismatch")
    if manifest.get("status") != "passed" or int(manifest.get("exit_code", -1)) != 0:
        raise SystemExit("static_quality_manifest_not_passed")
    if manifest.get("hf_export") is not False:
        raise SystemExit("static_quality_manifest_hf_export_not_false")
    command = str(manifest.get("command") or "")
    if "scripts\\run_static_quality_checks.ps1" not in command and "scripts/run_static_quality_checks.ps1" not in command:
        raise SystemExit("static_quality_manifest_command_mismatch")
    if "-RefreshEvidence" not in command:
        raise SystemExit("static_quality_manifest_not_from_explicit_refresh_evidence")
    try:
        generated = parse_utc(str(manifest.get("generated_at_utc") or ""))
    except Exception as exc:
        raise SystemExit(f"static_quality_manifest_invalid_generated_at_utc={exc}") from exc
    age_minutes = (datetime.now(timezone.utc) - generated).total_seconds() / 60.0
    if age_minutes < -1.0:
        raise SystemExit(f"static_quality_manifest_from_future age_minutes={age_minutes:.2f}")
    if age_minutes > max_age_minutes:
        raise SystemExit(f"static_quality_manifest_stale age_minutes={age_minutes:.2f} max={max_age_minutes:.2f}")
    for key, hash_key in (("log", "log_sha256"), ("runner", "runner_sha256")):
        rel = str(manifest.get(key) or "")
        if not rel:
            raise SystemExit(f"static_quality_manifest_{key}_missing")
        path = (ROOT / rel).resolve()
        try:
            path.relative_to(ROOT)
        except ValueError as exc:
            raise SystemExit(f"static_quality_manifest_{key}_escapes_root={rel}") from exc
        if not path.exists():
            raise SystemExit(f"static_quality_manifest_{key}_missing_path={rel}")
        if sha256_file(path) != str(manifest.get(hash_key) or ""):
            raise SystemExit(f"static_quality_manifest_{hash_key}_mismatch")
    return manifest


def tokenizer_policy() -> str:
    meta = read_json(ROOT / "data" / "tokenizer_public_char_vocab.meta.json")
    return str(meta.get("policy") or "")


def main() -> None:
    args = parse_args()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    release_state = release_workspace_state()
    static_manifest = require_static_quality_manifest(args.max_age_minutes)
    payload = {
        "schema": "old_japanese_0_1b_preflight_gate_v1",
        "status": "passed",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "static_quality_checks": True,
        "static_quality_manifest": STATIC_MANIFEST.relative_to(ROOT).as_posix(),
        "static_quality_manifest_sha256": sha256_file(STATIC_MANIFEST),
        "static_quality_log": static_manifest.get("log", ""),
        "static_quality_log_sha256": static_manifest.get("log_sha256", ""),
        "reviews_required": True,
        "hf_export": False,
        "release_workspace_clean": release_state["clean"],
        "release_workspace_files": release_state["files"],
        "tokenizer_policy": tokenizer_policy(),
        "inputs_sha256": collect_hashes(),
    }
    if not payload["release_workspace_clean"]:
        raise SystemExit(f"release_workspace_not_clean files={release_state['files']}")
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"preflight_gate_written={OUT.relative_to(ROOT)}")
    print(f"preflight_gate_schema={payload['schema']}")


if __name__ == "__main__":
    main()
