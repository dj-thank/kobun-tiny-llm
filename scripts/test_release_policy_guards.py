from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def require(path: Path, needle: str) -> None:
    text = path.read_text(encoding="utf-8")
    if needle not in text:
        raise SystemExit(f"missing {needle!r} in {path}")


def main() -> None:
    require(ROOT / "src" / "kobun_llm" / "release_policy.py", "NON_RELEASE_RUNS")
    require(ROOT / "src" / "kobun_llm" / "train.py", "release-shaped training must be from scratch")
    require(ROOT / "src" / "kobun_llm" / "train.py", "require_release_candidate_run(args.run_id")
    require(ROOT / "scripts" / "check_run_completion.py", "require_release_candidate_run(run_id")
    require(ROOT / "scripts" / "check_release_gate.py", "require_release_candidate_run(args.run_id")
    require(ROOT / "scripts" / "export_hf_release.py", "require_release_candidate_run")
    require(ROOT / "scripts" / "latest_valid_checkpoint.py", "known_non_release_run")
    require(ROOT / "scripts" / "finalize_old_japanese_0_1b_dml.ps1", "DML finalizer refuses known non-release")
    require(ROOT / "scripts" / "start_old_japanese_0_1b_dml_and_watch.ps1", "DML supervisor launcher refuses known non-release")
    require(ROOT / "scripts" / "train_old_japanese_0_1b_dml.ps1", "DML training wrapper refuses known non-release")
    require(ROOT / "scripts" / "start_old_japanese_0_1b_dml_and_watch.ps1", "Get-AnyActiveDmlTrainingProcesses")
    require(ROOT / "scripts" / "start_old_japanese_0_1b_dml_and_watch.ps1", "Test-DmlTrainingCommandLine")
    require(ROOT / "scripts" / "start_old_japanese_0_1b_dml_and_watch.ps1", "(cpu|cuda|hip)")
    require(ROOT / "scripts" / "check_run_completion.py", "final checkpoint")
    require(ROOT / "scripts" / "check_run_completion.py", "final checkpoint metadata")
    require(ROOT / "scripts" / "check_run_completion.py", "training log modification")
    require(ROOT / "scripts" / "check_run_completion.py", "st_mtime")
    require(ROOT / "scripts" / "report_old_japanese_0_1b_progress.ps1", "$CheckpointFresh")
    require(ROOT / "scripts" / "old_japanese_run_intel.py", "missing_exact_best_checkpoint_for_post_run_quality")
    require(ROOT / "scripts" / "report_old_japanese_0_1b_progress.ps1", "present_but_rejected_hf_export_recorded")
    require(ROOT / "scripts" / "eval_waka_meter_generation.py", 'choices=["greedy", "sample"]')
    require(ROOT / "scripts" / "eval_waka_meter_generation.py", 'default="greedy"')
    require(ROOT / "scripts" / "eval_waka_meter_generation.py", "torch.argmax")
    require(ROOT / "scripts" / "run_quality_checks_dml.ps1", "--decoding greedy")
    require(ROOT / "src" / "kobun_llm" / "train.py", "--overfit-stop-gap")
    require(ROOT / "src" / "kobun_llm" / "train.py", "early stopping: overfit signal")
    require(ROOT / "scripts" / "train_old_japanese_0_1b_dml.ps1", "--overfit-stop-gap")
    require(ROOT / "scripts" / "train_old_japanese_0_1b_gpu.ps1", "CUDA/HIP release-candidate training is disabled")
    require(ROOT / "scripts" / "train_old_japanese_0_1b_gpu.ps1", "zero-base review gate")
    require(ROOT / "scripts" / "export_hf_release.py", "save_model(model")
    require(ROOT / "scripts" / "export_hf_release.py", "check_release_gate.py")
    require(ROOT / "scripts" / "export_hf_release.py", "--confirm-explicit-user-request")
    require(ROOT / "scripts" / "export_hf_release.py", "HF package creation is manual-only")
    require(ROOT / "scripts" / "latest_valid_checkpoint.py", "check_release_gate.py")
    require(ROOT / "scripts" / "latest_valid_checkpoint.py", "not_exact_best_path")
    require(ROOT / "scripts" / "check_release_package.py", 'ALLOWED_FILES = REQUIRED_FILES | {"model.safetensors"}')
    require(ROOT / "scripts" / "check_release_package.py", "pickle-based .pt weights are forbidden")
    if "pytorch_model.pt" in (ROOT / "scripts" / "export_hf_release.py").read_text(encoding="utf-8"):
        raise SystemExit("export_hf_release.py must not write pytorch_model.pt")
    if "--allow-missing-safetensors" in (ROOT / "scripts" / "export_hf_release.py").read_text(encoding="utf-8"):
        raise SystemExit("export_hf_release.py must not allow missing safetensors")
    print("release_policy_guards_static_ok=true")


if __name__ == "__main__":
    main()
