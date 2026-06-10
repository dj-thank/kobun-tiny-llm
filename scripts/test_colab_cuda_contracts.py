from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "old_japanese_0_1b_colab_cuda.ipynb"
STARTER = ROOT / "scripts" / "start_old_japanese_0_1b_cuda_colab_and_watch.py"
ENV_CHECK = ROOT / "scripts" / "check_colab_cuda_environment.py"
QUALITY = ROOT / "scripts" / "run_quality_checks_cuda.py"
TRAIN = ROOT / "src" / "kobun_llm" / "train.py"
BUNDLE = ROOT / "scripts" / "create_colab_sync_bundle.py"


def require_contains(path: Path, needle: str) -> None:
    text = path.read_text(encoding="utf-8")
    if needle not in text:
        raise SystemExit(f"{path.relative_to(ROOT)} missing Colab CUDA contract: {needle!r}")


def require_not_contains(path: Path, needle: str) -> None:
    text = path.read_text(encoding="utf-8")
    if needle in text:
        raise SystemExit(f"{path.relative_to(ROOT)} must not contain forbidden Colab/HF action: {needle!r}")


def main() -> None:
    if NOTEBOOK.exists():
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        if notebook.get("nbformat") != 4:
            raise SystemExit("Colab notebook is not nbformat v4")
        source_text = "\n".join(
            "".join(cell.get("source") or [])
            for cell in notebook.get("cells") or []
            if isinstance(cell, dict)
        )
        for needle in (
            "drive.mount('/content/drive')",
            "PROJECT_ROOT = '/content/drive/MyDrive/kobun-tiny-llm'",
            "parts[0] != 'kobun-tiny-llm'",
            "scripts/check_colab_cuda_environment.py",
            "scripts/start_old_japanese_0_1b_cuda_colab_and_watch.py",
            "RUN_ID = 'old_japanese_0_1b_cuda_'",
            "'--run-id', RUN_ID",
            "--allow-start-training",
            "--reviews-passed",
            "validate_colab_sync_archive(archive)",
            "colab_sync_bundle_manifest.json",
            "contains_codex_state",
            "forbidden internal/secret text",
        ):
            if needle not in source_text:
                raise SystemExit(f"notebook missing expected Colab cell content: {needle!r}")

    for path in tuple(p for p in (NOTEBOOK, STARTER, ENV_CHECK, QUALITY) if p.exists()):
        for forbidden in (
            "export_hf_release.py",
            "push_to_hub",
            "HfApi",
            "create_repo",
            "huggingface_hub",
            "hf upload",
            "huggingface-cli",
        ):
            require_not_contains(path, forbidden)

    require_contains(STARTER, "old_japanese_0_1b_cuda_")
    require_contains(STARTER, "old_japanese_0_1b_cuda_colab_launch_context_v1")
    require_contains(STARTER, "old_japanese_0_1b_supervised_cuda_launch_context_v1")
    require_contains(STARTER, "--cuda-provider")
    require_contains(STARTER, "choices=[\"colab\", \"gcp\"]")
    require_contains(STARTER, "supervised_cuda_training")
    require_contains(STARTER, "gcp_active_old_japanese_0_1b_cuda")
    require_contains(STARTER, 'path.name.startswith("gcp_")')
    require_contains(STARTER, "google_credentials_read")
    require_contains(STARTER, "hf_export")
    require_contains(STARTER, "scripts/verify_preflight_gate.py")
    require_contains(STARTER, "scripts/verify_zero_base_review_gate.py")
    require_contains(STARTER, "scripts/assert_run_id_unused.py")
    require_contains(STARTER, "scripts/snapshot_training_inputs.py")
    require_contains(STARTER, "scripts/run_quality_checks_cuda.py")
    require_contains(STARTER, "parser.add_argument(\"--skip-post-run-quality\"")
    require_contains(STARTER, "post_run_quality_skipped_by_explicit_dry_run_flag")
    require_contains(STARTER, "run_checked([py, \"scripts/run_quality_checks_cuda.py\", \"--checkpoint\", best_out])")
    require_contains(STARTER, "refusing to update active CUDA lock not owned by this launcher")
    require_contains(STARTER, "def remove_owned_lock")
    require_contains(STARTER, "refusing to remove CUDA startup lock not owned by this launcher")
    require_contains(STARTER, "startup_or_training_supervision_failed")
    require_contains(STARTER, "archive_owned_lock")
    require_contains(STARTER, "def write_json_atomic")
    require_contains(STARTER, "os.replace(tmp, path)")
    require_contains(STARTER, "colab_active_old_japanese_0_1b_cuda")
    require_contains(STARTER, ".failed_non_release.")
    require_contains(STARTER, ".finished.")
    require_contains(STARTER, ".stale.")
    require_contains(STARTER, "old_japanese_0_1b_colab_cuda_active_lease_v1")
    require_contains(STARTER, "write_colab_lease")
    require_contains(STARTER, "archive_colab_lease")
    require_contains(STARTER, "def assert_no_active_colab_cuda_lease")
    require_contains(STARTER, "colab CUDA lease health blockers")
    require_contains(STARTER, "failed_non_release")
    require_contains(STARTER, "write_non_release_record")
    require_contains(STARTER, "non_release_runs")
    require_contains(STARTER, "suffix = \"finished\" if state == \"finished\" else \"failed_non_release\"")
    require_contains(STARTER, "active_old_japanese_0_1b_training.lock")
    require_contains(STARTER, "supervised_training_processes")
    require_contains(STARTER, "def current_process_tree_ids")
    require_contains(STARTER, "def supervised_wrapper_command")
    require_contains(STARTER, "assert_no_other_supervised_training(run_id)")
    require_contains(STARTER, "cuda_like_train_command")
    require_contains(STARTER, "kobun_llm.train")
    require_contains(STARTER, "--device(?:\\s+|=)")
    require_contains(STARTER, "device in {\"cuda\", \"auto\"}")
    require_contains(STARTER, "The train CLI default is auto")
    require_not_contains(STARTER, "if run_id and run_id in text")
    require_contains(STARTER, "\"simple-adamw\"")
    require_contains(STARTER, "write_train_exit_sentinel(train_exit, run_id, exit_code, message, out, best_out)")
    require_contains(STARTER, "scripts/verify_preflight_gate.py\", \"--gate\", args.preflight_gate")
    require_contains(STARTER, "scripts/verify_zero_base_review_gate.py\", \"--gate\", args.review_gate")
    starter_text = STARTER.read_text(encoding="utf-8")
    lock_index = starter_text.find("acquire_lock(active_lock")
    build_index = starter_text.find("\"scripts/build_waka_training_corpus.py\"")
    if lock_index < 0 or build_index < 0 or lock_index > build_index:
        raise SystemExit("CUDA active lock must be acquired before mutable corpus/snapshot build commands")
    refresh_calls = starter_text.count("refresh_colab_lease(") - 1  # exclude the function definition
    refresh_provider_args = starter_text.count("cuda_provider=cuda_provider")
    if refresh_calls < 2 or refresh_provider_args < refresh_calls:
        raise SystemExit("all CUDA lease refresh calls must bind cuda_provider")

    require_contains(ENV_CHECK, "torch.cuda.is_available()")
    require_contains(ENV_CHECK, "old_japanese_0_1b_supervised_cuda_environment_v1")
    require_contains(ENV_CHECK, "cuda_provider")
    require_contains(ENV_CHECK, "gcp_compute_hint")
    require_contains(ENV_CHECK, "torch_hip_version")
    require_contains(ENV_CHECK, "cuda_runtime_kind")
    require_contains(ENV_CHECK, "real_cuda_runtime")
    require_contains(ENV_CHECK, "hip_runtime_is_not_cuda")
    require_contains(ENV_CHECK, "google_credentials_read")
    require_contains(BUNDLE, "DOC_ALLOWLIST")
    require_contains(BUNDLE, "validate_no_internal_state(files)")
    require_contains(BUNDLE, "colab_bundle_internal_state_pattern")
    require_contains(BUNDLE, '"codex" + r"://"')
    require_contains(BUNDLE, "[A-Za-z]:[\\\\/]+Users")
    require_contains(BUNDLE, "local_project_path")
    require_contains(BUNDLE, "contains_training_corpus_text_for_private_colab")
    require_contains(BUNDLE, "Handoff docs, assistant thread URIs")
    require_contains(ROOT / "scripts" / "old_japanese_run_intel.py", "def colab_cuda_lease_health")
    require_contains(ROOT / "scripts" / "old_japanese_run_intel.py", "colab_cuda_lease_active")
    require_contains(ROOT / "scripts" / "old_japanese_run_intel.py", "gcp_active_old_japanese_0_1b_cuda")
    require_contains(ROOT / "scripts" / "old_japanese_run_intel.py", "old_japanese_0_1b_supervised_cuda_active_lease_v1")
    require_contains(ROOT / "scripts" / "check_run_completion.py", "gcp_active_old_japanese_0_1b_cuda")
    require_contains(ROOT / "scripts" / "update_evaluation_board.py", "colab_cuda_lease_health")
    spec = importlib.util.spec_from_file_location("create_colab_sync_bundle", BUNDLE)
    if spec is None or spec.loader is None:
        raise SystemExit("could not import create_colab_sync_bundle.py for contract validation")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    files = module.iter_files()
    rels = {module.rel_posix(path) for path in files}
    for forbidden in (
        "TRAINING_LOG.md",
        "docs/" + "HAND" + "OFF_TO_CODEX_THREAD_019e0c2a_2026-05-09.md",
    ):
        if forbidden in rels:
            raise SystemExit(f"Colab sync bundle must not include internal handoff/log artifact: {forbidden}")
    for rel in rels:
        if rel.startswith("docs/") and rel not in module.DOC_ALLOWLIST:
            raise SystemExit(f"Colab sync bundle includes non-allowlisted docs file: {rel}")
    module.validate_no_forbidden(files)
    module.validate_no_internal_state(files)
    local_marker_specs = (
        ("windows_home", ("C:", "\\Users\\", "example-user")),
        ("posix_home", ("C:", "/Users/", "example-user")),
        ("windows_project", ("ExampleWorkstation", "\\", "ExampleProjects")),
        ("posix_project", ("ExampleWorkstation", "/", "ExampleProjects")),
    )
    real_markers = tuple((label, "".join(parts)) for label, parts in local_marker_specs)
    for path in files:
        rel = module.rel_posix(path)
        if path.suffix.lower() not in module.TEXT_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8-sig", errors="replace")
        for label, marker in real_markers:
            if marker in text:
                raise SystemExit(f"Colab sync bundle includes real local marker {label} in {rel}")
    require_contains(QUALITY, "--require-backend")
    require_contains(QUALITY, "cuda")
    require_contains(QUALITY, "require_real_cuda_runtime('CUDA quality checks')")
    require_contains(QUALITY, "torch_hip_version")
    require_contains(QUALITY, "cuda_runtime_kind")
    require_contains(QUALITY, "real_cuda_runtime=true")
    require_contains(QUALITY, "run_id = checkpoint_base.removesuffix(\"_best\")")
    require_contains(QUALITY, "\"simple-adamw\"")
    require_contains(QUALITY, "scripts/check_run_completion.py")
    require_contains(QUALITY, "val_data = parse_path_from_output(input_output, \"val_data_path=\")")
    require_contains(QUALITY, "[py, \"scripts/validate_corpus.py\", train_data]")
    require_contains(QUALITY, "[py, \"scripts/validate_corpus.py\", val_data]")
    require_contains(QUALITY, "[py, \"scripts/validate_corpus.py\", test_data]")
    require_contains(QUALITY, "if completed.returncode != 0")
    require_contains(QUALITY, "eval JSON is not bound to checkpoint")
    require_contains(QUALITY, "scripts/parse_quality_log.py")
    require_contains(TRAIN, "old_japanese_0_1b_cuda_")
    require_contains(TRAIN, "start_old_japanese_0_1b_cuda_colab_and_watch.py")
    require_contains(TRAIN, "old_japanese_0_1b_cuda_colab_launch_context_v1")
    require_contains(TRAIN, "old_japanese_0_1b_supervised_cuda_launch_context_v1")
    require_contains(TRAIN, "supervised_cuda_training")
    print("colab_cuda_contracts_ok=true")


if __name__ == "__main__":
    main()
