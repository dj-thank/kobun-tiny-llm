from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
RUN_ID = "old_japanese_0_1b_dml_run_id_unused_guard_test"
STALE_ARTIFACTS = [
    ROOT / "logs" / f"eval_results_{RUN_ID}.json",
    ROOT / "logs" / "llm_review_packets" / f"{RUN_ID}.json",
    ROOT / "logs" / "upload_ready_evidence" / f"not_upload_ready_diagnostic_{RUN_ID}.json",
    ROOT / "logs" / f"autonomous_launch_context_{RUN_ID}.json",
    ROOT / "release" / f"hf_model_{RUN_ID}" / "config.json",
    ROOT / "checkpoints" / f"{RUN_ID}.pt.tmp",
]
PAYLOAD_ONLY_ACTIVE_LOCK = ROOT / "logs" / "active_old_japanese_0_1b_dml.stale.legacy_payload_only.json"
CANONICAL_ACTIVE_LOCK = ROOT / "logs" / "active_old_japanese_0_1b_dml.lock"
STARTUP_MUTEX_LOCK = ROOT / "logs" / "active_old_japanese_0_1b_training.lock"
TEST_PREFLIGHT = ROOT / "logs" / "preflight_gate_run_id_unused_guard_test.json"
TEST_REVIEW = ROOT / "logs" / "zero_base_review_gate_run_id_unused_guard_test.json"


def run(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=ROOT, text=True, capture_output=True, check=False)


def cleanup_test_artifact(path: Path) -> None:
    if path.is_dir():
        for child in sorted(path.rglob("*"), reverse=True):
            if child.is_file():
                child.unlink()
            elif child.is_dir():
                child.rmdir()
        path.rmdir()
        return
    path.unlink(missing_ok=True)
    parent = path.parent
    if parent.name == f"hf_model_{RUN_ID}" and parent.exists():
        parent.rmdir()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def assert_rejected_for(artifact: Path) -> None:
    artifact.parent.mkdir(parents=True, exist_ok=True)
    try:
        artifact.write_text('{"status":"stale"}\n', encoding="utf-8")
        stale = run([str(PYTHON), "scripts/assert_run_id_unused.py", "--run-id", RUN_ID])
        if stale.returncode == 0:
            raise SystemExit(f"stale_artifact_did_not_block_run_id_reuse path={artifact.relative_to(ROOT)}")
        if "run_id_reuse_artifacts=" not in stale.stdout:
            raise SystemExit(f"unexpected_run_id_reuse_output stdout={stale.stdout} stderr={stale.stderr}")
    finally:
        cleanup_test_artifact(artifact)


def assert_rejected_for_payload(artifact: Path, payload: str) -> None:
    artifact.parent.mkdir(parents=True, exist_ok=True)
    try:
        artifact.write_text(payload, encoding="utf-8")
        stale = run([str(PYTHON), "scripts/assert_run_id_unused.py", "--run-id", RUN_ID])
        if stale.returncode == 0:
            raise SystemExit(f"payload_artifact_did_not_block_run_id_reuse path={artifact.relative_to(ROOT)}")
        if "run_id_reuse_artifacts=" not in stale.stdout:
            raise SystemExit(f"unexpected_run_id_reuse_output stdout={stale.stdout} stderr={stale.stderr}")
    finally:
        cleanup_test_artifact(artifact)


def assert_allowed_supervisor_context_only() -> None:
    contexts = (
        ROOT / "logs" / f"autonomous_launch_context_{RUN_ID}.json",
        ROOT / "logs" / f"colab_cuda_launch_context_{RUN_ID}.json",
        ROOT / "logs" / f"gcp_cuda_launch_context_{RUN_ID}.json",
    )
    for context in contexts:
        cleanup_test_artifact(context)
        context.parent.mkdir(parents=True, exist_ok=True)
        try:
            context.write_text('{"run_id":"' + RUN_ID + '","hf_export":false}\n', encoding="utf-8")
            allowed = run(
                [
                    str(PYTHON),
                    "scripts/assert_run_id_unused.py",
                    "--run-id",
                    RUN_ID,
                    "--allow-supervisor-launch-artifacts",
                ]
            )
            if allowed.returncode != 0:
                raise SystemExit(
                    f"supervisor_launch_context_was_not_allowed path={context.relative_to(ROOT)} stdout={allowed.stdout} stderr={allowed.stderr}"
                )
            stale = run([str(PYTHON), "scripts/assert_run_id_unused.py", "--run-id", RUN_ID])
            if stale.returncode == 0:
                raise SystemExit(
                    f"supervisor_launch_context_did_not_block_without_allow_flag path={context.relative_to(ROOT)}"
                )
        finally:
            cleanup_test_artifact(context)


def assert_allowed_current_supervisor_active_lock() -> None:
    context = ROOT / "logs" / f"autonomous_launch_context_{RUN_ID}.json"
    token = "test-supervisor-token"
    for path in (context, TEST_PREFLIGHT, TEST_REVIEW, CANONICAL_ACTIVE_LOCK):
        cleanup_test_artifact(path)
    try:
        context.write_text('{"run_id":"' + RUN_ID + '","hf_export":false}\n', encoding="utf-8")
        TEST_PREFLIGHT.write_text('{"schema":"test_preflight"}\n', encoding="utf-8")
        TEST_REVIEW.write_text('{"schema":"test_review"}\n', encoding="utf-8")
        payload = {
            "run_id": RUN_ID,
            "backend": "dml",
            "state": "running",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "launcher_pid": os.getpid(),
            "hf_export": False,
            "launch_token_sha256": sha256_text(token),
            "launch_nonce_sha256": "0" * 64,
            "preflight_gate": str(TEST_PREFLIGHT),
            "preflight_gate_sha256": sha256_file(TEST_PREFLIGHT),
            "review_gate": str(TEST_REVIEW),
            "review_gate_sha256": sha256_file(TEST_REVIEW),
            "autonomous_launch_context": str(context),
            "autonomous_launch_context_sha256": sha256_file(context),
            "autonomous_script": "scripts/autonomous_old_japanese_0_1b_loop.ps1",
            "selected_action": "prepare_next_fresh_run_after_static_gate_and_zero_base_reviews",
        }
        CANONICAL_ACTIVE_LOCK.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        env = {
            **os.environ,
            "OLD_JAPANESE_SUPERVISOR_RUN_ID": RUN_ID,
            "OLD_JAPANESE_SUPERVISOR_TOKEN": token,
            "OLD_JAPANESE_ACTIVE_LOCK": str(CANONICAL_ACTIVE_LOCK),
            "OLD_JAPANESE_PREFLIGHT_GATE": str(TEST_PREFLIGHT),
            "OLD_JAPANESE_REVIEW_GATE": str(TEST_REVIEW),
            "OLD_JAPANESE_AUTONOMOUS_LAUNCH_CONTEXT": str(context),
        }
        allowed = subprocess.run(
            [
                str(PYTHON),
                "scripts/assert_run_id_unused.py",
                "--run-id",
                RUN_ID,
                "--allow-supervisor-launch-artifacts",
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
            env=env,
        )
        if allowed.returncode != 0:
            raise SystemExit(
                f"supervisor_active_lock_was_not_allowed stdout={allowed.stdout} stderr={allowed.stderr}"
            )
        stale = run([str(PYTHON), "scripts/assert_run_id_unused.py", "--run-id", RUN_ID])
        if stale.returncode == 0:
            raise SystemExit("supervisor_active_lock_did_not_block_without_allow_flag")
        payload["hf_export"] = True
        CANONICAL_ACTIVE_LOCK.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        rejected = subprocess.run(
            [
                str(PYTHON),
                "scripts/assert_run_id_unused.py",
                "--run-id",
                RUN_ID,
                "--allow-supervisor-launch-artifacts",
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
            env=env,
        )
        if rejected.returncode == 0:
            raise SystemExit("unsafe_supervisor_active_lock_was_allowed")
    finally:
        for path in (CANONICAL_ACTIVE_LOCK, context, TEST_PREFLIGHT, TEST_REVIEW):
            cleanup_test_artifact(path)


def assert_allowed_current_supervisor_startup_mutex() -> None:
    cleanup_test_artifact(STARTUP_MUTEX_LOCK)
    helper = subprocess.Popen(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            f"Start-Sleep -Seconds 30 # start_old_japanese_0_1b_dml_and_watch.ps1 {RUN_ID} -AllowStartTraining -ReviewsPassed",
        ],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        payload = {
            "run_id": RUN_ID,
            "backend": "dml",
            "launcher_pid": helper.pid,
            "state": "startup_mutex",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "hf_export": False,
        }
        STARTUP_MUTEX_LOCK.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        allowed = run(
            [
                str(PYTHON),
                "scripts/assert_run_id_unused.py",
                "--run-id",
                RUN_ID,
                "--allow-supervisor-launch-artifacts",
            ]
        )
        if allowed.returncode != 0:
            raise SystemExit(
                f"supervisor_startup_mutex_was_not_allowed stdout={allowed.stdout} stderr={allowed.stderr}"
            )
        stale = run([str(PYTHON), "scripts/assert_run_id_unused.py", "--run-id", RUN_ID])
        if stale.returncode == 0:
            raise SystemExit("supervisor_startup_mutex_did_not_block_without_allow_flag")
        payload["hf_export"] = True
        STARTUP_MUTEX_LOCK.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        rejected = run(
            [
                str(PYTHON),
                "scripts/assert_run_id_unused.py",
                "--run-id",
                RUN_ID,
                "--allow-supervisor-launch-artifacts",
            ]
        )
        if rejected.returncode == 0:
            raise SystemExit("unsafe_supervisor_startup_mutex_was_allowed")
    finally:
        helper.terminate()
        try:
            helper.wait(timeout=10)
        except subprocess.TimeoutExpired:
            helper.kill()
        cleanup_test_artifact(STARTUP_MUTEX_LOCK)


def main() -> None:
    for artifact in STALE_ARTIFACTS:
        cleanup_test_artifact(artifact)
    cleanup_test_artifact(PAYLOAD_ONLY_ACTIVE_LOCK)
    cleanup_test_artifact(CANONICAL_ACTIVE_LOCK)
    cleanup_test_artifact(STARTUP_MUTEX_LOCK)
    cleanup_test_artifact(TEST_PREFLIGHT)
    cleanup_test_artifact(TEST_REVIEW)

    clean = run([str(PYTHON), "scripts/assert_run_id_unused.py", "--run-id", RUN_ID])
    if clean.returncode != 0:
        raise SystemExit(f"clean_run_id_was_rejected stdout={clean.stdout} stderr={clean.stderr}")

    for artifact in STALE_ARTIFACTS:
        assert_rejected_for(artifact)
    assert_rejected_for_payload(
        PAYLOAD_ONLY_ACTIVE_LOCK,
        '{"run_id":"' + RUN_ID + '","state":"stale","hf_export":false}\n',
    )
    assert_allowed_supervisor_context_only()
    assert_allowed_current_supervisor_active_lock()
    assert_allowed_current_supervisor_startup_mutex()

    print("run_id_unused_guard_ok=true")


if __name__ == "__main__":
    main()
