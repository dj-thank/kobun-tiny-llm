from __future__ import annotations

import importlib.util
import json
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STARTER = ROOT / "scripts" / "start_old_japanese_0_1b_cuda_colab_and_watch.py"


def load_starter():
    spec = importlib.util.spec_from_file_location("cuda_colab_starter_contract", STARTER)
    if spec is None or spec.loader is None:
        raise SystemExit("could not import Colab CUDA starter")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    starter = load_starter()
    same_run = "old_japanese_0_1b_cuda_20990101_000000"
    command = f"python -m kobun_llm.train --run-id {same_run} --device cuda"
    if not starter.supervised_wrapper_command(command):
        raise SystemExit("cuda_same_run_train_command_not_detected")
    command = f"python -m kobun_llm.train --run-id {same_run}"
    if not starter.supervised_wrapper_command(command):
        raise SystemExit("cuda_auto_default_train_command_not_detected")
    command = f"python -m kobun_llm.train --run-id {same_run} --device auto"
    if not starter.supervised_wrapper_command(command):
        raise SystemExit("cuda_auto_device_train_command_not_detected")
    command = f"python -m kobun_llm.train --run-id {same_run} --device dml"
    if starter.supervised_wrapper_command(command):
        raise SystemExit("dml_train_command_should_not_be_cuda_conflict")

    with tempfile.TemporaryDirectory() as tmp:
        project_root = Path(tmp)
        logs = project_root / "logs"
        logs.mkdir()
        run_id = "old_japanese_0_1b_cuda_20990101_000001"
        token_sha = starter.sha256_text("token")
        active_lock = logs / "active_old_japanese_0_1b_cuda.lock"
        startup_lock = logs / "active_old_japanese_0_1b_training.lock"
        starter.acquire_lock(
            startup_lock,
            {
                "run_id": run_id,
                "backend": "cuda",
                "launcher_pid": 1234,
                "launch_token_sha256": token_sha,
                "state": "startup_mutex",
                "hf_export": False,
            },
        )
        removed = starter.remove_owned_lock(startup_lock, run_id, 1234, token_sha)
        if not removed or startup_lock.exists():
            raise SystemExit("cuda_startup_lock_owned_remove_failed")
        active_lock.write_text(
            json.dumps(
                {
                    "run_id": run_id,
                    "launcher_pid": 1234,
                    "launch_token_sha256": token_sha,
                    "state": "running",
                    "hf_export": False,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        sentinel = logs / f"train_exit_{run_id}.json"
        sentinel.write_text(
            json.dumps(
                {
                    "run_id": run_id,
                    "exit_code": 1,
                    "message": "intentional failure contract test",
                    "hf_export": False,
                },
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        archive = starter.archive_owned_lock(
            active_lock,
            run_id,
            1234,
            token_sha,
            state="failed_non_release",
            reason="contract_failure",
        )
        if archive is None or not archive.exists():
            raise SystemExit("cuda_failed_active_lock_not_archived")
        payload = json.loads(archive.read_text(encoding="utf-8-sig"))
        if payload.get("state") != "failed_non_release" or payload.get("hf_export") is not False:
            raise SystemExit("cuda_failed_archive_payload_not_fail_closed")
        starter.write_non_release_record(
            project_root,
            run_id=run_id,
            reason="contract_failure",
            train_exit=sentinel,
            active_lock_archive=archive,
        )
        record = project_root / "logs" / "non_release_runs" / f"{run_id}.json"
        if not record.exists():
            raise SystemExit("cuda_failed_non_release_record_missing")
        record_payload = json.loads(record.read_text(encoding="utf-8-sig"))
        if record_payload.get("release_status") != "non_release_artifact":
            raise SystemExit("cuda_failed_non_release_record_status_bad")
        if record_payload.get("hf_export") is not False:
            raise SystemExit("cuda_failed_non_release_record_hf_export_not_false")
        if not record_payload.get("source_archive_path") or not record_payload.get("source_archive_sha256"):
            raise SystemExit("cuda_failed_non_release_record_archive_binding_missing")
    print("colab_cuda_failure_contracts_ok=true")


if __name__ == "__main__":
    main()
