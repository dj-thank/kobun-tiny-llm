from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from old_japanese_run_intel import active_lock_health, classify_run, discover_run_ids


ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "logs" / "active_old_japanese_0_1b_cuda.lock"
RUN_ID = "old_japanese_0_1b_cuda_board_lock_test"


def cleanup() -> None:
    LOCK.unlink(missing_ok=True)
    for path in (ROOT / "logs").glob("active_old_japanese_0_1b_cuda.stale.*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
        except Exception:
            if "board_lock_test" in path.name:
                path.unlink(missing_ok=True)
            continue
        if str(payload.get("run_id") or "") == RUN_ID:
            path.unlink(missing_ok=True)


def write_lock(payload: dict[str, object]) -> None:
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    LOCK.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    if LOCK.exists():
        raise SystemExit("refusing to run CUDA active lock test while a real CUDA active lock exists")
    try:
        write_lock(
            {
                "run_id": RUN_ID,
                "backend": "cuda",
                "state": "launching",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "launcher_pid": os.getpid(),
                "train_pid": None,
                "watcher_pid": None,
                "hf_export": False,
                "launch_token_sha256": "0" * 64,
                "launch_nonce_sha256": "1" * 64,
                "preflight_gate": "logs/preflight_gate_old_japanese_0_1b.json",
                "preflight_gate_sha256": "2" * 64,
                "review_gate": "logs/zero_base_review_gate_old_japanese_0_1b.json",
                "review_gate_sha256": "3" * 64,
                "autonomous_launch_context": f"logs/autonomous_launch_context_{RUN_ID}.json",
                "autonomous_launch_context_sha256": "4" * 64,
                "autonomous_script": "scripts/start_old_japanese_0_1b_cuda_colab_and_watch.py",
                "selected_action": "colab_cuda_supervised_training",
            }
        )
        health = active_lock_health(ROOT)
        if health.get("backend") != "cuda" or health.get("run_id") != RUN_ID:
            raise SystemExit(f"cuda active lock health did not report lock: {health}")
        ids = discover_run_ids(ROOT)
        if RUN_ID not in ids:
            raise SystemExit(f"cuda active lock run id not discovered: {ids}")
        row = classify_run(ROOT, RUN_ID)
        if row.get("status") != "running":
            raise SystemExit(f"cuda active lock was not classified as running: {row}")
        if row.get("next_action") != "monitor":
            raise SystemExit(f"cuda active lock next_action should be monitor: {row}")
        LOCK.unlink()

        LOCK.write_text("{not json", encoding="utf-8")
        corrupt = active_lock_health(ROOT)
        if corrupt.get("exists") is not False or corrupt.get("invalid_json") is not True:
            raise SystemExit(f"stale corrupt CUDA lock was not quarantined safely: {corrupt}")
        quarantine = str(corrupt.get("quarantined_path") or "")
        if "active_old_japanese_0_1b_cuda.stale.invalid." not in quarantine:
            raise SystemExit(f"stale corrupt CUDA lock used wrong quarantine path: {corrupt}")
        if quarantine:
            (ROOT / "logs" / quarantine).unlink(missing_ok=True)
        if LOCK.exists():
            raise SystemExit("corrupt CUDA active lock still exists after quarantine")

        write_lock(
            {
                "run_id": RUN_ID,
                "backend": "cuda",
                "state": "launching",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "launcher_pid": os.getpid(),
                "train_pid": None,
                "watcher_pid": None,
                "hf_export": True,
                "launch_token_sha256": "0" * 64,
                "launch_nonce_sha256": "1" * 64,
                "preflight_gate": "logs/preflight_gate_old_japanese_0_1b.json",
                "preflight_gate_sha256": "2" * 64,
                "review_gate": "logs/zero_base_review_gate_old_japanese_0_1b.json",
                "review_gate_sha256": "3" * 64,
                "autonomous_launch_context": f"logs/autonomous_launch_context_{RUN_ID}.json",
                "autonomous_launch_context_sha256": "4" * 64,
                "autonomous_script": "scripts/start_old_japanese_0_1b_cuda_colab_and_watch.py",
                "selected_action": "colab_cuda_supervised_training",
            }
        )
        unsafe = active_lock_health(ROOT)
        if "invalid_active_lock_schema_with_live_cuda_or_supervised_process" not in unsafe.get("hard_blockers", []):
            raise SystemExit(f"CUDA active lock with hf_export=true was not a hard blocker: {unsafe}")
    finally:
        cleanup()
    print("cuda_active_lock_board_governance_ok=true")


if __name__ == "__main__":
    main()
