from __future__ import annotations

import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

from old_japanese_run_intel import active_lock_health


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        logs = root / "logs"
        logs.mkdir()
        lock = logs / "active_old_japanese_0_1b_dml.lock"
        lock.write_text("{not-json", encoding="utf-8")
        health = active_lock_health(root)
        if health.get("hard_blockers"):
            raise SystemExit(f"unexpected_hard_blockers={health.get('hard_blockers')}")
        if not health.get("invalid_json") or not health.get("quarantined_path"):
            raise SystemExit(f"invalid_lock_was_not_quarantined={health}")
        if lock.exists():
            raise SystemExit("invalid active lock was not removed after quarantine")
        quarantined = logs / str(health["quarantined_path"])
        if not quarantined.exists():
            raise SystemExit(f"missing quarantined invalid lock: {quarantined}")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        logs = root / "logs"
        logs.mkdir()
        lock = logs / "active_old_japanese_0_1b_dml.lock"
        lock.write_text(
            (
                '{"run_id":"old_japanese_0_1b_dml_20990101_000000",'
                '"backend":"dml","state":"running","created_at":"2026-05-11T00:00:00+09:00",'
                '"hf_export":false,'
                '"launch_token_sha256":"' + ("0" * 64) + '",'
                '"launch_nonce_sha256":"' + ("1" * 64) + '",'
                '"preflight_gate":"logs/preflight_gate_old_japanese_0_1b.json",'
                '"preflight_gate_sha256":"' + ("2" * 64) + '",'
                '"review_gate":"logs/zero_base_review_gate_old_japanese_0_1b.json",'
                '"review_gate_sha256":"' + ("3" * 64) + '",'
                '"autonomous_launch_context":"logs/autonomous_launch_context_old_japanese_0_1b_dml_20990101_000000.json",'
                '"autonomous_launch_context_sha256":"' + ("4" * 64) + '",'
                '"autonomous_script":"scripts/autonomous_old_japanese_0_1b_loop.ps1",'
                '"selected_action":"prepare_next_fresh_run_after_static_gate_and_zero_base_reviews",'
                '"launcher_pid":"not-a-pid","train_pid":null,"watcher_pid":null}'
            ),
            encoding="utf-8",
        )
        health = active_lock_health(root)
        if health.get("hard_blockers"):
            raise SystemExit(f"unexpected_schema_hard_blockers={health.get('hard_blockers')}")
        if not health.get("invalid_schema") or not health.get("quarantined_path"):
            raise SystemExit(f"invalid_schema_lock_was_not_quarantined={health}")
        if lock.exists():
            raise SystemExit("invalid schema active lock was not removed after quarantine")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        logs = root / "logs"
        logs.mkdir()
        lock = logs / "active_old_japanese_0_1b_cuda.lock"
        lock.write_text(
            (
                '{"run_id":"old_japanese_0_1b_cuda_20990101_000000",'
                '"backend":"cuda","state":"running","created_at":"'
                + (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
                + '",'
                '"hf_export":false,'
                '"launch_token_sha256":"' + ("0" * 64) + '",'
                '"launch_nonce_sha256":"' + ("1" * 64) + '",'
                '"preflight_gate":"logs/preflight_gate_old_japanese_0_1b.json",'
                '"preflight_gate_sha256":"' + ("2" * 64) + '",'
                '"review_gate":"logs/zero_base_review_gate_old_japanese_0_1b.json",'
                '"review_gate_sha256":"' + ("3" * 64) + '",'
                '"autonomous_launch_context":"logs/colab_cuda_launch_context_old_japanese_0_1b_cuda_20990101_000000.json",'
                '"autonomous_launch_context_sha256":"' + ("4" * 64) + '",'
                '"autonomous_script":"scripts/start_old_japanese_0_1b_cuda_colab_and_watch.py",'
                '"selected_action":"colab_cuda_supervised_training",'
                '"launcher_pid":99999999,"train_pid":99999998,"watcher_pid":99999997}'
            ),
            encoding="utf-8",
        )
        health = active_lock_health(root)
        if health.get("hard_blockers"):
            raise SystemExit(f"unexpected_valid_stale_lock_hard_blockers={health.get('hard_blockers')}")
        if health.get("exists") is not False or not health.get("quarantined_path"):
            raise SystemExit(f"valid stale active lock was not quarantined={health}")
        if lock.exists():
            raise SystemExit("valid stale active lock still exists after quarantine")
    print("active_lock_corruption_policy_ok=true")


if __name__ == "__main__":
    main()
