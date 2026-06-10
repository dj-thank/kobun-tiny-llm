from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from old_japanese_run_intel import startup_mutex_health


ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "logs" / "active_old_japanese_0_1b_training.lock"


def cleanup() -> None:
    LOCK.unlink(missing_ok=True)
    for path in (ROOT / "logs").glob("active_old_japanese_0_1b_training.stale.*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
        except Exception:
            continue
        if str(payload.get("run_id") or "") == "old_japanese_0_1b_dml_startup_mutex_health_test":
            path.unlink(missing_ok=True)


def write_lock(payload: dict[str, object]) -> None:
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    LOCK.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    if LOCK.exists():
        raise SystemExit("refusing to run startup mutex health test while a real startup mutex exists")
    try:
        write_lock(
            {
                "run_id": "old_japanese_0_1b_dml_startup_mutex_health_test",
                "backend": "dml",
                "state": "startup_mutex",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "launcher_pid": os.getpid(),
                "hf_export": False,
                "launch_token_sha256": "0" * 64,
                "launch_nonce_sha256": "4" * 64,
                "preflight_gate": "logs/preflight_gate_old_japanese_0_1b.json",
                "preflight_gate_sha256": "1" * 64,
                "review_gate": "logs/zero_base_review_gate_old_japanese_0_1b.json",
                "review_gate_sha256": "2" * 64,
                "autonomous_launch_context": "logs/autonomous_launch_context_old_japanese_0_1b_dml_startup_mutex_health_test.json",
                "autonomous_launch_context_sha256": "3" * 64,
            }
        )
        live = startup_mutex_health(ROOT)
        if "startup_mutex_live" not in (live.get("hard_blockers") or []):
            raise SystemExit(f"startup mutex live blocker was not reported: {live}")
        LOCK.unlink()

        write_lock(
            {
                "run_id": "old_japanese_0_1b_dml_startup_mutex_health_test",
                "backend": "dml",
                "state": "startup_mutex",
                "created_at": "2000-01-01T00:00:00+00:00",
                "launcher_pid": 99999999,
                "hf_export": False,
                "launch_token_sha256": "0" * 64,
            }
        )
        stale = startup_mutex_health(ROOT)
        if stale.get("exists") is not False or not str(stale.get("quarantined_path") or "").startswith("active_old_japanese_0_1b_training.stale."):
            raise SystemExit(f"stale startup mutex was not quarantined: {stale}")
        if LOCK.exists():
            raise SystemExit("stale startup mutex lock still exists after quarantine")
    finally:
        cleanup()
    print("startup_mutex_health_ok=true")


if __name__ == "__main__":
    main()
