from __future__ import annotations

import json
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

from old_japanese_run_intel import colab_cuda_lease_health


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        logs = root / "logs"
        logs.mkdir()
        run_id = "old_japanese_0_1b_cuda_20990101_000000"
        lease = logs / f"colab_active_old_japanese_0_1b_cuda.{run_id}.json"
        payload = {
            "schema": "old_japanese_0_1b_colab_cuda_active_lease_v1",
            "run_id": run_id,
            "backend": "cuda",
            "state": "running",
            "lease_expires_at_utc": (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
            "artifact_root": "/content/drive/MyDrive/kobun-tiny-llm",
            "train_pid": 123,
            "hf_export": False,
            "package_created": False,
            "upload_attempted": False,
        }
        lease.write_text(json.dumps(payload), encoding="utf-8")
        active = colab_cuda_lease_health(root)
        if "colab_cuda_lease_active" not in active.get("hard_blockers", []):
            raise SystemExit(f"active_colab_lease_not_blocking={active}")
        payload["state"] = "finished"
        lease.write_text(json.dumps(payload), encoding="utf-8")
        expired = colab_cuda_lease_health(root)
        if expired.get("hard_blockers") or lease.exists() or not expired.get("quarantined"):
            raise SystemExit(f"finished_colab_lease_not_quarantined={expired}")
    print("colab_cuda_lease_health_ok=true")


if __name__ == "__main__":
    main()
