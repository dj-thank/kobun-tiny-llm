from __future__ import annotations

import json
import tempfile
from pathlib import Path

from kobun_autonomy.non_release_registry import NonReleaseRecordError, is_non_release_recorded
from old_japanese_run_intel import classify_run, select_next_action


def main() -> None:
    run_id = "old_japanese_0_1b_dml_20990101_000000"
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        logs = root / "logs"
        logs.mkdir()
        checkpoints = root / "checkpoints"
        checkpoints.mkdir()
        (logs / f"{run_id}.out.log").write_text(
            "\n".join(
                [
                    "run_id=old_japanese_0_1b_dml_20990101_000000 data_sha256=aaa val_sha256=bbb test_sha256=ccc",
                    "config=vocab=100 block=16 layers=1 heads=1 kv_heads=1 embd=8 norm=rmsnorm mlp=swiglu rope=True qk_norm=True tied=True amp=False params=123456789 grad_accum=1",
                    "step=0 train_loss=10.0 val_loss=10.0",
                    "step=250 train_loss=0.1 val_loss=6.0",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        (logs / f"train_exit_{run_id}.json").write_text(
            json.dumps(
                {
                    "run_id": run_id,
                    "exit_code": -1,
                    "message": "training command failed with exit code -1",
                    "completed_at": "2099-01-01T00:00:00+09:00",
                    "checkpoint": f"checkpoints/{run_id}.pt",
                    "best_checkpoint": f"checkpoints/{run_id}_best.pt",
                    "hf_export": False,
                },
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        marker_dir = logs / "non_release_runs"
        marker_dir.mkdir()
        (marker_dir / f"{run_id}.json").write_text(
            json.dumps(
                {
                    "run_id": run_id,
                    "release_status": "non_release_artifact",
                    "reason": "autonomous_overfit_stop",
                    "created_at": "2099-01-01T00:00:10+09:00",
                    "hf_export": False,
                },
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        row = classify_run(root, run_id)
        if row["release_status"] != "non_release_artifact":
            raise SystemExit(f"release_status_not_non_release={row['release_status']}")
        if row["next_action"] != "ignore_for_release":
            raise SystemExit(f"next_action_not_ignore={row['next_action']}")
        board = {"runs": [row], "global_blockers": []}
        action = select_next_action(board)
        if action["action"] == "fix_blockers":
            raise SystemExit(f"non_release_marker_should_not_wedge_loop={action}")
        corrupt_id = "old_japanese_0_1b_dml_20990101_000001"
        corrupt_path = marker_dir / f"{corrupt_id}.json"
        corrupt_path.write_text("{not json", encoding="utf-8")
        try:
            is_non_release_recorded(corrupt_id, root)
        except NonReleaseRecordError:
            pass
        else:
            raise SystemExit("corrupt_non_release_record_did_not_fail_closed")

        mismatch_id = "old_japanese_0_1b_dml_20990101_000002"
        (marker_dir / f"{mismatch_id}.json").write_text(
            json.dumps(
                {
                    "run_id": "old_japanese_0_1b_dml_20990101_999999",
                    "release_status": "non_release_artifact",
                    "hf_export": False,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        try:
            is_non_release_recorded(mismatch_id, root)
        except NonReleaseRecordError:
            pass
        else:
            raise SystemExit("mismatched_non_release_record_did_not_fail_closed")

        bad_status_id = "old_japanese_0_1b_dml_20990101_000003"
        (marker_dir / f"{bad_status_id}.json").write_text(
            json.dumps(
                {
                    "run_id": bad_status_id,
                    "release_status": "maybe_release",
                    "hf_export": False,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        try:
            is_non_release_recorded(bad_status_id, root)
        except NonReleaseRecordError:
            pass
        else:
            raise SystemExit("bad_status_non_release_record_did_not_fail_closed")
    print("non_release_registry_ok=true")


if __name__ == "__main__":
    main()
