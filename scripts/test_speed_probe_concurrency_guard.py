from __future__ import annotations

import tempfile
from pathlib import Path

import probe_dml_training_speed as probe


def main() -> None:
    live_command = (
        r"C:\Python\python.exe -m kobun_llm.train "
        r"--run-id old_japanese_0_1b_dml_20990101_000000 --device dml"
    )
    watcher_command = (
        r"powershell -File scripts\watch_and_finalize_old_japanese_0_1b_dml.ps1 "
        r"-RunId old_japanese_0_1b_dml_20990101_000000"
    )
    cpu_command = (
        r"C:\Python\python.exe -m kobun_llm.train "
        r"--run-id old_japanese_0_1b_dml_20990101_000000 --device cpu"
    )
    unrelated_command = r"C:\Python\python.exe scripts\probe_dml_training_speed.py"

    if not probe._command_is_live_dml_training(live_command):
        raise SystemExit("DML train command was not detected")
    if not probe._command_is_live_dml_training(watcher_command):
        raise SystemExit("DML watcher command was not detected")
    if probe._command_is_live_dml_training(cpu_command):
        raise SystemExit("CPU training command was incorrectly detected as DML")
    if probe._command_is_live_dml_training(unrelated_command):
        raise SystemExit("probe command was incorrectly detected as live training")

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        old_cwd = Path.cwd()
        try:
            import os

            os.chdir(root)
            probe.assert_no_active_dml_training(allow_active_run=False)
            (root / "logs").mkdir()
            (root / "logs" / "active_old_japanese_0_1b_dml.lock").write_text("{}", encoding="utf-8")
            try:
                probe.assert_no_active_dml_training(allow_active_run=False)
            except SystemExit as exc:
                if "active DirectML lock exists" not in str(exc):
                    raise SystemExit(f"wrong active-lock failure: {exc}") from exc
            else:
                raise SystemExit("active lock did not fail speed probe guard")
        finally:
            os.chdir(old_cwd)

    print("speed_probe_concurrency_guard_ok=true")


if __name__ == "__main__":
    main()
