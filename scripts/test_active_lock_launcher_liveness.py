from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STARTER = ROOT / "scripts" / "start_old_japanese_0_1b_dml_and_watch.ps1"
INTEL = ROOT / "scripts" / "old_japanese_run_intel.py"
AUTO = ROOT / "scripts" / "autonomous_old_japanese_0_1b_loop.ps1"
WATCHER = ROOT / "scripts" / "watch_and_finalize_old_japanese_0_1b_dml.ps1"


def require(path: Path, needle: str) -> None:
    text = path.read_text(encoding="utf-8")
    if needle not in text:
        raise SystemExit(f"active_lock_launcher_liveness_missing={path.name}:{needle}")


def main() -> None:
    require(STARTER, "AllowLauncherWithoutRunId")
    require(STARTER, "train_started_watcher_pending")
    require(STARTER, "TotalMinutes -lt 30")
    require(STARTER, "Get-AncestorProcessIds")
    require(STARTER, "[IO.FileMode]::CreateNew, [IO.FileAccess]::Write, [IO.FileShare]::None")
    require(STARTER, "[IO.FileMode]::Open, [IO.FileAccess]::ReadWrite, [IO.FileShare]::None")
    require(STARTER, "Refusing to update active lock not owned by this launcher")
    require(STARTER, "stale.invalid")
    require(STARTER, "$Reason and live supervised old_japanese_0_1b process exists")
    require(STARTER, "Get-ActiveLockSchemaErrors")
    require(STARTER, "invalid schema")
    require(STARTER, "Get-AnySupervisedOldJapaneseProcesses")
    require(STARTER, "Test-SupervisedOldJapaneseCommandLine")
    require(STARTER, "start_old_japanese_0_1b_cuda_colab_and_watch.py")
    require(STARTER, "another supervised old_japanese_0_1b process is live")
    require(INTEL, "process_matches_launcher")
    require(INTEL, "recent_active_lock")
    require(INTEL, "train_started_watcher_pending")
    require(INTEL, "active_lock_health")
    require(INTEL, "startup_mutex_health")
    require(INTEL, "startup_mutex_live")
    require(INTEL, "active_old_japanese_0_1b_training.lock")
    require(INTEL, "invalid_startup_mutex_schema_with_live_process")
    require(INTEL, "live_supervised_process_summaries")
    require(INTEL, "invalid_active_lock_json_with_live_{backend}_or_supervised_process")
    require(INTEL, "invalid_active_lock_schema_with_live_{backend}_or_supervised_process")
    require(INTEL, "active_old_japanese_0_1b_cuda.lock")
    require(INTEL, "active_old_japanese_0_1b_{backend}.stale")
    require(INTEL, "process_exists(launcher_pid)")
    require(INTEL, "read_non_release_record")
    require(AUTO, "function New-DmlRunId")
    require(AUTO, "-RunId $RunId")
    require(AUTO, "Test-ExactDmlTrainingProcess")
    require(AUTO, "--device(?:\\s+|=)dml")
    require(AUTO, "failed active-lock archive")
    require(AUTO, "Write-NonReleaseRunRecord")
    require(AUTO, "Repair-DmlLaunchIfTrainingLive")
    require(AUTO, "Set-ActiveLockRunning")
    require(AUTO, "launch_repaired=1")
    require(WATCHER, "active_lock_complete_failed")
    require(WATCHER, "throw $OriginalError")
    print("active_lock_launcher_liveness_ok=true")


if __name__ == "__main__":
    main()
