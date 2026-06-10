from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path


SCRIPT = Path("scripts/start_old_japanese_0_1b_dml_and_watch.ps1")


def main() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    required_fragments = [
        "function Get-DmlRunIdFromCommandLine",
        "function Get-ActiveDmlRunProcesses",
        "function Get-AncestorProcessIds",
        "function Assert-NoOtherDmlRunProcess",
        "AllowLauncherWithoutRunId",
        "train_started_watcher_pending",
        "TotalMinutes -lt 30",
        "--run-id",
        "-RunId",
        "$RunIdInCommand",
        "$IgnorePids",
        "kobun_llm.train",
        "Test-DmlTrainingCommandLine",
        "Get-AnyActiveDmlTrainingProcesses",
        "(cpu|cuda|hip)",
        "train_started_watcher_pending",
        "Assert-NoOtherDmlRunProcess -RequestedRunId $RunId",
        "($ExistingRunProcesses.Count -gt 0)",
        "$IgnorePids -notcontains [int]$_.ProcessId",
    ]
    missing = [fragment for fragment in required_fragments if fragment not in text]
    if missing:
        raise SystemExit(f"active DML process scan hardening missing fragments: {missing}")

    command = (
        "$Tokens = $null; $Errors = $null; "
        "$null = [System.Management.Automation.Language.Parser]::ParseFile("
        "'scripts/start_old_japanese_0_1b_dml_and_watch.ps1', [ref]$Tokens, [ref]$Errors); "
        "if ($Errors.Count -gt 0) { $Errors | ForEach-Object { $_.ToString() }; exit 1 }; "
        "'parser_ok=true'"
    )
    result = subprocess.run(
        ["powershell", "-NoProfile", "-Command", command],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise SystemExit(result.stdout + result.stderr)

    intel_text = Path("scripts/old_japanese_run_intel.py").read_text(encoding="utf-8")
    intel_required = [
        "def is_dml_training_command",
        "kobun_llm.train",
        "--device(?:\\s+|=)(cpu|cuda|hip)",
        "relative script paths",
        "absolute root is not present",
    ]
    intel_missing = [fragment for fragment in intel_required if fragment not in intel_text]
    if intel_missing:
        raise SystemExit(f"old_japanese_run_intel broad DML scan missing fragments: {intel_missing}")

    # This is a structural test only: it must not start or stop training.
    with tempfile.TemporaryDirectory() as tmp:
        marker = Path(tmp) / "no_runtime_side_effects"
        marker.write_text("ok", encoding="utf-8")
        if marker.read_text(encoding="utf-8") != "ok":
            raise SystemExit("unexpected temp-file side effect")
    print("active_dml_process_scan_hardening_ok=true")


if __name__ == "__main__":
    main()
