from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "report_old_japanese_0_1b_progress.ps1"


def require(text: str, needle: str) -> None:
    if needle not in text:
        raise SystemExit(f"progress_reporter_pid_discovery_missing={needle}")


def main() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    require(text, "function Resolve-TrainingProcessId")
    require(text, "Get-CimInstance Win32_Process")
    require(text, "kobun_llm\\.train")
    require(text, "[regex]::Escape($TargetRunId)")
    require(text, "$ProcessId = $ResolvedProcessId")
    require(text, "function Test-ProcessMatchesRun")
    require(text, "Explicit ProcessId $ProcessId does not match live python kobun_llm.train command")
    require(text, "unknown_unverified_no_process_or_sentinel")
    print("progress_reporter_pid_discovery_static_ok=true")


if __name__ == "__main__":
    main()
