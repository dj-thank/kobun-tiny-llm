from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_static_quality_checks.ps1"


MUTATING_COMMANDS = (
    "scripts\\build_tokenizer_public_vocab.py",
    "scripts\\update_evaluation_board.py",
    "scripts\\build_llm_review_packet.py",
)
CHECK_ONLY_COMMANDS = {
    "scripts\\audit_public_manifest.py": "--check-only",
    "scripts\\score_source_quality.py": "--check-only",
    "scripts\\build_training_augmentation_manifest.py": "--audit-only",
}


def require(text: str, needle: str) -> None:
    if needle not in text:
        raise SystemExit(f"static_quality_contract_missing={needle!r}")


def main() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    require(text, "param(")
    require(text, "[switch]$RefreshEvidence")
    require(text, "function Invoke-RefreshChecked")
    require(text, "static_quality_check_only_ok=true")
    require(text, "if (-not $RefreshEvidence.IsPresent)")
    require(text, "scripts\\write_static_quality_manifest.py --log $StaticLog --status passed --exit-code 0 --command")
    require(text, "-RefreshEvidence")
    require(text, "Invoke-PostPreflightChecked")
    for command in MUTATING_COMMANDS:
        expected = f"Invoke-RefreshChecked $Python {command}"
        if expected not in text:
            raise SystemExit(f"mutating_static_command_not_refresh_only={command}")
        forbidden = f"Invoke-Checked $Python {command}"
        if forbidden in text:
            raise SystemExit(f"mutating_static_command_runs_in_checkonly={command}")
    for command, flag in CHECK_ONLY_COMMANDS.items():
        if f"Invoke-Checked $Python {command} {flag}" not in text:
            raise SystemExit(f"static_checkonly_command_missing_flag={command} {flag}")
    print("static_quality_checkonly_contract_ok=true")


if __name__ == "__main__":
    main()
