from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUPERVISOR = ROOT / "scripts" / "start_old_japanese_0_1b_dml_and_watch.ps1"
INTEL = ROOT / "scripts" / "old_japanese_run_intel.py"


def require(text: str, needle: str, label: str) -> None:
    if needle not in text:
        raise SystemExit(f"{label}_missing={needle!r}")


def function_body(text: str, name: str) -> str:
    pattern = rf"function\s+{re.escape(name)}\s*\{{(?P<body>.*?)(?=^function\s+|\Z)"
    match = re.search(pattern, text, flags=re.DOTALL | re.MULTILINE)
    if not match:
        raise SystemExit(f"missing_function={name}")
    return match.group("body")


def main() -> None:
    supervisor_text = SUPERVISOR.read_text(encoding="utf-8")
    intel_text = INTEL.read_text(encoding="utf-8")
    require(supervisor_text, 'backend = "dml"', "dml_active_lock_backend")
    require(supervisor_text, "backend_not_dml", "dml_active_lock_schema")
    require(supervisor_text, "hf_export_not_false", "dml_active_lock_schema")
    for needle in (
        "missing_launch_token_sha256",
        "missing_launch_nonce_sha256",
        "missing_preflight_gate",
        "missing_review_gate",
        "missing_autonomous_launch_context",
        "missing_autonomous_script",
        "missing_selected_action",
    ):
        require(supervisor_text, needle, "dml_active_lock_schema")

    invalid_lock_body = function_body(supervisor_text, "Move-InvalidActiveLockOrThrow")
    require(invalid_lock_body, "Get-AnySupervisedOldJapaneseProcesses", "invalid_lock_guard")
    require(invalid_lock_body, "live supervised old_japanese_0_1b process exists", "invalid_lock_guard")
    if "Get-AnyActiveDmlTrainingProcesses" in invalid_lock_body:
        raise SystemExit("invalid_lock_guard_still_uses_narrow_dml_training_scan")

    require(intel_text, "def active_lock_schema_errors(payload: dict[str, Any], expected_backend: str = \"\")", "intel_schema")
    require(intel_text, "backend_not_{expected_backend}", "intel_schema")
    require(intel_text, "active_lock_schema_errors(payload, expected_backend=backend)", "intel_schema")
    require(intel_text, "missing_launch_token_sha256", "intel_schema")
    require(intel_text, "missing_launch_nonce_sha256", "intel_schema")
    require(intel_text, "missing_autonomous_launch_context", "intel_schema")
    print("dml_invalid_lock_supervision_contract_ok=true")


if __name__ == "__main__":
    main()
