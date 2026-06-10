from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"


def run_checked(args: list[str]) -> str:
    result = subprocess.run(args, cwd=ROOT, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        raise SystemExit(
            "command_failed\n"
            f"args={args}\n"
            f"stdout={result.stdout}\n"
            f"stderr={result.stderr}"
        )
    return result.stdout


def main() -> None:
    preflight = ROOT / "logs" / "preflight_gate_old_japanese_0_1b.json"
    if not preflight.exists():
        raise SystemExit("missing_preflight_gate_for_review_gate_test")
    out_rel = "logs/zero_base_review_gate_test.json"
    out = ROOT / out_rel
    artifact_dir = ROOT / "logs" / "zero_base_review_artifacts"
    artifacts = {
        "safety": "logs/zero_base_review_artifacts/test_safety_review.json",
        "data": "logs/zero_base_review_artifacts/test_data_review.json",
        "backend": "logs/zero_base_review_artifacts/test_backend_review.json",
    }
    try:
        run_checked(
            [
                str(PYTHON),
                "scripts/write_zero_base_review_artifact.py",
                "--scope",
                "safety/release",
                "--agent-id",
                "019e17e9-a4bb-7b71-933f-a9abab3bc48b",
                "--decision",
                "no_blockers",
                "--blocking-findings-count",
                "0",
                "--summary",
                "No blocking findings in test safety scope.",
                "--prompt",
                "zero-base safety/release test prompt",
                "--review-model",
                "external-reviewer",
                "--review-reasoning-effort",
                "llm_witness",
                "--preflight-gate",
                preflight.relative_to(ROOT).as_posix(),
                "--out",
                artifacts["safety"],
            ]
        )
        run_checked(
            [
                str(PYTHON),
                "scripts/write_zero_base_review_artifact.py",
                "--scope",
                "data/eval",
                "--agent-id",
                "019e17e9-a55a-7d51-92bc-f62335888770",
                "--decision",
                "no_blockers",
                "--blocking-findings-count",
                "0",
                "--summary",
                "No blocking findings in test data scope.",
                "--prompt",
                "zero-base data/eval test prompt",
                "--review-model",
                "external-reviewer",
                "--review-reasoning-effort",
                "llm_witness",
                "--preflight-gate",
                preflight.relative_to(ROOT).as_posix(),
                "--out",
                artifacts["data"],
            ]
        )
        run_checked(
            [
                str(PYTHON),
                "scripts/write_zero_base_review_artifact.py",
                "--scope",
                "backend/runtime",
                "--agent-id",
                "019e17e9-a628-7873-b38c-ff1bf0bfd046",
                "--decision",
                "no_blockers",
                "--blocking-findings-count",
                "0",
                "--summary",
                "No blocking findings in test backend scope.",
                "--prompt",
                "zero-base backend/runtime test prompt",
                "--review-model",
                "external-reviewer",
                "--review-reasoning-effort",
                "llm_witness",
                "--preflight-gate",
                preflight.relative_to(ROOT).as_posix(),
                "--out",
                artifacts["backend"],
            ]
        )
        run_checked(
            [
                str(PYTHON),
                "scripts/write_zero_base_review_gate.py",
                "--out",
                out_rel,
                "--preflight-gate",
                preflight.relative_to(ROOT).as_posix(),
                "--safety-artifact",
                artifacts["safety"],
                "--data-artifact",
                artifacts["data"],
                "--backend-artifact",
                artifacts["backend"],
            ]
        )
        output = run_checked(
            [
                str(PYTHON),
                "scripts/verify_zero_base_review_gate.py",
                "--gate",
                out_rel,
                "--preflight-gate",
                preflight.relative_to(ROOT).as_posix(),
                "--max-age-minutes",
                "120",
            ]
        )
        if "zero_base_review_gate_ok=true" not in output:
            raise SystemExit(f"unexpected_verify_output={output}")
    finally:
        out.unlink(missing_ok=True)
        for rel in artifacts.values():
            (ROOT / rel).unlink(missing_ok=True)
        if artifact_dir.exists():
            try:
                artifact_dir.rmdir()
            except OSError:
                pass
    print("zero_base_review_gate_test_ok=true")


if __name__ == "__main__":
    main()
