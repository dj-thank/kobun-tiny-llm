from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_SCHEMA = "old_japanese_0_1b_preflight_gate_v1"
REQUIRED_TOKENIZER_POLICY = "train_split_plus_core_japanese_inventory_plus_utf8_byte_fallback_v1"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify a fresh static-check preflight gate before DML launch.")
    parser.add_argument("--gate", default="logs/preflight_gate_old_japanese_0_1b.json")
    parser.add_argument("--max-age-minutes", type=float, default=120.0)
    return parser.parse_args()


def load_gate(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"missing_preflight_gate={path}")
    return json.loads(path.read_text(encoding="utf-8-sig"))


def verify_static_quality_manifest(
    gate: dict[str, Any],
    issues: list[str],
    *,
    max_age_minutes: float,
    preflight_generated: datetime | None,
) -> None:
    manifest_rel = str(gate.get("static_quality_manifest") or "")
    manifest_hash = str(gate.get("static_quality_manifest_sha256") or "")
    if not manifest_rel:
        issues.append("static_quality_manifest_missing")
        return
    manifest_path = (ROOT / manifest_rel).resolve()
    try:
        manifest_path.relative_to(ROOT)
    except ValueError:
        issues.append(f"static_quality_manifest_escapes_root path={manifest_rel}")
        return
    if not manifest_path.exists():
        issues.append(f"static_quality_manifest_path_missing path={manifest_rel}")
        return
    if sha256_file(manifest_path) != manifest_hash:
        issues.append("static_quality_manifest_sha256_mismatch")
        return
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    except Exception as exc:
        issues.append(f"static_quality_manifest_unreadable error={exc}")
        return
    if manifest.get("schema") != "old_japanese_0_1b_static_quality_manifest_v1":
        issues.append("static_quality_manifest_schema_mismatch")
    if manifest.get("status") != "passed":
        issues.append("static_quality_manifest_not_passed")
    if int(manifest.get("exit_code", -1)) != 0:
        issues.append(f"static_quality_manifest_exit_code_not_zero got={manifest.get('exit_code')}")
    if manifest.get("hf_export") is not False:
        issues.append("static_quality_manifest_hf_export_not_false")
    command = str(manifest.get("command") or "")
    if (
        "scripts\\run_static_quality_checks.ps1" not in command
        and "scripts/run_static_quality_checks.ps1" not in command
    ):
        issues.append("static_quality_manifest_command_mismatch")
    if "-RefreshEvidence" not in command:
        issues.append("static_quality_manifest_not_from_explicit_refresh_evidence")
    try:
        static_generated = parse_utc(str(manifest.get("generated_at_utc") or ""))
        age_minutes = (datetime.now(timezone.utc) - static_generated).total_seconds() / 60.0
        if age_minutes < -1.0:
            issues.append(f"static_quality_manifest_from_future age_minutes={age_minutes:.2f}")
        if age_minutes > max_age_minutes:
            issues.append(
                f"static_quality_manifest_stale age_minutes={age_minutes:.2f} max={max_age_minutes:.2f}"
            )
        if preflight_generated and static_generated > preflight_generated:
            delta_seconds = (static_generated - preflight_generated).total_seconds()
            if delta_seconds > 5.0:
                issues.append(
                    f"static_quality_manifest_newer_than_preflight_gate delta_seconds={delta_seconds:.2f}"
                )
    except Exception as exc:
        issues.append(f"static_quality_manifest_invalid_generated_at_utc error={exc}")
    log_rel = str(manifest.get("log") or "")
    if gate.get("static_quality_log") != log_rel:
        issues.append("static_quality_log_path_mismatch")
    log_path = (ROOT / log_rel).resolve() if log_rel else None
    if not log_path:
        issues.append(f"static_quality_log_missing path={log_rel}")
    else:
        try:
            log_path.relative_to(ROOT)
        except ValueError:
            issues.append(f"static_quality_log_escapes_root path={log_rel}")
        if not log_path.exists():
            issues.append(f"static_quality_log_missing path={log_rel}")
        elif sha256_file(log_path) != str(manifest.get("log_sha256") or ""):
            issues.append("static_quality_manifest_log_sha256_mismatch")
    if gate.get("static_quality_log_sha256") != manifest.get("log_sha256"):
        issues.append("static_quality_log_sha256_mismatch")
    runner_rel = str(manifest.get("runner") or "")
    runner_path = (ROOT / runner_rel).resolve() if runner_rel else None
    if not runner_path:
        issues.append("static_quality_runner_missing")
    else:
        try:
            runner_path.relative_to(ROOT)
        except ValueError:
            issues.append(f"static_quality_runner_escapes_root path={runner_rel}")
        if not runner_path.exists():
            issues.append(f"static_quality_runner_missing path={runner_rel}")
        elif sha256_file(runner_path) != str(manifest.get("runner_sha256") or ""):
            issues.append("static_quality_manifest_runner_sha256_mismatch")


def parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("naive datetime")
    return parsed.astimezone(timezone.utc)


def main() -> None:
    args = parse_args()
    gate_path = (ROOT / args.gate).resolve()
    gate = load_gate(gate_path)
    issues: list[str] = []

    preflight_generated: datetime | None = None
    if gate.get("schema") != REQUIRED_SCHEMA:
        issues.append(f"schema_mismatch expected={REQUIRED_SCHEMA} got={gate.get('schema')}")
    if gate.get("status") != "passed":
        issues.append("status_not_passed")
    if gate.get("static_quality_checks") is not True:
        issues.append("static_quality_checks_not_true")
    try:
        preflight_generated = parse_utc(str(gate.get("generated_at_utc") or ""))
        age_minutes = (datetime.now(timezone.utc) - preflight_generated).total_seconds() / 60.0
        if age_minutes < -1.0:
            issues.append(f"preflight_gate_from_future age_minutes={age_minutes:.2f}")
        if age_minutes > args.max_age_minutes:
            issues.append(f"preflight_gate_stale age_minutes={age_minutes:.2f} max={args.max_age_minutes:.2f}")
    except Exception as exc:
        issues.append(f"invalid_generated_at_utc error={exc}")
    verify_static_quality_manifest(
        gate,
        issues,
        max_age_minutes=args.max_age_minutes,
        preflight_generated=preflight_generated,
    )
    if gate.get("reviews_required") is not True:
        issues.append("reviews_required_not_true")
    if gate.get("hf_export") is not False:
        issues.append("hf_export_not_false")
    if gate.get("release_workspace_clean") is not True:
        issues.append("release_workspace_not_clean")
    if gate.get("release_workspace_files"):
        issues.append("release_workspace_files_not_empty")
    if gate.get("tokenizer_policy") != REQUIRED_TOKENIZER_POLICY:
        issues.append(f"tokenizer_policy_mismatch got={gate.get('tokenizer_policy')}")

    inputs = gate.get("inputs_sha256") or {}
    if not isinstance(inputs, dict) or not inputs:
        issues.append("inputs_sha256_missing")
    else:
        for rel, expected_hash in sorted(inputs.items()):
            path = (ROOT / str(rel)).resolve()
            try:
                path.relative_to(ROOT)
            except ValueError:
                issues.append(f"input_path_escapes_root path={rel}")
                continue
            if not path.exists():
                issues.append(f"input_missing path={rel}")
                continue
            actual_hash = sha256_file(path)
            if str(expected_hash) != actual_hash:
                issues.append(f"input_hash_mismatch path={rel}")

    if issues:
        for issue in issues:
            print(issue)
        raise SystemExit(1)
    print(f"preflight_gate_ok=true gate={gate_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
