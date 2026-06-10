from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "old_japanese_0_1b_static_quality_manifest_v1"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("naive datetime")
    return parsed.astimezone(timezone.utc)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify the static quality command transcript manifest.")
    parser.add_argument("--manifest", default="logs/static_quality_manifest_old_japanese_0_1b.json")
    parser.add_argument("--max-age-minutes", type=float, default=120.0)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"missing_static_quality_manifest={path.relative_to(ROOT)}")
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main() -> None:
    args = parse_args()
    manifest_path = (ROOT / args.manifest).resolve()
    manifest_path.relative_to(ROOT)
    payload = load_json(manifest_path)
    issues: list[str] = []

    if payload.get("schema") != SCHEMA:
        issues.append(f"schema_mismatch got={payload.get('schema')}")
    if payload.get("status") != "passed":
        issues.append("status_not_passed")
    if int(payload.get("exit_code", -1)) != 0:
        issues.append(f"exit_code_not_zero got={payload.get('exit_code')}")
    if payload.get("hf_export") is not False:
        issues.append("hf_export_not_false")
    command = str(payload.get("command") or "")
    if "run_static_quality_checks.ps1" not in command:
        issues.append("command_does_not_reference_static_runner")
    if "-RefreshEvidence" not in command:
        issues.append("command_does_not_refresh_evidence")
    try:
        generated = parse_utc(str(payload.get("generated_at_utc") or ""))
        age_minutes = (datetime.now(timezone.utc) - generated).total_seconds() / 60.0
        if age_minutes < -1.0:
            issues.append(f"manifest_from_future age_minutes={age_minutes:.2f}")
        if age_minutes > args.max_age_minutes:
            issues.append(f"manifest_stale age_minutes={age_minutes:.2f} max={args.max_age_minutes:.2f}")
    except Exception as exc:
        issues.append(f"invalid_generated_at_utc error={exc}")

    for key, hash_key in (("log", "log_sha256"), ("runner", "runner_sha256")):
        rel = str(payload.get(key) or "")
        if not rel:
            issues.append(f"{key}_missing")
            continue
        path = (ROOT / rel).resolve()
        try:
            path.relative_to(ROOT)
        except ValueError:
            issues.append(f"{key}_escapes_root path={rel}")
            continue
        if not path.exists():
            issues.append(f"{key}_path_missing path={rel}")
            continue
        if sha256_file(path) != str(payload.get(hash_key) or ""):
            issues.append(f"{hash_key}_mismatch path={rel}")

    if issues:
        for issue in issues:
            print(issue)
        raise SystemExit(1)
    print(f"static_quality_manifest_ok=true manifest={manifest_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
