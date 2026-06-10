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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write the static quality command transcript manifest.")
    parser.add_argument("--out", default="logs/static_quality_manifest_old_japanese_0_1b.json")
    parser.add_argument("--log", required=True)
    parser.add_argument("--status", choices=("passed", "failed"), required=True)
    parser.add_argument("--exit-code", type=int, required=True)
    parser.add_argument(
        "--command",
        default="powershell -NoProfile -ExecutionPolicy Bypass -File scripts\\run_static_quality_checks.ps1 -RefreshEvidence",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = (ROOT / args.out).resolve()
    log = (ROOT / args.log).resolve()
    out.relative_to(ROOT)
    log.relative_to(ROOT)
    if not log.exists():
        raise SystemExit(f"missing_static_quality_log={log.relative_to(ROOT)}")
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "status": args.status,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": args.command,
        "exit_code": args.exit_code,
        "log": log.relative_to(ROOT).as_posix(),
        "log_sha256": sha256_file(log),
        "runner": "scripts/run_static_quality_checks.ps1",
        "runner_sha256": sha256_file(ROOT / "scripts" / "run_static_quality_checks.ps1"),
        "hf_export": False,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"static_quality_manifest_written={out.relative_to(ROOT)}")
    print(f"static_quality_manifest_status={args.status}")


if __name__ == "__main__":
    main()
