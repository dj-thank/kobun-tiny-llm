from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


NON_RELEASE_STATE_MARKERS = (
    "failed",
    "nonrelease",
    "non_release",
    "stopped_non_release",
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def is_non_release_archive(path: Path, payload: dict[str, Any]) -> bool:
    name = path.name.lower()
    state = str(payload.get("state") or "").lower()
    return any(marker in name or marker in state for marker in NON_RELEASE_STATE_MARKERS)


def iter_non_release_archives(root: Path) -> list[tuple[Path, dict[str, Any]]]:
    archives: list[tuple[Path, dict[str, Any]]] = []
    logs_dir = root / "logs"
    for pattern in ("active_old_japanese_0_1b_dml*.json", "active_old_japanese_0_1b_cuda*.json"):
        for path in sorted(logs_dir.glob(pattern)):
            try:
                payload = load_json(path)
            except (OSError, json.JSONDecodeError) as exc:
                raise SystemExit(f"non_release_registry_archive_json_error path={path} error={exc}") from exc
            if is_non_release_archive(path, payload):
                archives.append((path, payload))
    return archives


def make_record(root: Path, archive_path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    rel_archive = archive_path.relative_to(root).as_posix()
    reason = (
        str(payload.get("stop_reason") or "")
        or str(payload.get("reason") or "")
        or str(payload.get("state") or "")
        or "archived_non_release_active_lock"
    )
    return {
        "run_id": str(payload.get("run_id") or ""),
        "release_status": "non_release_artifact",
        "reason": reason,
        "active_lock_state": str(payload.get("state") or ""),
        "created_at": str(payload.get("created_at") or ""),
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_archive_path": rel_archive,
        "source_archive_sha256": sha256_file(archive_path),
        "hf_export": False,
    }


def record_path(root: Path, run_id: str) -> Path:
    return root / "logs" / "non_release_runs" / f"{run_id}.json"


def validate_record(
    *,
    root: Path,
    archive_path: Path,
    archive_payload: dict[str, Any],
    backfill: bool,
) -> list[str]:
    issues: list[str] = []
    run_id = str(archive_payload.get("run_id") or "")
    if not run_id:
        return [f"archive_missing_run_id path={archive_path}"]
    if archive_payload.get("hf_export") is not False:
        issues.append(f"archive_hf_export_not_false run_id={run_id} path={archive_path}")

    path = record_path(root, run_id)
    if not path.exists():
        if backfill:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(make_record(root, archive_path, archive_payload), ensure_ascii=False, indent=2)
                + "\n",
                encoding="utf-8",
            )
        else:
            issues.append(f"missing_non_release_record run_id={run_id} archive={archive_path.name}")
            return issues

    try:
        record = load_json(path)
    except (OSError, json.JSONDecodeError) as exc:
        issues.append(f"invalid_non_release_record_json run_id={run_id} path={path} error={exc}")
        return issues

    if str(record.get("run_id") or "") != run_id:
        issues.append(f"record_run_id_mismatch expected={run_id} path={path}")
    if record.get("release_status") != "non_release_artifact":
        issues.append(f"record_release_status_not_non_release run_id={run_id} path={path}")
    if record.get("hf_export") is not False:
        issues.append(f"record_hf_export_not_false run_id={run_id} path={path}")
    if not str(record.get("reason") or record.get("stop_reason") or ""):
        issues.append(f"record_missing_reason run_id={run_id} path={path}")

    rel_archive = archive_path.relative_to(root).as_posix()
    recorded_archive = str(record.get("source_archive_path") or "")
    legacy_archive = str(record.get("active_lock_archive") or "")
    if recorded_archive:
        if recorded_archive.replace("\\", "/") != rel_archive:
            issues.append(
                f"record_archive_path_mismatch run_id={run_id} expected={rel_archive} got={recorded_archive}"
            )
        if str(record.get("source_archive_sha256") or "") != sha256_file(archive_path):
            issues.append(f"record_archive_hash_mismatch run_id={run_id} path={path}")
    elif legacy_archive != archive_path.name:
        issues.append(f"record_missing_archive_path run_id={run_id} path={path}")

    return issues


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--backfill", action="store_true")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    archives = iter_non_release_archives(root)
    issues: list[str] = []
    for archive_path, payload in archives:
        issues.extend(
            validate_record(
                root=root,
                archive_path=archive_path,
                archive_payload=payload,
                backfill=args.backfill,
            )
        )
    if issues:
        for issue in issues:
            print(issue)
        raise SystemExit(1)
    print(f"non_release_registry_archives_ok=true archives={len(archives)}")


if __name__ == "__main__":
    main()
