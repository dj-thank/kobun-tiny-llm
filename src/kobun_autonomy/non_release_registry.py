from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class NonReleaseRecordError(RuntimeError):
    """Raised when a durable non-release record exists but is not trustworthy."""


def non_release_dir(root: Path | None = None) -> Path:
    base = root or Path.cwd()
    return base / "logs" / "non_release_runs"


def non_release_record_path(run_id: str, root: Path | None = None) -> Path:
    return non_release_dir(root) / f"{run_id}.json"


def read_non_release_record(run_id: str, root: Path | None = None) -> dict[str, Any]:
    path = non_release_record_path(run_id, root)
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise NonReleaseRecordError(f"invalid_non_release_record_json path={path}") from exc
    if str(payload.get("run_id") or "") != run_id:
        raise NonReleaseRecordError(
            f"non_release_record_run_id_mismatch path={path} expected={run_id} got={payload.get('run_id')}"
        )
    if payload.get("release_status") != "non_release_artifact":
        raise NonReleaseRecordError(
            f"non_release_record_status_invalid path={path} status={payload.get('release_status')}"
        )
    if payload.get("hf_export") is not False:
        raise NonReleaseRecordError(f"non_release_record_hf_export_not_false path={path}")
    return payload


def list_non_release_run_ids(root: Path | None = None) -> list[str]:
    directory = non_release_dir(root)
    if not directory.exists():
        return []
    run_ids: list[str] = []
    for path in sorted(directory.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError):
            continue
        run_id = str(payload.get("run_id") or path.stem)
        if payload.get("release_status") == "non_release_artifact" and run_id == path.stem:
            run_ids.append(run_id)
    return sorted(set(run_ids))


def is_non_release_recorded(run_id: str, root: Path | None = None) -> bool:
    payload = read_non_release_record(run_id, root)
    return bool(payload) and payload.get("release_status") == "non_release_artifact"
