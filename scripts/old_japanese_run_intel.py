from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from kobun_autonomy.types import AutonomousAction, BoardRunRow, EvaluationBoard, RunClassification, TrainingLogStatus
from kobun_autonomy.non_release_registry import read_non_release_record
from kobun_autonomy.release_policy import NON_RELEASE_RUNS

INTERNAL_EVIDENCE_MARKERS = (
    "failtest",
    "negative_test",
)

DML_RUN_RE = re.compile(r"^old_japanese_0_1b_dml_[0-9A-Za-z][0-9A-Za-z_-]{0,63}$")
CUDA_RUN_RE = re.compile(r"^old_japanese_0_1b_cuda_[0-9A-Za-z][0-9A-Za-z_-]{0,63}$")
RUN_RE = re.compile(r"^old_japanese_0_1b_(?:dml|cuda)_[0-9A-Za-z][0-9A-Za-z_-]{0,63}$")
STEP_RE = re.compile(r"step=(?P<step>\d+)\s+train_loss=(?P<train>[0-9.]+)\s+val_loss=(?P<val>[0-9.]+)")
CONFIG_RE = re.compile(r"config=.*?vocab=(?P<vocab>\d+).*?params=(?P<params>\d+)")
CONFIG_BLOCK_RE = re.compile(r"config=.*?block=(?P<block>\d+)")
DEVICE_RE = re.compile(r"device=(?P<device>.+)")
RELEASE_TOKENIZER_POLICY = "train_split_plus_core_japanese_inventory_plus_utf8_byte_fallback_v1"
RELEASE_MAX_VOCAB_SIZE = 10_000
RELEASE_BLOCK_SIZE = 384


@dataclass(frozen=True)
class StepRecord:
    step: int
    train_loss: float
    val_loss: float


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def short_sha256(path: Path) -> str:
    if not path.exists() or not path.is_file():
        return ""
    return sha256_file(path)


def load_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return read_json(path)
    except (json.JSONDecodeError, OSError):
        return {}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def process_exists(pid: int) -> bool:
    if pid <= 0:
        return False
    if sys.platform == "win32":
        result = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                f"if (Get-Process -Id {pid} -ErrorAction SilentlyContinue) {{ 'true' }}",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        return "true" in result.stdout.lower()
    try:
        import os

        os.kill(pid, 0)
    except OSError:
        return False
    return True


def current_process_tree_ids(limit: int = 12) -> set[int]:
    """Return this process and ancestors so process scans ignore their own test shell."""
    ignored = {os.getpid()}
    pid = os.getpid()
    for _ in range(limit):
        parent = 0
        if sys.platform == "win32":
            result = subprocess.run(
                [
                    "powershell",
                    "-NoProfile",
                    "-Command",
                    f"$p=Get-CimInstance Win32_Process -Filter 'ProcessId={pid}'; if($p){{ $p.ParentProcessId }}",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            try:
                parent = int(result.stdout.strip() or "0")
            except ValueError:
                parent = 0
        else:
            stat_path = Path("/proc") / str(pid) / "stat"
            try:
                parts = stat_path.read_text(encoding="utf-8", errors="replace").split()
                parent = int(parts[3])
            except (OSError, ValueError, IndexError):
                parent = 0
        if parent <= 0 or parent in ignored:
            break
        ignored.add(parent)
        pid = parent
    return ignored


def process_matches_run(pid: int, run_id: str) -> bool:
    if pid <= 0:
        return False
    markers = (
        "kobun_llm.train",
        "train_old_japanese_0_1b_dml.ps1",
        "train_old_japanese_0_1b_gpu.ps1",
        "watch_and_finalize_old_japanese_0_1b_dml.ps1",
        "finalize_old_japanese_0_1b_dml.ps1",
        "start_old_japanese_0_1b_dml_and_watch.ps1",
        "start_old_japanese_0_1b_cuda_colab_and_watch.py",
    )
    if sys.platform == "win32":
        result = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                f"$p=Get-CimInstance Win32_Process -Filter 'ProcessId={pid}'; if($p){{ $p.CommandLine }}",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        text = result.stdout
    else:
        cmdline = Path("/proc") / str(pid) / "cmdline"
        try:
            text = cmdline.read_bytes().replace(b"\x00", b" ").decode("utf-8", errors="replace")
        except OSError:
            return False
    return run_id in text and any(marker in text for marker in markers)


def process_matches_launcher(pid: int) -> bool:
    if pid <= 0:
        return False
    if sys.platform == "win32":
        result = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                f"$p=Get-CimInstance Win32_Process -Filter 'ProcessId={pid}'; if($p){{ $p.CommandLine }}",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        text = result.stdout
    else:
        cmdline = Path("/proc") / str(pid) / "cmdline"
        try:
            text = cmdline.read_bytes().replace(b"\x00", b" ").decode("utf-8", errors="replace")
        except OSError:
            return False
    return "start_old_japanese_0_1b_dml_and_watch.ps1" in text or "start_old_japanese_0_1b_cuda_colab_and_watch.py" in text


def recent_active_lock(payload: dict[str, Any], minutes: float = 30.0) -> bool:
    try:
        created = datetime.fromisoformat(str(payload.get("created_at", "")).replace("Z", "+00:00"))
    except ValueError:
        return False
    if created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - created.astimezone(timezone.utc)).total_seconds() < minutes * 60.0


def live_process_for_run(run_id: str) -> bool:
    markers = (
        "kobun_llm.train",
        "train_old_japanese_0_1b_dml.ps1",
        "train_old_japanese_0_1b_gpu.ps1",
        "watch_and_finalize_old_japanese_0_1b_dml.ps1",
        "start_old_japanese_0_1b_dml_and_watch.ps1",
        "start_old_japanese_0_1b_cuda_colab_and_watch.py",
    )
    if sys.platform == "win32":
        escaped = run_id.replace("'", "''")
        result = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "Get-CimInstance Win32_Process | "
                f"Where-Object {{ $_.CommandLine -like '*{escaped}*' }} | "
                "ForEach-Object { $_.CommandLine }",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        lines = result.stdout.splitlines()
    else:
        lines = []
        proc_root = Path("/proc")
        for cmdline in proc_root.glob("[0-9]*/cmdline"):
            try:
                lines.append(cmdline.read_bytes().replace(b"\x00", b" ").decode("utf-8", errors="replace"))
            except OSError:
                continue
    for text in lines:
        if run_id in text and any(marker in text for marker in markers):
            return True
    return False


def live_dml_process_summaries(root: Path | None = None) -> list[dict[str, Any]]:
    markers = (
        "train_old_japanese_0_1b_dml.ps1",
        "watch_and_finalize_old_japanese_0_1b_dml.ps1",
        "finalize_old_japanese_0_1b_dml.ps1",
        "start_old_japanese_0_1b_dml_and_watch.ps1",
    )
    root_token = ""
    if root is not None:
        root_token = str(root.resolve(strict=False)).casefold()

    def belongs_to_root(command_line: str) -> bool:
        if not root_token:
            return True
        # Supervised launches commonly use relative script paths. Treat the
        # project's unique script names as belonging to this workflow even when
        # the absolute root is not present in the command line.
        lowered = command_line.casefold()
        return root_token in lowered or any(marker.casefold() in lowered for marker in markers)

    def is_dml_training_command(command_line: str) -> bool:
        if not command_line:
            return False
        if any(marker in command_line for marker in markers):
            return True
        if "kobun_llm.train" not in command_line:
            return False
        # Direct unsupervised DML probes may omit the release run id. Treat
        # train commands as DML-like unless they explicitly select CPU/CUDA/HIP.
        return not re.search(r'(?i)(?:^|\s)--device(?:\s+|=)(cpu|cuda|hip)(?=\s|$|"|\')', command_line)

    summaries: list[dict[str, Any]] = []
    ignored_pids = current_process_tree_ids()
    if sys.platform == "win32":
        result = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "Get-CimInstance Win32_Process | "
                "Where-Object { $_.CommandLine -match 'kobun_llm\\.train|train_old_japanese_0_1b_dml\\.ps1|watch_and_finalize_old_japanese_0_1b_dml\\.ps1|finalize_old_japanese_0_1b_dml\\.ps1|start_old_japanese_0_1b_dml_and_watch\\.ps1' } | "
                "ForEach-Object { [pscustomobject]@{ProcessId=$_.ProcessId; Name=$_.Name; CommandLine=$_.CommandLine} | ConvertTo-Json -Compress }",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        for line in result.stdout.splitlines():
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            pid = safe_int(payload.get("ProcessId"), 0) or 0
            if pid in ignored_pids:
                continue
            text = str(payload.get("CommandLine") or "")
            if belongs_to_root(text) and is_dml_training_command(text):
                summaries.append(
                    {
                        "pid": pid,
                        "name": payload.get("Name"),
                        "command_preview": text[:260],
                    }
                )
    else:
        for cmdline in Path("/proc").glob("[0-9]*/cmdline"):
            try:
                text = cmdline.read_bytes().replace(b"\x00", b" ").decode("utf-8", errors="replace")
            except OSError:
                continue
            pid = safe_int(cmdline.parent.name, 0) or 0
            if pid in ignored_pids:
                continue
            if belongs_to_root(text) and is_dml_training_command(text):
                summaries.append(
                    {
                        "pid": pid,
                        "name": "process",
                        "command_preview": text[:260],
                    }
                )
    return summaries


def live_supervised_process_summaries(root: Path | None = None) -> list[dict[str, Any]]:
    markers = (
        "kobun_llm.train",
        "train_old_japanese_0_1b_dml.ps1",
        "watch_and_finalize_old_japanese_0_1b_dml.ps1",
        "finalize_old_japanese_0_1b_dml.ps1",
        "start_old_japanese_0_1b_dml_and_watch.ps1",
        "start_old_japanese_0_1b_cuda_colab_and_watch.py",
    )
    root_token = str(root.resolve(strict=False)).casefold() if root is not None else ""

    def belongs_to_root(command_line: str) -> bool:
        lowered = command_line.casefold()
        return not root_token or root_token in lowered or any(marker.casefold() in lowered for marker in markers)

    def interesting(command_line: str) -> bool:
        if "Get-CimInstance Win32_Process" in command_line and "Where-Object" in command_line:
            return False
        return any(marker in command_line for marker in markers)

    summaries: list[dict[str, Any]] = []
    ignored_pids = current_process_tree_ids()
    if sys.platform == "win32":
        result = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "Get-CimInstance Win32_Process | "
                "Where-Object { $_.CommandLine -match 'kobun_llm\\.train|train_old_japanese_0_1b_dml\\.ps1|watch_and_finalize_old_japanese_0_1b_dml\\.ps1|finalize_old_japanese_0_1b_dml\\.ps1|start_old_japanese_0_1b_dml_and_watch\\.ps1|start_old_japanese_0_1b_cuda_colab_and_watch\\.py' } | "
                "ForEach-Object { [pscustomobject]@{ProcessId=$_.ProcessId; Name=$_.Name; CommandLine=$_.CommandLine} | ConvertTo-Json -Compress }",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        for line in result.stdout.splitlines():
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            pid = safe_int(payload.get("ProcessId"), 0) or 0
            if pid in ignored_pids:
                continue
            text = str(payload.get("CommandLine") or "")
            if belongs_to_root(text) and interesting(text):
                summaries.append(
                    {
                        "pid": pid,
                        "name": payload.get("Name"),
                        "command_preview": text[:260],
                    }
                )
        return summaries
    for cmdline in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            text = cmdline.read_bytes().replace(b"\x00", b" ").decode("utf-8", errors="replace")
        except OSError:
            continue
        pid = safe_int(cmdline.parent.name, 0) or 0
        if pid in ignored_pids:
            continue
        if belongs_to_root(text) and interesting(text):
            summaries.append(
                {
                    "pid": pid,
                    "name": "process",
                    "command_preview": text[:260],
                }
            )
    return summaries


def startup_mutex_schema_errors(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    run_id = str(payload.get("run_id") or "")
    if not RUN_RE.fullmatch(run_id):
        errors.append("invalid_or_missing_run_id")
    if str(payload.get("state") or "") != "startup_mutex":
        errors.append("invalid_or_missing_state")
    if str(payload.get("backend") or "") not in {"dml", "cuda"}:
        errors.append("invalid_or_missing_backend")
    if payload.get("hf_export") is not False:
        errors.append("hf_export_not_false")
    try:
        created = datetime.fromisoformat(str(payload.get("created_at", "")).replace("Z", "+00:00"))
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
    except ValueError:
        errors.append("invalid_or_missing_created_at")
    try:
        launcher_pid = int(payload.get("launcher_pid") or 0)
    except (TypeError, ValueError):
        launcher_pid = 0
    if launcher_pid <= 0:
        errors.append("invalid_or_missing_launcher_pid")
    for key in (
        "launch_token_sha256",
        "preflight_gate_sha256",
        "review_gate_sha256",
        "autonomous_launch_context_sha256",
    ):
        value = str(payload.get(key) or "")
        if value and not re.fullmatch(r"[0-9a-f]{64}", value):
            errors.append(f"invalid_{key}")
    return errors


def archive_startup_mutex(path: Path, reason: str) -> str:
    archive = path.with_name(f"active_old_japanese_0_1b_training.stale.{reason}.{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    path.replace(archive)
    return archive.name


def startup_mutex_health(root: Path) -> dict[str, Any]:
    path = root / "logs" / "active_old_japanese_0_1b_training.lock"
    if not path.exists():
        return {"exists": False, "invalid_json": False, "hard_blockers": []}
    live = live_supervised_process_summaries(root)
    try:
        payload = read_json(path)
    except (json.JSONDecodeError, OSError) as exc:
        if live:
            return {
                "exists": True,
                "invalid_json": True,
                "path": path.name,
                "hard_blockers": ["invalid_startup_mutex_json_with_live_process"],
                "live_processes": live,
                "error": str(exc),
            }
        try:
            archived = archive_startup_mutex(path, "invalid")
        except OSError as replace_exc:
            return {
                "exists": True,
                "invalid_json": True,
                "path": path.name,
                "hard_blockers": ["invalid_startup_mutex_json_unquarantined"],
                "live_processes": [],
                "error": str(replace_exc),
            }
        return {
            "exists": False,
            "invalid_json": True,
            "quarantined_path": archived,
            "hard_blockers": [],
            "live_processes": [],
            "error": str(exc),
        }
    schema_errors = startup_mutex_schema_errors(payload)
    if schema_errors:
        if live:
            return {
                "exists": True,
                "invalid_json": False,
                "invalid_schema": True,
                "path": path.name,
                "hard_blockers": ["invalid_startup_mutex_schema_with_live_process"],
                "schema_errors": schema_errors,
                "live_processes": live,
            }
        try:
            archived = archive_startup_mutex(path, "invalid")
        except OSError as replace_exc:
            return {
                "exists": True,
                "invalid_json": False,
                "invalid_schema": True,
                "path": path.name,
                "hard_blockers": ["invalid_startup_mutex_schema_unquarantined"],
                "schema_errors": schema_errors,
                "live_processes": [],
                "error": str(replace_exc),
            }
        return {
            "exists": False,
            "invalid_json": False,
            "invalid_schema": True,
            "quarantined_path": archived,
            "hard_blockers": [],
            "schema_errors": schema_errors,
            "live_processes": [],
        }
    launcher_pid = safe_int(payload.get("launcher_pid"), 0) or 0
    launcher_live = process_exists(launcher_pid)
    recent = recent_active_lock(payload, minutes=30.0)
    if launcher_live or recent:
        return {
            "exists": True,
            "invalid_json": False,
            "invalid_schema": False,
            "run_id": str(payload.get("run_id") or ""),
            "backend": str(payload.get("backend") or ""),
            "state": str(payload.get("state") or ""),
            "launcher_pid": launcher_pid,
            "launcher_live": launcher_live,
            "recent": recent,
            "hard_blockers": ["startup_mutex_live"],
        }
    try:
        archived = archive_startup_mutex(path, "expired")
    except OSError as exc:
        return {
            "exists": True,
            "invalid_json": False,
            "invalid_schema": False,
            "path": path.name,
            "run_id": str(payload.get("run_id") or ""),
            "launcher_live": False,
            "recent": False,
            "hard_blockers": ["stale_startup_mutex_unquarantined"],
            "error": str(exc),
        }
    return {
        "exists": False,
        "invalid_json": False,
        "invalid_schema": False,
        "quarantined_path": archived,
        "run_id": str(payload.get("run_id") or ""),
        "launcher_live": False,
        "recent": False,
        "hard_blockers": [],
    }


def active_lock_health(root: Path) -> dict[str, Any]:
    candidates = [
        root / "logs" / "active_old_japanese_0_1b_dml.lock",
        root / "logs" / "active_old_japanese_0_1b_cuda.lock",
    ]
    existing = [path for path in candidates if path.exists()]
    if not existing:
        return {"exists": False, "invalid_json": False, "hard_blockers": []}
    if len(existing) > 1:
        return {
            "exists": True,
            "invalid_json": False,
            "path": [path.name for path in existing],
            "hard_blockers": ["multiple_active_backend_locks"],
        }
    path = existing[0]
    backend = "cuda" if path.name == "active_old_japanese_0_1b_cuda.lock" else "dml"

    def stale_archive_path(reason: str) -> Path:
        return root / "logs" / f"active_old_japanese_0_1b_{backend}.stale.{reason}.{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    try:
        payload = read_json(path)
    except (json.JSONDecodeError, OSError) as exc:
        live = live_supervised_process_summaries(root)
        if live:
            return {
                "exists": True,
                "invalid_json": True,
                "path": path.name,
                "backend": backend,
                "hard_blockers": [f"invalid_active_lock_json_with_live_{backend}_or_supervised_process"],
                "live_processes": live,
                "error": str(exc),
            }
        stale = stale_archive_path("invalid")
        try:
            path.replace(stale)
        except OSError as replace_exc:
            return {
                "exists": True,
                "invalid_json": True,
                "path": path.name,
                "hard_blockers": ["invalid_active_lock_json_unquarantined"],
                "live_processes": [],
                "error": str(replace_exc),
            }
        return {
            "exists": False,
            "invalid_json": True,
            "quarantined_path": stale.name,
            "hard_blockers": [],
            "live_processes": [],
            "error": str(exc),
        }
    schema_errors = active_lock_schema_errors(payload, expected_backend=backend)
    if schema_errors:
        live = live_supervised_process_summaries(root)
        recent = recent_active_lock(payload)
        if live or recent:
            return {
                "exists": True,
                "invalid_json": False,
                "invalid_schema": True,
                "path": path.name,
                "backend": backend,
                "hard_blockers": [f"invalid_active_lock_schema_with_live_{backend}_or_supervised_process"],
                "schema_errors": schema_errors,
                "live_processes": live,
                "recent": recent,
            }
        stale = stale_archive_path("invalid")
        try:
            path.replace(stale)
        except OSError as replace_exc:
            return {
                "exists": True,
                "invalid_json": False,
                "invalid_schema": True,
                "path": path.name,
                "hard_blockers": ["invalid_active_lock_schema_unquarantined"],
                "schema_errors": schema_errors,
                "live_processes": [],
                "error": str(replace_exc),
            }
        return {
            "exists": False,
            "invalid_json": False,
            "invalid_schema": True,
            "quarantined_path": stale.name,
            "hard_blockers": [],
            "schema_errors": schema_errors,
            "live_processes": [],
        }
    run_id = str(payload.get("run_id") or "")
    launcher_pid = safe_int(payload.get("launcher_pid"), 0) or 0
    train_pid = safe_int(payload.get("train_pid"), 0) or 0
    watcher_pid = safe_int(payload.get("watcher_pid"), 0) or 0
    launcher_live = process_matches_launcher(launcher_pid)
    train_live = process_matches_run(train_pid, run_id)
    watcher_live = process_matches_run(watcher_pid, run_id)
    recent = recent_active_lock(payload)
    if launcher_live or train_live or watcher_live or recent:
        return {
            "exists": True,
            "invalid_json": False,
            "invalid_schema": False,
            "run_id": run_id,
            "backend": backend,
            "state": str(payload.get("state") or ""),
            "launcher_pid": launcher_pid,
            "train_pid": train_pid,
            "watcher_pid": watcher_pid,
            "launcher_live": launcher_live,
            "train_live": train_live,
            "watcher_live": watcher_live,
            "recent": recent,
            "hard_blockers": [],
        }
    stale = stale_archive_path("expired")
    try:
        path.replace(stale)
    except OSError as exc:
        return {
            "exists": True,
            "invalid_json": False,
            "invalid_schema": False,
            "path": path.name,
            "run_id": run_id,
            "backend": backend,
            "state": str(payload.get("state") or ""),
            "launcher_live": False,
            "train_live": False,
            "watcher_live": False,
            "recent": False,
            "hard_blockers": ["stale_active_lock_unquarantined"],
            "error": str(exc),
        }
    return {
        "exists": False,
        "invalid_json": False,
        "invalid_schema": False,
        "quarantined_path": stale.name,
        "run_id": run_id,
        "backend": backend,
        "state": str(payload.get("state") or ""),
        "launcher_live": False,
        "train_live": False,
        "watcher_live": False,
        "recent": False,
        "hard_blockers": [],
    }


def colab_cuda_lease_health(root: Path) -> dict[str, Any]:
    paths = sorted((root / "logs").glob("colab_active_old_japanese_0_1b_cuda.*.json"))
    paths.extend(sorted((root / "logs").glob("gcp_active_old_japanese_0_1b_cuda.*.json")))
    active: list[dict[str, Any]] = []
    quarantined: list[str] = []
    errors: list[str] = []
    now = datetime.now(timezone.utc)
    for path in paths:
        if ".finished." in path.name or ".failed_non_release." in path.name or ".stale." in path.name:
            continue
        try:
            payload = read_json(path)
        except (json.JSONDecodeError, OSError) as exc:
            stale = path.with_name(path.name.replace(".json", f".stale.invalid.{now.strftime('%Y%m%d_%H%M%S')}.json"))
            try:
                path.replace(stale)
                quarantined.append(stale.name)
            except OSError as replace_exc:
                errors.append(f"{path.name}:unquarantined_invalid:{replace_exc}")
            continue
        run_id = str(payload.get("run_id") or "")
        state = str(payload.get("state") or "")
        expires_text = str(payload.get("lease_expires_at_utc") or "")
        if payload.get("schema") not in {
            "old_japanese_0_1b_colab_cuda_active_lease_v1",
            "old_japanese_0_1b_supervised_cuda_active_lease_v1",
        }:
            errors.append(f"{path.name}:invalid_schema")
            continue
        cuda_provider = str(payload.get("cuda_provider") or ("gcp" if path.name.startswith("gcp_") else "colab"))
        if cuda_provider not in {"colab", "gcp"}:
            errors.append(f"{path.name}:invalid_cuda_provider")
            continue
        if not CUDA_RUN_RE.fullmatch(run_id):
            errors.append(f"{path.name}:invalid_run_id")
            continue
        if payload.get("hf_export") is not False or payload.get("package_created") is not False or payload.get("upload_attempted") is not False:
            errors.append(f"{path.name}:unsafe_release_action")
            continue
        try:
            expires = datetime.fromisoformat(expires_text.replace("Z", "+00:00"))
            if expires.tzinfo is None:
                expires = expires.replace(tzinfo=timezone.utc)
        except ValueError:
            errors.append(f"{path.name}:invalid_expiry")
            continue
        train_pid = safe_int(payload.get("train_pid"), 0) or 0
        launcher_pid = safe_int(payload.get("launcher_pid"), 0) or 0
        train_live = process_matches_run(train_pid, run_id)
        launcher_live = process_matches_launcher(launcher_pid)
        any_live = train_live or launcher_live
        if state in {"finished", "failed_non_release"} or (expires <= now and not any_live):
            stale = path.with_name(path.name.replace(".json", f".stale.expired.{now.strftime('%Y%m%d_%H%M%S')}.json"))
            try:
                path.replace(stale)
                quarantined.append(stale.name)
            except OSError as replace_exc:
                errors.append(f"{path.name}:unquarantined_expired:{replace_exc}")
            continue
        if expires <= now and any_live:
            errors.append(f"{path.name}:expired_but_live")
        active.append(
            {
                "path": path.name,
                "run_id": run_id,
                "state": state,
                "train_pid": train_pid,
                "launcher_pid": launcher_pid,
                "train_live": train_live,
                "launcher_live": launcher_live,
                "cuda_provider": cuda_provider,
                "lease_expires_at_utc": expires.isoformat(),
                "artifact_root": str(payload.get("artifact_root") or ""),
            }
        )
    blockers: list[str] = []
    if active:
        blockers.append("colab_cuda_lease_active")
    if errors:
        blockers.append("colab_cuda_lease_invalid_or_unquarantined")
    return {
        "exists": bool(active),
        "active": active,
        "quarantined": quarantined,
        "errors": errors,
        "hard_blockers": blockers,
    }


def active_lock_schema_errors(payload: dict[str, Any], expected_backend: str = "") -> list[str]:
    errors: list[str] = []
    run_id = str(payload.get("run_id") or "")
    if not RUN_RE.fullmatch(run_id):
        errors.append("invalid_or_missing_run_id")
    backend = str(payload.get("backend") or "")
    if expected_backend and backend != expected_backend:
        errors.append(f"backend_not_{expected_backend}")
    elif not expected_backend and backend and backend not in {"dml", "cuda"}:
        errors.append("invalid_backend")
    state = str(payload.get("state") or "")
    if not state:
        errors.append("missing_state")
    if payload.get("hf_export") is not False:
        errors.append("hf_export_not_false")
    try:
        created = datetime.fromisoformat(str(payload.get("created_at", "")).replace("Z", "+00:00"))
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
    except ValueError:
        errors.append("invalid_or_missing_created_at")
    for key in ("launcher_pid", "train_pid", "watcher_pid"):
        value = payload.get(key)
        if value is None:
            continue
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            errors.append(f"invalid_{key}")
            continue
        if parsed < 0:
            errors.append(f"invalid_{key}")
    launch_token = str(payload.get("launch_token_sha256") or "")
    if not launch_token:
        errors.append("missing_launch_token_sha256")
    elif not re.fullmatch(r"[0-9a-f]{64}", launch_token):
        errors.append("invalid_launch_token_sha256")
    launch_nonce = str(payload.get("launch_nonce_sha256") or "")
    if not launch_nonce:
        errors.append("missing_launch_nonce_sha256")
    elif not re.fullmatch(r"[0-9a-f]{64}", launch_nonce):
        errors.append("invalid_launch_nonce_sha256")
    required_evidence_fields = {
        "preflight_gate": "missing_preflight_gate",
        "preflight_gate_sha256": "missing_preflight_gate_sha256",
        "review_gate": "missing_review_gate",
        "review_gate_sha256": "missing_review_gate_sha256",
        "autonomous_launch_context": "missing_autonomous_launch_context",
        "autonomous_launch_context_sha256": "missing_autonomous_launch_context_sha256",
    }
    for key, error_label in required_evidence_fields.items():
        if not str(payload.get(key) or ""):
            errors.append(error_label)
    for key in ("preflight_gate_sha256", "review_gate_sha256", "autonomous_launch_context_sha256"):
        value = str(payload.get(key) or "")
        if value and not re.fullmatch(r"[0-9a-f]{64}", value):
            errors.append(f"invalid_{key}")
    if not str(payload.get("autonomous_script") or ""):
        errors.append("missing_autonomous_script")
    if not str(payload.get("selected_action") or ""):
        errors.append("missing_selected_action")
    return errors


def read_text_lossy(path: Path) -> str:
    data = path.read_bytes()
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        text = data.decode("utf-16", errors="replace")
    elif b"\x00" in data[:400]:
        text = data.decode("utf-16-le", errors="replace")
    else:
        try:
            text = data.decode("utf-8-sig")
        except UnicodeDecodeError:
            text = data.decode("cp932", errors="replace")
    return text.replace("\x00", "")


def basename(path_text: str) -> str:
    return Path(str(path_text)).name


def is_internal_evidence_run(run_id: str) -> bool:
    lower = run_id.lower()
    return any(marker in lower for marker in INTERNAL_EVIDENCE_MARKERS)


def backend_for_run_id(run_id: str) -> str:
    if DML_RUN_RE.fullmatch(run_id):
        return "dml"
    if CUDA_RUN_RE.fullmatch(run_id):
        return "cuda"
    return "unknown"


def safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value: Any, default: int | None = None) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def discover_run_ids(root: Path) -> list[str]:
    ids: set[str] = set()
    for path in (root / "logs").glob("old_japanese_0_1b_*.out.log"):
        run_id = path.name[: -len(".out.log")]
        if RUN_RE.fullmatch(run_id):
            ids.add(run_id)
    for path in (root / "logs").glob("train_exit_old_japanese_0_1b_*.json"):
        run_id = path.name.removeprefix("train_exit_").removesuffix(".json")
        if RUN_RE.fullmatch(run_id):
            ids.add(run_id)
    for pattern, prefix, suffix in (
        ("launch_old_japanese_0_1b_*.out.log", "launch_", ".out.log"),
        ("launch_old_japanese_0_1b_*.err.log", "launch_", ".err.log"),
        ("watch_start_old_japanese_0_1b_*.out.log", "watch_start_", ".out.log"),
        ("watch_start_old_japanese_0_1b_*.err.log", "watch_start_", ".err.log"),
    ):
        for path in (root / "logs").glob(pattern):
            run_id = path.name.removeprefix(prefix).removesuffix(suffix)
            if RUN_RE.fullmatch(run_id):
                ids.add(run_id)
    for lock_path in (
        root / "logs" / "active_old_japanese_0_1b_dml.lock",
        root / "logs" / "active_old_japanese_0_1b_cuda.lock",
    ):
        if lock_path.exists():
            try:
                payload = read_json(lock_path)
            except (json.JSONDecodeError, OSError):
                payload = {}
            run_id = str(payload.get("run_id") or "")
            if RUN_RE.fullmatch(run_id):
                ids.add(run_id)
    for path in (root / "logs").glob("colab_active_old_japanese_0_1b_cuda.*.json"):
        try:
            payload = read_json(path)
        except (json.JSONDecodeError, OSError):
            payload = {}
        run_id = str(payload.get("run_id") or "")
        if CUDA_RUN_RE.fullmatch(run_id):
            ids.add(run_id)
    for path in (root / "checkpoints").glob("old_japanese_0_1b_*_best.pt"):
        run_id = path.name[: -len("_best.pt")]
        if RUN_RE.fullmatch(run_id):
            ids.add(run_id)
    for path in (root / "logs").glob("eval_results_old_japanese_0_1b_*.json"):
        run_id = path.name.removeprefix("eval_results_").removesuffix(".json")
        if RUN_RE.fullmatch(run_id):
            ids.add(run_id)
    return sorted(ids)


def parse_training_log(path: Path) -> TrainingLogStatus:
    if not path.exists():
        return {
            "exists": False,
            "steps": [],
            "latest": None,
            "best": None,
            "completed": False,
            "stop_reason": "",
        }
    text = read_text_lossy(path)
    records = [
        StepRecord(
            step=int(match.group("step")),
            train_loss=float(match.group("train")),
            val_loss=float(match.group("val")),
        )
        for match in STEP_RE.finditer(text)
    ]
    latest = max(records, key=lambda row: row.step) if records else None
    best = min(records, key=lambda row: (row.val_loss, row.step)) if records else None
    non_improving_evals = 0
    best_so_far = float("inf")
    for record in sorted(records, key=lambda row: row.step):
        if record.val_loss < best_so_far:
            best_so_far = record.val_loss
            non_improving_evals = 0
        else:
            non_improving_evals += 1
    train_val_gap = None
    best_val_regression = None
    if latest is not None:
        train_val_gap = latest.val_loss - latest.train_loss
    if latest is not None and best is not None:
        best_val_regression = latest.val_loss - best.val_loss
    overfit_stop_signal = bool(
        latest is not None
        and best is not None
        and latest.step >= 1000
        and non_improving_evals >= 5
        and (train_val_gap or 0.0) >= 3.0
        and (best_val_regression or 0.0) >= 0.5
    )
    early_stop = ""
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("early stopping:"):
            early_stop = line
    completed = bool(early_stop) or "launcher_completed=" in text
    config: dict[str, Any] = {}
    for line in text.splitlines():
        clean = line.strip()
        config_match = CONFIG_RE.search(clean)
        if config_match:
            config["vocab_size"] = int(config_match.group("vocab"))
            config["params"] = int(config_match.group("params"))
        block_match = CONFIG_BLOCK_RE.search(clean)
        if block_match:
            config["block_size"] = int(block_match.group("block"))
        device_match = DEVICE_RE.match(clean)
        if device_match and "device" not in config:
            config["device"] = device_match.group("device").strip()
    return {
        "exists": True,
        "path": str(path),
        "path_name": path.name,
        "steps": [record.__dict__ for record in records],
        "latest": latest.__dict__ if latest else None,
        "best": best.__dict__ if best else None,
        "completed": completed,
        "stop_reason": early_stop,
        "non_improving_evals": non_improving_evals,
        "train_val_gap": train_val_gap,
        "best_val_regression": best_val_regression,
        "overfit_stop_signal": overfit_stop_signal,
        "last_write_time": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
        "config": config,
    }


def active_run_release_policy_issues(root: Path, run_id: str, train_log: dict[str, Any]) -> list[str]:
    """Return reasons a live run is obsolete under the current release policy.

    This is intentionally limited to active-run supersession decisions. It does
    not try to repair old snapshots or old checkpoints; it marks them as
    non-release so a fresh run can be started under the current gates.
    """

    issues: list[str] = []
    config = dict(train_log.get("config") or {})
    vocab_size = safe_int(config.get("vocab_size"), 0) or 0
    block_size = safe_int(config.get("block_size"), 0) or 0
    if vocab_size >= RELEASE_MAX_VOCAB_SIZE:
        issues.append(f"active_vocab_too_large:{vocab_size}")
    if block_size and block_size != RELEASE_BLOCK_SIZE:
        issues.append(f"active_block_size_obsolete:{block_size}")

    snapshot_meta = (
        root
        / "data"
        / "run_snapshots"
        / run_id
        / "provenance"
        / "tokenizer_public_char_vocab.meta.json"
    )
    if not snapshot_meta.exists():
        issues.append("active_snapshot_tokenizer_meta_missing")
        return issues
    try:
        meta = read_json(snapshot_meta)
    except (json.JSONDecodeError, OSError):
        issues.append("active_snapshot_tokenizer_meta_unreadable")
        return issues
    policy = str(meta.get("policy") or "")
    if policy != RELEASE_TOKENIZER_POLICY:
        issues.append(f"active_tokenizer_policy_obsolete:{policy or 'missing'}")
    if meta.get("byte_fallback") is not True:
        issues.append("active_tokenizer_missing_byte_fallback")
    if safe_int(meta.get("byte_fallback_tokens"), 0) != 256:
        issues.append("active_tokenizer_byte_token_count_invalid")
    estimated_vocab = safe_int(meta.get("estimated_total_vocab_with_byte_fallback_and_unk"), 0) or 0
    total_chars = safe_int(meta.get("total_chars"), 0) or 0
    direct_vocab = safe_int(meta.get("direct_vocab_chars"), total_chars) or total_chars
    comparable_vocab = estimated_vocab or direct_vocab
    if comparable_vocab >= RELEASE_MAX_VOCAB_SIZE:
        issues.append(f"active_tokenizer_vocab_policy_obsolete:{comparable_vocab}")
    return issues


def load_eval_json(root: Path, run_id: str) -> tuple[Path, dict[str, Any] | None]:
    path = root / "logs" / f"eval_results_{run_id}.json"
    if not path.exists():
        return path, None
    return path, read_json(path)


def score_from_eval(eval_payload: dict[str, Any] | None) -> dict[str, Any]:
    if not eval_payload:
        return {
            "test_lm_token_nll": None,
            "grammar_score": None,
            "waka_score": None,
            "morphology_score": None,
            "eval_contamination_hits": None,
            "eval_source_overlap_hits": None,
            "split_leaks": None,
            "waka_leaks": None,
            "tokenizer_leakage": None,
            "source_record_audit": None,
        }
    metrics = eval_payload.get("metrics") or {}
    smoke = eval_payload.get("smoke_metrics") or {}
    model = eval_payload.get("model_metrics") or {}
    leakage = eval_payload.get("leakage") or {}
    contamination = eval_payload.get("eval_contamination") or {}
    source_overlap = eval_payload.get("eval_source_overlap") or {}
    tokenizer = eval_payload.get("tokenizer_vocab_scope") or {}
    grammar_score = (
        (smoke.get("primary_contrastive_preference_accuracy") or metrics.get("primary_contrastive_preference_accuracy") or {}).get("value")
    )
    waka_static_score = (
        smoke.get("waka_meter_constraint_static_accuracy")
        or metrics.get("waka_meter_constraint_static_accuracy")
        or {}
    ).get("value")
    waka_generation_score = (
        smoke.get("waka_meter_constrained_generation_accuracy")
        or metrics.get("waka_meter_constrained_generation_accuracy")
        or {}
    ).get("value")
    return {
        "test_lm_token_nll": (model.get("test_lm_token_nll") or metrics.get("test_lm_token_nll") or {}).get("value"),
        "grammar_score": grammar_score,
        "waka_score": waka_generation_score,
        "waka_static_score": waka_static_score,
        "waka_generation_score": waka_generation_score,
        "morphology_score": (metrics.get("morphology_adversarial_accuracy") or {}).get("value"),
        "eval_contamination_hits": contamination.get("hits"),
        "eval_source_overlap_hits": source_overlap.get("hits"),
        "split_leaks": leakage.get("leaks"),
        "waka_leaks": leakage.get("waka_leaks"),
        "tokenizer_leakage": tokenizer.get("forbidden_heldout_tokenizer_leakage"),
        "source_record_audit": eval_payload.get("source_record_audit") or "see_quality_log",
    }


def source_quality_status(root: Path) -> dict[str, Any]:
    payload = load_json_if_exists(root / "logs" / "source_quality_board.json")
    if not payload:
        return {
            "source_quality_present": False,
            "source_quality_average": None,
            "source_quality_hard_blocker_rows": None,
        }
    return {
        "source_quality_present": True,
        "source_quality_average": payload.get("average_included_score"),
        "source_quality_hard_blocker_rows": payload.get("hard_blocker_rows"),
    }


def generation_diagnostic_status(root: Path, run_id: str) -> dict[str, Any]:
    policy = load_json_if_exists(root / "data" / "rules" / "generation_diagnostic_policy.json")
    plan_path = root / "logs" / "generation_diagnostics" / f"{run_id}_generation_diagnostic_plan.json"
    return {
        "generation_diagnostic_policy_present": bool(policy),
        "generation_diagnostic_release_metric": policy.get("release_metric") if policy else None,
        "generation_diagnostic_plan_present": plan_path.exists(),
    }


def classify_run(root: Path, run_id: str) -> RunClassification:
    log_path = root / "logs" / f"{run_id}.out.log"
    stderr_path = root / "logs" / f"{run_id}.err.log"
    launch_out_path = root / "logs" / f"launch_{run_id}.out.log"
    launch_err_path = root / "logs" / f"launch_{run_id}.err.log"
    watch_start_out_path = root / "logs" / f"watch_start_{run_id}.out.log"
    watch_start_err_path = root / "logs" / f"watch_start_{run_id}.err.log"
    sentinel_path = root / "logs" / f"train_exit_{run_id}.json"
    checkpoint_path = root / "checkpoints" / f"{run_id}_best.pt"
    active_lock_name = "active_old_japanese_0_1b_cuda.lock" if backend_for_run_id(run_id) == "cuda" else "active_old_japanese_0_1b_dml.lock"
    active_lock_path = root / "logs" / active_lock_name
    active_lock: dict[str, Any] = {}
    active_lock_live = False
    if active_lock_path.exists():
        try:
            payload = read_json(active_lock_path)
        except (json.JSONDecodeError, OSError):
            payload = {}
        if str(payload.get("run_id") or "") == run_id:
            active_lock = payload
            active_lock_live = any(
                process_matches_run(safe_int(payload.get(key), 0) or 0, run_id)
                for key in ("train_pid", "watcher_pid", "launcher_pid")
            )
            if not active_lock_live:
                state = str(payload.get("state") or "")
                launcher_pid = safe_int(payload.get("launcher_pid"), 0) or 0
                train_pid = safe_int(payload.get("train_pid"), 0) or 0
                watcher_pid = safe_int(payload.get("watcher_pid"), 0) or 0
                if state in {"launching", "train_started_watcher_pending", "running"} and recent_active_lock(payload):
                    active_lock_live = (
                        process_matches_launcher(launcher_pid)
                        or process_matches_run(train_pid, run_id)
                        or process_matches_run(watcher_pid, run_id)
                        or process_exists(launcher_pid)
                        or process_exists(train_pid)
                        or process_exists(watcher_pid)
                    )
    live_process = active_lock_live or live_process_for_run(run_id)
    eval_path, eval_payload = load_eval_json(root, run_id)
    train_log = parse_training_log(log_path)
    latest = train_log.get("latest") or {}
    best = train_log.get("best") or {}
    active_policy_issues = active_run_release_policy_issues(root, run_id, train_log) if live_process and not sentinel_path.exists() else []
    eval_metrics = score_from_eval(eval_payload)
    governance_metrics = {
        **source_quality_status(root),
        **generation_diagnostic_status(root, run_id),
    }
    hard_blockers: list[str] = []
    soft_warnings: list[str] = []

    if is_internal_evidence_run(run_id):
        if eval_payload and eval_payload.get("hf_export"):
            hard_blockers.append("hf_export_was_requested_or_recorded")
        hard_blockers.append("internal_negative_test_evidence")
        return {
            "run_id": run_id,
            "status": "internal_evidence",
            "backend": backend_for_run_id(run_id),
            "params": None,
            "vocab_size": None,
            "best_step": None,
            "best_val_loss": None,
            "latest_step": None,
            "latest_val_loss": None,
            **eval_metrics,
            **governance_metrics,
            "checkpoint": checkpoint_path.name if checkpoint_path.exists() else "",
            "checkpoint_sha256": eval_payload.get("checkpoint_sha256", "") if eval_payload else "",
            "checkpoint_bytes": checkpoint_path.stat().st_size if checkpoint_path.exists() else 0,
            "eval_json": eval_path.name if eval_payload else "",
            "train_exit_sentinel": sentinel_path.name if sentinel_path.exists() else "",
            "release_status": "internal_failtest_evidence",
            "upload_ready": False,
            "overall_score": 0.0,
            "hard_blockers": hard_blockers,
            "soft_warnings": soft_warnings,
            "next_action": "ignore_for_release",
            "training_log": train_log,
            "stderr_bytes": stderr_path.stat().st_size if stderr_path.exists() else 0,
        }

    non_release_record = read_non_release_record(run_id, root)
    durable_non_release = bool(non_release_record) and non_release_record.get("release_status") == "non_release_artifact"
    if run_id in NON_RELEASE_RUNS or durable_non_release:
        hard_blockers.append("known_non_release_artifact")
    if active_policy_issues:
        hard_blockers.extend(active_policy_issues)
    if backend_for_run_id(run_id) == "unknown":
        hard_blockers.append("invalid_or_unknown_run_backend")
    if eval_payload and eval_payload.get("hf_export"):
        hard_blockers.append("hf_export_was_requested_or_recorded")
    if train_log.get("completed") and not sentinel_path.exists():
        hard_blockers.append("missing_train_exit_sentinel")
    sentinel_exit_zero = False
    if sentinel_path.exists():
        try:
            sentinel = read_json(sentinel_path)
        except json.JSONDecodeError:
            hard_blockers.append("invalid_train_exit_sentinel_json")
            sentinel = {}
        if sentinel.get("run_id") and sentinel.get("run_id") != run_id:
            hard_blockers.append("train_exit_sentinel_run_id_mismatch")
        sentinel_exit_zero = safe_int(sentinel.get("exit_code"), -1) == 0
        if not sentinel_exit_zero:
            hard_blockers.append("train_exit_sentinel_nonzero")
        if sentinel.get("hf_export"):
            hard_blockers.append("train_exit_sentinel_reports_hf_export")
    if eval_payload:
        if eval_payload.get("status") != "passed":
            hard_blockers.append("quality_eval_status_not_passed")
        if not eval_payload.get("checkpoint_sha256"):
            hard_blockers.append("missing_checkpoint_sha256")
        if eval_metrics["test_lm_token_nll"] is None:
            hard_blockers.append("missing_checkpoint_bound_test_lm_token_nll")
        if safe_int(eval_metrics["eval_contamination_hits"], 0) != 0:
            hard_blockers.append("eval_contamination_hits_nonzero")
        if safe_int(eval_metrics["eval_source_overlap_hits"], 0) != 0:
            hard_blockers.append("eval_source_overlap_hits_nonzero")
        if safe_int(eval_metrics["split_leaks"], 0) != 0:
            hard_blockers.append("split_leaks_nonzero")
        if safe_int(eval_metrics["waka_leaks"], 0) != 0:
            hard_blockers.append("waka_leaks_nonzero")
        if safe_int(eval_metrics["tokenizer_leakage"], 0) != 0:
            hard_blockers.append("tokenizer_heldout_leakage_nonzero")
    elif sentinel_path.exists() and run_id not in NON_RELEASE_RUNS:
        if not sentinel_exit_zero:
            hard_blockers.append("missing_post_run_eval_json_after_failed_sentinel")
        elif not checkpoint_path.exists():
            hard_blockers.append("missing_exact_best_checkpoint_for_post_run_quality")
        else:
            # This is an actionable state, not a release blocker: the finalizer
            # should run quality checks for the exact best checkpoint.
            pass

    best_val = safe_float(best.get("val_loss"))
    latest_val = safe_float(latest.get("val_loss"))
    latest_train = safe_float(latest.get("train_loss"))
    if best_val is not None and latest_val is not None and latest_val - best_val >= 0.5:
        soft_warnings.append(f"overfitting_gap={latest_val - best_val:.4f}")
    if latest_train is not None and latest_val is not None and latest_val - latest_train >= 2.0:
        soft_warnings.append(f"train_val_gap={latest_val - latest_train:.4f}")
    if train_log.get("overfit_stop_signal"):
        soft_warnings.append(
            "overfit_stop_signal="
            f"non_improving_evals={train_log.get('non_improving_evals')} "
            f"train_val_gap={safe_float(train_log.get('train_val_gap'), 0.0):.4f}"
        )
    stderr_bytes = stderr_path.stat().st_size if stderr_path.exists() else 0
    launch_stderr_bytes = launch_err_path.stat().st_size if launch_err_path.exists() else 0
    watch_start_stderr_bytes = watch_start_err_path.stat().st_size if watch_start_err_path.exists() else 0
    if stderr_bytes > 8:
        soft_warnings.append(f"stderr_bytes={stderr_bytes}")
    if launch_stderr_bytes > 8:
        soft_warnings.append(f"launch_stderr_bytes={launch_stderr_bytes}")
    if watch_start_stderr_bytes > 8:
        soft_warnings.append(f"watch_start_stderr_bytes={watch_start_stderr_bytes}")

    release_gate_verified = False
    release_gate_error = ""
    if eval_payload and sentinel_path.exists() and checkpoint_path.exists() and not hard_blockers and run_id not in NON_RELEASE_RUNS:
        gate = subprocess.run(
            [
                sys.executable,
                str(root / "scripts" / "check_release_gate.py"),
                "--run-id",
                run_id,
                "--checkpoint",
                str(checkpoint_path),
            ],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
            timeout=300,
        )
        release_gate_verified = gate.returncode == 0
        if not release_gate_verified:
            release_gate_error = (gate.stderr.strip() or gate.stdout.strip() or f"exit={gate.returncode}").splitlines()[-1]
            hard_blockers.append(f"release_gate_failed:{release_gate_error}")

    if run_id in NON_RELEASE_RUNS or durable_non_release:
        release_status = "non_release_artifact"
        upload_ready = False
        next_action = "ignore_for_release"
    elif active_lock and active_lock_live and not train_log.get("exists") and not sentinel_path.exists():
        release_status = "training_active_or_unverified"
        upload_ready = False
        next_action = "monitor"
    elif (launch_out_path.exists() or launch_err_path.exists() or watch_start_out_path.exists() or watch_start_err_path.exists()) and not train_log.get("exists") and not sentinel_path.exists():
        release_status = "not_upload_ready"
        upload_ready = False
        hard_blockers.append("launch_started_but_training_log_and_sentinel_missing")
        next_action = "investigate_failed_or_stuck_launch"
    elif not train_log.get("completed") and not sentinel_path.exists() and live_process:
        upload_ready = False
        if active_policy_issues:
            release_status = "training_active_superseded_non_release"
            next_action = "supersede_non_release_run"
        elif train_log.get("overfit_stop_signal"):
            release_status = "training_active_overfit_stop_recommended"
            next_action = "stop_overfit_run"
        else:
            release_status = "training_active_or_unverified"
            next_action = "monitor"
    elif train_log.get("exists") and not train_log.get("completed") and not sentinel_path.exists() and not live_process:
        release_status = "not_upload_ready"
        upload_ready = False
        hard_blockers.append("stale_incomplete_training_log_without_live_process")
        next_action = "investigate_missing_train_exit_sentinel"
    elif train_log.get("completed") and not sentinel_path.exists():
        release_status = "not_upload_ready"
        upload_ready = False
        next_action = "investigate_missing_train_exit_sentinel"
    elif sentinel_path.exists() and not eval_payload and not hard_blockers:
        release_status = "needs_post_run_quality"
        upload_ready = False
        next_action = "run_post_run_quality_checks"
    elif hard_blockers:
        release_status = "not_upload_ready"
        upload_ready = False
        next_action = "fix_blockers_then_rerun_checks"
    elif release_gate_verified:
        release_status = "upload_ready_not_exported"
        upload_ready = True
        next_action = "await_explicit_hf_export_request"
    else:
        release_status = "not_upload_ready"
        upload_ready = False
        hard_blockers.append("release_gate_not_verified")
        next_action = "run_or_fix_release_gate"

    score = 100.0
    if run_id in NON_RELEASE_RUNS or durable_non_release:
        score = 0.0
    score -= 18.0 * len([item for item in hard_blockers if item != "known_non_release_artifact"])
    score -= 4.0 * len(soft_warnings)
    test_loss = safe_float(eval_metrics.get("test_lm_token_nll"))
    if test_loss is not None:
        score -= max(0.0, test_loss - 4.0) * 5.0
    if release_status == "training_active_or_unverified":
        score = min(score, 65.0)
    score = max(0.0, round(score, 2))

    if sentinel_path.exists():
        status = "completed"
    elif train_log.get("completed"):
        status = "completed_unverified"
    elif live_process:
        status = "running"
    elif train_log.get("exists"):
        status = "stale_incomplete"
    elif launch_out_path.exists() or launch_err_path.exists() or watch_start_out_path.exists() or watch_start_err_path.exists():
        status = "launch_incomplete"
    else:
        status = "unknown"

    return {
        "run_id": run_id,
        "status": status,
        "backend": backend_for_run_id(run_id),
        "params": train_log.get("config", {}).get("params"),
        "vocab_size": train_log.get("config", {}).get("vocab_size"),
        "block_size": train_log.get("config", {}).get("block_size"),
        "best_step": best.get("step"),
        "best_val_loss": best.get("val_loss"),
        "latest_step": latest.get("step"),
        "latest_val_loss": latest.get("val_loss"),
        **eval_metrics,
        **governance_metrics,
        "checkpoint": checkpoint_path.name if checkpoint_path.exists() else "",
        "checkpoint_sha256": eval_payload.get("checkpoint_sha256", "") if eval_payload else "",
        "checkpoint_bytes": checkpoint_path.stat().st_size if checkpoint_path.exists() else 0,
        "eval_json": eval_path.name if eval_payload else "",
        "train_exit_sentinel": sentinel_path.name if sentinel_path.exists() else "",
        "active_lock": active_lock_path.name if active_lock else "",
        "active_lock_live": active_lock_live,
        "release_gate_verified": release_gate_verified,
        "release_gate_error": release_gate_error,
        "release_status": release_status,
        "upload_ready": upload_ready,
        "overall_score": score,
        "hard_blockers": hard_blockers,
        "soft_warnings": soft_warnings,
        "active_policy_issues": active_policy_issues,
        "next_action": next_action,
        "training_log": train_log,
        "stderr_bytes": stderr_bytes,
    }


def public_board_row(run: RunClassification) -> BoardRunRow:
    keys = [
        "run_id",
        "status",
        "backend",
        "params",
        "vocab_size",
        "block_size",
        "best_step",
        "best_val_loss",
        "latest_step",
        "latest_val_loss",
        "test_lm_token_nll",
        "grammar_score",
        "waka_score",
        "waka_static_score",
        "waka_generation_score",
        "morphology_score",
        "eval_contamination_hits",
        "eval_source_overlap_hits",
        "split_leaks",
        "waka_leaks",
        "tokenizer_leakage",
        "source_quality_present",
        "source_quality_average",
        "source_quality_hard_blocker_rows",
        "generation_diagnostic_policy_present",
        "generation_diagnostic_release_metric",
        "generation_diagnostic_plan_present",
        "checkpoint_sha256",
        "release_status",
        "upload_ready",
        "overall_score",
        "hard_blockers",
        "soft_warnings",
        "active_policy_issues",
        "next_action",
    ]
    return {key: run.get(key) for key in keys}


def load_board(root: Path) -> EvaluationBoard | None:
    path = root / "logs" / "evaluation_board.json"
    if not path.exists():
        return None
    return read_json(path)


def select_next_action(board: EvaluationBoard) -> AutonomousAction:
    global_blockers = list(board.get("global_blockers") or [])
    if global_blockers:
        return {
            "action": "fix_blockers",
            "run_id": "",
            "reason": "global_release_loop_blockers_present",
            "hard_blockers": global_blockers,
        }
    runs = board.get("runs", [])
    active = [run for run in runs if run.get("next_action") == "monitor"]
    superseded = [run for run in runs if run.get("next_action") == "supersede_non_release_run"]
    overfit = [run for run in runs if run.get("next_action") == "stop_overfit_run"]
    if superseded:
        superseded.sort(key=lambda item: safe_int(item.get("latest_step"), -1) or -1, reverse=True)
        run = superseded[0]
        return {
            "action": "supersede_non_release_run",
            "run_id": run.get("run_id"),
            "reason": "active_training_run_obsolete_under_current_release_policy",
            "details": {
                "latest_step": run.get("latest_step"),
                "best_step": run.get("best_step"),
                "best_val_loss": run.get("best_val_loss"),
                "latest_val_loss": run.get("latest_val_loss"),
                "active_policy_issues": run.get("active_policy_issues", []),
            },
        }
    if overfit:
        overfit.sort(key=lambda item: safe_int(item.get("latest_step"), -1) or -1, reverse=True)
        run = overfit[0]
        return {
            "action": "stop_overfit_run",
            "run_id": run.get("run_id"),
            "reason": "active_training_run_crossed_overfit_stop_signal",
            "details": {
                "latest_step": run.get("latest_step"),
                "best_step": run.get("best_step"),
                "best_val_loss": run.get("best_val_loss"),
                "latest_val_loss": run.get("latest_val_loss"),
                "soft_warnings": run.get("soft_warnings", []),
            },
        }
    if active:
        active.sort(key=lambda item: safe_int(item.get("latest_step"), -1) or -1, reverse=True)
        run = active[0]
        return {
            "action": "monitor",
            "run_id": run.get("run_id"),
            "reason": "active_training_run_present",
            "details": {
                "latest_step": run.get("latest_step"),
                "best_step": run.get("best_step"),
                "best_val_loss": run.get("best_val_loss"),
                "latest_val_loss": run.get("latest_val_loss"),
                "soft_warnings": run.get("soft_warnings", []),
            },
        }
    needs_quality = [run for run in runs if run.get("next_action") == "run_post_run_quality_checks"]
    if needs_quality:
        run = needs_quality[0]
        return {
            "action": "run_post_run_quality_checks",
            "run_id": run.get("run_id"),
            "reason": "completed_run_missing_eval_json",
        }
    blockers = [
        run
        for run in runs
        if run.get("hard_blockers") and run.get("release_status") not in {"non_release_artifact", "internal_failtest_evidence"}
    ]
    if blockers:
        run = blockers[0]
        return {
            "action": "fix_blockers",
            "run_id": run.get("run_id"),
            "reason": "release_candidate_has_hard_blockers",
            "hard_blockers": run.get("hard_blockers", []),
        }
    ready = [run for run in runs if run.get("upload_ready")]
    if ready:
        ready.sort(key=lambda item: safe_float(item.get("overall_score"), 0.0) or 0.0, reverse=True)
        run = ready[0]
        return {
            "action": "stop_and_report_upload_ready",
            "run_id": run.get("run_id"),
            "reason": "upload_ready_not_exported",
            "overall_score": run.get("overall_score"),
        }
    return {
        "action": "prepare_next_fresh_run_after_static_gate_and_zero_base_reviews",
        "run_id": "",
        "reason": "no_active_or_upload_ready_release_candidate",
    }
