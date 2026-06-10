from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from kobun_llm.checkpoint_io import load_trusted_checkpoint
from kobun_autonomy.release_policy import require_release_candidate_run
from old_japanese_run_intel import startup_mutex_health


DML_RUN_ID_RE = re.compile(r"^old_japanese_0_1b_dml_[0-9A-Za-z][0-9A-Za-z_-]{0,63}$")
CUDA_RUN_ID_RE = re.compile(r"^old_japanese_0_1b_cuda_[0-9A-Za-z][0-9A-Za-z_-]{0,63}$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify that a training run completed before release-quality use.")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--backend", choices=["dml", "cuda"], default="")
    parser.add_argument("--min-step", type=int, default=1)
    parser.add_argument("--require-no-active-process", action="store_true")
    parser.add_argument(
        "--require-no-active-lock",
        action="store_true",
        help="Fail release readiness if any canonical old-japanese active lock still exists.",
    )
    parser.add_argument(
        "--active-process-scope",
        choices=["training", "supervision"],
        default="training",
        help="training ignores watcher/finalizer processes; supervision also blocks same-RunId watcher/finalizer/export-facing supervision.",
    )
    parser.add_argument("--ignore-pid", type=int, default=0, help="Ignore this supervisor PID when scanning live processes.")
    return parser.parse_args()


def repo_path(path: str | Path) -> Path:
    return (Path.cwd() / path).resolve(strict=False)


def normalize_path(path: str | Path) -> str:
    return str(repo_path(path)).rstrip("\\/").casefold()


def parse_timestamp(value: Any, label: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise SystemExit(f"run completion sentinel missing {label}")
    normalized = value.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise SystemExit(f"run completion sentinel has invalid {label}: {value!r}") from exc


def infer_run_id(checkpoint: Path | None, requested: str) -> str:
    if requested:
        return requested
    if checkpoint is None:
        raise SystemExit("--run-id or --checkpoint is required")
    name = checkpoint.stem
    if name.endswith("_best"):
        name = name[: -len("_best")]
    return name


def validate_run_id(run_id: str, backend: str) -> str:
    if run_id.startswith("old_japanese_0_1b_dml_"):
        if not DML_RUN_ID_RE.fullmatch(run_id):
            raise SystemExit(f"invalid DML run id: {run_id!r}")
        if backend and backend != "dml":
            raise SystemExit(f"run id/backend mismatch: run_id={run_id!r} backend={backend!r}")
        return "dml"
    if run_id.startswith("old_japanese_0_1b_cuda_"):
        if not CUDA_RUN_ID_RE.fullmatch(run_id):
            raise SystemExit(f"invalid CUDA run id: {run_id!r}")
        if backend and backend != "cuda":
            raise SystemExit(f"run id/backend mismatch: run_id={run_id!r} backend={backend!r}")
        return "cuda"
    raise SystemExit(
        "invalid release run id: expected old_japanese_0_1b_dml_* or "
        f"old_japanese_0_1b_cuda_*, got {run_id!r}"
    )


def active_markers(scope: str) -> tuple[str, ...]:
    training_markers = (
        "kobun_llm.train",
        "train_old_japanese_0_1b_dml.ps1",
        "train_old_japanese_0_1b_gpu.ps1",
        "start_old_japanese_0_1b_cuda_colab_and_watch.py",
    )
    if scope == "training":
        return training_markers
    return training_markers + (
        "watch_and_finalize_old_japanese_0_1b_dml.ps1",
        "finalize_old_japanese_0_1b_dml.ps1",
        "start_old_japanese_0_1b_dml_and_watch.ps1",
    )


def active_process_command_lines(run_id: str, ignore_pid: int = 0, scope: str = "training") -> list[str]:
    markers = active_markers(scope)
    if sys.platform != "win32":
        proc_root = Path("/proc")
        if not proc_root.exists():
            raise SystemExit("--require-no-active-process cannot scan active processes on this platform")
        lines = []
        for cmdline in proc_root.glob("[0-9]*/cmdline"):
            try:
                pid = int(cmdline.parent.name)
            except ValueError:
                continue
            if ignore_pid and pid == ignore_pid:
                continue
            try:
                raw = cmdline.read_bytes()
            except OSError:
                continue
            text = raw.replace(b"\x00", b" ").decode("utf-8", errors="replace").strip()
            if run_id not in text:
                continue
            if any(marker in text for marker in markers):
                lines.append(text)
        return lines
    command = (
        "Get-CimInstance Win32_Process | "
        "Where-Object { $_.CommandLine -like '*" + run_id.replace("'", "''") + "*'"
        + (f" -and $_.ProcessId -ne {ignore_pid}" if ignore_pid else "")
        + " } | "
        "ForEach-Object { $_.CommandLine }"
    )
    result = subprocess.run(
        ["powershell", "-NoProfile", "-Command", command],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise SystemExit(f"could not scan active processes for run id {run_id}: {result.stderr.strip()}")
    lines = []
    for line in result.stdout.splitlines():
        text = line.strip()
        if not text:
            continue
        if not any(marker in text for marker in markers):
            continue
        lines.append(text)
    return lines


def canonical_active_locks() -> list[Path]:
    locks = [
        path
        for path in (
            Path("logs") / "active_old_japanese_0_1b_dml.lock",
            Path("logs") / "active_old_japanese_0_1b_cuda.lock",
            Path("logs") / "active_old_japanese_0_1b_training.lock",
        )
        if path.exists()
    ]
    for path in sorted(Path("logs").glob("colab_active_old_japanese_0_1b_cuda.*.json")):
        name = path.name
        if ".finished." in name or ".failed_non_release." in name or ".stale." in name:
            continue
        locks.append(path)
    return locks


def main() -> None:
    args = parse_args()
    run_id = infer_run_id(args.checkpoint, args.run_id)
    backend = validate_run_id(run_id, args.backend)
    require_release_candidate_run(run_id, context="run completion gate")
    checkpoint = args.checkpoint or Path("checkpoints") / f"{run_id}_best.pt"
    expected_checkpoint = Path("checkpoints") / f"{run_id}.pt"
    expected_best = Path("checkpoints") / f"{run_id}_best.pt"
    sentinel_path = Path("logs") / f"train_exit_{run_id}.json"
    log_path = Path("logs") / f"{run_id}.out.log"

    if normalize_path(checkpoint) != normalize_path(expected_best):
        raise SystemExit(f"checkpoint must be the exact best checkpoint for this run: expected={expected_best} actual={checkpoint}")
    for label, path in (
        ("sentinel", sentinel_path),
        ("training log", log_path),
        ("final checkpoint", expected_checkpoint),
        ("best checkpoint", expected_best),
    ):
        if not path.exists():
            raise SystemExit(f"missing {label}: {path}")

    sentinel = json.loads(sentinel_path.read_text(encoding="utf-8-sig"))
    if sentinel.get("run_id") != run_id:
        raise SystemExit(f"sentinel run_id mismatch: expected={run_id} actual={sentinel.get('run_id')!r}")
    if int(sentinel.get("exit_code", -1)) != 0:
        raise SystemExit(f"sentinel exit_code is not zero: {sentinel.get('exit_code')!r}")
    if bool(sentinel.get("hf_export")):
        raise SystemExit("sentinel unexpectedly reports hf_export=true")
    if normalize_path(str(sentinel.get("checkpoint") or "")) != normalize_path(expected_checkpoint):
        raise SystemExit("sentinel checkpoint path does not match expected final checkpoint")
    if normalize_path(str(sentinel.get("best_checkpoint") or "")) != normalize_path(expected_best):
        raise SystemExit("sentinel best_checkpoint path does not match expected best checkpoint")

    completed_at = parse_timestamp(sentinel.get("completed_at"), "completed_at")
    log_text = log_path.read_text(encoding="utf-8", errors="replace").replace("\x00", "")
    if f"run_id={run_id} " not in log_text:
        raise SystemExit(f"training log does not prove run id: {log_path}")
    if "launcher_completed=" not in log_text:
        raise SystemExit(f"training log has no launcher_completed marker: {log_path}")
    if completed_at.timestamp() + 2.0 < log_path.stat().st_mtime:
        raise SystemExit("sentinel completed_at predates training log modification")
    if completed_at.timestamp() + 2.0 < expected_checkpoint.stat().st_mtime:
        raise SystemExit("sentinel completed_at predates final checkpoint modification")
    if completed_at.timestamp() + 2.0 < expected_best.stat().st_mtime:
        raise SystemExit("sentinel completed_at predates best checkpoint modification")

    payload = load_trusted_checkpoint(expected_best, map_location="cpu")
    metadata = dict(payload.get("metadata", {}) or {})
    if metadata.get("run_id") != run_id:
        raise SystemExit(f"checkpoint metadata run_id mismatch: expected={run_id} actual={metadata.get('run_id')!r}")
    if metadata.get("backend") != backend:
        raise SystemExit(f"checkpoint metadata backend mismatch: expected={backend} actual={metadata.get('backend')!r}")
    final_payload = load_trusted_checkpoint(expected_checkpoint, map_location="cpu")
    final_metadata = dict(final_payload.get("metadata", {}) or {})
    if final_metadata.get("run_id") != run_id:
        raise SystemExit(
            f"final checkpoint metadata run_id mismatch: expected={run_id} actual={final_metadata.get('run_id')!r}"
        )
    if final_metadata.get("backend") != backend:
        raise SystemExit(
            f"final checkpoint metadata backend mismatch: expected={backend} actual={final_metadata.get('backend')!r}"
        )
    step = int(payload.get("step", -1))
    if step < args.min_step:
        raise SystemExit(f"checkpoint step is below release completion threshold: step={step} min_step={args.min_step}")

    if args.require_no_active_process:
        active = active_process_command_lines(run_id, ignore_pid=args.ignore_pid, scope=args.active_process_scope)
        if active:
            preview = "\n".join(active[:5])
            raise SystemExit(f"run still has active {args.active_process_scope} processes for {run_id}:\n{preview}")
    if args.require_no_active_lock:
        startup_health = startup_mutex_health(Path.cwd())
        startup_blockers = list(startup_health.get("hard_blockers") or [])
        if startup_blockers or startup_health.get("exists"):
            raise SystemExit(
                "startup mutex still exists during release completion gate:\n"
                + json.dumps(startup_health, ensure_ascii=False, indent=2)
            )
        locks = canonical_active_locks()
        if locks:
            details: list[str] = []
            for path in locks:
                try:
                    payload = json.loads(path.read_text(encoding="utf-8-sig"))
                    details.append(f"{path}: run_id={payload.get('run_id')!r} backend={payload.get('backend')!r} state={payload.get('state')!r}")
                except Exception as exc:
                    details.append(f"{path}: unreadable active lock: {exc}")
            raise SystemExit(
                "canonical active lock still exists during release completion gate:\n" + "\n".join(details)
            )

    print(f"run_completion_ok=true run_id={run_id} backend={backend} step={step} sentinel={sentinel_path}")


if __name__ == "__main__":
    main()
