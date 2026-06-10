from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from datetime import timedelta
from pathlib import Path
from typing import Any

from old_japanese_run_intel import active_lock_health, colab_cuda_lease_health, startup_mutex_health


ROOT = Path(__file__).resolve().parents[1]
RUN_ID_RE = re.compile(r"^old_japanese_0_1b_cuda_[0-9A-Za-z][0-9A-Za-z_-]{0,63}$")
RUN_PATTERN = re.compile(r"old_japanese_0_1b_(?:dml|cuda)_[0-9A-Za-z][0-9A-Za-z_-]{0,63}")
DEVICE_RE = re.compile(r'(?i)(?:^|\s)--device(?:\s+|=)(?P<device>[^\s"\']+)')
SUPERVISED_MARKERS = (
    "kobun_llm.train",
    "train_old_japanese_0_1b_dml.ps1",
    "watch_and_finalize_old_japanese_0_1b_dml.ps1",
    "finalize_old_japanese_0_1b_dml.ps1",
    "start_old_japanese_0_1b_dml_and_watch.ps1",
    "start_old_japanese_0_1b_cuda_colab_and_watch.py",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_id_now() -> str:
    return "old_japanese_0_1b_cuda_" + datetime.now().strftime("%Y%m%d_%H%M%S")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Supervised CUDA launcher for old-japanese-0.1B.")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--preflight-gate", default="logs/preflight_gate_old_japanese_0_1b.json")
    parser.add_argument("--review-gate", default="logs/zero_base_review_gate_old_japanese_0_1b.json")
    parser.add_argument(
        "--cuda-provider",
        choices=["colab", "gcp"],
        default="colab",
        help="Execution provider for the supervised CUDA backend. GCP uses the same release gates as Colab.",
    )
    parser.add_argument("--allow-start-training", action="store_true")
    parser.add_argument("--reviews-passed", action="store_true")
    parser.add_argument("--steps", type=int, default=8000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--skip-post-run-quality", action="store_true", help="Dry-run only: forbid success and skip CUDA quality checks.")
    parser.add_argument("--allow-no-cuda", action="store_true", help="Dry-run gate checks without starting training.")
    return parser.parse_args()


def run_checked(command: list[str], *, env: dict[str, str] | None = None) -> str:
    completed = subprocess.run(command, text=True, capture_output=True, check=False, env=env)
    output = completed.stdout + completed.stderr
    if output:
        print(output, end="" if output.endswith("\n") else "\n")
    if completed.returncode != 0:
        raise SystemExit(f"command failed exit={completed.returncode}: {' '.join(command)}")
    return output


def ensure_under_repo(project_root: Path, path_text: str) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = project_root / path
    resolved = path.resolve()
    try:
        resolved.relative_to(project_root)
    except ValueError as exc:
        raise SystemExit(f"path escapes project root: {path_text}") from exc
    return resolved


def write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    with tmp.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(tmp, path)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def command_is_scanner(text: str) -> bool:
    return "Get-CimInstance Win32_Process" in text and "Where-Object" in text


def cuda_like_train_command(text: str) -> bool:
    if "kobun_llm.train" not in text:
        return False
    matches = list(DEVICE_RE.finditer(text))
    if not matches:
        # The train CLI default is auto, which can select CUDA in Colab.
        return True
    device = matches[-1].group("device").strip().strip("\"'").lower()
    return device in {"cuda", "auto"}


def supervised_wrapper_command(text: str) -> bool:
    if any(marker in text for marker in SUPERVISED_MARKERS if marker != "kobun_llm.train") and RUN_PATTERN.search(text):
        return True
    return cuda_like_train_command(text)


def cuda_lease_patterns() -> tuple[str, ...]:
    return (
        "logs/colab_active_old_japanese_0_1b_cuda.*.json",
        "logs/gcp_active_old_japanese_0_1b_cuda.*.json",
    )


def active_locks(project_root: Path) -> list[Path]:
    locks = [
        path
        for pattern in ("logs/active_old_japanese_0_1b_dml.lock", "logs/active_old_japanese_0_1b_cuda.lock")
        for path in [project_root / pattern]
        if path.exists()
    ]
    for pattern in cuda_lease_patterns():
        for path in sorted(project_root.glob(pattern)):
            name = path.name
            if ".finished." in name or ".failed_non_release." in name or ".stale." in name:
                continue
            locks.append(path)
    return locks


def quarantine_stale_active_locks(project_root: Path) -> None:
    health = active_lock_health(project_root)
    blockers = list(health.get("hard_blockers") or [])
    if blockers:
        raise SystemExit("active lock health blockers: " + ", ".join(str(item) for item in blockers))
    quarantined = health.get("quarantined_path")
    if quarantined:
        print(f"quarantined_stale_active_lock={quarantined}")


def assert_no_active_colab_cuda_lease(project_root: Path) -> None:
    health = colab_cuda_lease_health(project_root)
    blockers = list(health.get("hard_blockers") or [])
    if blockers or health.get("exists"):
        raise SystemExit("colab CUDA lease health blockers: " + json.dumps(health, ensure_ascii=False))
    quarantined = health.get("quarantined")
    if quarantined:
        print(f"quarantined_colab_cuda_lease={quarantined}")


def quarantine_stale_startup_mutex(project_root: Path) -> None:
    health = startup_mutex_health(project_root)
    blockers = list(health.get("hard_blockers") or [])
    if blockers:
        raise SystemExit("startup mutex health blockers: " + ", ".join(str(item) for item in blockers))
    if health.get("exists"):
        raise SystemExit("startup mutex still exists: " + json.dumps(health, ensure_ascii=False))
    quarantined = health.get("quarantined_path")
    if quarantined:
        print(f"quarantined_stale_startup_mutex={quarantined}")


def supervised_training_processes(run_id: str = "", ignore_pids: set[int] | None = None) -> list[str]:
    ignore_pids = ignore_pids or set()
    lines: list[str] = []
    if sys.platform == "win32":
        ps = (
            "Get-CimInstance Win32_Process | "
            "Where-Object { $_.CommandLine -match 'old_japanese_0_1b_(dml|cuda)_|kobun_llm\\.train|start_old_japanese_0_1b_cuda_colab_and_watch\\.py|start_old_japanese_0_1b_dml_and_watch\\.ps1|train_old_japanese_0_1b_dml\\.ps1|watch_and_finalize_old_japanese_0_1b_dml\\.ps1|finalize_old_japanese_0_1b_dml\\.ps1' } | "
            "ForEach-Object { [pscustomobject]@{ProcessId=$_.ProcessId; CommandLine=$_.CommandLine} | ConvertTo-Json -Compress }"
        )
        result = subprocess.run(["powershell", "-NoProfile", "-Command", ps], text=True, capture_output=True, check=False)
        if result.returncode != 0:
            raise SystemExit(f"could not scan supervised training processes: {result.stderr.strip()}")
        records = [line for line in result.stdout.splitlines() if line.strip()]
        for line in records:
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            pid = int(payload.get("ProcessId") or 0)
            text = str(payload.get("CommandLine") or "")
            if pid in ignore_pids:
                continue
            if command_is_scanner(text):
                continue
            if supervised_wrapper_command(text):
                lines.append(f"pid={pid} command={text}")
        return lines

    proc_root = Path("/proc")
    if not proc_root.exists():
        return []
    for cmdline in proc_root.glob("[0-9]*/cmdline"):
        try:
            pid = int(cmdline.parent.name)
        except ValueError:
            continue
        if pid in ignore_pids:
            continue
        try:
            text = cmdline.read_bytes().replace(b"\x00", b" ").decode("utf-8", errors="replace").strip()
        except OSError:
            continue
        if command_is_scanner(text):
            continue
        if supervised_wrapper_command(text):
            lines.append(f"pid={pid} command={text}")
    return lines


def current_process_tree_ids(limit: int = 12) -> set[int]:
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
                text=True,
                capture_output=True,
                check=False,
            )
            try:
                parent = int(result.stdout.strip() or "0")
            except ValueError:
                parent = 0
        else:
            stat_path = Path("/proc") / str(pid) / "stat"
            try:
                parent = int(stat_path.read_text(encoding="utf-8", errors="replace").split()[3])
            except (OSError, ValueError, IndexError):
                parent = 0
        if parent <= 0 or parent in ignored:
            break
        ignored.add(parent)
        pid = parent
    return ignored


def assert_no_other_supervised_training(run_id: str) -> None:
    live = supervised_training_processes(run_id=run_id, ignore_pids=current_process_tree_ids())
    if live:
        raise SystemExit("refusing to start CUDA run while another supervised old_japanese_0_1b process is live:\n" + "\n".join(live[:5]))


def write_train_exit_sentinel(train_exit: Path, run_id: str, exit_code: int, message: str, out: str, best_out: str) -> None:
    train_exit.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "exit_code": exit_code,
                "message": message,
                "completed_at": datetime.now().astimezone().isoformat(),
                "checkpoint": out,
                "best_checkpoint": best_out,
                "hf_export": False,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def acquire_lock(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def update_owned_lock(path: Path, run_id: str, launcher_pid: int, launch_token_sha256: str, payload: dict[str, Any]) -> None:
    existing = read_json(path)
    if (
        existing.get("run_id") != run_id
        or int(existing.get("launcher_pid") or 0) != launcher_pid
        or existing.get("launch_token_sha256") != launch_token_sha256
    ):
        raise SystemExit("refusing to update active CUDA lock not owned by this launcher")
    write_json_atomic(path, payload)


def remove_owned_lock(path: Path, run_id: str, launcher_pid: int, launch_token_sha256: str) -> bool:
    if not path.exists():
        return False
    existing = read_json(path)
    if (
        existing.get("run_id") != run_id
        or int(existing.get("launcher_pid") or 0) != launcher_pid
        or existing.get("launch_token_sha256") != launch_token_sha256
    ):
        raise SystemExit("refusing to remove CUDA startup lock not owned by this launcher")
    path.unlink()
    return True


def archive_owned_lock(
    path: Path,
    run_id: str,
    launcher_pid: int,
    launch_token_sha256: str,
    *,
    state: str,
    reason: str,
) -> Path | None:
    if not path.exists():
        return None
    try:
        existing = read_json(path)
    except Exception:
        return None
    if (
        existing.get("run_id") == run_id
        and int(existing.get("launcher_pid") or 0) == launcher_pid
        and existing.get("launch_token_sha256") == launch_token_sha256
    ):
        existing["state"] = state
        existing["final_state"] = state
        existing["final_reason"] = reason
        existing["archived_at"] = datetime.now().astimezone().isoformat()
        existing["hf_export"] = False
        suffix = "finished" if state == "finished" else "failed_non_release"
        archive = path.with_name(f"active_old_japanese_0_1b_cuda.{run_id}.{suffix}.{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        write_json_atomic(archive, existing)
        path.unlink()
        return archive
    return None


def colab_lease_path(project_root: Path, run_id: str) -> Path:
    return project_root / "logs" / f"colab_active_old_japanese_0_1b_cuda.{run_id}.json"


def cuda_lease_path(project_root: Path, run_id: str, provider: str) -> Path:
    if provider == "gcp":
        return project_root / "logs" / f"gcp_active_old_japanese_0_1b_cuda.{run_id}.json"
    return colab_lease_path(project_root, run_id)


def lease_payload(
    *,
    run_id: str,
    project_root: Path,
    state: str,
    active_lock: Path,
    train_pid: int | None,
    preflight: Path,
    review: Path,
    context_path: Path,
    cuda_provider: str,
    expires_minutes: int = 360,
) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    schema = (
        "old_japanese_0_1b_colab_cuda_active_lease_v1"
        if cuda_provider == "colab"
        else "old_japanese_0_1b_supervised_cuda_active_lease_v1"
    )
    return {
        "schema": schema,
        "run_id": run_id,
        "backend": "cuda",
        "cuda_provider": cuda_provider,
        "state": state,
        "created_or_updated_at_utc": now.isoformat(),
        "heartbeat_at_utc": now.isoformat(),
        "lease_expires_at_utc": (now + timedelta(minutes=expires_minutes)).isoformat(),
        "project_root": str(project_root),
        "artifact_root": str(project_root),
        "active_lock": active_lock.relative_to(project_root).as_posix(),
        "train_pid": train_pid,
        "preflight_gate": preflight.relative_to(project_root).as_posix(),
        "preflight_gate_sha256": sha256_file(preflight),
        "review_gate": review.relative_to(project_root).as_posix(),
        "review_gate_sha256": sha256_file(review),
        "autonomous_launch_context": context_path.relative_to(project_root).as_posix(),
        "autonomous_launch_context_sha256": sha256_file(context_path),
        "hf_export": False,
        "package_created": False,
        "upload_attempted": False,
        "google_credentials_read": False,
    }


def write_colab_lease(path: Path, payload: dict[str, Any]) -> None:
    write_json_atomic(path, payload)


def refresh_colab_lease(
    path: Path,
    *,
    run_id: str,
    project_root: Path,
    state: str,
    active_lock: Path,
    train_pid: int | None,
    preflight: Path,
    review: Path,
    context_path: Path,
    cuda_provider: str,
) -> None:
    write_colab_lease(
        path,
        lease_payload(
            run_id=run_id,
            project_root=project_root,
            state=state,
            active_lock=active_lock,
            train_pid=train_pid,
            preflight=preflight,
            review=review,
            context_path=context_path,
            cuda_provider=cuda_provider,
        ),
    )


def archive_colab_lease(path: Path, *, state: str, reason: str) -> Path | None:
    if not path.exists():
        return None
    payload = read_json(path)
    payload["state"] = state
    payload["final_state"] = state
    payload["final_reason"] = reason
    payload["archived_at_utc"] = utc_now()
    payload["hf_export"] = False
    suffix = "finished" if state == "finished" else "failed_non_release"
    run_id = str(payload.get("run_id") or "unknown")
    prefix = "gcp_active_old_japanese_0_1b_cuda" if path.name.startswith("gcp_") else "colab_active_old_japanese_0_1b_cuda"
    archive = path.with_name(f"{prefix}.{run_id}.{suffix}.{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    write_json_atomic(archive, payload)
    path.unlink()
    return archive


def write_non_release_record(
    project_root: Path,
    *,
    run_id: str,
    reason: str,
    train_exit: Path,
    active_lock_archive: Path | None,
) -> None:
    record_dir = project_root / "logs" / "non_release_runs"
    record_dir.mkdir(parents=True, exist_ok=True)
    train_exit_path = train_exit if train_exit.is_absolute() else project_root / train_exit
    sentinel_sha256 = sha256_file(train_exit_path) if train_exit_path.exists() else ""
    archive_rel = ""
    archive_sha256 = ""
    archive_name = ""
    if active_lock_archive is not None and active_lock_archive.exists():
        archive_rel = active_lock_archive.relative_to(project_root).as_posix()
        archive_sha256 = sha256_file(active_lock_archive)
        archive_name = active_lock_archive.name
    payload = {
        "run_id": run_id,
        "release_status": "non_release_artifact",
        "reason": reason,
        "created_at": datetime.now().astimezone().isoformat(),
        "train_exit_sentinel": train_exit_path.relative_to(project_root).as_posix(),
        "train_exit_sentinel_sha256": sentinel_sha256,
        "active_lock_archive": archive_name,
        "active_lock_archive_sha256": archive_sha256,
        "source_archive_path": archive_rel,
        "source_archive_sha256": archive_sha256,
        "hf_export": False,
    }
    record_path = record_dir / f"{run_id}.json"
    record_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_snapshot_output(output: str) -> tuple[str, str, str, list[str], list[str]]:
    train = val = test = ""
    tokenizer: list[str] = []
    provenance: list[str] = []
    for line in output.splitlines():
        if line.startswith("snapshot_train_data="):
            train = line.split("=", 1)[1].strip()
        elif line.startswith("snapshot_val_data="):
            val = line.split("=", 1)[1].strip()
        elif line.startswith("snapshot_test_data="):
            test = line.split("=", 1)[1].strip()
        elif line.startswith("snapshot_tokenizer_extra_data="):
            tokenizer.append(line.split("=", 1)[1].strip())
        elif line.startswith("snapshot_provenance_file="):
            provenance.append(line.split("=", 1)[1].strip())
    if not train or not val or not test or not tokenizer:
        raise SystemExit("snapshot output missing required train/validation/test/tokenizer paths")
    return train, val, test, tokenizer, provenance


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    os.chdir(project_root)
    cuda_provider = args.cuda_provider
    cuda_context = f"supervised CUDA {cuda_provider}"
    if not args.run_id:
        raise SystemExit("Refusing supervised CUDA launch without explicit --run-id.")
    if args.skip_post_run_quality and not args.allow_no_cuda:
        raise SystemExit("Refusing real supervised CUDA launch with --skip-post-run-quality; that flag is dry-run only.")
    run_id = args.run_id
    if not RUN_ID_RE.fullmatch(run_id):
        raise SystemExit(f"invalid CUDA RunId: {run_id}")
    if not args.allow_start_training:
        raise SystemExit("Refusing supervised CUDA launch without --allow-start-training.")
    if not args.reviews_passed:
        raise SystemExit("Refusing supervised CUDA launch without --reviews-passed.")

    py = sys.executable
    run_checked([py, "-c", "from kobun_autonomy.release_policy import require_release_candidate_run; import sys; require_release_candidate_run(sys.argv[1], context=sys.argv[2])", run_id, cuda_context])
    run_checked([py, "scripts/verify_preflight_gate.py", "--gate", args.preflight_gate, "--max-age-minutes", "240"])
    run_checked([py, "scripts/verify_zero_base_review_gate.py", "--gate", args.review_gate, "--preflight-gate", args.preflight_gate, "--max-age-minutes", "240"])
    run_checked(
        [
            py,
            "scripts/check_colab_cuda_environment.py",
            "--project-root",
            str(project_root),
            "--preflight-gate",
            args.preflight_gate,
            "--review-gate",
            args.review_gate,
            "--cuda-provider",
            cuda_provider,
        ]
        + (["--allow-no-cuda"] if args.allow_no_cuda else [])
    )
    run_checked([py, "scripts/assert_run_id_unused.py", "--run-id", run_id])
    assert_no_active_colab_cuda_lease(project_root)
    assert_no_other_supervised_training(run_id)
    quarantine_stale_startup_mutex(project_root)
    quarantine_stale_active_locks(project_root)
    existing_locks = active_locks(project_root)
    if existing_locks:
        raise SystemExit(f"refusing to start CUDA run while active lock exists: {existing_locks}")
    if args.allow_no_cuda:
        print(f"supervised_cuda_supervisor_dry_run_ok=true provider={cuda_provider} run_id={run_id}")
        return

    preflight = ensure_under_repo(project_root, args.preflight_gate)
    review = ensure_under_repo(project_root, args.review_gate)
    launch_nonce = hashlib.sha256(os.urandom(32)).hexdigest()
    launch_token = hashlib.sha256(os.urandom(32)).hexdigest() + hashlib.sha256(os.urandom(32)).hexdigest()
    launch_token_sha256 = sha256_text(launch_token)
    context_rel = (
        f"logs/colab_cuda_launch_context_{run_id}.json"
        if cuda_provider == "colab"
        else f"logs/gcp_cuda_launch_context_{run_id}.json"
    )
    context_path = project_root / context_rel
    context_schema = (
        "old_japanese_0_1b_cuda_colab_launch_context_v1"
        if cuda_provider == "colab"
        else "old_japanese_0_1b_supervised_cuda_launch_context_v1"
    )
    selected_action = "colab_cuda_supervised_training" if cuda_provider == "colab" else "supervised_cuda_training"
    context = {
        "schema": context_schema,
        "generated_at_utc": utc_now(),
        "run_id": run_id,
        "backend": "cuda",
        "cuda_provider": cuda_provider,
        "selected_action": selected_action,
        "launcher_pid": os.getpid(),
        "autonomous_pid": os.getpid(),
        "autonomous_script": "scripts/start_old_japanese_0_1b_cuda_colab_and_watch.py",
        "preflight_gate": args.preflight_gate,
        "preflight_gate_sha256": sha256_file(preflight),
        "review_gate": args.review_gate,
        "review_gate_sha256": sha256_file(review),
        "launch_nonce_sha256": sha256_text(launch_nonce),
        "hf_export": False,
        "google_credentials_read": False,
    }
    write_json_exclusive(context_path, context)
    run_checked([py, "scripts/assert_run_id_unused.py", "--run-id", run_id, "--allow-supervisor-launch-artifacts"])

    out = f"checkpoints/{run_id}.pt"
    best_out = f"checkpoints/{run_id}_best.pt"
    stdout_log = Path("logs") / f"{run_id}.out.log"
    stderr_log = Path("logs") / f"{run_id}.err.log"
    train_exit = Path("logs") / f"train_exit_{run_id}.json"
    startup_lock = project_root / "logs/active_old_japanese_0_1b_training.lock"
    active_lock = project_root / "logs/active_old_japanese_0_1b_cuda.lock"
    active_lease = cuda_lease_path(project_root, run_id, cuda_provider)
    lock_base = {
        "run_id": run_id,
        "backend": "cuda",
        "cuda_provider": cuda_provider,
        "launcher_pid": os.getpid(),
        "watcher_pid": None,
        "created_at": datetime.now().astimezone().isoformat(),
        "launch_token_sha256": launch_token_sha256,
        "launch_nonce_sha256": sha256_text(launch_nonce),
        "preflight_gate": args.preflight_gate,
        "preflight_gate_sha256": sha256_file(preflight),
        "review_gate": args.review_gate,
        "review_gate_sha256": sha256_file(review),
        "autonomous_launch_context": context_rel,
        "autonomous_launch_context_sha256": sha256_file(context_path),
        "autonomous_pid": os.getpid(),
        "autonomous_script": "scripts/start_old_japanese_0_1b_cuda_colab_and_watch.py",
        "selected_action": selected_action,
        "hf_export": False,
    }

    proc: subprocess.Popen[str] | None = None
    exit_code = 1
    message = "startup failed before training process exit"
    failure_reason = ""
    lease_archive: Path | None = None
    try:
        quarantine_stale_startup_mutex(project_root)
        if active_lease.exists():
            raise SystemExit(f"refusing to reuse active supervised CUDA lease: {active_lease.relative_to(project_root).as_posix()}")
        acquire_lock(startup_lock, {**lock_base, "train_pid": None, "state": "startup_mutex"})
        try:
            quarantine_stale_active_locks(project_root)
            assert_no_active_colab_cuda_lease(project_root)
            assert_no_other_supervised_training(run_id)
            existing_locks = active_locks(project_root)
            if existing_locks:
                raise SystemExit(f"refusing to start CUDA run while active lock exists: {existing_locks}")
            acquire_lock(active_lock, {**lock_base, "train_pid": None, "state": "launching"})
            write_colab_lease(
                active_lease,
                lease_payload(
                    run_id=run_id,
                    project_root=project_root,
                    state="launching",
                    active_lock=active_lock,
                    train_pid=None,
                    preflight=preflight,
                    review=review,
                    context_path=context_path,
                    cuda_provider=cuda_provider,
                ),
            )
        finally:
            remove_owned_lock(startup_lock, run_id, os.getpid(), launch_token_sha256)

        for command in (
            [py, "scripts/build_waka_training_corpus.py"],
            [py, "scripts/audit_source_records.py", "data/aozora/sources.json", "data/waka/sources.json"],
            [py, "scripts/build_manifest.py"],
            [py, "scripts/audit_public_manifest.py"],
            [py, "scripts/build_waka_meter_corpus.py"],
            [py, "scripts/build_tokenizer_public_vocab.py"],
            [py, "scripts/build_training_corpus.py"],
            [py, "scripts/build_preference_boost_corpus.py"],
            [py, "scripts/build_external_knowledge_surface_patterns.py"],
            [py, "scripts/build_worldclass_corpus.py"],
            [py, "scripts/build_training_augmentation_manifest.py"],
        ):
            run_checked(command)

        run_checked([py, "scripts/verify_preflight_gate.py", "--gate", args.preflight_gate, "--max-age-minutes", "240"])
        run_checked([py, "scripts/verify_zero_base_review_gate.py", "--gate", args.review_gate, "--preflight-gate", args.preflight_gate, "--max-age-minutes", "240"])

        snapshot_output = run_checked(
            [
                py,
                "scripts/snapshot_training_inputs.py",
                "--run-id",
                run_id,
                "--data",
                "data/kobun_worldclass_corpus.txt",
                "--val-data",
                "data/kobun_labeled_grammar_val.txt",
                "--test-data",
                "data/kobun_labeled_grammar_test.txt",
                "--tokenizer-extra-data",
                "data/tokenizer_public_char_vocab.txt",
                "--provenance-file",
                "data/corpus_manifest.jsonl",
                "--provenance-file",
                "logs/public_manifest_summary.json",
                "--provenance-file",
                "data/aozora/sources.json",
                "--provenance-file",
                "data/waka/sources.json",
                "--provenance-file",
                "data/tokenizer_public_char_vocab.meta.json",
                "--provenance-file",
                "data/training_augmentation_manifest.json",
            ]
        )
        train_data, val_data, test_data, tokenizer_extras, provenance_files = parse_snapshot_output(snapshot_output)
        tokenizer_extra_args = [arg for path in tokenizer_extras for arg in ("--tokenizer-extra-data", path)]
        provenance_args = [arg for path in provenance_files for arg in ("--provenance-file", path)]

        env = os.environ.copy()
        env.update(
            {
                "PYTHONUTF8": "1",
                "PYTHONIOENCODING": "utf-8",
                "OLD_JAPANESE_SUPERVISOR_RUN_ID": run_id,
                "OLD_JAPANESE_SUPERVISOR_TOKEN": launch_token,
                "OLD_JAPANESE_ACTIVE_LOCK": str(active_lock),
                "OLD_JAPANESE_PREFLIGHT_GATE": str(preflight),
                "OLD_JAPANESE_REVIEW_GATE": str(review),
                "OLD_JAPANESE_AUTONOMOUS_LAUNCH_CONTEXT": str(context_path),
                "OLD_JAPANESE_AUTONOMOUS_LAUNCH_NONCE": launch_nonce,
            }
        )
        train_command = [
            py,
            "-u",
            "-m",
            "kobun_llm.train",
            "--data",
            train_data,
            "--val-data",
            val_data,
            "--test-data",
            test_data,
            *tokenizer_extra_args,
            *provenance_args,
            "--tokenizer-source-label",
            "train_split_plus_core_japanese_inventory_plus_utf8_byte_fallback_v1",
            "--tokenizer-type",
            "byte_fallback_char_v1",
            "--fail-on-val-oov",
            "--out",
            out,
            "--best-out",
            best_out,
            "--run-id",
            run_id,
            "--require-supervisor",
            "--seed",
            "20260509",
            "--steps",
            str(args.steps),
            "--batch-size",
            str(args.batch_size),
            "--grad-accum-steps",
            str(args.grad_accum_steps),
            "--block-size",
            "384",
            "--n-layer",
            "16",
            "--n-head",
            "12",
            "--num-key-value-heads",
            "6",
            "--n-embd",
            "768",
            "--intermediate-size",
            "2304",
            "--dropout",
            "0.05",
            "--eval-every",
            str(args.eval_every),
            "--log-every",
            str(args.log_every),
            "--save-every",
            "1000",
            "--early-stop-patience",
            "10",
            "--overfit-stop-gap",
            "3.0",
            "--overfit-stop-after-evals",
            "5",
            "--overfit-stop-min-step",
            "1000",
            "--optimizer",
            "simple-adamw",
            "--lr",
            "2e-4",
            "--min-lr",
            "2e-5",
            "--warmup-steps",
            "400",
            "--cosine-lr",
            "--grad-clip",
            "1.0",
            "--qwen3-style",
            "--qk-norm",
            "--device",
            "cuda",
            "--amp",
            "--release-name",
            "old-japanese-0.1B-preview",
        ]

        stdout_log.parent.mkdir(parents=True, exist_ok=True)
        with stdout_log.open("w", encoding="utf-8") as stdout_handle, stderr_log.open("w", encoding="utf-8") as stderr_handle:
            stdout_handle.write(f"run_id={run_id} launcher=start_old_japanese_0_1b_cuda_colab_and_watch.py started={utc_now()}\n")
            stdout_handle.flush()
            proc = subprocess.Popen(train_command, stdout=stdout_handle, stderr=stderr_handle, env=env, text=True)
            update_owned_lock(active_lock, run_id, os.getpid(), launch_token_sha256, {**lock_base, "train_pid": proc.pid, "state": "running"})
            refresh_colab_lease(
                active_lease,
                run_id=run_id,
                project_root=project_root,
                state="running",
                active_lock=active_lock,
                train_pid=proc.pid,
                preflight=preflight,
                review=review,
                context_path=context_path,
                cuda_provider=cuda_provider,
            )
            while True:
                exit_code = proc.poll()
                if exit_code is not None:
                    break
                refresh_colab_lease(
                    active_lease,
                    run_id=run_id,
                    project_root=project_root,
                    state="running",
                    active_lock=active_lock,
                    train_pid=proc.pid,
                    preflight=preflight,
                    review=review,
                    context_path=context_path,
                    cuda_provider=cuda_provider,
                )
                time.sleep(60)
            stdout_handle.write(f"launcher_completed={utc_now()} exit_code={exit_code}\n")
            stdout_handle.flush()
        message = "completed" if exit_code == 0 else f"training command failed with exit code {exit_code}"
        if exit_code != 0:
            failure_reason = message
            raise SystemExit(message)
        write_train_exit_sentinel(train_exit, run_id, exit_code, message, out, best_out)
        if args.skip_post_run_quality:
            failure_reason = "post_run_quality_skipped_by_explicit_dry_run_flag"
            raise SystemExit(failure_reason)
        run_checked([py, "scripts/run_quality_checks_cuda.py", "--checkpoint", best_out])
    except BaseException as exc:
        message = f"startup_or_training_supervision_failed: {exc}"
        if not failure_reason:
            failure_reason = message
        if proc is not None and proc.poll() is None:
            try:
                proc.send_signal(signal.SIGTERM)
                proc.wait(timeout=30)
            except Exception:
                proc.kill()
        raise
    finally:
        if not train_exit.exists():
            write_train_exit_sentinel(train_exit, run_id, exit_code, message, out, best_out)
        final_state = "finished" if exit_code == 0 and not failure_reason else "failed_non_release"
        archive = archive_owned_lock(
            active_lock,
            run_id,
            os.getpid(),
            launch_token_sha256,
            state=final_state,
            reason=failure_reason or message,
        )
        if active_lease.exists():
            lease_archive = archive_colab_lease(active_lease, state=final_state, reason=failure_reason or message)
        if final_state != "finished":
            write_non_release_record(
                project_root,
                run_id=run_id,
                reason=failure_reason or message,
                train_exit=train_exit,
                active_lock_archive=archive or lease_archive,
            )
    if exit_code != 0:
        raise SystemExit(message)

    print(
        json.dumps(
            {
                "run_id": run_id,
                "best_checkpoint": best_out,
                "stdout_log": str(stdout_log),
                "stderr_log": str(stderr_log),
                "train_exit_sentinel": str(train_exit),
                "hf_export": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
