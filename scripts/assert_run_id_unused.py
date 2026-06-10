from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN_ID_RE = re.compile(r"^old_japanese_0_1b_(?:dml|cuda)_[0-9A-Za-z][0-9A-Za-z_-]{0,63}$")
ALLOWED_ACTIVE_LOCK_STATES = {"launching", "train_started_watcher_pending", "running"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fail closed if any artifact already exists for a supervised release RunId.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--allow-supervisor-launch-artifacts",
        action="store_true",
        help="Allow launch/watch-start logs that the supervisor creates before invoking the train wrapper.",
    )
    return parser.parse_args()


def exact_paths(run_id: str) -> list[Path]:
    return [
        ROOT / "checkpoints" / f"{run_id}.pt",
        ROOT / "checkpoints" / f"{run_id}_best.pt",
        ROOT / "logs" / f"{run_id}.out.log",
        ROOT / "logs" / f"{run_id}.err.log",
        ROOT / "logs" / f"launch_{run_id}.out.log",
        ROOT / "logs" / f"launch_{run_id}.err.log",
        ROOT / "logs" / f"watch_start_{run_id}.out.log",
        ROOT / "logs" / f"watch_start_{run_id}.err.log",
        ROOT / "logs" / f"colab_cuda_launch_context_{run_id}.json",
        ROOT / "logs" / f"colab_active_old_japanese_0_1b_cuda.{run_id}.json",
        ROOT / "logs" / f"watch_finalize_{run_id}.log",
        ROOT / "logs" / f"finalize_{run_id}.log",
        ROOT / "logs" / f"quality_{run_id}.log",
        ROOT / "logs" / f"eval_results_{run_id}.json",
        ROOT / "logs" / f"train_exit_{run_id}.json",
        ROOT / "logs" / "non_release_runs" / f"{run_id}.json",
        ROOT / "logs" / "llm_review_packets" / f"{run_id}.json",
        ROOT / "logs" / "upload_ready_evidence" / f"{run_id}.json",
        ROOT / "logs" / "generation_samples" / f"{run_id}.json",
        ROOT / "logs" / "generation_diagnostics" / f"{run_id}_generation_diagnostic_plan.json",
        ROOT / "data" / "run_snapshots" / run_id,
        ROOT / "release" / f"hf_model_{run_id}",
        ROOT / f"docs/UPLOAD_READY_REPORT_{run_id}.md",
        ROOT / f"docs/GENERATION_REVIEW_{run_id}.md",
    ]


def glob_paths(run_id: str) -> list[Path]:
    patterns = [
        f"checkpoints/{run_id}*.tmp",
        f"checkpoints/{run_id}*.pt.tmp",
        f"logs/eval_snapshots/{run_id}*",
        f"logs/llm_review_packets/*{run_id}*",
        f"logs/upload_ready_evidence/*{run_id}*",
        f"logs/generation_diagnostics/*{run_id}*",
        f"logs/generation_samples/*{run_id}*",
        f"logs/active_old_japanese_0_1b_dml.{run_id}.*.json",
        f"logs/active_old_japanese_0_1b_cuda.{run_id}.*.json",
        f"logs/**/*{run_id}*.json",
        f"logs/**/*{run_id}*.log",
        f"release/*{run_id}*",
        f"release/**/*{run_id}*",
        f"data/run_snapshots/{run_id}*",
        f"logs/*{run_id}*.json",
        f"logs/*{run_id}*.log",
        f"docs/*{run_id}*.md",
    ]
    found: list[Path] = []
    for pattern in patterns:
        found.extend(ROOT.glob(pattern))
    return found


def active_lock_payload_paths(run_id: str) -> list[Path]:
    found: list[Path] = []
    for pattern in ("logs/active_old_japanese_0_1b_dml*.json", "logs/active_old_japanese_0_1b_cuda*.json"):
        for path in ROOT.glob(pattern):
            if not path.is_file():
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8-sig"))
            except Exception:
                continue
            if str(payload.get("run_id") or "") == run_id:
                found.append(path)
    for path in ROOT.glob("logs/active_old_japanese_0_1b_*.lock"):
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
        except Exception:
            continue
        if str(payload.get("run_id") or "") == run_id:
            found.append(path)
    return found


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical(path: Path) -> Path:
    return path.resolve(strict=False)


def paths_equivalent(left: str, right: str) -> bool:
    if not left or not right:
        return False
    return str(canonical(Path(left))).casefold() == str(canonical(Path(right))).casefold()


def active_lock_age_minutes(payload: dict[str, object]) -> float:
    raw = str(payload.get("created_at") or "")
    try:
        created = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return float("inf")
    if created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - created.astimezone(timezone.utc)).total_seconds() / 60.0


def process_command_line(pid: int) -> str:
    if pid <= 0:
        return ""
    if os.name == "nt":
        command = (
            "$p=Get-CimInstance Win32_Process -Filter \"ProcessId="
            + str(int(pid))
            + "\" -ErrorAction SilentlyContinue; if($p){$p.CommandLine}"
        )
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", command],
            check=False,
            text=True,
            capture_output=True,
        )
        return result.stdout.strip() if result.returncode == 0 else ""
    try:
        return (Path("/proc") / str(int(pid)) / "cmdline").read_bytes().replace(b"\x00", b" ").decode(
            "utf-8", errors="replace"
        )
    except OSError:
        return ""


def allowed_supervisor_startup_mutex(path: Path, run_id: str) -> bool:
    if path.name != "active_old_japanese_0_1b_training.lock":
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return False
    if str(payload.get("run_id") or "") != run_id:
        return False
    if str(payload.get("state") or "") != "startup_mutex":
        return False
    if payload.get("hf_export") is not False:
        return False
    if str(payload.get("backend") or "") not in {"dml", "cuda"}:
        return False
    if active_lock_age_minutes(payload) > 30:
        return False
    launcher_pid = int(payload.get("launcher_pid") or 0)
    command_line = process_command_line(launcher_pid)
    if not command_line or run_id not in command_line:
        return False
    return (
        "start_old_japanese_0_1b_dml_and_watch.ps1" in command_line
        or "start_old_japanese_0_1b_cuda_colab_and_watch.py" in command_line
    )


def allowed_supervisor_released_archive(path: Path, run_id: str) -> bool:
    import re
    pattern = re.compile(rf"^active_old_japanese_0_1b_training\.{re.escape(run_id)}\.released\.\d{{8}}_\d{{6}}\.json$")
    return bool(pattern.fullmatch(path.name))


def allowed_supervisor_active_lock(path: Path, run_id: str) -> bool:
    """Allow only the currently owned launch lock created by the supervisor.

    This keeps stale/archived locks as RunId reuse blockers while allowing the
    supervised train wrapper to start after its parent has created the active
    lock that proves supervision.
    """

    active_lock_env = str(os.environ.get("OLD_JAPANESE_ACTIVE_LOCK") or "")
    supervisor_run_id = str(os.environ.get("OLD_JAPANESE_SUPERVISOR_RUN_ID") or "")
    supervisor_token = str(os.environ.get("OLD_JAPANESE_SUPERVISOR_TOKEN") or "")
    preflight_gate = str(os.environ.get("OLD_JAPANESE_PREFLIGHT_GATE") or "")
    review_gate = str(os.environ.get("OLD_JAPANESE_REVIEW_GATE") or "")
    launch_context = str(os.environ.get("OLD_JAPANESE_AUTONOMOUS_LAUNCH_CONTEXT") or "")
    if supervisor_run_id != run_id or not supervisor_token or not active_lock_env:
        return False
    if canonical(path) != canonical(Path(active_lock_env)):
        return False
    if path.name not in {"active_old_japanese_0_1b_dml.lock", "active_old_japanese_0_1b_cuda.lock"}:
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return False
    if str(payload.get("run_id") or "") != run_id:
        return False
    if payload.get("hf_export") is not False:
        return False
    if str(payload.get("state") or "") not in ALLOWED_ACTIVE_LOCK_STATES:
        return False
    if active_lock_age_minutes(payload) > 240:
        return False
    if str(payload.get("launch_token_sha256") or "") != sha256_text(supervisor_token):
        return False
    for field, expected_path in (
        ("preflight_gate", preflight_gate),
        ("review_gate", review_gate),
        ("autonomous_launch_context", launch_context),
    ):
        recorded = str(payload.get(field) or "")
        if expected_path and not paths_equivalent(recorded, expected_path):
            return False
    for field, expected_path in (
        ("preflight_gate_sha256", preflight_gate),
        ("review_gate_sha256", review_gate),
        ("autonomous_launch_context_sha256", launch_context),
    ):
        if expected_path:
            expected = Path(expected_path)
            if not expected.exists() or str(payload.get(field) or "") != sha256_file(expected):
                return False
    launcher_pid = int(payload.get("launcher_pid") or 0)
    if launcher_pid <= 0:
        return False
    return True


def main() -> None:
    args = parse_args()
    run_id = args.run_id
    if not RUN_ID_RE.fullmatch(run_id):
        raise SystemExit(f"invalid_release_run_id={run_id}")
    found: list[Path] = []
    for path in exact_paths(run_id) + glob_paths(run_id) + active_lock_payload_paths(run_id):
        if path.exists():
            found.append(path)
    if args.allow_supervisor_launch_artifacts:
        allowed = {
            (ROOT / "logs" / f"launch_{run_id}.out.log").resolve(),
            (ROOT / "logs" / f"launch_{run_id}.err.log").resolve(),
            (ROOT / "logs" / f"watch_start_{run_id}.out.log").resolve(),
            (ROOT / "logs" / f"watch_start_{run_id}.err.log").resolve(),
            (ROOT / "logs" / f"watch_finalize_{run_id}.log").resolve(),
            (ROOT / "logs" / f"autonomous_launch_context_{run_id}.json").resolve(),
            (ROOT / "logs" / f"colab_cuda_launch_context_{run_id}.json").resolve(),
        }
        found = [
            path
            for path in found
            if path.resolve() not in allowed
            and not allowed_supervisor_active_lock(path, run_id)
            and not allowed_supervisor_startup_mutex(path, run_id)
            and not allowed_supervisor_released_archive(path, run_id)
        ]
    unique = sorted({path.resolve() for path in found})
    if unique:
        rels = [path.relative_to(ROOT).as_posix() for path in unique]
        print("run_id_reuse_artifacts=" + ",".join(rels))
        raise SystemExit(1)
    print(f"run_id_unused_ok=true run_id={run_id}")


if __name__ == "__main__":
    main()
