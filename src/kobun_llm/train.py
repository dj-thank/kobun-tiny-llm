from __future__ import annotations

import argparse
import gc
import hashlib
import json
import random
from dataclasses import asdict
import math
import os
import re
import subprocess
import time
from pathlib import Path
from time import monotonic, strftime
from typing import Any

import torch

from .checkpoint_io import load_trusted_checkpoint
from .device import describe_device, device_backend, is_cuda_device, resolve_device
from .model import GPT, GPTConfig
from .optimizer_state import validate_simple_adamw_state
from kobun_autonomy.release_policy import require_release_candidate_run
from .release_resume import (
    file_records_signature,
    optimizer_checkpoint_step,
    validate_release_resume_chain_from_payload,
)
from .tokenizer import BYTE_FALLBACK_TOKENIZER_TYPE, CharTokenizer, tokenizer_from_text


class SimpleAdamW:
    def __init__(
        self,
        params,
        lr: float,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ) -> None:
        self.params = [param for param in params if param.requires_grad]
        self.param_groups = [
            {
                "params": self.params,
                "lr": lr,
                "betas": betas,
                "eps": eps,
                "weight_decay": weight_decay,
            }
        ]
        self.exp_avg = [torch.zeros_like(param, memory_format=torch.preserve_format) for param in self.params]
        self.exp_avg_sq = [torch.zeros_like(param, memory_format=torch.preserve_format) for param in self.params]
        self.step_count = 0

    def zero_grad(self, set_to_none: bool = True) -> None:
        for param in self.params:
            if param.grad is None:
                continue
            if set_to_none:
                param.grad = None
            else:
                param.grad.detach_()
                param.grad.zero_()

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        group = self.param_groups[0]
        lr = float(group["lr"])
        beta1, beta2 = group["betas"]
        eps = float(group["eps"])
        weight_decay = float(group["weight_decay"])
        self.step_count += 1
        bias_correction1 = 1.0 - beta1**self.step_count
        bias_correction2 = 1.0 - beta2**self.step_count
        step_size = lr / bias_correction1
        for index, param in enumerate(self.params):
            grad = param.grad
            if grad is None:
                continue
            if weight_decay:
                param.mul_(1.0 - lr * weight_decay)
            exp_avg = self.exp_avg[index]
            exp_avg_sq = self.exp_avg_sq[index]
            exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
            denom = exp_avg_sq.sqrt().div_(math.sqrt(bias_correction2)).add_(eps)
            param.addcdiv_(exp_avg, denom, value=-step_size)
        return loss

    def state_dict(self) -> dict[str, Any]:
        group = self.param_groups[0]
        return {
            "optimizer_type": "simple-adamw",
            "step_count": self.step_count,
            "lr": float(group["lr"]),
            "betas": tuple(group["betas"]),
            "eps": float(group["eps"]),
            "weight_decay": float(group["weight_decay"]),
            "exp_avg": [tensor.detach().cpu() for tensor in self.exp_avg],
            "exp_avg_sq": [tensor.detach().cpu() for tensor in self.exp_avg_sq],
        }

    def load_state_dict(self, payload: dict[str, Any]) -> None:
        validate_simple_adamw_state(payload, self.params)
        exp_avg_payload = payload["exp_avg"]
        exp_avg_sq_payload = payload["exp_avg_sq"]
        betas = tuple(payload["betas"])
        self.step_count = int(payload["step_count"])
        group = self.param_groups[0]
        group["lr"] = float(payload["lr"])
        group["betas"] = betas
        group["eps"] = float(payload["eps"])
        group["weight_decay"] = float(payload["weight_decay"])
        for target, source in zip(self.exp_avg, exp_avg_payload, strict=True):
            target.copy_(source.to(device=target.device))
        for target, source in zip(self.exp_avg_sq, exp_avg_sq_payload, strict=True):
            target.copy_(source.to(device=target.device))


def save_checkpoint(
    path: Path,
    model: GPT,
    config: GPTConfig,
    tokenizer: CharTokenizer,
    step: int,
    optimizer: Any | None = None,
    scaler: torch.amp.GradScaler | None = None,
    best_val: float | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    payload: dict[str, Any] = {
        "model": model.state_dict(),
        "config": asdict(config),
        "tokenizer": tokenizer.to_dict(),
        "step": step,
        "best_val": best_val,
        "metadata": metadata or {},
        "torch_rng_state": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        payload["cuda_rng_state_all"] = [state.cpu() for state in torch.cuda.get_rng_state_all()]
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scaler is not None:
        payload["scaler"] = scaler.state_dict()
    try:
        torch.save(payload, tmp_path)
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a tiny GPT-style model on classical Japanese text.")
    parser.add_argument("--data", type=Path, default=Path("data/kobun_sample.txt"))
    parser.add_argument("--out", type=Path, default=Path("checkpoints/kobun_tiny.pt"))
    parser.add_argument("--best-out", type=Path, default=None)
    parser.add_argument("--resume", type=Path, default=None, help="Resume a full training checkpoint with optimizer/RNG state.")
    parser.add_argument("--init-from", type=Path, default=None, help="Initialize weights/tokenizer/config from a checkpoint and start a new run.")
    parser.add_argument("--val-data", type=Path, default=None, help="Separate validation corpus. Required for serious runs.")
    parser.add_argument("--test-data", type=Path, default=None, help="Independent test corpus. Saved in metadata, never used for training or checkpoint selection.")
    parser.add_argument(
        "--tokenizer-extra-data",
        action="append",
        type=Path,
        default=[],
        help="Additional files included only for tokenizer vocabulary coverage.",
    )
    parser.add_argument("--tokenizer-source-label", default="", help="Human-readable tokenizer provenance label.")
    parser.add_argument(
        "--tokenizer-type",
        choices=["char", BYTE_FALLBACK_TOKENIZER_TYPE],
        default="char",
        help="Tokenizer implementation. byte_fallback_char_v1 keeps heldout-only chars out of direct vocab.",
    )
    parser.add_argument(
        "--provenance-file",
        action="append",
        type=Path,
        default=[],
        help="Run-specific provenance files whose paths and hashes are saved in checkpoint metadata.",
    )
    parser.add_argument("--fail-on-val-oov", action="store_true", help="Fail if validation text has tokenizer OOV chars.")
    parser.add_argument("--split-fraction", type=float, default=0.9)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--grad-accum-steps",
        type=int,
        default=1,
        help="Accumulate this many micro-batches before each optimizer step.",
    )
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--n-layer", type=int, default=4)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--n-embd", type=int, default=128)
    parser.add_argument("--intermediate-size", type=int, default=None)
    parser.add_argument("--num-key-value-heads", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=3e-5)
    parser.add_argument("--optimizer", choices=["adamw", "simple-adamw"], default="adamw")
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--cosine-lr", action="store_true")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument(
        "--log-every",
        type=int,
        default=0,
        help="Print lightweight optimizer-step progress every N steps. 0 disables.",
    )
    parser.add_argument("--early-stop-patience", type=int, default=0, help="Stop after this many non-improving evals. 0 disables.")
    parser.add_argument(
        "--overfit-stop-gap",
        type=float,
        default=0.0,
        help=(
            "If >0, stop after repeated non-improving evals once "
            "val_loss - train_loss is at least this large. 0 disables."
        ),
    )
    parser.add_argument(
        "--overfit-stop-after-evals",
        type=int,
        default=0,
        help="Number of non-improving evals required for --overfit-stop-gap. 0 falls back to --early-stop-patience.",
    )
    parser.add_argument(
        "--overfit-stop-min-step",
        type=int,
        default=0,
        help="Do not apply --overfit-stop-gap before this optimizer step.",
    )
    parser.add_argument("--amp", action="store_true", help="Use CUDA mixed precision training.")
    parser.add_argument("--qwen3-style", action="store_true", help="Use Qwen3-like RoPE/RMSNorm/SwiGLU/GQA/tied embeddings.")
    parser.add_argument("--qk-norm", action="store_true", help="Normalize Q/K heads with RMSNorm.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "dml"], default="auto")
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--require-supervisor", action="store_true")
    parser.add_argument("--seed", type=int, default=20260509)
    parser.add_argument("--release-name", type=str, default="old-japanese-0.1B-preview")
    return parser.parse_args()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_runtime_path(path_text: str) -> str:
    path = Path(path_text)
    if not path.is_absolute():
        path = Path.cwd() / path
    return os.path.normcase(os.path.abspath(path))


def read_sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def query_windows_process(pid: int) -> dict[str, Any]:
    if pid <= 0:
        return {}
    if os.name != "nt":
        proc_root = Path("/proc") / str(int(pid))
        try:
            stat_text = (proc_root / "stat").read_text(encoding="utf-8", errors="replace")
            # /proc/<pid>/stat wraps comm in parentheses; parent pid is the
            # first integer after the state field following the final ")".
            after_comm = stat_text.rsplit(")", 1)[1].strip().split()
            parent_pid = int(after_comm[1]) if len(after_comm) > 1 else 0
            raw_cmd = (proc_root / "cmdline").read_bytes()
            command_line = raw_cmd.replace(b"\x00", b" ").decode("utf-8", errors="replace").strip()
            return {"ProcessId": int(pid), "ParentProcessId": parent_pid, "CommandLine": command_line}
        except OSError:
            return {}
    command = (
        "$p=Get-CimInstance Win32_Process -Filter \"ProcessId="
        + str(int(pid))
        + "\" -ErrorAction SilentlyContinue; "
        "if($p){[pscustomobject]@{ProcessId=$p.ProcessId;ParentProcessId=$p.ParentProcessId;"
        "CommandLine=$p.CommandLine}|ConvertTo-Json -Compress}"
    )
    result = subprocess.run(
        ["powershell", "-NoProfile", "-Command", command],
        check=False,
        text=True,
        capture_output=True,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return {}
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def windows_ancestor_process_ids(pid: int) -> list[int]:
    ancestors: list[int] = []
    current = pid
    while current > 0:
        payload = query_windows_process(current)
        if not payload:
            break
        try:
            parent = int(payload.get("ParentProcessId") or 0)
        except (TypeError, ValueError):
            break
        if parent <= 0 or parent in ancestors:
            break
        ancestors.append(parent)
        current = parent
    return ancestors


def require_python_parent_is_train_wrapper(lock: dict[str, Any], run_id: str) -> None:
    lock_backend = str(lock.get("backend") or ("cuda" if run_id.startswith("old_japanese_0_1b_cuda_") else "dml"))
    train_pid = int(lock.get("train_pid") or 0)
    if train_pid <= 0:
        raise SystemExit("active-run lock is missing train_pid.")
    if lock_backend == "cuda":
        if train_pid != os.getpid():
            raise SystemExit(f"active-run lock train_pid does not match CUDA training process: lock={train_pid} pid={os.getpid()}")
        current = query_windows_process(os.getpid())
        command_line = str(current.get("CommandLine") or "")
        if "kobun_llm.train" not in command_line or run_id not in command_line or "--require-supervisor" not in command_line:
            raise SystemExit("active-run lock train_pid command line is not the supervised CUDA train process.")
        return
    ancestors = windows_ancestor_process_ids(os.getpid())
    if train_pid not in ancestors:
        raise SystemExit(
            f"active-run lock train_pid is not in Python ancestor process chain: lock={train_pid} ancestors={ancestors}"
        )
    wrapper = query_windows_process(train_pid)
    command_line = str(wrapper.get("CommandLine") or "")
    if (
        "train_old_japanese_0_1b_dml.ps1" not in command_line
        or run_id not in command_line
        or "-LaunchedBySupervisor" not in command_line
    ):
        raise SystemExit("active-run lock train_pid parent command line is not the supervised DML train wrapper.")


def verify_autonomous_launch_context_payload(lock: dict[str, Any], run_id: str, nonce: str) -> None:
    context_env = os.environ.get("OLD_JAPANESE_AUTONOMOUS_LAUNCH_CONTEXT", "")
    if not context_env:
        raise SystemExit("release-shaped training requires autonomous launch context supervisor env.")
    context_path = Path(context_env)
    if not context_path.is_absolute():
        context_path = Path.cwd() / context_path
    try:
        context = json.loads(context_path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"autonomous launch context is not readable JSON: {exc}") from exc
    schema = str(context.get("schema") or "")
    if schema not in {"old_japanese_0_1b_autonomous_launch_context_v1", "old_japanese_0_1b_cuda_colab_launch_context_v1"}:
        raise SystemExit("autonomous launch context schema mismatch.")
    if str(context.get("run_id") or "") != run_id:
        raise SystemExit("autonomous launch context RunId mismatch.")
    expected_action = (
        "colab_cuda_supervised_training"
        if schema == "old_japanese_0_1b_cuda_colab_launch_context_v1"
        else "prepare_next_fresh_run_after_static_gate_and_zero_base_reviews"
    )
    if str(context.get("selected_action") or "") != expected_action:
        raise SystemExit("autonomous launch context selected_action is not authorized for training.")
    if schema == "old_japanese_0_1b_cuda_colab_launch_context_v1" and str(context.get("backend") or "") != "cuda":
        raise SystemExit("CUDA Colab launch context backend mismatch.")
    if context.get("hf_export") is not False:
        raise SystemExit("autonomous launch context must attest hf_export=false.")
    if str(context.get("launch_nonce_sha256") or "") != hashlib.sha256(nonce.encode("utf-8")).hexdigest():
        raise SystemExit("autonomous launch context nonce hash mismatch.")
    for context_key, lock_key in (
        ("preflight_gate_sha256", "preflight_gate_sha256"),
        ("review_gate_sha256", "review_gate_sha256"),
    ):
        if str(context.get(context_key) or "") != str(lock.get(lock_key) or ""):
            raise SystemExit(f"autonomous launch context {context_key} is not bound to active-run lock.")


def verify_bound_runtime_file(lock: dict[str, Any], label: str, env_key: str, path_key: str, hash_key: str) -> None:
    env_value = os.environ.get(env_key, "")
    if not env_value:
        raise SystemExit(f"release-shaped training requires {label} supervisor context.")
    lock_path_value = str(lock.get(path_key) or "")
    if not lock_path_value:
        raise SystemExit(f"active-run lock missing {path_key}.")
    if normalize_runtime_path(env_value) != normalize_runtime_path(lock_path_value):
        raise SystemExit(f"active-run lock {label} path does not match supervisor context.")
    file_path = Path(env_value)
    if not file_path.is_absolute():
        file_path = Path.cwd() / file_path
    if not file_path.exists():
        raise SystemExit(f"active-run lock bound {label} file is missing.")
    expected_hash = str(lock.get(hash_key) or "")
    if not expected_hash or expected_hash != read_sha256_file(file_path):
        raise SystemExit(f"active-run lock {label} hash mismatch.")


def verify_supervisor_context(run_id: str) -> None:
    env_run_id = os.environ.get("OLD_JAPANESE_SUPERVISOR_RUN_ID", "")
    token = os.environ.get("OLD_JAPANESE_SUPERVISOR_TOKEN", "")
    lock_text = os.environ.get("OLD_JAPANESE_ACTIVE_LOCK", "")
    if env_run_id != run_id:
        raise SystemExit("release-shaped training requires matching supervisor RunId context.")
    if not token:
        raise SystemExit("release-shaped training requires a supervisor launch token.")
    lock_path = Path(lock_text) if lock_text else Path("logs/active_old_japanese_0_1b_dml.lock")
    lock: dict[str, Any] | None = None
    deadline = time.monotonic() + 10.0
    last_error: Exception | None = None
    while time.monotonic() <= deadline:
        if not lock_path.exists():
            last_error = FileNotFoundError(str(lock_path))
            time.sleep(0.1)
            continue
        try:
            candidate = json.loads(lock_path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError) as exc:
            last_error = exc
            time.sleep(0.1)
            continue
        if str(candidate.get("run_id") or "") == run_id and str(candidate.get("state") or "") == "launching":
            time.sleep(0.1)
            continue
        lock = candidate
        break
    if lock is None:
        raise SystemExit(f"release-shaped training requires an active-run lock ready for training: {last_error}") from last_error
    if str(lock.get("run_id") or "") != run_id:
        raise SystemExit("active-run lock RunId does not match training RunId.")
    expected_hash = str(lock.get("launch_token_sha256") or "")
    actual_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
    if not expected_hash or actual_hash != expected_hash:
        raise SystemExit("active-run lock token hash does not match supervisor token.")
    nonce = os.environ.get("OLD_JAPANESE_AUTONOMOUS_LAUNCH_NONCE", "")
    if not nonce:
        raise SystemExit("release-shaped training requires an autonomous launch nonce.")
    expected_nonce_hash = str(lock.get("launch_nonce_sha256") or "")
    actual_nonce_hash = hashlib.sha256(nonce.encode("utf-8")).hexdigest()
    if not expected_nonce_hash or actual_nonce_hash != expected_nonce_hash:
        raise SystemExit("active-run lock autonomous launch nonce hash mismatch.")
    if str(lock.get("state") or "") not in {"train_started_watcher_pending", "running"}:
        raise SystemExit("active-run lock state is not valid for supervised training.")
    require_python_parent_is_train_wrapper(lock, run_id)
    verify_autonomous_launch_context_payload(lock, run_id, nonce)
    launcher_pid = int(lock.get("launcher_pid") or 0)
    if launcher_pid <= 0:
        raise SystemExit("active-run lock is missing launcher_pid.")
    launcher = query_windows_process(launcher_pid)
    launcher_command = str(launcher.get("CommandLine") or "")
    lock_backend = str(lock.get("backend") or ("cuda" if run_id.startswith("old_japanese_0_1b_cuda_") else "dml"))
    # The launcher may have exited after starting the watcher.
    # If still alive, verify its command line; otherwise the train_pid
    # ancestor check and lock token already prove supervision.
    # Legacy hard error: "active-run lock launcher_pid is not in Python ancestor process chain."
    if launcher_command:
        if lock_backend == "cuda":
            if (
                "start_old_japanese_0_1b_cuda_colab_and_watch.py" not in launcher_command
                or run_id not in launcher_command
                or "--allow-start-training" not in launcher_command
                or "--reviews-passed" not in launcher_command
            ):
                raise SystemExit("active-run lock launcher_pid is not the authorized CUDA Colab supervisor for this RunId.")
        elif (
            "start_old_japanese_0_1b_dml_and_watch.ps1" not in launcher_command
            or run_id not in launcher_command
            or "-AllowStartTraining" not in launcher_command
            or "-ReviewsPassed" not in launcher_command
        ):
            raise SystemExit("active-run lock launcher_pid is not the authorized DML supervisor for this RunId.")
    if lock.get("hf_export") is not False:
        raise SystemExit("active-run lock must attest hf_export=false.")
    expected_action = "colab_cuda_supervised_training" if lock_backend == "cuda" else "prepare_next_fresh_run_after_static_gate_and_zero_base_reviews"
    if str(lock.get("selected_action") or "") != expected_action:
        raise SystemExit("active-run lock selected_action is not authorized for training.")
    expected_script = "scripts/start_old_japanese_0_1b_cuda_colab_and_watch.py" if lock_backend == "cuda" else "scripts\\autonomous_old_japanese_0_1b_loop.ps1"
    if str(lock.get("autonomous_script") or "") != expected_script:
        raise SystemExit("active-run lock autonomous_script mismatch.")
    if int(lock.get("autonomous_pid") or 0) <= 0:
        raise SystemExit("active-run lock is missing autonomous_pid.")
    verify_bound_runtime_file(
        lock,
        "preflight gate",
        "OLD_JAPANESE_PREFLIGHT_GATE",
        "preflight_gate",
        "preflight_gate_sha256",
    )
    verify_bound_runtime_file(
        lock,
        "zero-base review gate",
        "OLD_JAPANESE_REVIEW_GATE",
        "review_gate",
        "review_gate_sha256",
    )
    verify_bound_runtime_file(
        lock,
        "autonomous launch context",
        "OLD_JAPANESE_AUTONOMOUS_LAUNCH_CONTEXT",
        "autonomous_launch_context",
        "autonomous_launch_context_sha256",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count_parameters(model: torch.nn.Module) -> int:
    return sum(param.numel() for param in model.parameters())


def validate_release_resume(
    payload: dict[str, Any],
    args: argparse.Namespace,
    run_id: str,
    checkpoint_path: Path,
    backend: str,
    config: GPTConfig,
    data_sha256: str,
    val_data_sha256: str,
    test_data_sha256: str,
    tokenizer_extra_metadata: list[dict[str, Any]],
    provenance_files: list[dict[str, Any]],
) -> None:
    metadata = dict(payload.get("metadata", {}) or {})
    if str(metadata.get("run_id") or "") != run_id:
        raise SystemExit(f"resume run_id mismatch: checkpoint={metadata.get('run_id')!r} requested={run_id!r}")
    for key, expected in (
        ("data_sha256", data_sha256),
        ("val_data_sha256", val_data_sha256),
        ("test_data_sha256", test_data_sha256),
    ):
        if str(metadata.get(key) or "") != expected:
            raise SystemExit(f"resume metadata mismatch for {key}: checkpoint={metadata.get(key)!r} requested={expected!r}")
    requested_tokenizer_source = args.tokenizer_source_label or (
        "train_text_plus_extra_vocab_with_unk" if tokenizer_extra_metadata else "train_text_only_with_unk"
    )
    for key, expected in (
        ("tokenizer_source", requested_tokenizer_source),
        ("release_name", args.release_name),
    ):
        if str(metadata.get(key) or "") != str(expected):
            raise SystemExit(f"resume metadata mismatch for {key}: checkpoint={metadata.get(key)!r} requested={expected!r}")
    if str(metadata.get("val_oov_chars") or ""):
        raise SystemExit(f"resume checkpoint has validation OOV chars: {metadata.get('val_oov_chars')!r}")
    if str(metadata.get("test_oov_chars") or ""):
        raise SystemExit(f"resume checkpoint has test OOV chars: {metadata.get('test_oov_chars')!r}")
    try:
        if file_records_signature(metadata.get("tokenizer_extra_data")) != file_records_signature(tokenizer_extra_metadata):
            raise SystemExit("resume tokenizer_extra_data metadata does not match requested tokenizer files.")
        if file_records_signature(metadata.get("provenance_files")) != file_records_signature(provenance_files):
            raise SystemExit("resume provenance_files metadata does not match requested provenance files.")
    except ValueError as exc:
        raise SystemExit(f"resume file metadata is unsafe: {exc}") from exc
    try:
        validate_release_resume_chain_from_payload(
            payload,
            checkpoint_path,
            allow_same_run_resume=True,
            expected_backend=backend,
            expected_seed=args.seed,
            expected_optimizer=args.optimizer,
            expected_config=asdict(config),
            expected_tokenizer=payload.get("tokenizer"),
            expected_tokenizer_extra_data=tokenizer_extra_metadata,
            expected_provenance_files=provenance_files,
            load_checkpoint=lambda path: load_trusted_checkpoint(path, map_location="cpu"),
            resolve_path=lambda raw: Path(raw) if Path(raw).is_absolute() else Path.cwd() / raw,
        )
    except ValueError as exc:
        raise SystemExit(f"release resume chain is unsafe: {exc}") from exc


def learning_rate(step: int, args: argparse.Namespace) -> float:
    if not args.cosine_lr:
        return args.lr
    if step < args.warmup_steps:
        return args.lr * (step + 1) / max(1, args.warmup_steps)
    progress = (step - args.warmup_steps) / max(1, args.steps - args.warmup_steps)
    progress = min(1.0, max(0.0, progress))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return args.min_lr + cosine * (args.lr - args.min_lr)


def get_batch(data: torch.Tensor, batch_size: int, block_size: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


def split_data(
    encoded: torch.Tensor,
    block_size: int,
    split_fraction: float,
    val_encoded: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    minimum = block_size + 2
    if len(encoded) < minimum:
        raise SystemExit("Training text is too short for the selected block size.")
    if val_encoded is not None:
        if len(val_encoded) < minimum:
            raise SystemExit("Validation text is too short for the selected block size.")
        return encoded, val_encoded

    if not 0.0 < split_fraction < 1.0:
        raise SystemExit("--split-fraction must be between 0 and 1.")
    split = int(len(encoded) * split_fraction)
    if split < minimum:
        raise SystemExit("Training split is too short; provide --val-data or lower --block-size.")
    if len(encoded) - split < minimum:
        raise SystemExit("Validation split is too short; provide --val-data instead of falling back to train data.")
    return encoded[:split], encoded[split:]


@torch.no_grad()
def estimate_loss(
    model: GPT,
    data: torch.Tensor,
    batch_size: int,
    block_size: int,
    device: str,
    amp: bool,
) -> float:
    model.eval()
    losses = []
    for _ in range(10):
        xb, yb = get_batch(data, batch_size, block_size, device)
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=amp):
            _, loss = model(xb, yb)
        assert loss is not None
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite eval loss: {float(loss.item())}")
        losses.append(float(loss.item()))
    model.train()
    return sum(losses) / len(losses)


def main() -> None:
    args = parse_args()
    if args.resume is not None and args.init_from is not None:
        raise SystemExit("Use only one of --resume or --init-from.")
    if args.grad_accum_steps < 1:
        raise SystemExit("--grad-accum-steps must be >= 1.")
    if args.overfit_stop_gap < 0:
        raise SystemExit("--overfit-stop-gap must be >= 0.")
    if args.overfit_stop_after_evals < 0:
        raise SystemExit("--overfit-stop-after-evals must be >= 0.")
    if args.overfit_stop_min_step < 0:
        raise SystemExit("--overfit-stop-min-step must be >= 0.")
    if args.log_every < 0:
        raise SystemExit("--log-every must be >= 0.")
    if args.run_id and not re.fullmatch(r"old_japanese_0_1b(?:_dml|_cuda|_hip)?_[0-9A-Za-z][0-9A-Za-z_-]{0,63}", args.run_id):
        raise SystemExit(f"--run-id is not a safe release run id: {args.run_id!r}")
    if args.run_id and args.run_id.startswith("old_japanese_0_1b_"):
        if not (args.run_id.startswith("old_japanese_0_1b_dml_") or args.run_id.startswith("old_japanese_0_1b_cuda_")):
            raise SystemExit(
                "Only supervised DirectML or CUDA Colab release-shaped runs are enabled; "
                "HIP release training is disabled until equivalent gates exist."
            )
        require_release_candidate_run(args.run_id, context="release-shaped training")
        if not args.require_supervisor:
            raise SystemExit("release-shaped training requires --require-supervisor and the supervised launcher context.")
        verify_supervisor_context(args.run_id)
        if args.init_from is not None:
            raise SystemExit("release-shaped training must be from scratch; --init-from is forbidden.")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = resolve_device(args.device)
    backend = device_backend(device)
    device_description = describe_device(device)
    print(f"device={device_description}")
    use_amp = args.amp and is_cuda_device(device)
    if args.amp and not use_amp:
        print("amp=requested but disabled because device is not cuda")
    run_id = args.run_id or strftime("%Y%m%d_%H%M%S")
    if run_id.startswith("old_japanese_0_1b_dml_") and backend != "dml":
        raise SystemExit(f"DML release run id requires DirectML backend: run_id={run_id!r} backend={backend!r}")
    if run_id.startswith("old_japanese_0_1b_cuda_") and backend != "cuda":
        raise SystemExit(f"CUDA release run id requires CUDA backend: run_id={run_id!r} backend={backend!r}")
    if run_id.startswith("old_japanese_0_1b_") and not (
        run_id.startswith("old_japanese_0_1b_dml_") or run_id.startswith("old_japanese_0_1b_cuda_")
    ):
        raise SystemExit("Only DirectML and supervised Colab CUDA release-shaped training are currently enabled.")

    text = args.data.read_text(encoding="utf-8")
    val_text = args.val_data.read_text(encoding="utf-8") if args.val_data is not None else None
    test_text = args.test_data.read_text(encoding="utf-8") if args.test_data is not None else None
    tokenizer_extra_text = "".join(path.read_text(encoding="utf-8") for path in args.tokenizer_extra_data)
    tokenizer_extra_metadata = [
        {
            "path": str(path),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for path in args.tokenizer_extra_data
    ]
    provenance_files = [
        {
            "path": str(path),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for path in args.provenance_file
    ]
    checkpoint_path = args.resume or args.init_from
    if checkpoint_path is not None:
        payload = load_trusted_checkpoint(checkpoint_path, map_location="cpu")
        tokenizer = CharTokenizer.from_dict(payload["tokenizer"])
        config = GPTConfig(**payload["config"])
        if args.resume is not None and ("optimizer" not in payload or "scaler" not in payload):
            raise SystemExit(
                f"{args.resume} is not a full resume checkpoint. "
                "Use --init-from for legacy/init-only checkpoints."
            )
        mode = "resuming" if args.resume is not None else "initializing_from"
        print(f"{mode}={checkpoint_path}")
        if args.resume is not None:
            validate_release_resume(
                payload,
                args,
                run_id,
                checkpoint_path,
                backend,
                config,
                sha256_text(text),
                sha256_text(val_text) if val_text is not None else "",
                sha256_text(test_text) if test_text is not None else "",
                tokenizer_extra_metadata,
                provenance_files,
            )
    else:
        payload = None
        tokenizer = tokenizer_from_text(text + tokenizer_extra_text, tokenizer_type=args.tokenizer_type)
        qwen3_kv_heads = args.num_key_value_heads
        if args.qwen3_style and qwen3_kv_heads is None:
            qwen3_kv_heads = max(1, args.n_head // 2)
        config = GPTConfig(
            vocab_size=tokenizer.vocab_size,
            block_size=args.block_size,
            n_layer=args.n_layer,
            n_head=args.n_head,
            n_embd=args.n_embd,
            dropout=args.dropout,
            intermediate_size=args.intermediate_size or (3 * args.n_embd if args.qwen3_style else None),
            num_key_value_heads=qwen3_kv_heads,
            norm_type="rmsnorm" if args.qwen3_style else "layernorm",
            mlp_type="swiglu" if args.qwen3_style else "gelu",
            use_rope=args.qwen3_style,
            rope_theta=1000000.0 if args.qwen3_style else 10000.0,
            tie_word_embeddings=args.qwen3_style,
            attention_bias=not args.qwen3_style,
            mlp_bias=not args.qwen3_style,
            qk_norm=args.qk_norm,
        )
    encoded = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    val_encoded = None
    if val_text is not None:
        val_oov_chars = tokenizer.missing_chars(val_text)
        if val_oov_chars and args.fail_on_val_oov:
            shown = "".join(val_oov_chars[:50])
            raise SystemExit(f"validation text contains tokenizer OOV characters: {shown!r}")
        val_encoded = torch.tensor(tokenizer.encode(val_text), dtype=torch.long)
    test_oov_chars = tokenizer.missing_chars(test_text or "")
    if test_oov_chars and args.fail_on_val_oov:
        shown = "".join(test_oov_chars[:50])
        raise SystemExit(f"test text contains tokenizer OOV characters: {shown!r}")
    train_data, val_data = split_data(encoded, config.block_size, args.split_fraction, val_encoded)

    model = GPT(config).to(device)
    if config.tie_word_embeddings:
        model.tie_weights()
    if payload is not None:
        model.load_state_dict(payload["model"])
        if config.tie_word_embeddings:
            model.tie_weights()
    param_count = count_parameters(model)
    if args.optimizer == "simple-adamw":
        optimizer = SimpleAdamW(model.parameters(), lr=args.lr)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    start_step = 0
    best_val = float("inf")
    if args.resume is not None and payload is not None:
        checkpoint_metadata = dict(payload.get("metadata", {}) or {})
        checkpoint_optimizer = str(checkpoint_metadata.get("optimizer") or "")
        if checkpoint_optimizer and checkpoint_optimizer != args.optimizer:
            raise SystemExit(
                f"resume optimizer mismatch: checkpoint optimizer={checkpoint_optimizer!r} "
                f"requested optimizer={args.optimizer!r}"
            )
        optimizer.load_state_dict(payload["optimizer"])
        scaler.load_state_dict(payload["scaler"])
        start_step = int(payload.get("step", 0))
        try:
            optimizer_step = optimizer_checkpoint_step(payload["optimizer"], args.optimizer)
        except ValueError as exc:
            raise SystemExit(f"resume optimizer state is unsafe: {exc}") from exc
        if int(optimizer_step) != start_step:
            raise SystemExit(
                f"resume checkpoint step mismatch: checkpoint_step={start_step} "
                f"optimizer_step_count={optimizer_step}. Refusing unsafe resume."
            )
        best_val = float(payload.get("best_val", float("inf")) or float("inf"))
        torch.set_rng_state(payload["torch_rng_state"].cpu())
        if is_cuda_device(device) and "cuda_rng_state_all" in payload:
            torch.cuda.set_rng_state_all([state.cpu() for state in payload["cuda_rng_state_all"]])
    if payload is not None:
        payload.clear()
        payload = None
        gc.collect()
        if is_cuda_device(device):
            torch.cuda.empty_cache()
    print(
        "config="
        f"vocab={config.vocab_size} block={config.block_size} layers={config.n_layer} "
        f"heads={config.n_head} kv_heads={config.num_key_value_heads or config.n_head} "
        f"embd={config.n_embd} norm={config.norm_type} mlp={config.mlp_type} "
        f"rope={config.use_rope} qk_norm={config.qk_norm} tied={config.tie_word_embeddings} "
        f"amp={use_amp} params={param_count} grad_accum={args.grad_accum_steps}"
    )
    metadata = {
        "run_id": run_id,
        "backend": backend,
        "device_description": device_description,
        "data_path": str(args.data),
        "data_sha256": sha256_text(text),
        "data_chars": len(text),
        "val_data_path": str(args.val_data) if args.val_data is not None else "",
        "val_data_sha256": sha256_text(val_text) if val_text is not None else "",
        "val_data_chars": len(val_text) if val_text is not None else 0,
        "test_data_path": str(args.test_data) if args.test_data is not None else "",
        "test_data_sha256": sha256_text(test_text) if test_text is not None else "",
        "test_data_chars": len(test_text) if test_text is not None else 0,
        "tokenizer_source": args.tokenizer_source_label
        or ("train_text_plus_extra_vocab_with_unk" if tokenizer_extra_text else "train_text_only_with_unk"),
        "tokenizer_type": getattr(tokenizer, "tokenizer_type", "char"),
        "byte_fallback": getattr(tokenizer, "tokenizer_type", "char") == BYTE_FALLBACK_TOKENIZER_TYPE,
        "tokenizer_extra_data_paths": [str(path) for path in args.tokenizer_extra_data],
        "tokenizer_extra_data": tokenizer_extra_metadata,
        "provenance_files": provenance_files,
        "val_oov_chars": "".join(tokenizer.missing_chars(val_text or ""))[:200],
        "test_oov_chars": "".join(test_oov_chars)[:200],
        "init_from": str(args.init_from) if args.init_from is not None else "",
        "resume": str(args.resume) if args.resume is not None else "",
        "param_count": param_count,
        "param_count_b": param_count / 1_000_000_000,
        "optimizer": args.optimizer,
        "seed": args.seed,
        "determinism_note": "Seed fixes Python/Torch initialization and batch sampling; DirectML kernels may still be nondeterministic.",
        "batch_size": args.batch_size,
        "grad_accum_steps": args.grad_accum_steps,
        "effective_batch_size": args.batch_size * args.grad_accum_steps,
        "overfit_stop_gap": args.overfit_stop_gap,
        "overfit_stop_after_evals": args.overfit_stop_after_evals,
        "overfit_stop_min_step": args.overfit_stop_min_step,
        "license_policy": "code MIT; model weights intended CC BY-SA 4.0; training text not bundled in release artifacts",
        "release_name": args.release_name,
    }
    print(
        f"run_id={metadata['run_id']} data_sha256={metadata['data_sha256'][:12]} "
        f"val_sha256={str(metadata['val_data_sha256'])[:12]} "
        f"test_sha256={str(metadata['test_data_sha256'])[:12]}"
    )
    best_out = args.best_out or args.out.with_name(args.out.stem + "_best.pt")
    non_improving_evals = 0
    completed_step = start_step
    train_started_at = monotonic()
    last_progress_log_at = train_started_at

    for step in range(start_step, args.steps + 1):
        completed_step = step
        if step % args.eval_every == 0:
            train_loss = estimate_loss(model, train_data, args.batch_size, config.block_size, device, use_amp)
            val_loss = estimate_loss(model, val_data, args.batch_size, config.block_size, device, use_amp)
            if not math.isfinite(train_loss) or not math.isfinite(val_loss):
                raise SystemExit(f"non-finite eval loss at step={step}: train={train_loss} val={val_loss}")
            print(f"step={step} train_loss={train_loss:.4f} val_loss={val_loss:.4f}")
            if val_loss < best_val:
                best_val = val_loss
                non_improving_evals = 0
                save_checkpoint(best_out, model, config, tokenizer, step, optimizer, scaler, best_val, metadata)
                print(f"saved best checkpoint: {best_out} val_loss={best_val:.4f} step={step}")
            else:
                non_improving_evals += 1
                overfit_gap = val_loss - train_loss
                overfit_stop_after = args.overfit_stop_after_evals or args.early_stop_patience
                if (
                    args.overfit_stop_gap > 0
                    and overfit_stop_after > 0
                    and step >= args.overfit_stop_min_step
                    and non_improving_evals >= overfit_stop_after
                    and overfit_gap >= args.overfit_stop_gap
                ):
                    print(
                        "early stopping: overfit signal "
                        f"non_improving_evals={non_improving_evals} "
                        f"train_val_gap={overfit_gap:.4f} "
                        f"threshold={args.overfit_stop_gap:.4f} "
                        f"best_val={best_val:.4f}"
                    )
                    break
                if args.early_stop_patience > 0 and non_improving_evals >= args.early_stop_patience:
                    print(
                        f"early stopping: no val improvement for {non_improving_evals} evals "
                        f"(best_val={best_val:.4f})"
                    )
                    break

        if step >= args.steps:
            break
        lr = learning_rate(step, args)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        try:
            for _ in range(args.grad_accum_steps):
                xb, yb = get_batch(train_data, args.batch_size, config.block_size, device)
                with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                    _, loss = model(xb, yb)
                    assert loss is not None
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"non-finite train loss at step={step}: {float(loss.item())}")
                    loss = loss / args.grad_accum_steps
                scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            for param in model.parameters():
                if param.grad is not None and not torch.isfinite(param.grad).all():
                    raise FloatingPointError(f"non-finite gradient at step={step}")
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                if is_cuda_device(device):
                    torch.cuda.empty_cache()
                raise SystemExit(
                    "CUDA out of memory during training. "
                    "No checkpoint was overwritten for this failed step. "
                    "Retry with a smaller --batch-size, --block-size, or model size."
                ) from exc
            raise

        completed_step = step + 1
        if args.log_every > 0 and completed_step > 0 and completed_step % args.log_every == 0:
            now = monotonic()
            elapsed = max(1e-9, now - train_started_at)
            recent_elapsed = max(1e-9, now - last_progress_log_at)
            steps_per_min = (completed_step - start_step) / elapsed * 60.0
            recent_steps_per_min = args.log_every / recent_elapsed * 60.0
            print(
                f"progress step={completed_step} "
                f"lr={lr:.6g} "
                f"steps_per_min={steps_per_min:.4f} "
                f"recent_steps_per_min={recent_steps_per_min:.4f}"
            )
            last_progress_log_at = now
        if args.save_every > 0 and completed_step > 0 and completed_step % args.save_every == 0:
            save_checkpoint(args.out, model, config, tokenizer, completed_step, optimizer, scaler, best_val, metadata)
            print(f"saved checkpoint: {args.out} step={completed_step}")

    final_step = getattr(optimizer, "step_count", completed_step)
    if isinstance(final_step, int) and final_step != completed_step and completed_step >= args.steps:
        completed_step = final_step
    save_checkpoint(args.out, model, config, tokenizer, completed_step, optimizer, scaler, best_val, metadata)
    print(f"saved checkpoint: {args.out} step={completed_step}")


if __name__ == "__main__":
    main()
