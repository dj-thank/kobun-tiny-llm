from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Callable

import torch


SIMPLE_ADAMW_RELEASE_KEYS = {
    "optimizer_type",
    "step_count",
    "lr",
    "betas",
    "eps",
    "weight_decay",
    "exp_avg",
    "exp_avg_sq",
}


def file_records_signature(records: Any) -> list[tuple[str, str, int]]:
    if records is None:
        return []
    if not isinstance(records, list):
        raise ValueError("file records metadata must be a list.")
    signature: list[tuple[str, str, int]] = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"file record {index} must be an object.")
        path_name = Path(str(record.get("path") or "")).name
        digest = str(record.get("sha256") or "")
        try:
            bytes_value = int(record.get("bytes"))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"file record {index} has invalid bytes.") from exc
        if not path_name or not digest or bytes_value < 0:
            raise ValueError(f"file record {index} missing path, sha256, or bytes.")
        signature.append((path_name, digest, bytes_value))
    return sorted(signature)


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric, got {type(value).__name__}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite, got {value!r}")
    return result


def _simple_adamw_step(optimizer_state: dict[str, Any]) -> int:
    if not isinstance(optimizer_state, dict):
        raise ValueError("simple-adamw optimizer state must be a dict.")
    actual_keys = set(optimizer_state)
    if actual_keys != SIMPLE_ADAMW_RELEASE_KEYS:
        missing = sorted(SIMPLE_ADAMW_RELEASE_KEYS - actual_keys)
        extra = sorted(actual_keys - SIMPLE_ADAMW_RELEASE_KEYS)
        raise ValueError(f"simple-adamw optimizer keys mismatch: missing={missing} extra={extra}")
    if optimizer_state.get("optimizer_type") != "simple-adamw":
        raise ValueError("optimizer_kind=simple-adamw but checkpoint state is not simple-adamw.")
    step_count = optimizer_state.get("step_count")
    if isinstance(step_count, bool) or not isinstance(step_count, int) or step_count < 0:
        raise ValueError(f"simple-adamw step_count must be a non-negative integer, got {step_count!r}")
    lr = _finite_number(optimizer_state.get("lr"), "simple-adamw lr")
    eps = _finite_number(optimizer_state.get("eps"), "simple-adamw eps")
    weight_decay = _finite_number(optimizer_state.get("weight_decay"), "simple-adamw weight_decay")
    if lr < 0.0:
        raise ValueError(f"simple-adamw lr must be >= 0, got {lr}")
    if eps <= 0.0:
        raise ValueError(f"simple-adamw eps must be > 0, got {eps}")
    if weight_decay < 0.0:
        raise ValueError(f"simple-adamw weight_decay must be >= 0, got {weight_decay}")
    betas = optimizer_state.get("betas")
    if not isinstance(betas, (list, tuple)) or len(betas) != 2:
        raise ValueError(f"simple-adamw betas must be a 2-item list/tuple, got {betas!r}")
    beta1 = _finite_number(betas[0], "simple-adamw beta1")
    beta2 = _finite_number(betas[1], "simple-adamw beta2")
    if not 0.0 <= beta1 < 1.0 or not 0.0 <= beta2 < 1.0:
        raise ValueError(f"simple-adamw betas must be in [0, 1), got {(beta1, beta2)!r}")
    exp_avg = optimizer_state.get("exp_avg")
    exp_avg_sq = optimizer_state.get("exp_avg_sq")
    if not isinstance(exp_avg, list) or not isinstance(exp_avg_sq, list):
        raise ValueError("simple-adamw exp_avg and exp_avg_sq must be lists.")
    if len(exp_avg) != len(exp_avg_sq):
        raise ValueError(
            f"simple-adamw state length mismatch: exp_avg={len(exp_avg)} exp_avg_sq={len(exp_avg_sq)}"
        )
    if not exp_avg:
        raise ValueError("simple-adamw optimizer tensor lists are empty.")
    for index, (avg, avg_sq) in enumerate(zip(exp_avg, exp_avg_sq, strict=True)):
        if not torch.is_tensor(avg) or not torch.is_tensor(avg_sq):
            raise ValueError(f"simple-adamw state tensor pair {index} must contain tensors.")
        if not avg.is_floating_point() or not avg_sq.is_floating_point():
            raise ValueError(f"simple-adamw state tensor pair {index} must be floating point.")
        if tuple(avg.shape) != tuple(avg_sq.shape):
            raise ValueError(
                f"simple-adamw state tensor pair {index} shape mismatch: "
                f"exp_avg={tuple(avg.shape)} exp_avg_sq={tuple(avg_sq.shape)}"
            )
        if avg.dtype != avg_sq.dtype:
            raise ValueError(
                f"simple-adamw state tensor pair {index} dtype mismatch: "
                f"exp_avg={avg.dtype} exp_avg_sq={avg_sq.dtype}"
            )
        if not bool(torch.isfinite(avg.detach().cpu()).all().item()):
            raise ValueError(f"simple-adamw exp_avg[{index}] contains non-finite values.")
        if not bool(torch.isfinite(avg_sq.detach().cpu()).all().item()):
            raise ValueError(f"simple-adamw exp_avg_sq[{index}] contains non-finite values.")
    return step_count


def _adamw_step(optimizer_state: dict[str, Any]) -> int:
    if not isinstance(optimizer_state, dict):
        raise ValueError("AdamW optimizer state must be a dict.")
    if optimizer_state.get("optimizer_type") == "simple-adamw":
        raise ValueError("Cannot validate a simple-adamw checkpoint as torch AdamW.")
    state = optimizer_state.get("state")
    if not isinstance(state, dict) or not state:
        raise ValueError("AdamW optimizer state is missing or empty.")
    steps: list[int] = []
    for param_index, param_state in state.items():
        if not isinstance(param_state, dict):
            raise ValueError(f"AdamW state[{param_index!r}] must be an object.")
        if "step" not in param_state:
            raise ValueError(f"AdamW state[{param_index!r}] missing step.")
        raw_step = param_state["step"]
        if torch.is_tensor(raw_step):
            raw_step = raw_step.detach().cpu().item()
        if isinstance(raw_step, bool) or not isinstance(raw_step, (int, float)):
            raise ValueError(f"AdamW state[{param_index!r}] has invalid step {raw_step!r}.")
        step = int(raw_step)
        if step < 0 or float(raw_step) != float(step):
            raise ValueError(f"AdamW state[{param_index!r}] has invalid step {raw_step!r}.")
        for tensor_key in ("exp_avg", "exp_avg_sq"):
            tensor = param_state.get(tensor_key)
            if not torch.is_tensor(tensor):
                raise ValueError(f"AdamW state[{param_index!r}] missing tensor {tensor_key}.")
            if not tensor.is_floating_point():
                raise ValueError(f"AdamW state[{param_index!r}] {tensor_key} must be floating point.")
            if not bool(torch.isfinite(tensor.detach().cpu()).all().item()):
                raise ValueError(f"AdamW state[{param_index!r}] {tensor_key} contains non-finite values.")
        exp_avg = param_state["exp_avg"]
        exp_avg_sq = param_state["exp_avg_sq"]
        if tuple(exp_avg.shape) != tuple(exp_avg_sq.shape) or exp_avg.dtype != exp_avg_sq.dtype:
            raise ValueError(f"AdamW state[{param_index!r}] exp_avg/exp_avg_sq shape or dtype mismatch.")
        steps.append(step)
    unique_steps = sorted(set(steps))
    if len(unique_steps) != 1:
        raise ValueError(f"AdamW optimizer state has inconsistent per-parameter steps: {unique_steps[:10]}")
    return unique_steps[0]


def optimizer_checkpoint_step(optimizer_state: Any, optimizer_kind: str) -> int:
    if not isinstance(optimizer_state, dict):
        raise ValueError("optimizer state must be a dict.")
    if optimizer_kind == "simple-adamw":
        return _simple_adamw_step(optimizer_state)
    if optimizer_kind == "adamw":
        return _adamw_step(optimizer_state)
    raise ValueError(f"Unsupported optimizer kind for release resume validation: {optimizer_kind!r}")


def validate_checkpoint_step_and_optimizer(payload: dict[str, Any], *, label: str = "checkpoint") -> int:
    if not isinstance(payload, dict):
        raise ValueError(f"{label} payload must be a dict.")
    step_raw = payload.get("step")
    if isinstance(step_raw, bool) or not isinstance(step_raw, int) or step_raw < 0:
        raise ValueError(f"{label} step must be a non-negative integer, got {step_raw!r}")
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError(f"{label} metadata must be a dict.")
    optimizer_kind = str(metadata.get("optimizer") or "")
    if optimizer_kind not in {"simple-adamw", "adamw"}:
        raise ValueError(f"{label} metadata optimizer is missing or unsupported: {optimizer_kind!r}")
    optimizer_state = payload.get("optimizer")
    optimizer_step = optimizer_checkpoint_step(optimizer_state, optimizer_kind)
    if optimizer_step != step_raw:
        raise ValueError(f"{label} optimizer step mismatch: checkpoint_step={step_raw} optimizer_step={optimizer_step}")
    return step_raw


def _metadata_field(metadata: dict[str, Any], key: str) -> Any:
    return metadata.get(key, "")


def _compare_metadata(current: dict[str, Any], ancestor: dict[str, Any], key: str) -> None:
    if _metadata_field(current, key) != _metadata_field(ancestor, key):
        raise ValueError(
            f"resume checkpoint metadata mismatch for {key}: "
            f"resume={ancestor.get(key)!r} current={current.get(key)!r}"
        )


def _compare_expected(metadata: dict[str, Any], key: str, expected: Any) -> None:
    if expected is None:
        return
    if _metadata_field(metadata, key) != expected:
        raise ValueError(
            f"checkpoint metadata mismatch for {key}: "
            f"checkpoint={metadata.get(key)!r} expected={expected!r}"
        )


def _resolve_resume_path(
    resume: str,
    *,
    current_path: Path,
    resolve_path: Callable[[str], Path],
) -> Path:
    candidate = resolve_path(resume)
    if candidate.is_absolute():
        return candidate
    return current_path.parent / candidate


def validate_release_resume_chain_from_payload(
    payload: dict[str, Any],
    checkpoint_path: Path | str,
    *,
    allow_same_run_resume: bool,
    expected_backend: str | None = None,
    expected_seed: int | None = None,
    expected_optimizer: str | None = None,
    expected_config: dict[str, Any] | None = None,
    expected_tokenizer: dict[str, Any] | None = None,
    expected_tokenizer_extra_data: Any = None,
    expected_provenance_files: Any = None,
    load_checkpoint: Callable[[Path], dict[str, Any]],
    resolve_path: Callable[[str], Path] | None = None,
    seen: set[str] | None = None,
) -> None:
    checkpoint = Path(checkpoint_path)
    current_id = os.path.normcase(str(checkpoint.resolve(strict=False)))
    seen = seen or set()
    if current_id in seen:
        raise ValueError(f"resume chain loop detected at {checkpoint}")
    seen.add(current_id)

    current_step = validate_checkpoint_step_and_optimizer(payload, label=str(checkpoint))
    metadata = dict(payload.get("metadata", {}) or {})
    _compare_expected(metadata, "backend", expected_backend)
    _compare_expected(metadata, "seed", expected_seed)
    _compare_expected(metadata, "optimizer", expected_optimizer)
    if expected_config is not None and payload.get("config") != expected_config:
        raise ValueError("checkpoint config does not match expected launch config.")
    if expected_tokenizer is not None and payload.get("tokenizer") != expected_tokenizer:
        raise ValueError("checkpoint tokenizer payload does not match expected launch tokenizer.")
    if expected_tokenizer_extra_data is not None:
        if file_records_signature(metadata.get("tokenizer_extra_data")) != file_records_signature(expected_tokenizer_extra_data):
            raise ValueError("checkpoint tokenizer_extra_data records do not match expected launch records.")
    if expected_provenance_files is not None:
        if file_records_signature(metadata.get("provenance_files")) != file_records_signature(expected_provenance_files):
            raise ValueError("checkpoint provenance_files records do not match expected launch records.")

    resume = str(metadata.get("resume") or "")
    if not resume:
        return
    if not allow_same_run_resume:
        raise ValueError(f"checkpoint was resumed from {resume!r}; same-run resume is not allowed here.")

    resolver = resolve_path or (lambda raw: Path(raw))
    resume_path = _resolve_resume_path(resume, current_path=checkpoint, resolve_path=resolver)
    resume_payload = load_checkpoint(resume_path)
    resume_step = validate_checkpoint_step_and_optimizer(resume_payload, label=str(resume_path))
    if resume_step >= current_step:
        raise ValueError(f"resume chain is not monotonic: resume_step={resume_step} current_step={current_step}")
    resume_metadata = dict(resume_payload.get("metadata", {}) or {})
    if str(resume_metadata.get("init_from") or ""):
        raise ValueError("resume chain is not from scratch: resume checkpoint has init_from metadata.")

    for key in (
        "run_id",
        "backend",
        "seed",
        "optimizer",
        "data_sha256",
        "val_data_sha256",
        "test_data_sha256",
        "tokenizer_source",
        "tokenizer_type",
        "byte_fallback",
        "val_oov_chars",
        "test_oov_chars",
        "release_name",
    ):
        _compare_metadata(metadata, resume_metadata, key)
    if payload.get("config") != resume_payload.get("config"):
        raise ValueError("resume checkpoint model config does not match current checkpoint config.")
    if payload.get("tokenizer") != resume_payload.get("tokenizer"):
        raise ValueError("resume checkpoint tokenizer payload does not match current checkpoint tokenizer payload.")
    for key in ("tokenizer_extra_data", "provenance_files"):
        if file_records_signature(metadata.get(key)) != file_records_signature(resume_metadata.get(key)):
            raise ValueError(f"resume checkpoint metadata mismatch for {key}.")

    validate_release_resume_chain_from_payload(
        resume_payload,
        resume_path,
        allow_same_run_resume=allow_same_run_resume,
        expected_backend=expected_backend,
        expected_seed=expected_seed,
        expected_optimizer=expected_optimizer,
        expected_config=expected_config,
        expected_tokenizer=expected_tokenizer,
        expected_tokenizer_extra_data=expected_tokenizer_extra_data,
        expected_provenance_files=expected_provenance_files,
        load_checkpoint=load_checkpoint,
        resolve_path=resolver,
        seen=seen,
    )
