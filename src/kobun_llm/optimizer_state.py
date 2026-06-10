from __future__ import annotations

import math
from typing import Any, Iterable

import torch


SIMPLE_ADAMW_KEYS = {
    "optimizer_type",
    "step_count",
    "lr",
    "betas",
    "eps",
    "weight_decay",
    "exp_avg",
    "exp_avg_sq",
}


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"SimpleAdamW {name} must be numeric, got {type(value).__name__}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"SimpleAdamW {name} must be finite, got {value!r}")
    return result


def _validate_tensor_list(
    *,
    name: str,
    tensors: Any,
    params: list[torch.nn.Parameter],
) -> list[torch.Tensor]:
    if not isinstance(tensors, list):
        raise ValueError(f"SimpleAdamW {name} must be a list")
    if len(tensors) != len(params):
        raise ValueError(f"SimpleAdamW {name} length mismatch: expected={len(params)} actual={len(tensors)}")
    validated: list[torch.Tensor] = []
    for index, (tensor, param) in enumerate(zip(tensors, params, strict=True)):
        if not torch.is_tensor(tensor):
            raise ValueError(f"SimpleAdamW {name}[{index}] is not a tensor")
        if not tensor.is_floating_point():
            raise ValueError(f"SimpleAdamW {name}[{index}] must be a floating point tensor")
        if tuple(tensor.shape) != tuple(param.shape):
            raise ValueError(
                f"SimpleAdamW {name}[{index}] shape mismatch: "
                f"expected={tuple(param.shape)} actual={tuple(tensor.shape)}"
            )
        if tensor.dtype != param.dtype:
            raise ValueError(
                f"SimpleAdamW {name}[{index}] dtype mismatch: "
                f"expected={param.dtype} actual={tensor.dtype}"
            )
        if not bool(torch.isfinite(tensor.detach()).all().item()):
            raise ValueError(f"SimpleAdamW {name}[{index}] contains non-finite values")
        validated.append(tensor)
    return validated


def validate_simple_adamw_state(
    payload: dict[str, Any],
    params: Iterable[torch.nn.Parameter],
    *,
    require_step: int | None = None,
) -> int:
    if not isinstance(payload, dict):
        raise ValueError("SimpleAdamW state must be a dict")
    actual_keys = set(payload)
    if actual_keys != SIMPLE_ADAMW_KEYS:
        missing = sorted(SIMPLE_ADAMW_KEYS - actual_keys)
        extra = sorted(actual_keys - SIMPLE_ADAMW_KEYS)
        raise ValueError(f"SimpleAdamW state keys mismatch: missing={missing} extra={extra}")
    if payload.get("optimizer_type") != "simple-adamw":
        raise ValueError("SimpleAdamW optimizer_type must be 'simple-adamw'")

    step_count_raw = payload.get("step_count")
    if isinstance(step_count_raw, bool) or not isinstance(step_count_raw, int):
        raise ValueError(f"SimpleAdamW step_count must be an integer, got {step_count_raw!r}")
    if step_count_raw < 0:
        raise ValueError(f"SimpleAdamW step_count must be non-negative, got {step_count_raw}")
    if require_step is not None and step_count_raw != int(require_step):
        raise ValueError(f"SimpleAdamW step_count mismatch: expected={require_step} actual={step_count_raw}")

    lr = _finite_number(payload.get("lr"), "lr")
    eps = _finite_number(payload.get("eps"), "eps")
    weight_decay = _finite_number(payload.get("weight_decay"), "weight_decay")
    if lr < 0.0:
        raise ValueError(f"SimpleAdamW lr must be >= 0, got {lr}")
    if eps <= 0.0:
        raise ValueError(f"SimpleAdamW eps must be > 0, got {eps}")
    if weight_decay < 0.0:
        raise ValueError(f"SimpleAdamW weight_decay must be >= 0, got {weight_decay}")

    betas_raw = payload.get("betas")
    if not isinstance(betas_raw, (list, tuple)) or len(betas_raw) != 2:
        raise ValueError(f"SimpleAdamW betas must be a 2-item list/tuple, got {betas_raw!r}")
    beta1 = _finite_number(betas_raw[0], "beta1")
    beta2 = _finite_number(betas_raw[1], "beta2")
    if not 0.0 <= beta1 < 1.0 or not 0.0 <= beta2 < 1.0:
        raise ValueError(f"SimpleAdamW betas must be in [0, 1), got {(beta1, beta2)!r}")

    trainable_params = [param for param in params if param.requires_grad]
    _validate_tensor_list(name="exp_avg", tensors=payload.get("exp_avg"), params=trainable_params)
    _validate_tensor_list(name="exp_avg_sq", tensors=payload.get("exp_avg_sq"), params=trainable_params)
    return step_count_raw
