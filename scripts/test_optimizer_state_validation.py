from __future__ import annotations

import copy

import torch

from kobun_llm.optimizer_state import validate_simple_adamw_state


def valid_payload(param: torch.nn.Parameter) -> dict[str, object]:
    return {
        "optimizer_type": "simple-adamw",
        "step_count": 3,
        "lr": 1e-4,
        "betas": (0.9, 0.999),
        "eps": 1e-8,
        "weight_decay": 0.01,
        "exp_avg": [torch.zeros_like(param)],
        "exp_avg_sq": [torch.zeros_like(param)],
    }


def must_fail(payload: dict[str, object], param: torch.nn.Parameter, reason: str) -> None:
    try:
        validate_simple_adamw_state(payload, [param], require_step=3)
    except ValueError:
        return
    raise SystemExit(f"optimizer validation unexpectedly accepted bad state: {reason}")


def main() -> None:
    param = torch.nn.Parameter(torch.zeros(2, 3, dtype=torch.float32))
    payload = valid_payload(param)
    step = validate_simple_adamw_state(payload, [param], require_step=3)
    if step != 3:
        raise SystemExit(f"unexpected optimizer step: {step}")

    bad = copy.deepcopy(payload)
    bad["extra"] = True
    must_fail(bad, param, "extra key")

    bad = copy.deepcopy(payload)
    bad["step_count"] = 2
    must_fail(bad, param, "step mismatch")

    bad = copy.deepcopy(payload)
    bad["exp_avg"] = [torch.zeros(2, 3, dtype=torch.float64)]
    must_fail(bad, param, "dtype mismatch")

    bad = copy.deepcopy(payload)
    bad["exp_avg_sq"] = [torch.zeros(3, 2, dtype=torch.float32)]
    must_fail(bad, param, "shape mismatch")

    bad = copy.deepcopy(payload)
    bad["exp_avg"] = [torch.full((2, 3), float("nan"), dtype=torch.float32)]
    must_fail(bad, param, "non-finite tensor")

    print("optimizer_state_validation_ok=true")


if __name__ == "__main__":
    main()
