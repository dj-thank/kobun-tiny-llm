from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from kobun_llm.checkpoint_io import load_trusted_checkpoint
from kobun_llm.model import GPT, GPTConfig
from kobun_llm.optimizer_state import validate_simple_adamw_state


EXPECTED_CONFIG: dict[str, Any] = {
    "block_size": 384,
    "n_layer": 16,
    "n_head": 12,
    "num_key_value_heads": 6,
    "n_embd": 768,
    "intermediate_size": 2304,
    "norm_type": "rmsnorm",
    "mlp_type": "swiglu",
    "use_rope": True,
    "rope_theta": 1_000_000.0,
    "tie_word_embeddings": True,
    "attention_bias": False,
    "mlp_bias": False,
    "qk_norm": True,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check that a checkpoint matches the old-japanese-0.1B target.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--min-params", type=int, default=100_000_000)
    parser.add_argument("--max-params", type=int, default=250_000_000)
    parser.add_argument("--strict-config", action="store_true")
    parser.add_argument("--require-release-prefix", default="")
    parser.add_argument("--fail-on-val-oov", action="store_true")
    parser.add_argument("--require-from-scratch", action="store_true")
    parser.add_argument("--require-seed", action="store_true")
    parser.add_argument("--require-optimizer", default="")
    parser.add_argument("--require-backend", choices=["cpu", "cuda", "dml", "hip"], default="")
    return parser.parse_args()


def optimizer_checkpoint_step(
    optimizer_payload: dict[str, Any],
    optimizer_kind: str,
    model: torch.nn.Module,
    require_step: int,
) -> int:
    if optimizer_kind == "simple-adamw":
        try:
            return validate_simple_adamw_state(optimizer_payload, model.parameters(), require_step=require_step)
        except ValueError as exc:
            raise SystemExit(f"simple-adamw optimizer state is unsafe: {exc}") from exc
    if optimizer_kind == "adamw":
        if optimizer_payload.get("optimizer_type") == "simple-adamw":
            raise SystemExit("checkpoint metadata says adamw but optimizer state is simple-adamw")
        state = optimizer_payload.get("state")
        if not isinstance(state, dict) or not state:
            raise SystemExit("adamw optimizer state is missing or empty")
        steps: list[int] = []
        for param_state in state.values():
            if not isinstance(param_state, dict) or "step" not in param_state:
                continue
            raw_step = param_state["step"]
            if torch.is_tensor(raw_step):
                raw_step = raw_step.detach().cpu().item()
            steps.append(int(raw_step))
        if not steps:
            raise SystemExit("adamw optimizer state has no per-parameter step values")
        unique_steps = sorted(set(steps))
        if len(unique_steps) != 1:
            raise SystemExit(f"adamw optimizer state has inconsistent per-parameter steps: {unique_steps[:10]}")
        return unique_steps[0]
    raise SystemExit(f"unsupported optimizer kind in checkpoint metadata: {optimizer_kind!r}")


def main() -> None:
    args = parse_args()
    payload = load_trusted_checkpoint(args.checkpoint, map_location="cpu")
    missing = {"model", "config", "tokenizer"} - set(payload)
    if missing:
        raise SystemExit(f"checkpoint missing required keys: {sorted(missing)}")

    config_payload = payload["config"]
    config = GPTConfig(**config_payload)
    model = GPT(config)
    params = sum(param.numel() for param in model.parameters())
    metadata_params = payload.get("metadata", {}).get("param_count")
    metadata = dict(payload.get("metadata", {}) or {})

    print(f"checkpoint={args.checkpoint}")
    print(f"vocab_size={config.vocab_size}")
    print(f"params={params}")
    print(f"params_b={params / 1_000_000_000:.4f}")
    print(f"checkpoint_step={payload.get('step')}")
    print(f"checkpoint_best_val={payload.get('best_val')}")
    print(f"checkpoint_backend={metadata.get('backend', '')}")
    print(f"checkpoint_device_description={metadata.get('device_description', '')}")
    optimizer_payload = payload.get("optimizer")
    metadata_optimizer = str(metadata.get("optimizer") or "")
    checkpoint_step = int(payload.get("step", -1))
    if args.require_optimizer and metadata_optimizer != args.require_optimizer:
        raise SystemExit(
            f"checkpoint metadata optimizer={metadata_optimizer!r} does not match required "
            f"optimizer={args.require_optimizer!r}"
        )
    if args.require_backend:
        backend = str(metadata.get("backend") or "")
        if backend != args.require_backend:
            raise SystemExit(f"checkpoint metadata backend={backend!r} does not match required backend={args.require_backend!r}")
    if isinstance(optimizer_payload, dict) and metadata_optimizer:
        optimizer_step = optimizer_checkpoint_step(optimizer_payload, metadata_optimizer, model, checkpoint_step)
        if int(optimizer_step) != checkpoint_step:
            raise SystemExit(
                f"checkpoint step mismatch: checkpoint_step={checkpoint_step} "
                f"optimizer_step_count={optimizer_step}"
            )
    elif args.require_optimizer:
        raise SystemExit("checkpoint missing optimizer state or optimizer metadata")

    if metadata_params is not None and int(metadata_params) != params:
        raise SystemExit(f"metadata param_count mismatch: metadata={metadata_params} actual={params}")
    if not args.min_params <= params <= args.max_params:
        raise SystemExit(
            f"parameter count outside expected range: {params} "
            f"not in [{args.min_params}, {args.max_params}]"
        )

    if args.strict_config:
        mismatches = []
        for key, expected in EXPECTED_CONFIG.items():
            actual = config_payload.get(key)
            if actual != expected:
                mismatches.append(f"{key}: actual={actual!r} expected={expected!r}")
        if mismatches:
            raise SystemExit("checkpoint config mismatch:\n" + "\n".join(mismatches))
    if args.require_release_prefix:
        release_name = str(metadata.get("release_name") or "")
        if not release_name.startswith(args.require_release_prefix):
            raise SystemExit(
                f"checkpoint metadata release_name={release_name!r} does not start with "
                f"{args.require_release_prefix!r}"
            )
    if args.fail_on_val_oov and str(metadata.get("val_oov_chars") or ""):
        raise SystemExit(f"checkpoint metadata val_oov_chars is not empty: {metadata.get('val_oov_chars')!r}")
    if args.fail_on_val_oov and str(metadata.get("test_oov_chars") or ""):
        raise SystemExit(f"checkpoint metadata test_oov_chars is not empty: {metadata.get('test_oov_chars')!r}")
    if args.require_from_scratch and str(metadata.get("init_from") or ""):
        raise SystemExit(f"checkpoint was initialized from another checkpoint: init_from={metadata.get('init_from')!r}")
    if args.require_seed and metadata.get("seed") is None:
        raise SystemExit("checkpoint metadata missing seed")


if __name__ == "__main__":
    main()
