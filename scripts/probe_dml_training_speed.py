from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from kobun_llm.device import describe_device, device_backend, resolve_device
from kobun_llm.model import GPT, GPTConfig
from kobun_llm.tokenizer import BYTE_FALLBACK_TOKENIZER_TYPE, tokenizer_from_text
from kobun_llm.train import SimpleAdamW, get_batch

DML_PROCESS_MARKERS = (
    "kobun_llm.train",
    "train_old_japanese_0_1b_dml.ps1",
    "watch_and_finalize_old_japanese_0_1b_dml.ps1",
    "start_old_japanese_0_1b_dml_and_watch.ps1",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Short, non-release DirectML speed probe for tokenizer/block/optimizer choices."
    )
    parser.add_argument("--data", type=Path, default=Path("data/kobun_worldclass_corpus.txt"))
    parser.add_argument("--tokenizer-extra-data", type=Path, default=Path("data/tokenizer_public_char_vocab.txt"))
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--device", choices=["auto", "dml", "cuda", "cpu"], default="dml")
    parser.add_argument("--blocks", type=int, nargs="+", default=[512, 384, 256])
    parser.add_argument("--optimizers", nargs="+", choices=["simple-adamw", "adamw"], default=["simple-adamw", "adamw"])
    parser.add_argument("--steps", type=int, default=2, help="Measured optimizer steps after warmup.")
    parser.add_argument("--warmup-steps", type=int, default=1, help="Unmeasured warmup steps per configuration.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--n-layer", type=int, default=16)
    parser.add_argument("--n-head", type=int, default=12)
    parser.add_argument("--num-key-value-heads", type=int, default=6)
    parser.add_argument("--n-embd", type=int, default=768)
    parser.add_argument("--intermediate-size", type=int, default=2304)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--allow-active-run", action="store_true")
    return parser.parse_args()


def active_lock_exists() -> bool:
    return (Path("logs") / "active_old_japanese_0_1b_dml.lock").exists()


def _is_self_or_parent(pid: int) -> bool:
    if pid <= 0:
        return False
    current = os.getpid()
    if pid == current:
        return True
    if sys.platform == "win32":
        probe = current
        seen: set[int] = set()
        while probe > 0 and probe not in seen:
            seen.add(probe)
            result = subprocess.run(
                [
                    "powershell",
                    "-NoProfile",
                    "-Command",
                    f"$p=Get-CimInstance Win32_Process -Filter 'ProcessId={probe}'; if($p){{ $p.ParentProcessId }}",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            try:
                parent = int(result.stdout.strip())
            except ValueError:
                return False
            if parent == pid:
                return True
            probe = parent
        return False
    return False


def _command_is_live_dml_training(command_line: str) -> bool:
    if not command_line or "old_japanese_0_1b_dml_" not in command_line:
        return False
    if not any(marker in command_line for marker in DML_PROCESS_MARKERS):
        return False
    lowered = command_line.lower()
    if "--device cpu" in lowered or "--device cuda" in lowered or "--device hip" in lowered:
        return False
    return True


def live_dml_training_processes() -> list[dict[str, Any]]:
    live: list[dict[str, Any]] = []
    if sys.platform == "win32":
        result = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "Get-CimInstance Win32_Process | "
                "Where-Object { $_.CommandLine -like '*old_japanese_0_1b_dml_*' } | "
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
            pid = int(payload.get("ProcessId") or 0)
            command_line = str(payload.get("CommandLine") or "")
            if _is_self_or_parent(pid):
                continue
            if _command_is_live_dml_training(command_line):
                live.append(
                    {
                        "pid": pid,
                        "name": payload.get("Name"),
                        "command_preview": command_line[:260],
                    }
                )
        return live

    for cmdline in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            command_line = cmdline.read_bytes().replace(b"\x00", b" ").decode("utf-8", errors="replace")
        except OSError:
            continue
        pid = int(cmdline.parent.name)
        if _is_self_or_parent(pid):
            continue
        if _command_is_live_dml_training(command_line):
            live.append({"pid": pid, "name": "process", "command_preview": command_line[:260]})
    return live


def assert_no_active_dml_training(*, allow_active_run: bool) -> None:
    if allow_active_run:
        return
    reasons: list[str] = []
    if active_lock_exists():
        reasons.append("active DirectML lock exists")
    live = live_dml_training_processes()
    if live:
        preview = "; ".join(f"pid={row['pid']} command={row['command_preview']}" for row in live[:5])
        reasons.append(f"live DirectML training/supervisor process exists: {preview}")
    if reasons:
        raise SystemExit(
            "refusing to run a GPU speed probe concurrently; "
            + " | ".join(reasons)
            + ". Stop/supersede the active run first, or pass --allow-active-run for a deliberate non-release probe."
        )


def make_tokenizer(args: argparse.Namespace) -> Any:
    text = args.data.read_text(encoding="utf-8")
    if args.tokenizer_extra_data.exists():
        text += args.tokenizer_extra_data.read_text(encoding="utf-8")
    return tokenizer_from_text(text, tokenizer_type=BYTE_FALLBACK_TOKENIZER_TYPE)


def encoded_data(tokenizer: Any, data_path: Path) -> torch.Tensor:
    ids = tokenizer.encode(data_path.read_text(encoding="utf-8"))
    return torch.tensor(ids, dtype=torch.long)


def make_model(vocab_size: int, block_size: int, args: argparse.Namespace) -> GPT:
    config = GPTConfig(
        vocab_size=vocab_size,
        block_size=block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        num_key_value_heads=args.num_key_value_heads,
        n_embd=args.n_embd,
        dropout=args.dropout,
        intermediate_size=args.intermediate_size,
        norm_type="rmsnorm",
        mlp_type="swiglu",
        use_rope=True,
        rope_theta=1_000_000.0,
        tie_word_embeddings=True,
        attention_bias=False,
        mlp_bias=False,
        qk_norm=True,
    )
    return GPT(config)


def make_optimizer(name: str, model: GPT, lr: float) -> Any:
    if name == "simple-adamw":
        return SimpleAdamW(model.parameters(), lr=lr)
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr)
    raise ValueError(f"unknown optimizer: {name}")


def probe_one(
    data: torch.Tensor,
    vocab_size: int,
    block_size: int,
    optimizer_name: str,
    device: Any,
    args: argparse.Namespace,
) -> dict[str, Any]:
    model = make_model(vocab_size, block_size, args)
    params = sum(param.numel() for param in model.parameters())
    model = model.to(device)
    if model.config.tie_word_embeddings:
        model.tie_weights()
    optimizer = make_optimizer(optimizer_name, model, lr=args.lr)
    model.train()

    losses: list[float] = []
    for _ in range(max(args.warmup_steps, 0)):
        optimizer.zero_grad(set_to_none=True)
        xb, yb = get_batch(data, args.batch_size, block_size, device)
        _, loss = model(xb, yb)
        if loss is None or not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss for block={block_size} optimizer={optimizer_name}")
        loss.backward()
        optimizer.step()

    step_seconds: list[float] = []
    start = time.perf_counter()
    for _ in range(args.steps):
        step_start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        xb, yb = get_batch(data, args.batch_size, block_size, device)
        _, loss = model(xb, yb)
        if loss is None or not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss for block={block_size} optimizer={optimizer_name}")
        loss.backward()
        optimizer.step()
        step_seconds.append(time.perf_counter() - step_start)
        losses.append(float(loss.detach().cpu().item()))
    elapsed = max(time.perf_counter() - start, 1e-9)
    sorted_seconds = sorted(step_seconds)
    median_step_seconds = sorted_seconds[len(sorted_seconds) // 2] if sorted_seconds else None

    del optimizer
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "block_size": block_size,
        "optimizer": optimizer_name,
        "steps": args.steps,
        "warmup_steps": args.warmup_steps,
        "batch_size": args.batch_size,
        "params": params,
        "elapsed_seconds": elapsed,
        "steps_per_min": args.steps * 60.0 / elapsed,
        "median_step_seconds": median_step_seconds,
        "step_seconds": step_seconds,
        "last_loss": losses[-1] if losses else None,
    }


def main() -> None:
    args = parse_args()
    assert_no_active_dml_training(allow_active_run=args.allow_active_run)
    device = resolve_device(args.device)
    tokenizer = make_tokenizer(args)
    data = encoded_data(tokenizer, args.data)
    results = []
    for block_size in args.blocks:
        if len(data) <= block_size + 1:
            raise SystemExit(f"data is too short for block_size={block_size}")
        for optimizer_name in args.optimizers:
            results.append(probe_one(data, tokenizer.vocab_size, block_size, optimizer_name, device, args))

    by_key = {(row["block_size"], row["optimizer"]): row for row in results}
    fastest = min(results, key=lambda row: (-float(row["steps_per_min"]), int(row["block_size"])))
    block384_simple = by_key.get((384, "simple-adamw"))
    recommendation = {
        "block_size": 384 if block384_simple else fastest["block_size"],
        "speed_fallback_block_size": fastest["block_size"],
        "optimizer": "simple-adamw",
        "reason": (
            "prefer block=384 for release-candidate context length; use the fastest "
            "block only as an explicit speed fallback. Default to checkpoint-safe "
            "SimpleAdamW unless AdamW is >=1.20x faster for block=384."
        ),
    }
    simple_384 = block384_simple
    adamw_384 = by_key.get((384, "adamw"))
    if simple_384 and adamw_384:
        speedup = float(adamw_384["steps_per_min"]) / max(float(simple_384["steps_per_min"]), 1e-9)
        if speedup >= 1.2:
            recommendation["optimizer"] = "adamw"
            recommendation["adamw_speedup_vs_simple_adamw_block384"] = speedup

    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "backend": device_backend(device),
        "device_description": describe_device(device),
        "tokenizer_type": BYTE_FALLBACK_TOKENIZER_TYPE,
        "vocab_size": tokenizer.vocab_size,
        "data": args.data.name,
        "results": results,
        "recommendation": recommendation,
        "release_policy": "speed probe only; no checkpoint, no export, no package, no upload",
    }
    out = args.out or Path("logs") / f"dml_speed_probe_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"speed_probe_json={out}")
    print(json.dumps({"vocab_size": tokenizer.vocab_size, "recommendation": recommendation}, ensure_ascii=False))


if __name__ == "__main__":
    main()
