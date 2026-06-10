from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from kobun_llm.checkpoint_io import load_trusted_checkpoint
from kobun_llm.device import describe_device, resolve_device
from kobun_llm.model import GPT, GPTConfig
from kobun_llm.tokenizer import CharTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate heldout character-level LM loss for a checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "dml"], default="auto")
    parser.add_argument("--split-name", default="heldout", choices=["heldout", "validation", "test"])
    parser.add_argument("--max-loss", type=float, default=8.0)
    return parser.parse_args()


def sha256_text(path: Path) -> str:
    return hashlib.sha256(path.read_text(encoding="utf-8").encode("utf-8")).hexdigest()


@torch.no_grad()
def eval_loss(model: GPT, tokenizer: CharTokenizer, text: str, device: object) -> tuple[float, int]:
    ids = tokenizer.encode(text)
    if len(ids) < 2:
        raise SystemExit("heldout eval text is too short.")
    block_size = model.config.block_size
    total_loss = 0.0
    total_tokens = 0
    for start in range(0, len(ids) - 1, block_size):
        chunk = ids[start : start + block_size + 1]
        if len(chunk) < 2:
            continue
        idx = torch.tensor([chunk[:-1]], dtype=torch.long, device=device)
        targets = torch.tensor([chunk[1:]], dtype=torch.long, device=device)
        logits, _ = model(idx)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), reduction="sum")
        token_count = targets.numel()
        total_loss += float(loss.item())
        total_tokens += token_count
    if total_tokens <= 0:
        raise SystemExit("heldout eval produced zero target tokens.")
    return total_loss / total_tokens, total_tokens


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    print(f"device={describe_device(device)}")
    payload = load_trusted_checkpoint(args.checkpoint, map_location="cpu")
    tokenizer = CharTokenizer.from_dict(payload["tokenizer"])
    model = GPT(GPTConfig(**payload["config"])).to(device)
    if model.config.tie_word_embeddings:
        model.tie_weights()
    model.load_state_dict(payload["model"])
    if model.config.tie_word_embeddings:
        model.tie_weights()
    model.eval()
    text = args.data.read_text(encoding="utf-8")
    loss, tokens = eval_loss(model, tokenizer, text, device)
    metric_prefix = f"{args.split_name}_lm"
    if not math.isfinite(loss):
        raise SystemExit(f"{metric_prefix}_token_nll is not finite: {loss}")
    perplexity = math.exp(min(50.0, loss))
    print(
        f"{metric_prefix}_token_nll={loss:.6f} {metric_prefix}_perplexity={perplexity:.6f} "
        f"{metric_prefix}_tokens={tokens} {args.split_name}_data={args.data} "
        f"{args.split_name}_sha256={sha256_text(args.data)}"
    )
    if loss > args.max_loss:
        raise SystemExit(f"{metric_prefix}_token_nll {loss:.6f} exceeds --max-loss {args.max_loss:.6f}")


if __name__ == "__main__":
    main()
