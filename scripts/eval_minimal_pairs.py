from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from kobun_llm.checkpoint_io import load_trusted_checkpoint
from kobun_llm.device import describe_device, resolve_device
from kobun_llm.grammar import grammar_score
from kobun_llm.morphology import annotate, morphology_score
from kobun_llm.model import GPT, GPTConfig
from kobun_llm.tokenizer import CharTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate grammar minimal pairs.")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/kobun_qwen3_style_gpu_best.pt"))
    parser.add_argument("--pairs", type=Path, default=Path("data/eval/grammar_minimal_pairs.jsonl"))
    parser.add_argument("--min-cases", type=int, default=1)
    parser.add_argument("--min-accuracy", type=float, default=0.0)
    parser.add_argument(
        "--metric-prefix",
        default="contrastive",
        help="Prefix for emitted metric keys, e.g. primary or heldout.",
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "dml"], default="auto")
    return parser.parse_args()


@torch.no_grad()
def continuation_nll(model: GPT, tokenizer: CharTokenizer, prompt: str, continuation: str, device: str) -> float:
    prompt_ids = tokenizer.encode(prompt)
    ids = tokenizer.encode(prompt + continuation)
    if len(ids) < 2:
        return float("inf")
    idx = torch.tensor([ids[:-1]], dtype=torch.long, device=device)
    targets = torch.tensor([ids[1:]], dtype=torch.long, device=device)
    logits, _ = model(idx)
    start = max(0, len(prompt_ids) - 1)
    logits = logits[:, start:, :]
    targets = targets[:, start:]
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), reduction="mean")
    return float(loss.item())


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    print(f"device={describe_device(device)}")
    payload = load_trusted_checkpoint(args.checkpoint, map_location=device)
    tokenizer = CharTokenizer.from_dict(payload["tokenizer"])
    model = GPT(GPTConfig(**payload["config"])).to(device)
    if model.config.tie_word_embeddings:
        model.tie_weights()
    model.load_state_dict(payload["model"])
    if model.config.tie_word_embeddings:
        model.tie_weights()
    model.eval()

    total = 0
    preferred = 0
    grammar_preferred = 0
    for line in args.pairs.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        prompt = str(row["prompt"])
        good = prompt + row["good"]
        bad = prompt + row["bad"]
        good_nll = continuation_nll(model, tokenizer, prompt, str(row["good"]), device)
        bad_nll = continuation_nll(model, tokenizer, prompt, str(row["bad"]), device)
        good_grammar = grammar_score(good, style="genji")
        bad_grammar = grammar_score(bad, style="genji")
        good_morph = morphology_score(good)
        bad_morph = morphology_score(bad)
        model_ok = good_nll < bad_nll
        grammar_ok = good_grammar > bad_grammar
        total += 1
        preferred += int(model_ok)
        grammar_preferred += int(grammar_ok)
        print(
            f"{row['rule_ids']} model_ok={model_ok} grammar_ok={grammar_ok} "
            f"good_nll={good_nll:.3f} bad_nll={bad_nll:.3f} "
            f"good_g={good_grammar} bad_g={bad_grammar} "
            f"good_morph={good_morph} bad_morph={bad_morph}"
        )
    print(f"{args.metric_prefix}_contrastive_preference_accuracy={preferred}/{total}={preferred / max(1, total):.3f}")
    print(f"{args.metric_prefix}_grammar_rule_preference={grammar_preferred}/{total}={grammar_preferred / max(1, total):.3f}")
    if total < args.min_cases:
        raise SystemExit(f"case count {total} is below required --min-cases {args.min_cases}")
    accuracy = preferred / max(1, total)
    if accuracy < args.min_accuracy:
        raise SystemExit(f"accuracy {accuracy:.3f} is below required --min-accuracy {args.min_accuracy:.3f}")


if __name__ == "__main__":
    main()
