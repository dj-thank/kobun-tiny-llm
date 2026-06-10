from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from kobun_llm.checkpoint_io import load_trusted_checkpoint
from kobun_llm.device import describe_device, resolve_device
from kobun_llm.genre_rules import waka_meter
from kobun_llm.grammar import grammar_score
from kobun_llm.grammar_constraints import GrammarLogitsProcessor
from kobun_llm.model import GPT, GPTConfig
from kobun_llm.tokenizer import CharTokenizer
from kobun_llm.waka_meter_constraints import (
    WakaMeterLogitsProcessor,
    parse_meter_pattern,
    validate_tokenizer_coverage,
    validate_waka_prefix,
)
from latest_valid_checkpoint import is_valid_checkpoint


DEFAULT_PROMPTS_FILE = Path("data/eval/waka_generation_prompts.jsonl")


def latest_worldclass_checkpoint() -> Path:
    candidates = sorted(
        Path("checkpoints").glob("kobun_qwen3_12l_worldclass_*_best.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for candidate in candidates:
        if is_valid_checkpoint(candidate):
            return candidate
    raise SystemExit("No valid kobun_qwen3_12l_worldclass_*_best.pt checkpoint found.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate waka with strict 5-7-5-7-7 meter constraints and verify exact meter.")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument(
        "--prompts-file",
        type=Path,
        default=DEFAULT_PROMPTS_FILE,
        help="JSONL prompt set for release-gate constrained-generation smoke evidence.",
    )
    parser.add_argument(
        "--prompts",
        nargs="*",
        default=None,
        help="Inline diagnostic prompts. Release quality runners use --prompts-file instead.",
    )
    parser.add_argument("--meter", type=str, default="5,7,5,7,7")
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--presence-penalty", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260509)
    parser.add_argument(
        "--decoding",
        choices=["greedy", "sample"],
        default="greedy",
        help="Use greedy argmax for release-gate evidence; sampling is diagnostic only.",
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "dml"], default="auto")
    parser.add_argument("--min-cases", type=int, default=4)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_prompt_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        prompt = str(row.get("prompt") or "")
        if not prompt:
            raise ValueError(f"{path}:{line_no} missing prompt")
        if row.get("llm_generated_eval_answer_text") is not False:
            raise ValueError(f"{path}:{line_no} must attest llm_generated_eval_answer_text=false")
        rows.append(row)
    if not rows:
        raise ValueError(f"{path} has no prompt rows")
    return rows


def load_prompts(args: argparse.Namespace) -> list[str]:
    if args.prompts is not None:
        print("waka_generation_prompts_source=inline_diagnostic")
        return [str(prompt) for prompt in args.prompts]
    rows = read_prompt_rows(args.prompts_file)
    print(
        "waka_generation_prompts_file="
        f"{args.prompts_file} sha256={sha256_file(args.prompts_file)} rows={len(rows)}"
    )
    return [str(row["prompt"]) for row in rows]


@torch.no_grad()
def generate_with_constraints(
    model: GPT,
    idx: torch.Tensor,
    *,
    tokenizer: CharTokenizer,
    max_new_tokens: int,
    decoding: str,
    temperature: float,
    top_k: int | None,
    top_p: float | None,
    presence_penalty: float,
    logits_processor,
) -> torch.Tensor:
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.config.block_size :]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :]
        if presence_penalty > 0:
            for batch_index in range(idx.size(0)):
                used = torch.unique(idx[batch_index])
                logits[batch_index, used] -= presence_penalty
        logits = logits_processor(idx, logits)
        if not torch.isfinite(logits).any(dim=-1).all():
            raise RuntimeError("Waka constrained generation produced no finite legal token.")

        if decoding == "greedy":
            next_id = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            logits = logits / max(temperature, 1e-6)
            if top_k is not None:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits = logits.masked_fill(logits < values[:, [-1]], -float("inf"))
            if top_p is not None and 0 < top_p < 1:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                sorted_probs = F.softmax(sorted_logits, dim=-1)
                cumulative = torch.cumsum(sorted_probs, dim=-1)
                remove = cumulative > top_p
                remove[..., 1:] = remove[..., :-1].clone()
                remove[..., 0] = False
                sorted_logits = sorted_logits.masked_fill(remove, -float("inf"))
                logits = torch.full_like(logits, -float("inf"))
                logits.scatter_(dim=-1, index=sorted_indices, src=sorted_logits)
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)

        idx = torch.cat((idx, next_id), dim=1)
        next_text = tokenizer.decode(next_id[0].tolist())
        if "\n" in next_text:
            break
    return idx


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint or latest_worldclass_checkpoint()
    target = parse_meter_pattern(args.meter)
    device = resolve_device(args.device)
    torch.manual_seed(args.seed)
    print(f"device={describe_device(device)}")
    print(f"checkpoint={checkpoint}")
    print(f"checkpoint_sha256={sha256_file(checkpoint)}")
    prompts = load_prompts(args)
    print(
        "waka_generation_decoding="
        f"{args.decoding} seed={args.seed} temperature={args.temperature} "
        f"top_k={args.top_k} top_p={args.top_p} presence_penalty={args.presence_penalty}"
    )

    payload = load_trusted_checkpoint(checkpoint, map_location=device)
    tokenizer = CharTokenizer.from_dict(payload["tokenizer"])
    model = GPT(GPTConfig(**payload["config"])).to(device)
    if model.config.tie_word_embeddings:
        model.tie_weights()
    model.load_state_dict(payload["model"])
    if model.config.tie_word_embeddings:
        model.tie_weights()
    model.eval()

    total = 0
    passed = 0
    for prompt in prompts:
        validate_waka_prefix(prompt, target, kana_only=True)
        validate_tokenizer_coverage(prompt, tokenizer)
        processors = [
            GrammarLogitsProcessor(tokenizer=tokenizer, hard=False, bias=8.0),
            WakaMeterLogitsProcessor(tokenizer=tokenizer, target=target, kana_only=True),
        ]

        def process_logits(idx: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
            for processor in processors:
                logits = processor(idx, logits)
            return logits

        idx = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
        out = generate_with_constraints(
            model,
            idx,
            tokenizer=tokenizer,
            max_new_tokens=args.max_new_tokens,
            decoding=args.decoding,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            presence_penalty=args.presence_penalty,
            logits_processor=process_logits,
        )
        text = tokenizer.decode(out[0].tolist())
        poem = text.splitlines()[0]
        meter = waka_meter(poem)
        ok = meter == target
        total += 1
        passed += int(ok)
        print(f"ok={ok} meter={meter} grammar_score={grammar_score(poem, style='waka')} poem={poem}")
    print(f"waka_meter_constrained_generation_accuracy={passed}/{total}={passed / max(1, total):.3f}")
    if total < args.min_cases:
        raise SystemExit(f"only {total} cases found, below --min-cases {args.min_cases}")
    if passed != total:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
