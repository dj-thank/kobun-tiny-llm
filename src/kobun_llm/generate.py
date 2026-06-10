from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .checkpoint_io import load_trusted_checkpoint
from .device import describe_device, resolve_device
from .grammar_constraints import GrammarLogitsProcessor
from .grammar import grammar_score
from .model import GPT, GPTConfig
from .tokenizer import CharTokenizer
from .waka_meter_constraints import (
    WakaMeterLogitsProcessor,
    parse_meter_pattern,
    validate_tokenizer_coverage,
    validate_waka_prefix,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text with a trained tiny GPT-style model.")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/kobun_tiny.pt"))
    parser.add_argument("--prompt", type=str, default="春は")
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--presence-penalty", type=float, default=0.0)
    parser.add_argument("--candidates", type=int, default=1)
    parser.add_argument("--grammar-rerank", action="store_true")
    parser.add_argument("--grammar-constraints", action="store_true")
    parser.add_argument("--soft-grammar-constraints", action="store_true")
    parser.add_argument("--grammar-constraint-bias", type=float, default=8.0)
    parser.add_argument("--stop-sequence", action="append", default=[])
    parser.add_argument("--stop-at-newline", action="store_true")
    parser.add_argument("--waka-meter-constraints", action="store_true")
    parser.add_argument("--waka-meter-pattern", type=str, default="5,7,5,7,7")
    parser.add_argument("--waka-allow-kanji", action="store_true")
    parser.add_argument("--style", choices=["all", "genji", "makura", "setsuwa", "waka"], default="genji")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "dml"], default="auto")
    return parser.parse_args()


def truncate_at_stop(text: str, prompt: str, stops: list[str]) -> str:
    if not stops:
        return text
    start = len(prompt)
    best = None
    for stop in stops:
        if not stop:
            continue
        index = text.find(stop, start)
        if index >= 0:
            end = index + len(stop)
            best = end if best is None else min(best, end)
    return text[:best] if best is not None else text


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    print(f"device={describe_device(device)}")

    payload = load_trusted_checkpoint(args.checkpoint, map_location=device)
    tokenizer = CharTokenizer.from_dict(payload["tokenizer"])
    config = GPTConfig(**payload["config"])
    model = GPT(config).to(device)
    if config.tie_word_embeddings:
        model.tie_weights()
    model.load_state_dict(payload["model"])
    if config.tie_word_embeddings:
        model.tie_weights()
    model.eval()
    logits_processors = []
    if args.grammar_constraints or args.soft_grammar_constraints:
        logits_processors.append(
            GrammarLogitsProcessor(
                tokenizer=tokenizer,
                hard=not args.soft_grammar_constraints,
                bias=args.grammar_constraint_bias,
            )
        )
    if args.waka_meter_constraints:
        target = parse_meter_pattern(args.waka_meter_pattern)
        validate_waka_prefix(args.prompt, target, kana_only=not args.waka_allow_kanji)
        validate_tokenizer_coverage(args.prompt, tokenizer)
        logits_processors.append(
            WakaMeterLogitsProcessor(tokenizer=tokenizer, target=target, kana_only=not args.waka_allow_kanji)
        )

    def process_logits(idx: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        for processor in logits_processors:
            logits = processor(idx, logits)
        return logits

    stops = list(args.stop_sequence)
    if args.stop_at_newline or args.waka_meter_constraints:
        stops.append("\n")

    best_text = ""
    best_score = -10**9
    for _ in range(max(1, args.candidates)):
        if logits_processors:
            validate_tokenizer_coverage(args.prompt, tokenizer)
        idx = torch.tensor([tokenizer.encode(args.prompt)], dtype=torch.long, device=device)
        out = model.generate(
            idx,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            presence_penalty=args.presence_penalty,
            logits_processor=process_logits if logits_processors else None,
        )
        text = truncate_at_stop(tokenizer.decode(out[0].tolist()), args.prompt, stops)
        score = grammar_score(text, style=args.style) if args.grammar_rerank else 0
        if score > best_score:
            best_text = text
            best_score = score
    if args.grammar_rerank:
        print(f"grammar_score={best_score}")
    print(best_text)


if __name__ == "__main__":
    main()
