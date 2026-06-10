from __future__ import annotations

import argparse
import re
from pathlib import Path

import torch

from kobun_llm.checkpoint_io import load_trusted_checkpoint
from kobun_llm.device import describe_device, resolve_device
from kobun_llm.genre_rules import MORA_JOINERS, waka_score
from kobun_llm.grammar import grammar_score
from kobun_llm.grammar_constraints import GrammarLogitsProcessor
from kobun_llm.model import GPT, GPTConfig
from kobun_llm.tokenizer import CharTokenizer


TARGET_METER = (5, 7, 5, 7, 7)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate waka candidates and rerank by Kobun/waka rules.")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--prompt", type=str, default="あしひきの")
    parser.add_argument("--candidates", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--presence-penalty", type=float, default=0.15)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "dml"], default="auto")
    return parser.parse_args()


def latest_checkpoint() -> Path:
    candidates = sorted(Path("checkpoints").glob("kobun_qwen3_12l_worldclass_*_best.pt"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not candidates:
        raise SystemExit("No kobun_qwen3_12l_worldclass_*_best.pt checkpoint found.")
    return candidates[0]


def clean_first_line(text: str) -> str:
    line = text.splitlines()[0]
    line = re.sub(r"[「」『』（）()\[\]【】]", "", line)
    return line.strip()


def trim_to_mora(text: str, limit: int = 31) -> str:
    count = 0
    kept = []
    for ch in text:
        if ch in "、。 　\t\r\n/|":
            continue
        kept.append(ch)
        if ch not in MORA_JOINERS:
            count += 1
        if count >= limit:
            break
    return "".join(kept)


def mora_len(text: str) -> int:
    count = 0
    for ch in text:
        if ch in "、。 　\t\r\n/|":
            continue
        if ch in MORA_JOINERS:
            continue
        count += 1
    return count


def segment_rough_57577(text: str) -> str:
    compact = re.sub(r"[、。\s/|]+", "", text)
    parts: list[str] = []
    current = ""
    count = 0
    target_index = 0
    for ch in compact:
        current += ch
        if ch not in MORA_JOINERS:
            count += 1
        if target_index < len(TARGET_METER) and count == TARGET_METER[target_index]:
            parts.append(current)
            current = ""
            count = 0
            target_index += 1
    if target_index == len(TARGET_METER) and not current:
        return "/".join(parts)
    return text


def repetition_penalty(text: str) -> int:
    penalty = 0
    for size in (2, 3, 4):
        seen: set[str] = set()
        for index in range(0, max(0, len(text) - size + 1)):
            chunk = text[index : index + size]
            if chunk in seen:
                penalty += 1
            seen.add(chunk)
    return penalty


def combined_score(text: str) -> tuple[int, str]:
    segmented = segment_rough_57577(text)
    waka = waka_score(segmented)
    total = mora_len(text)
    meter_bonus = 8 if "/" in segmented else 0
    meter_penalty = min(14, abs(total - 31))
    score = waka.score + grammar_score(segmented, style="waka") + meter_bonus - meter_penalty - repetition_penalty(text)
    return score, segmented


@torch.no_grad()
def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint or latest_checkpoint()
    device = resolve_device(args.device)
    print(f"device={describe_device(device)}")
    print(f"checkpoint={checkpoint}")

    payload = load_trusted_checkpoint(checkpoint, map_location=device)
    tokenizer = CharTokenizer.from_dict(payload["tokenizer"])
    model = GPT(GPTConfig(**payload["config"])).to(device)
    if model.config.tie_word_embeddings:
        model.tie_weights()
    model.load_state_dict(payload["model"])
    if model.config.tie_word_embeddings:
        model.tie_weights()
    model.eval()
    logits_processor = GrammarLogitsProcessor(tokenizer=tokenizer, hard=False, bias=6.0)

    best: tuple[int, str, str] | None = None
    prompt_ids = tokenizer.encode(args.prompt)
    remaining = max(1, args.candidates)
    while remaining > 0:
        batch = min(max(1, args.batch_size), remaining)
        idx = torch.tensor([prompt_ids for _ in range(batch)], dtype=torch.long, device=device)
        out = model.generate(
            idx,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            presence_penalty=args.presence_penalty,
            logits_processor=logits_processor,
        )
        for row in out:
            raw = trim_to_mora(clean_first_line(tokenizer.decode(row.tolist())), 31)
            score, segmented = combined_score(raw)
            if best is None or score > best[0]:
                best = (score, raw, segmented)
        remaining -= batch
    assert best is not None
    score, raw, segmented = best
    waka = waka_score(segmented)
    print(f"combined_score={score}")
    print(f"waka_score={waka.score} meter={waka.meter} reasons={list(waka.reasons)}")
    print(f"raw={raw}")
    print(segmented)


if __name__ == "__main__":
    main()
