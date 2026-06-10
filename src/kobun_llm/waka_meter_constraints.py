from __future__ import annotations

from dataclasses import dataclass

import torch

from .genre_rules import MORA_JOINERS, rough_mora_count, waka_meter
from .tokenizer import CharTokenizer


HIRAGANA = set("ぁあぃいぅうぇえぉおかがきぎくぐけげこごさざしじすずせぜそぞただちぢっつづてでとど")
HIRAGANA.update("なにぬねのはばぱひびぴふぶぷへべぺほぼぽまみむめもゃやゅゆょよらりるれろゎわゐゑをん")
HIRAGANA.update("ゔゝゞ")
SMALL_NONINITIAL = set("ぁぃぅぇぉっゃゅょゎ")
BOUNDARY = "/"
FINAL_STOP = "\n"


@dataclass(frozen=True)
class WakaMeterState:
    phrases: tuple[str, ...]
    phrase_index: int
    current_morae: int
    current_target: int
    complete: bool


def parse_meter_pattern(pattern: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in pattern.split(",") if part.strip())
    if not values or any(value <= 0 for value in values):
        raise ValueError(f"Invalid waka meter pattern: {pattern!r}")
    return values


def mora_delta(ch: str) -> int:
    if ch in MORA_JOINERS:
        return 0
    if ch in HIRAGANA:
        return 1
    return 1


def can_follow_phrase(prefix: str, ch: str, kana_only: bool) -> bool:
    if ch in {BOUNDARY, FINAL_STOP, "\r"}:
        return False
    if kana_only and ch not in HIRAGANA:
        return False
    if ch in SMALL_NONINITIAL:
        return bool(prefix) and prefix[-1] not in SMALL_NONINITIAL and prefix[-1] not in {BOUNDARY, FINAL_STOP}
    if ch in MORA_JOINERS:
        return bool(prefix) and prefix[-1] not in MORA_JOINERS and prefix[-1] not in {BOUNDARY, FINAL_STOP}
    return True


def split_strict_waka_prefix(line: str) -> tuple[str, ...]:
    if "|" in line or "　" in line:
        raise ValueError("Waka meter constraints use '/' as the only phrase boundary.")
    if not line:
        return ("",)
    parts = tuple(line.split(BOUNDARY))
    for index, part in enumerate(parts):
        is_allowed_trailing_empty = index == len(parts) - 1 and part == "" and line.endswith(BOUNDARY)
        if not part and not is_allowed_trailing_empty:
            raise ValueError("Empty waka phrase boundary is not allowed.")
    return parts


def waka_meter_state(text: str, target: tuple[int, ...]) -> WakaMeterState:
    line = text.splitlines()[-1] if text.splitlines() else text
    phrases = split_strict_waka_prefix(line)
    phrase_index = min(len(phrases) - 1, len(target) - 1)
    current = phrases[phrase_index] if phrases else ""
    current_morae = rough_mora_count(current)
    complete = len(phrases) >= len(target) and current_morae == target[-1]
    return WakaMeterState(
        phrases=phrases,
        phrase_index=phrase_index,
        current_morae=current_morae,
        current_target=target[phrase_index],
        complete=complete,
    )


def validate_waka_prefix(text: str, target: tuple[int, ...], kana_only: bool = True) -> None:
    line = text.splitlines()[-1] if text.splitlines() else text
    phrases = split_strict_waka_prefix(line)
    if len(phrases) > len(target):
        raise ValueError(f"Prompt has too many waka phrases: {len(phrases)} > {len(target)}")
    for index, phrase in enumerate(phrases):
        if not phrase and line:
            continue
        if kana_only:
            bad_chars = sorted({ch for ch in phrase if ch not in HIRAGANA})
            if bad_chars:
                raise ValueError(f"Waka meter constraints require kana-only prompt. Bad chars: {''.join(bad_chars)!r}")
        if phrase and phrase[0] in SMALL_NONINITIAL:
            raise ValueError(f"Waka phrase {index + 1} starts with a non-initial kana: {phrase[0]!r}")
        morae = rough_mora_count(phrase)
        limit = target[index]
        is_complete_phrase = index < len(phrases) - 1 or line.endswith(BOUNDARY)
        if morae > limit:
            raise ValueError(f"Prompt phrase {index + 1} exceeds target morae: {morae} > {limit}")
        if is_complete_phrase and morae != limit:
            raise ValueError(f"Prompt phrase {index + 1} must be exactly {limit} morae before '/': {morae}")


def validate_tokenizer_coverage(text: str, tokenizer: CharTokenizer) -> None:
    missing = tokenizer.missing_chars(text)
    if missing:
        shown = "".join(missing[:20])
        raise ValueError(f"Prompt contains characters outside tokenizer vocabulary: {shown!r}")


class WakaMeterLogitsProcessor:
    def __init__(
        self,
        tokenizer: CharTokenizer,
        target: tuple[int, ...] = (5, 7, 5, 7, 7),
        kana_only: bool = True,
    ) -> None:
        self.tokenizer = tokenizer
        self.target = target
        self.kana_only = kana_only
        missing = [ch for ch in (BOUNDARY, FINAL_STOP) if ch not in tokenizer.stoi]
        if missing:
            raise ValueError(f"Tokenizer is missing required waka boundary tokens: {missing}")

    def __call__(self, idx: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        logits = logits.clone()
        for batch_index in range(idx.size(0)):
            text = self.tokenizer.decode(idx[batch_index].tolist())
            state = waka_meter_state(text, self.target)
            allowed: list[int] = []
            current_phrase = state.phrases[state.phrase_index] if state.phrases else ""

            if state.complete:
                allowed = [self.tokenizer.stoi[FINAL_STOP]]
            elif state.current_morae == state.current_target:
                boundary = FINAL_STOP if state.phrase_index == len(self.target) - 1 else BOUNDARY
                allowed = [self.tokenizer.stoi[boundary]]
            else:
                remaining = state.current_target - state.current_morae
                for ch, token_id in self.tokenizer.stoi.items():
                    if len(ch) != 1:
                        continue
                    if not can_follow_phrase(current_phrase, ch, self.kana_only):
                        continue
                    if mora_delta(ch) <= remaining:
                        allowed.append(token_id)

            if not allowed:
                raise RuntimeError(f"No legal waka-meter next token for text={text!r}")
            masked = torch.full_like(logits[batch_index], -float("inf"))
            masked[allowed] = logits[batch_index, allowed]
            logits[batch_index] = masked
        return logits


def exact_waka_meter_ok(text: str, target: tuple[int, ...] = (5, 7, 5, 7, 7)) -> bool:
    line = text.strip().splitlines()[0] if text.strip() else ""
    return waka_meter(line) == target
