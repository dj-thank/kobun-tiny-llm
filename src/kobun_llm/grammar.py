from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

from .genre_rules import waka_score
from .morphology import morphology_score


RENTAI_ENDINGS = ("く", "き", "む", "る", "たる", "ける", "なる", "ぬる", "ざる", "べき", "らむ", "める")
IZEN_ENDINGS = ("れ", "たれ", "けれ", "なれ", "ぬれ", "ざれ", "しか")
NATURAL_ENDINGS = ("けり", "なり", "たり", "べし", "らむ", "めり", "ぬ", "む", "。", "といふ")
HONORIFICS = ("たまふ", "思す", "おはす", "おはします", "御覧ず", "聞こゆ", "聞こえ")
COLLOCATIONS_PATH = Path("data/grammar/collocations.jsonl")
GRAMMAR_RULES_PATH = Path("data/grammar/rules.jsonl")
HIGH_CONFIDENCE_KA_PREFIXES = ("いかでか", "いづれか", "誰か", "何か", "いかにか")
HIGH_CONFIDENCE_YA_PREFIXES = ("誰や", "何や", "いづれや", "いかにや")


@lru_cache(maxsize=1)
def load_collocations() -> tuple[dict[str, object], ...]:
    if not COLLOCATIONS_PATH.exists():
        return ()
    rows = []
    for line in COLLOCATIONS_PATH.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return tuple(rows)


@lru_cache(maxsize=1)
def load_grammar_rules() -> tuple[dict[str, object], ...]:
    if not GRAMMAR_RULES_PATH.exists():
        return ()
    rows = []
    for line in GRAMMAR_RULES_PATH.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return tuple(rows)


def collocation_score(text: str, style: str = "all") -> int:
    score = 0
    for row in load_collocations():
        row_style = str(row.get("style", "all"))
        if row_style not in {"all", style}:
            continue
        weight = int(row.get("weight", 1))
        for example in row.get("positive_examples", []):
            example_text = str(example).rstrip("。")
            if example_text and example_text in text:
                score += weight
        for example in row.get("negative_examples", []):
            example_text = str(example).rstrip("。")
            if example_text and example_text in text:
                score -= weight * 2
    return score


def rule_table_score(text: str) -> int:
    score = 0
    for row in load_grammar_rules():
        weight = int(row.get("weight", 1))
        for pattern in row.get("bad_patterns", []):
            if re.search(str(pattern), text):
                score -= weight * 2
        for example in row.get("negative_examples", []):
            example_text = str(example).rstrip("。")
            if example_text and example_text in text:
                score -= weight * 2
        for example in row.get("positive_examples", []):
            example_text = str(example).rstrip("。")
            if example_text and example_text in text:
                score += weight
    return score


def _window_after(text: str, marker: str, width: int = 28) -> str:
    index = text.rfind(marker)
    if index < 0:
        return ""
    return text[index : index + width]


def _has_high_confidence_marker(text: str, marker: str) -> bool:
    if marker == "か":
        return any(prefix in text for prefix in HIGH_CONFIDENCE_KA_PREFIXES)
    if marker == "や":
        return any(prefix in text for prefix in HIGH_CONFIDENCE_YA_PREFIXES)
    return marker in text


def grammar_score(text: str, style: str = "all") -> int:
    score = 0
    if any(text.rstrip().endswith(ending) for ending in NATURAL_ENDINGS):
        score += 3
    if "こそ" in text:
        window = _window_after(text, "こそ")
        if any(ending in window for ending in IZEN_ENDINGS):
            score += 4
        else:
            score -= 3
    for marker in ("ぞ", "なむ", "や", "か"):
        if _has_high_confidence_marker(text, marker):
            window = _window_after(text, marker)
            if any(ending in window for ending in RENTAI_ENDINGS):
                score += 2
    score += min(4, sum(text.count(word) for word in HONORIFICS))
    # Penalize unnatural chained honorific repeats such as "たまふたまふたまふ".
    for word in HONORIFICS:
        pattern = re.compile(f"(?:{re.escape(word)}){{2,}}")
        for match in pattern.finditer(text):
            run = match.group(0)
            repeats = len(run) // len(word)
            if repeats >= 2:
                score -= (repeats - 1) * 4
    score += min(3, text.count("。"))
    score -= text.count("、、") * 3
    score -= text.count("。。") * 3
    for repeated in ("ことこと", "なほなほなほ", "かくかくかく"):
        if repeated in text:
            score -= 4
    if style == "waka":
        score += waka_score(text).score
    return score + morphology_score(text) + collocation_score(text, style) + rule_table_score(text)
