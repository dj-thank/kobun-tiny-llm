from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


GENRE_RULES_PATH = Path("data/grammar/genre_rules.jsonl")
SMALL_KANA = set("ぁぃぅぇぉゃゅょゎァィゥェォャュョヮ")
MORA_JOINERS = set("ゃゅょャュョ")
PUNCT = "、。 「」『』（）()[]【】・"


@dataclass(frozen=True)
class WakaScore:
    score: int
    meter: tuple[int, ...]
    reasons: tuple[str, ...]


@lru_cache(maxsize=1)
def load_genre_rules(path: Path = GENRE_RULES_PATH) -> tuple[dict[str, object], ...]:
    if not path.exists():
        return ()
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return tuple(rows)


def normalize_for_mora(text: str) -> str:
    return "".join(ch for ch in text if ch not in PUNCT and not ch.isspace())


def rough_mora_count(text: str) -> int:
    text = normalize_for_mora(text)
    count = 0
    for ch in text:
        if ch in MORA_JOINERS:
            continue
        count += 1
    return count


def split_waka_phrases(text: str) -> tuple[str, ...]:
    if "/" in text:
        return tuple(part for part in text.split("/") if part)
    if "|" in text:
        return tuple(part for part in text.split("|") if part)
    if "　" in text:
        return tuple(part for part in text.split("　") if part)
    return (text,)


def waka_meter(text: str) -> tuple[int, ...]:
    return tuple(rough_mora_count(part) for part in split_waka_phrases(text))


def _near_after(text: str, phrase: str, targets: list[str], width: int = 12) -> bool:
    index = text.find(phrase)
    if index < 0:
        return False
    window = text[index + len(phrase) : index + len(phrase) + width]
    return any(target in window for target in targets)


def _has_kakekotoba_evidence(text: str, row: dict[str, object]) -> bool:
    surface = str(row.get("surface", ""))
    variants = [surface, *[str(item) for item in row.get("readings", [])]]
    if not any(variant and variant in text for variant in variants):
        return False

    sense_markers = row.get("sense_markers")
    if isinstance(sense_markers, dict):
        hit_senses = 0
        for markers in sense_markers.values():
            marker_list = [str(marker) for marker in markers]
            if any(marker and marker in text for marker in marker_list):
                hit_senses += 1
        return hit_senses >= 2

    related = [str(item) for item in row.get("related_words", [])]
    return any(word in text for word in related)


def _row_words(row: dict[str, object], key: str = "words") -> list[str]:
    return [str(item) for item in row.get(key, [])]


def _min_hits(row: dict[str, object], default: int = 2) -> int:
    return int(row.get("min_hits", default))


def _hit_terms(text: str, terms: object) -> list[str]:
    return [str(term) for term in terms if str(term) and str(term) in text]


def waka_score(text: str, reading: str | None = None) -> WakaScore:
    score = 0
    reasons: list[str] = []
    meter_text = reading or text
    meter = waka_meter(meter_text)
    rules = load_genre_rules()
    meter_rule = next((row for row in rules if row.get("rule_id") == "waka_meter_57577"), None)
    if meter_rule is not None:
        target = tuple(int(n) for n in meter_rule.get("target_morae", []))
        if len(meter) == len(target):
            diff = sum(abs(actual - expected) for actual, expected in zip(meter, target))
            weight = int(meter_rule.get("weight", 1))
            score += max(-weight, weight - diff)
            reasons.append(f"meter={meter} target={target} diff={diff}")
        else:
            total = rough_mora_count(meter_text)
            if not 29 <= total <= 33:
                score -= min(4, abs(total - 31))
            reasons.append(f"low_confidence_unsplit_total_morae={total}")

    for row in rules:
        category = str(row.get("category", ""))
        weight = int(row.get("weight", 1))
        if category == "makurakotoba":
            phrase = str(row.get("phrase", ""))
            leads_to = [str(item) for item in row.get("leads_to", [])]
            if phrase and phrase in text:
                if _near_after(text, phrase, leads_to):
                    score += weight
                    reasons.append(f"makurakotoba={phrase}")
                else:
                    score -= weight
                    reasons.append(f"unresolved_makurakotoba={phrase}")
        elif category == "kakekotoba":
            surface = str(row.get("surface", ""))
            if _has_kakekotoba_evidence(text, row):
                score += weight
                reasons.append(f"kakekotoba={surface}")
        elif category == "engo":
            cluster = [str(item) for item in row.get("cluster", [])]
            hits = [word for word in cluster if word in text]
            if len(hits) >= 2:
                score += weight
                reasons.append(f"engo={','.join(hits)}")
        elif category == "season_word":
            words = _row_words(row)
            hits = [word for word in words if word in text]
            if len(hits) >= _min_hits(row, 1):
                score += weight
                reasons.append(f"season={row.get('season')}:{','.join(hits)}")
        elif category == "theme":
            words = _row_words(row)
            hits = [word for word in words if word in text]
            if len(hits) >= _min_hits(row, 2):
                score += weight
                reasons.append(f"theme={row.get('theme')}:{','.join(hits)}")
        elif category == "utamakura":
            place = str(row.get("place", ""))
            associations = _row_words(row, "associations")
            hits = [word for word in associations if word in text]
            if place and place in text and len(hits) >= _min_hits(row, 1):
                score += weight
                reasons.append(f"utamakura={place}:{','.join(hits)}")
        elif category == "honkadori_signal":
            source = str(row.get("source", ""))
            motifs = _row_words(row, "motifs")
            hits = [word for word in motifs if word in text]
            if len(hits) >= _min_hits(row, 2):
                score += weight
                reasons.append(f"honkadori_signal={source}:{','.join(hits)}")
        elif category == "ending":
            endings = [str(item) for item in row.get("surface_endings", [])]
            stripped = re.sub(r"[。 、\s]+$", "", text)
            if any(stripped.endswith(ending) for ending in endings):
                score += weight
                reasons.append("taigendome")
    return WakaScore(score=score, meter=meter, reasons=tuple(reasons))
