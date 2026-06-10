from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


DEFAULT_LEXICON = Path("data/grammar/morph_lexicon.jsonl")
CLAUSE_BOUNDARIES = ("。", "、", "\n", "」", "「")


@dataclass(frozen=True)
class MorphToken:
    surface: str
    lemma: str
    pos: str
    subpos: str
    conjugation_type: str
    conjugation_form: str
    reading: str
    pronunciation: str
    orthographic_base: str
    period: str
    style: str
    grammar_tags: tuple[str, ...]
    start: int
    end: int

    @classmethod
    def from_row(cls, row: dict[str, object], start: int, end: int) -> "MorphToken":
        return cls(
            surface=str(row.get("surface", "")),
            lemma=str(row.get("lemma", "")),
            pos=str(row.get("pos", "")),
            subpos=str(row.get("subpos", "")),
            conjugation_type=str(row.get("conjugation_type", "")),
            conjugation_form=str(row.get("conjugation_form", "")),
            reading=str(row.get("reading", "")),
            pronunciation=str(row.get("pronunciation", "")),
            orthographic_base=str(row.get("orthographic_base", "")),
            period=str(row.get("period", "")),
            style=str(row.get("style", "")),
            grammar_tags=tuple(str(tag) for tag in row.get("grammar_tags", [])),
            start=start,
            end=end,
        )

    def to_chj_line(self) -> str:
        tags = ",".join(self.grammar_tags)
        return (
            f"{self.surface}\t{self.lemma}\t{self.pos}\t{self.subpos}\t"
            f"{self.conjugation_type}\t{self.conjugation_form}\t{self.reading}\t"
            f"{self.period}\t{self.style}\t{tags}"
        )


def load_lexicon(path: Path = DEFAULT_LEXICON) -> list[dict[str, object]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    rows.sort(key=lambda row: len(str(row["surface"])), reverse=True)
    return rows


def is_hiragana(ch: str) -> bool:
    return "ぁ" <= ch <= "ゖ"


def allowed_context(text: str, index: int, end: int, row: dict[str, object]) -> bool:
    surface = str(row["surface"])
    pos = str(row.get("pos", ""))
    if pos == "助詞" and surface == "か":
        return text[:end].endswith(("いかでか", "いづれか", "誰か", "何か", "いかにか", "いづくにか"))
    if pos == "助詞" and surface == "や":
        return text[:end].endswith(("誰や", "何や", "何をや", "いづれや", "いかにや"))
    if pos == "助動詞" and surface in {"き", "し"}:
        local = text[max(0, index - 1) : end]
        if local in {"明き", "よし"}:
            return False
    if len(surface) == 1 and pos in {"助詞", "助動詞"}:
        prev_ch = text[index - 1] if index > 0 else ""
        next_ch = text[end] if end < len(text) else ""
        if prev_ch and next_ch and is_hiragana(prev_ch) and is_hiragana(next_ch):
            return False
    return True


def annotate(text: str, lexicon: list[dict[str, object]] | None = None) -> list[MorphToken]:
    lexicon = lexicon or load_lexicon()
    candidates: list[tuple[int, int, dict[str, object]]] = []
    for row in lexicon:
        surface = str(row["surface"])
        start = 0
        while True:
            index = text.find(surface, start)
            if index < 0:
                break
            end = index + len(surface)
            if allowed_context(text, index, end, row):
                candidates.append((index, end, row))
            start = index + 1
    candidates.sort(key=lambda item: (item[0], -(item[1] - item[0]), str(item[2]["surface"])))

    tokens: list[MorphToken] = []
    occupied = [False] * len(text)
    for index, end, row in candidates:
        if not any(occupied[index:end]):
            tokens.append(MorphToken.from_row(row, index, end))
            for pos in range(index, end):
                occupied[pos] = True
    tokens.sort(key=lambda token: (token.start, token.end))
    return tokens


def required_musubi_form(marker: MorphToken) -> str | None:
    if "係り結び" not in marker.grammar_tags:
        return None
    if "已然形要求" in marker.grammar_tags:
        return "已然形"
    if "連体形要求" in marker.grammar_tags:
        return "連体形"
    return None


def candidate_musubi(tokens: list[MorphToken], marker: MorphToken, text: str, window: int = 40) -> MorphToken | None:
    boundary = marker.end + window
    for boundary_char in CLAUSE_BOUNDARIES:
        boundary_index = text.find(boundary_char, marker.end)
        if boundary_index >= 0:
            boundary = min(boundary, boundary_index)
    candidates = [
        token
        for token in tokens
        if token.start >= marker.end
        and token.start <= boundary
        and (token.pos in {"動詞", "助動詞", "形容詞", "形容動詞"} or token.conjugation_form)
    ]
    if not candidates:
        return None
    return candidates[-1]


def form_matches(actual: str, expected: str) -> bool:
    if expected == "連体形":
        return "連体形" in actual or actual == "終止形-連体形"
    if expected == "已然形":
        return "已然形" in actual
    return expected in actual


def morphology_score(text: str) -> int:
    tokens = annotate(text)
    score = 0
    for token in tokens:
        expected = required_musubi_form(token)
        if not expected:
            continue
        musubi = candidate_musubi(tokens, token, text)
        if musubi is None:
            score -= 2
            continue
        score += 4 if form_matches(musubi.conjugation_form, expected) else -4
    honorifics = [token for token in tokens if "敬語" in token.grammar_tags]
    score += min(4, len(honorifics))
    if len(honorifics) >= 3 and len({token.lemma for token in honorifics}) == 1:
        score -= 4
    return score
