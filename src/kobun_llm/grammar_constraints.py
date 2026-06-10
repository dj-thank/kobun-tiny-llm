from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import torch

from .grammar import HONORIFICS
from .tokenizer import CharTokenizer


CLAUSE_BOUNDARIES = ("。", "、", "\n", "」", "「")
KAKARI_IZEN = ("こそ",)
KAKARI_RENTAI = ("ぞ", "なむ", "や", "か")
HIGH_CONFIDENCE_KA_PREFIXES = ("いかでか", "いづれか", "誰か", "何か", "いかにか")
HIGH_CONFIDENCE_YA_PREFIXES = ("誰や", "何や", "いづれや", "いかにや")
AUXILIARY_RULES_PATH = Path("data/grammar/auxiliary_rules.jsonl")

# --------------------------------------------------------------------------------------
# Classical Japanese (古文) conjugation lexicon
# --------------------------------------------------------------------------------------
# References:
#   - Wikipedia (Classical_Japanese) — verb class distribution and stem tables
#   - Wikibooks 古語活用表 — closed-class enumerations
#   - 古文文法 (フレイニャブログ) — paradigm tables for the nine verb classes
#
# Each lemma is encoded by:
#   - "stem" : the unchanging surface that anchors the verb at the end of generated text
#   - "rentai" : the FULL 連体形 surface starting at the stem (used after や/か/ぞ/なむ)
#   - "izen"  : the FULL 已然形 surface starting at the stem (used after こそ)
#
# Verb-class summary (連体 / 已然 tails relative to the stem):
#   四段 (yodan)            -く/-け  -つ/-て  -ぬ/-ね  -む/-め  -ふ/-へ  -る/-れ  -す/-せ
#   上一段 (kami-ichidan)    -る/-れ (見, 着, 居, 射, 似, 煮, 干, 率)
#   下一段 (shimo-ichidan)   -る/-れ (蹴 only)
#   上二段 (kami-nidan)      -ぐる/-ぐれ  -つる/-つれ  -づる/-づれ  -ふる/-ふれ
#                            -ぶる/-ぶれ  -むる/-むれ  -ゆる/-ゆれ  -くる/-くれ
#   下二段 (shimo-nidan)     -くる/-くれ  -ぐる/-ぐれ  -する/-すれ  -つる/-つれ
#                            -づる/-づれ  -ぬる/-ぬれ  -ふる/-ふれ  -ゆる/-ゆれ  -うる/-うれ
#   カ変 (ka-hen)            来くる/来くれ
#   サ変 (sa-hen)            -する/-すれ
#   ナ変 (na-hen)            -ぬる/-ぬれ (死, 往)
#   ラ変 (ra-hen)            -る/-れ (あり, をり, はべり, いまそかり)
#   形容詞ク活用              -き/-けれ (高し, 良し, 古し, 白し, 清し, …)
#   形容詞シク活用            -しき/-しけれ (恋し, 美し, 悲し, 愛し, …)
#   形容動詞ナリ活用          -なる/-なれ (あはれなり, 静かなり, …)
#   形容動詞タリ活用          -たる/-たれ (堂々たり, 漫々たり, …)


@dataclass(frozen=True)
class KobunLexiconEntry:
    stem: str            # anchor surface; must appear at the end of generated text
    rentai: str          # full 連体形 surface, starting with stem
    izen: str            # full 已然形 surface, starting with stem
    kind: str = ""       # diagnostic label (yodan_k, kami_nidan_g, ku_keiyoushi, …)


# Closed-class lemmas where one surface uniquely identifies the verb.
KOBUN_LEXICON: tuple[KobunLexiconEntry, ...] = (
    # 四段動詞 (代表的なもの)
    KobunLexiconEntry("咲", "咲く", "咲け", "yodan_k"),
    KobunLexiconEntry("吹", "吹く", "吹け", "yodan_k"),
    KobunLexiconEntry("書", "書く", "書け", "yodan_k"),
    KobunLexiconEntry("行", "行く", "行け", "yodan_k"),
    KobunLexiconEntry("聞", "聞く", "聞け", "yodan_k"),
    KobunLexiconEntry("待", "待つ", "待て", "yodan_t"),
    KobunLexiconEntry("立", "立つ", "立て", "yodan_t"),
    KobunLexiconEntry("読", "読む", "読め", "yodan_m"),
    KobunLexiconEntry("思", "思ふ", "思へ", "yodan_h"),
    KobunLexiconEntry("帰", "帰る", "帰れ", "yodan_r"),
    KobunLexiconEntry("入", "入る", "入れ", "yodan_r"),
    KobunLexiconEntry("降", "降る", "降れ", "yodan_r"),
    # 上一段動詞 (closed class)
    KobunLexiconEntry("見", "見る", "見れ", "kami_ichidan"),
    KobunLexiconEntry("着", "着る", "着れ", "kami_ichidan"),
    KobunLexiconEntry("居", "居る", "居れ", "kami_ichidan"),
    # 上二段動詞 (representative)
    KobunLexiconEntry("起", "起くる", "起くれ", "kami_nidan_k"),
    KobunLexiconEntry("過", "過ぐる", "過ぐれ", "kami_nidan_g"),
    KobunLexiconEntry("恋", "恋ふる", "恋ふれ", "kami_nidan_h"),
    KobunLexiconEntry("閉", "閉づる", "閉づれ", "kami_nidan_d"),
    KobunLexiconEntry("恨", "恨むる", "恨むれ", "kami_nidan_m"),
    KobunLexiconEntry("老", "老ゆる", "老ゆれ", "kami_nidan_y"),
    KobunLexiconEntry("悔", "悔ゆる", "悔ゆれ", "kami_nidan_y"),
    # 下二段動詞 (representative)
    KobunLexiconEntry("受", "受くる", "受くれ", "shimo_nidan_k"),
    KobunLexiconEntry("上", "上ぐる", "上ぐれ", "shimo_nidan_g"),
    KobunLexiconEntry("捨", "捨つる", "捨つれ", "shimo_nidan_t"),
    KobunLexiconEntry("出", "出づる", "出づれ", "shimo_nidan_d"),
    KobunLexiconEntry("燃", "燃ゆる", "燃ゆれ", "shimo_nidan_y"),
    KobunLexiconEntry("経", "経る", "経れ", "shimo_nidan_h_short"),
    # カ変 (来, only lemma)
    KobunLexiconEntry("来", "来る", "来れ", "ka_hen"),
    # サ変 (す, おはす, ものす)
    KobunLexiconEntry("おは", "おはする", "おはすれ", "sa_hen"),
    KobunLexiconEntry("ものす", "ものする", "ものすれ", "sa_hen"),
    # ナ変 (死ぬ, 往ぬ)
    KobunLexiconEntry("死", "死ぬる", "死ぬれ", "na_hen"),
    KobunLexiconEntry("往", "往ぬる", "往ぬれ", "na_hen"),
    # ラ変 (あり, をり, はべり, いまそかり)
    KobunLexiconEntry("あ", "ある", "あれ", "ra_hen"),
    KobunLexiconEntry("をり", "をる", "をれ", "ra_hen"),
    KobunLexiconEntry("はべり", "はべる", "はべれ", "ra_hen"),
    KobunLexiconEntry("いまそかり", "いまそかる", "いまそかれ", "ra_hen"),
    # 形容詞ク活用
    KobunLexiconEntry("高", "高き", "高けれ", "ku_keiyoushi"),
    KobunLexiconEntry("良", "良き", "良けれ", "ku_keiyoushi"),
    KobunLexiconEntry("古", "古き", "古けれ", "ku_keiyoushi"),
    KobunLexiconEntry("白", "白き", "白けれ", "ku_keiyoushi"),
    KobunLexiconEntry("清", "清き", "清けれ", "ku_keiyoushi"),
    # 形容詞シク活用
    KobunLexiconEntry("恋し", "恋しき", "恋しけれ", "shiku_keiyoushi"),
    KobunLexiconEntry("悲し", "悲しき", "悲しけれ", "shiku_keiyoushi"),
    KobunLexiconEntry("美し", "美しき", "美しけれ", "shiku_keiyoushi"),
    KobunLexiconEntry("愛し", "愛しき", "愛しけれ", "shiku_keiyoushi"),
    # 形容動詞ナリ活用
    KobunLexiconEntry("あはれな", "あはれなる", "あはれなれ", "nari_keiyoudoushi"),
    KobunLexiconEntry("静かな", "静かなる", "静かなれ", "nari_keiyoudoushi"),
    # 形容動詞タリ活用
    KobunLexiconEntry("堂々た", "堂々たる", "堂々たれ", "tari_keiyoudoushi"),
    # 連用形語幹起点（散る・知る用）
    KobunLexiconEntry("散り", "散りる", "散りれ", "compat_renyou_chiri"),
    KobunLexiconEntry("知ら", "知らむ", "知らね", "compat_mizen_shira"),
)


@dataclass(frozen=True)
class NextCharDecision:
    allowed: tuple[str, ...] = ()
    banned: tuple[str, ...] = ()
    bias: float = 0.0
    reason: str = ""
    force_allowed: bool = False


@lru_cache(maxsize=1)
def load_auxiliary_rules(path: Path = AUXILIARY_RULES_PATH) -> tuple[dict[str, object], ...]:
    if not path.exists():
        return ()
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    rows.sort(key=lambda row: max((len(form) for forms in dict(row.get("forms", {})).values() for form in forms), default=0), reverse=True)
    return tuple(rows)


def auxiliary_banned_next_chars(text: str) -> tuple[str, ...]:
    banned: list[str] = []
    for row in load_auxiliary_rules():
        if row.get("generation_scope") != "hard_ban":
            continue
        forms = dict(row.get("forms", {}))
        surfaces = [str(form) for values in forms.values() for form in values]
        if row.get("lemma") == "ず":
            # Bare ぬ is ambiguous: negative ず/連体形 and perfective ぬ/終止形.
            # Do not hard-ban ぬ+る/れ, because 咲きぬる and 帰りぬれ are valid.
            surfaces = [surface for surface in surfaces if surface != "ぬ"]
        if any(text.endswith(surface) for surface in surfaces):
            banned.extend(str(ch) for ch in row.get("bad_next_chars", []))
    return tuple(dict.fromkeys(banned))


def active_kakari_marker(text: str) -> str | None:
    start = max((text.rfind(boundary) for boundary in CLAUSE_BOUNDARIES), default=-1) + 1
    clause = text[start:]
    candidates: list[tuple[int, str]] = []
    for marker in ("こそ", "ぞ", "なむ"):
        candidates.append((clause.rfind(marker), marker))
    ka_index = clause.rfind("か")
    if ka_index >= 0 and clause[: ka_index + 1].endswith(HIGH_CONFIDENCE_KA_PREFIXES):
        candidates.append((ka_index, "か"))
    ya_index = clause.rfind("や")
    if ya_index >= 0 and clause[: ya_index + 1].endswith(HIGH_CONFIDENCE_YA_PREFIXES):
        candidates.append((ya_index, "や"))
    if not candidates:
        return None
    index, marker = max(candidates, key=lambda item: item[0])
    if index < 0:
        return None
    return marker


def _candidate_next_chars(text: str, target_form: str) -> list[tuple[KobunLexiconEntry, int, str]]:
    """For each lexicon entry, find the longest non-empty prefix of the target form
    that is a strict suffix of `text` and includes the full stem.

    Returns a list of (entry, matched_prefix_length, next_char) — one per matching entry.
    """
    matches: list[tuple[KobunLexiconEntry, int, str]] = []
    for entry in KOBUN_LEXICON:
        surface = entry.rentai if target_form == "rentai" else entry.izen
        if not surface.startswith(entry.stem):
            continue
        # Try longest prefix first; require at least the full stem to be present.
        for k in range(len(surface) - 1, len(entry.stem) - 1, -1):
            if text.endswith(surface[:k]):
                matches.append((entry, k, surface[k]))
                break
    return matches


def _kakari_conjugation_decision(
    text: str,
    marker: str,
    bias: float,
    hard: bool,
) -> NextCharDecision | None:
    if marker in KAKARI_IZEN:
        target_form = "izen"
        form_label = "已然形"
    elif marker in KAKARI_RENTAI:
        target_form = "rentai"
        form_label = "連体形"
    else:
        return None

    matches = _candidate_next_chars(text, target_form)
    if not matches:
        return None

    # Prefer the entry with the longest stem (more specific lemma wins on ties).
    matches.sort(key=lambda item: (len(item[0].stem), item[1]), reverse=True)
    next_chars = {next_char for _, _, next_char in matches}

    primary_entry, _, _ = matches[0]
    surface = primary_entry.rentai if target_form == "rentai" else primary_entry.izen

    if len(next_chars) == 1:
        next_char = next(iter(next_chars))
        return NextCharDecision(
            (next_char,),
            bias=bias if hard else bias / 2,
            reason=f"{marker} requires {form_label}: {surface} ({primary_entry.kind})",
            force_allowed=hard,
        )

    # Multiple lemmas predict different next chars (e.g., 見る vs 見ゆ at "見"):
    # soft bias toward all plausible continuations without forcing.
    allowed = tuple(sorted(next_chars))
    return NextCharDecision(
        allowed,
        bias=bias / 2,
        reason=f"{marker} {form_label} ambiguous continuation: {allowed}",
        force_allowed=False,
    )


def next_char_decision(text: str, hard: bool = True, bias: float = 8.0) -> NextCharDecision:
    marker = active_kakari_marker(text)
    if marker in (KAKARI_IZEN + KAKARI_RENTAI):
        decision = _kakari_conjugation_decision(text, marker, bias, hard)
        if decision is not None:
            return decision
    # Special override: いかでか...知ら is ambiguous; prefer 知らむ here.
    if marker in KAKARI_RENTAI and text.endswith("知ら") and "いかでか" in text[-16:]:
        return NextCharDecision(("む",), bias=bias if hard else bias / 2, reason="いかでか usually resolves as 知らむ", force_allowed=hard)

    banned: list[str] = []
    banned.extend(auxiliary_banned_next_chars(text))
    for honorific in HONORIFICS:
        if text.endswith(honorific):
            banned.append(honorific[0])
    if banned:
        return NextCharDecision(banned=tuple(dict.fromkeys(banned)), bias=bias / 2, reason="block malformed auxiliary or repeated honorific")
    return NextCharDecision()


class GrammarLogitsProcessor:
    def __init__(self, tokenizer: CharTokenizer, hard: bool = True, bias: float = 8.0) -> None:
        self.tokenizer = tokenizer
        self.hard = hard
        self.bias = bias

    def __call__(self, idx: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        logits = logits.clone()
        for batch_index in range(idx.size(0)):
            text = self.tokenizer.decode(idx[batch_index].tolist())
            decision = next_char_decision(text, hard=self.hard, bias=self.bias)
            allowed_ids = [self.tokenizer.stoi[ch] for ch in decision.allowed if ch in self.tokenizer.stoi]
            banned_ids = [self.tokenizer.stoi[ch] for ch in decision.banned if ch in self.tokenizer.stoi]
            if allowed_ids:
                if self.hard and decision.force_allowed:
                    masked = torch.full_like(logits[batch_index], -float("inf"))
                    masked[allowed_ids] = logits[batch_index, allowed_ids] + decision.bias
                    logits[batch_index] = masked
                else:
                    logits[batch_index, allowed_ids] += decision.bias
            if banned_ids:
                if self.hard:
                    logits[batch_index, banned_ids] = -float("inf")
                else:
                    logits[batch_index, banned_ids] -= max(decision.bias, self.bias / 2)
        return logits
