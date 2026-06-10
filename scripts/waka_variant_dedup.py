from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher


DEFAULT_WAKA_VARIANT_THRESHOLD = 0.86
MIN_WAKA_VARIANT_CHARS = 18
WAKA_GRAM_SIZE = 5


def normalize_waka(value: str) -> str:
    value = value.replace("－", "-").replace("ー", "")
    return re.sub(r"[\s/|,，、。・「」『』（）()［］\[\]{}<>《》\-]+", "", value)


def waka_grams(value: str, size: int = WAKA_GRAM_SIZE) -> set[str]:
    if len(value) < size:
        return set()
    return {value[index : index + size] for index in range(len(value) - size + 1)}


def waka_variant_match(left: str, right: str, threshold: float = DEFAULT_WAKA_VARIANT_THRESHOLD) -> bool:
    if len(left) < MIN_WAKA_VARIANT_CHARS or len(right) < MIN_WAKA_VARIANT_CHARS:
        return False
    if abs(len(left) - len(right)) > max(6, int(max(len(left), len(right)) * 0.25)):
        return False
    ratio = SequenceMatcher(None, left, right, autojunk=False).ratio()
    if ratio >= threshold:
        return True
    left_grams = waka_grams(left)
    right_grams = waka_grams(right)
    if not left_grams or not right_grams:
        return False
    jaccard = len(left_grams & right_grams) / len(left_grams | right_grams)
    return jaccard >= 0.62


@dataclass(frozen=True)
class WakaVariantMatch:
    role: str
    label: str
    value: str
    kind: str
    ratio: float


@dataclass(frozen=True)
class WakaVariantItem:
    role: str
    label: str
    value: str
    grams: frozenset[str]


class WakaVariantIndex:
    def __init__(self, threshold: float = DEFAULT_WAKA_VARIANT_THRESHOLD) -> None:
        self.threshold = threshold
        self._items: list[WakaVariantItem] = []
        self._exact: dict[str, list[int]] = defaultdict(list)
        self._gram_index: dict[str, list[int]] = defaultdict(list)

    def add(self, role: str, label: str, value: str) -> None:
        normalized = normalize_waka(value)
        if not normalized:
            return
        item = WakaVariantItem(role=role, label=label, value=normalized, grams=frozenset(waka_grams(normalized)))
        item_index = len(self._items)
        self._items.append(item)
        self._exact[normalized].append(item_index)
        for gram in item.grams:
            self._gram_index[gram].append(item_index)

    def find_cross_role(self, role: str, value: str) -> WakaVariantMatch | None:
        normalized = normalize_waka(value)
        if not normalized:
            return None
        for item_index in self._exact.get(normalized, []):
            item = self._items[item_index]
            if item.role != role:
                return WakaVariantMatch(item.role, item.label, item.value, "exact", 1.0)
        if len(normalized) < MIN_WAKA_VARIANT_CHARS:
            return None
        grams = waka_grams(normalized)
        if not grams:
            return None
        candidate_counts: Counter[int] = Counter()
        for gram in grams:
            candidate_counts.update(self._gram_index.get(gram, ()))
        for item_index, shared_count in candidate_counts.most_common(80):
            item = self._items[item_index]
            if item.role == role:
                continue
            if shared_count < 3:
                break
            if not waka_variant_match(normalized, item.value, self.threshold):
                continue
            ratio = SequenceMatcher(None, normalized, item.value, autojunk=False).ratio()
            return WakaVariantMatch(item.role, item.label, item.value, "variant", ratio)
        return None
