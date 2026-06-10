from __future__ import annotations

import hashlib
import re
from typing import Any


SPLIT_POLICY = "work_group_genji_reference_v1"

# Release-candidate grammar scope is the Genji-era / chuko classical register.
# Later medieval prose is retained as source provenance only until a separate
# diachronic model/eval plan exists.
TRAIN_WORK_IDS = {"work:源氏物語", "work:古今和歌集", "work:和泉式部日記", "work:蜻蛉日記"}
VALIDATION_WORK_IDS = {"work:土佐日記", "work:更級日記", "work:後撰和歌集"}
TEST_WORK_IDS = {"work:枕草子", "work:紫式部日記", "work:拾遺和歌集"}
REFERENCE_ONLY_WORK_IDS = {"work:方丈記", "work:宇治拾遺物語", "work:伊勢物語"}
EXCLUDED_WORK_IDS = {"work:竹取物語"}


def stable_bucket(value: str) -> float:
    digest = hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") / float(2**64)


def canonical_work_id(title: str) -> str:
    normalized = re.sub(r"\s+", " ", title.replace("/", " ")).strip()
    if normalized.startswith("源氏物語"):
        return "work:源氏物語"
    if normalized.startswith("古今和歌集"):
        return "work:古今和歌集"
    if normalized.startswith("後撰和歌集"):
        return "work:後撰和歌集"
    if normalized.startswith("拾遺和歌集"):
        return "work:拾遺和歌集"
    if normalized.startswith("小倉百人一首"):
        return "work:小倉百人一首"
    if normalized.startswith("枕草子"):
        return "work:枕草子"
    if normalized.startswith("宇治拾遺物語"):
        return "work:宇治拾遺物語"
    if normalized.startswith("更級日記") or normalized.startswith("さらしな日記"):
        return "work:更級日記"
    if normalized.startswith("紫式部日記"):
        return "work:紫式部日記"
    if normalized.startswith("和泉式部日記"):
        return "work:和泉式部日記"
    if normalized.startswith("蜻蛉日記"):
        return "work:蜻蛉日記"
    if normalized.startswith("伊勢物語"):
        return "work:伊勢物語"
    for exact in ("土佐日記", "方丈記", "竹取物語"):
        if normalized == exact or normalized.startswith(exact + " "):
            return f"work:{exact}"
    return "work:" + (normalized.split(" ")[0] if normalized else "unknown")


def grammar_scope(work_id: str) -> str:
    if work_id in TRAIN_WORK_IDS | VALIDATION_WORK_IDS | TEST_WORK_IDS:
        return "genji-era-reference"
    if work_id in REFERENCE_ONLY_WORK_IDS:
        return "reference_only_outside_genji_era_scope"
    return "unregistered_outside_genji_era_scope"


def split_role_for_work(work_id: str, include_in_training: bool = True) -> str:
    if work_id in REFERENCE_ONLY_WORK_IDS:
        return "reference"
    if not include_in_training or work_id in EXCLUDED_WORK_IDS:
        return "excluded"
    if work_id in TRAIN_WORK_IDS:
        return "train"
    if work_id in VALIDATION_WORK_IDS:
        return "validation"
    if work_id in TEST_WORK_IDS:
        return "test"
    return "excluded"


def split_group_key(row: dict[str, Any]) -> str:
    return str(row.get("split_group_key") or row.get("work_id") or canonical_work_id(str(row.get("title", ""))))


def row_work_id(row: dict[str, Any]) -> str:
    return str(row.get("work_id") or canonical_work_id(str(row.get("title", ""))))


def expected_split_role(row: dict[str, Any]) -> str:
    include = bool(row.get("include_in_training", True))
    return split_role_for_work(row_work_id(row), include)


def expected_grammar_scope(row: dict[str, Any]) -> str:
    return grammar_scope(row_work_id(row))


def manifest_policy_errors(row: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    work_id = row_work_id(row)
    group_key = split_group_key(row)
    if group_key != work_id:
        errors.append(f"split_group_key_mismatch expected={work_id!r} actual={group_key!r}")
    role = str(row.get("split_role") or "")
    expected_role = expected_split_role(row)
    if role and role != expected_role:
        errors.append(f"split_role_mismatch expected={expected_role!r} actual={role!r}")
    scope = str(row.get("grammar_scope") or "")
    expected_scope = expected_grammar_scope(row)
    if scope and scope != expected_scope:
        errors.append(f"grammar_scope_mismatch expected={expected_scope!r} actual={scope!r}")
    policy = str(row.get("split_policy") or "")
    if policy and policy != SPLIT_POLICY:
        errors.append(f"split_policy_mismatch expected={SPLIT_POLICY!r} actual={policy!r}")
    return errors


def split_name(row: dict[str, Any], val_ratio: float = 0.1, test_ratio: float = 0.05) -> str:
    del val_ratio, test_ratio
    errors = manifest_policy_errors(row)
    if errors:
        source_id = row.get("source_id") or row.get("title") or row_work_id(row)
        raise ValueError(f"manifest split policy mismatch source={source_id!r}: {'; '.join(errors)}")
    role = str(row.get("split_role") or "")
    if role:
        return role
    return expected_split_role(row)


def is_model_split(role: str) -> bool:
    return role in {"train", "validation", "test"}
