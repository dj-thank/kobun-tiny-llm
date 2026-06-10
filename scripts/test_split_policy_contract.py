from __future__ import annotations

from split_policy import (
    EXCLUDED_WORK_IDS,
    REFERENCE_ONLY_WORK_IDS,
    SPLIT_POLICY,
    TEST_WORK_IDS,
    TRAIN_WORK_IDS,
    VALIDATION_WORK_IDS,
)


EXPECTED_TRAIN = {"work:源氏物語", "work:古今和歌集", "work:和泉式部日記", "work:蜻蛉日記"}
EXPECTED_VALIDATION = {"work:土佐日記", "work:更級日記", "work:後撰和歌集"}
EXPECTED_TEST = {"work:枕草子", "work:紫式部日記", "work:拾遺和歌集"}
EXPECTED_REFERENCE = {"work:方丈記", "work:宇治拾遺物語", "work:伊勢物語"}
EXPECTED_EXCLUDED = {"work:竹取物語"}


def require_set(name: str, actual: set[str], expected: set[str]) -> None:
    if actual != expected:
        raise SystemExit(
            f"split_policy_{name}_mismatch "
            f"missing={sorted(expected - actual)} unexpected={sorted(actual - expected)}"
        )


def main() -> None:
    if SPLIT_POLICY != "work_group_genji_reference_v1":
        raise SystemExit(f"split_policy_name_mismatch={SPLIT_POLICY}")
    require_set("train_work_ids", set(TRAIN_WORK_IDS), EXPECTED_TRAIN)
    require_set("validation_work_ids", set(VALIDATION_WORK_IDS), EXPECTED_VALIDATION)
    require_set("test_work_ids", set(TEST_WORK_IDS), EXPECTED_TEST)
    require_set("reference_only_work_ids", set(REFERENCE_ONLY_WORK_IDS), EXPECTED_REFERENCE)
    require_set("excluded_work_ids", set(EXCLUDED_WORK_IDS), EXPECTED_EXCLUDED)
    model_groups = set(TRAIN_WORK_IDS) | set(VALIDATION_WORK_IDS) | set(TEST_WORK_IDS)
    if model_groups & set(REFERENCE_ONLY_WORK_IDS):
        raise SystemExit("split_policy_reference_group_overlaps_model_split")
    if set(TRAIN_WORK_IDS) & set(VALIDATION_WORK_IDS):
        raise SystemExit("split_policy_train_validation_overlap")
    if set(TRAIN_WORK_IDS) & set(TEST_WORK_IDS):
        raise SystemExit("split_policy_train_test_overlap")
    if set(VALIDATION_WORK_IDS) & set(TEST_WORK_IDS):
        raise SystemExit("split_policy_validation_test_overlap")
    print("split_policy_contract_ok=true")


if __name__ == "__main__":
    main()
