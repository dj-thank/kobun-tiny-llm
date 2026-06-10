from __future__ import annotations

from split_policy import grammar_scope, manifest_policy_errors, split_name, split_role_for_work


def main() -> None:
    work_id = "work:未知作品"
    if grammar_scope(work_id) != "unregistered_outside_genji_era_scope":
        raise SystemExit("unknown work_id must not be treated as Genji-era grammar evidence")
    if split_role_for_work(work_id, include_in_training=True) != "excluded":
        raise SystemExit("unknown work_id must fail closed outside model train/validation/test splits")
    forged_unknown = {
        "source_id": "forged_unknown",
        "work_id": work_id,
        "split_group_key": work_id,
        "include_in_training": True,
        "split_role": "train",
        "grammar_scope": "genji-era-reference",
        "split_policy": "work_group_genji_reference_v1",
    }
    if not manifest_policy_errors(forged_unknown):
        raise SystemExit("forged unknown-work manifest row must report policy errors")
    try:
        split_name(forged_unknown)
    except ValueError:
        pass
    else:
        raise SystemExit("forged unknown-work split_role override must fail closed")
    forged_reference = {
        "source_id": "forged_reference",
        "work_id": "work:方丈記",
        "split_group_key": "work:方丈記",
        "include_in_training": True,
        "split_role": "train",
        "grammar_scope": "genji-era-reference",
        "split_policy": "work_group_genji_reference_v1",
    }
    if not manifest_policy_errors(forged_reference):
        raise SystemExit("forged reference-only manifest row must report policy errors")
    try:
        split_name(forged_reference)
    except ValueError:
        pass
    else:
        raise SystemExit("forged reference-only split_role override must fail closed")
    print("split_policy_unknown_work_fail_closed=true")


if __name__ == "__main__":
    main()
