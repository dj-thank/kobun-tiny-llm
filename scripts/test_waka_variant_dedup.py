from __future__ import annotations

from waka_variant_dedup import WakaVariantIndex, normalize_waka, waka_variant_match


def main() -> None:
    left = normalize_waka("ながらへばまたこのごろやしのばれむ憂しと見し世ぞ今は恋しき")
    right = normalize_waka("ながらへばまた此の頃やしのばれむうしと見し世ぞ今は恋しき")
    unrelated = normalize_waka("春霞立つを見すてて行く雁は花なき里に住みやならへる")
    if not waka_variant_match(left, right):
        raise SystemExit("waka_variant_match_failed_for_known_near_duplicate")
    if waka_variant_match(left, unrelated):
        raise SystemExit("waka_variant_match_false_positive_for_unrelated_poem")
    index = WakaVariantIndex()
    index.add("train", "train:sample", left)
    match = index.find_cross_role("validation", right)
    if match is None or match.role != "train" or match.kind != "variant":
        raise SystemExit(f"waka_variant_index_failed_cross_role_match={match}")
    if index.find_cross_role("train", right) is not None:
        raise SystemExit("waka_variant_index_should_allow_same_role_variant")
    print("waka_variant_dedup_ok=true")


if __name__ == "__main__":
    main()
