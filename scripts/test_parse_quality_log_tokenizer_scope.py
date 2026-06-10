from __future__ import annotations

from parse_quality_log import parse_log


def main() -> None:
    text = (
        "tokenizer_vocab_scope "
        "policy=train_split_plus_core_japanese_inventory_plus_utf8_byte_fallback_v1 "
        "tokenizer_chars=3317 "
        "direct_vocab_chars=3031 "
        "byte_fallback=true "
        "byte_fallback_tokens=256 "
        "train_chars=2222 "
        "heldout_exclusive_chars=206 "
        "covered_by_static_inventory=27 "
        "heldout_covered_by_byte_fallback=179 "
        "forbidden_heldout_tokenizer_leakage=0 "
        "heldout_missing_from_tokenizer=0 "
        "meta_verified=true "
        "manifest_sha256=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa "
        "vocab_sha256=bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb "
        "tokenizer_meta_sha256=cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc\n"
    )
    parsed = parse_log(text)
    scope = parsed.get("tokenizer_vocab_scope") or {}
    expected = {
        "policy": "train_split_plus_core_japanese_inventory_plus_utf8_byte_fallback_v1",
        "tokenizer_chars": 3317,
        "direct_vocab_chars": 3031,
        "byte_fallback": True,
        "byte_fallback_tokens": 256,
        "heldout_covered_by_byte_fallback": 179,
        "forbidden_heldout_tokenizer_leakage": 0,
        "heldout_missing_from_tokenizer": 0,
        "tokenizer_meta_verified": True,
    }
    for key, value in expected.items():
        if scope.get(key) != value:
            raise SystemExit(f"tokenizer_scope_parse_mismatch key={key} expected={value!r} got={scope.get(key)!r}")
    checkpoint_text = (
        text
        + "checkpoint_tokenizer_vocab_scope "
        + "policy=train_split_plus_core_japanese_inventory_plus_utf8_byte_fallback_v1 "
        + "tokenizer_chars=3317 "
        + "direct_vocab_chars=3031 "
        + "byte_fallback=true "
        + "byte_fallback_tokens=256 "
        + "train_chars=2222 "
        + "heldout_exclusive_chars=206 "
        + "covered_by_static_inventory=27 "
        + "heldout_covered_by_byte_fallback=179 "
        + "forbidden_heldout_tokenizer_leakage=0 "
        + "heldout_missing_from_tokenizer=0 "
        + "meta_verified=true "
        + "manifest_sha256=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa "
        + "vocab_sha256=bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb "
        + "tokenizer_meta_sha256=cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc "
        + "core_inventory_sha256=dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd "
        + "checkpoint_bound=true\n"
    )
    checkpoint_scope = parse_log(checkpoint_text).get("checkpoint_tokenizer_vocab_scope") or {}
    if checkpoint_scope.get("checkpoint_bound") is not True:
        raise SystemExit("checkpoint_tokenizer_scope_parse_missing_checkpoint_bound")
    if checkpoint_scope.get("core_inventory_sha256") != "d" * 64:
        raise SystemExit("checkpoint_tokenizer_scope_parse_missing_core_inventory_sha256")
    print("parse_quality_log_tokenizer_scope_ok=true")


if __name__ == "__main__":
    main()
