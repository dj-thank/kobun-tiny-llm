from __future__ import annotations

import json
import tempfile
from pathlib import Path

from old_japanese_run_intel import active_run_release_policy_issues, parse_training_log


RUN_ID = "old_japanese_0_1b_dml_20990101_000000"


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_meta(root: Path, payload: dict[str, object]) -> None:
    path = root / "data" / "run_snapshots" / RUN_ID / "provenance" / "tokenizer_public_char_vocab.meta.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        log = root / "logs" / f"{RUN_ID}.out.log"
        write(log, "config=vocab=22450 block=512 layers=16 heads=12 params=130515200\n")
        write_meta(
            root,
            {
                "policy": "train_split_plus_fixed_unicode_japanese_inventory",
                "total_chars": 22449,
            },
        )
        issues = active_run_release_policy_issues(root, RUN_ID, parse_training_log(log))
        expected = {
            "active_vocab_too_large:22450",
            "active_block_size_obsolete:512",
            "active_tokenizer_policy_obsolete:train_split_plus_fixed_unicode_japanese_inventory",
            "active_tokenizer_missing_byte_fallback",
            "active_tokenizer_byte_token_count_invalid",
            "active_tokenizer_vocab_policy_obsolete:22449",
        }
        if not expected.issubset(set(issues)):
            raise SystemExit(f"obsolete active run was not fully detected: {issues}")

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        log = root / "logs" / f"{RUN_ID}.out.log"
        write(log, "config=vocab=3317 block=384 layers=16 heads=12 params=115821056\n")
        write_meta(
            root,
            {
                "policy": "train_split_plus_core_japanese_inventory_plus_utf8_byte_fallback_v1",
                "byte_fallback": True,
                "byte_fallback_tokens": 256,
                "estimated_total_vocab_with_byte_fallback_and_unk": 3288,
                "direct_vocab_chars": 3031,
            },
        )
        issues = active_run_release_policy_issues(root, RUN_ID, parse_training_log(log))
        if issues:
            raise SystemExit(f"release-shaped active run was incorrectly marked obsolete: {issues}")
    print("active_run_release_policy_ok=true")


if __name__ == "__main__":
    main()
