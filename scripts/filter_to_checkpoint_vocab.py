from __future__ import annotations

import argparse
from pathlib import Path

import torch

from kobun_llm.checkpoint_io import load_trusted_checkpoint
from kobun_llm.tokenizer import CharTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter a corpus to the fixed character vocabulary of a checkpoint.")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/kobun_qwen3_12l_worldclass_20260508_090747_best.pt"))
    parser.add_argument("--input", type=Path, default=Path("data/kobun_worldclass_corpus.txt"))
    parser.add_argument("--out", type=Path, default=Path("data/kobun_worldclass_resume_vocab_corpus.txt"))
    parser.add_argument("--replacement", type=str, default="")
    parser.add_argument(
        "--allow-drop-oov",
        action="store_true",
        help="Allow deleting missing characters when --replacement is empty. Unsafe for training quality.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = load_trusted_checkpoint(args.checkpoint, map_location="cpu")
    tokenizer = CharTokenizer.from_dict(payload["tokenizer"])
    allowed = set(tokenizer.stoi)
    text = args.input.read_text(encoding="utf-8")
    missing = sorted(set(text) - allowed)
    if missing and not args.replacement and not args.allow_drop_oov:
        shown = "".join(missing[:80])
        raise SystemExit(
            "Refusing to silently delete OOV characters. "
            "Retrain the tokenizer/model fresh, provide --replacement, or pass --allow-drop-oov explicitly. "
            f"missing_chars={len(missing)} sample={shown!r}"
        )
    filtered = "".join(ch if ch in allowed else args.replacement for ch in text)
    args.out.write_text(filtered, encoding="utf-8")
    print(f"wrote {args.out} bytes={args.out.stat().st_size}")
    print(f"allowed_chars={len(allowed)} missing_chars={len(missing)}")
    if missing:
        print("missing_sample=" + "".join(missing[:80]))


if __name__ == "__main__":
    main()
