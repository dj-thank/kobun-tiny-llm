from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

from kobun_llm.model import GPT, GPTConfig
from kobun_llm.tokenizer import BYTE_FALLBACK_TOKENIZER_TYPE, tokenizer_from_text, tokenizer_vocab_source_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check a GPT config parameter count before launching a long run.")
    parser.add_argument("--data", type=Path, default=Path("data/kobun_worldclass_corpus.txt"))
    parser.add_argument(
        "--tokenizer-extra-data",
        action="append",
        type=Path,
        default=[],
        help="Additional files included only for tokenizer vocabulary coverage.",
    )
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--n-layer", type=int, default=16)
    parser.add_argument("--n-head", type=int, default=12)
    parser.add_argument("--num-key-value-heads", type=int, default=6)
    parser.add_argument("--n-embd", type=int, default=768)
    parser.add_argument("--intermediate-size", type=int, default=2304)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--tokenizer-type", choices=["char", BYTE_FALLBACK_TOKENIZER_TYPE], default="char")
    parser.add_argument("--min-params", type=int, default=100_000_000)
    parser.add_argument("--max-params", type=int, default=180_000_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    text = args.data.read_text(encoding="utf-8")
    tokenizer_extra_text = "".join(path.read_text(encoding="utf-8") for path in args.tokenizer_extra_data)
    tokenizer_text = tokenizer_vocab_source_text(text, tokenizer_extra_text, args.tokenizer_type)
    tokenizer = tokenizer_from_text(tokenizer_text, tokenizer_type=args.tokenizer_type)
    config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        num_key_value_heads=args.num_key_value_heads,
        n_embd=args.n_embd,
        intermediate_size=args.intermediate_size,
        dropout=args.dropout,
        norm_type="rmsnorm",
        mlp_type="swiglu",
        use_rope=True,
        rope_theta=1_000_000.0,
        tie_word_embeddings=True,
        attention_bias=False,
        mlp_bias=False,
        qk_norm=True,
    )
    model = GPT(config)
    param_count = sum(param.numel() for param in model.parameters())
    print(f"params={param_count} params_b={param_count / 1_000_000_000:.3f}")
    print(
        f"vocab_size={tokenizer.vocab_size} tokenizer_type={getattr(tokenizer, 'tokenizer_type', 'char')} "
        f"tokenizer_extra_files={len(args.tokenizer_extra_data)}"
    )
    print(f"config={asdict(config)}")
    if not args.min_params <= param_count < args.max_params:
        raise SystemExit(
            f"parameter count {param_count} outside required range "
            f"[{args.min_params}, {args.max_params})"
        )


if __name__ == "__main__":
    main()
