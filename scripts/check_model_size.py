from __future__ import annotations

import argparse

from kobun_llm.model import GPT, GPTConfig
from kobun_llm.tokenizer import CharTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check parameter count for a Kobun GPT config.")
    parser.add_argument("--vocab-size", type=int, default=2304)
    parser.add_argument("--data", type=str, default="", help="Optional corpus path used to infer train tokenizer vocab size.")
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--n-layer", type=int, default=16)
    parser.add_argument("--n-head", type=int, default=12)
    parser.add_argument("--num-key-value-heads", type=int, default=6)
    parser.add_argument("--n-embd", type=int, default=768)
    parser.add_argument("--intermediate-size", type=int, default=2304)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--min-params", type=int, default=100_000_000)
    parser.add_argument("--max-params", type=int, default=250_000_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    vocab_size = args.vocab_size
    if args.data:
        text = open(args.data, "r", encoding="utf-8").read()
        vocab_size = CharTokenizer.from_text(text).vocab_size
    config = GPTConfig(
        vocab_size=vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
        intermediate_size=args.intermediate_size,
        num_key_value_heads=args.num_key_value_heads,
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
    params = sum(param.numel() for param in model.parameters())
    print(f"vocab_size={vocab_size}")
    print(f"params={params}")
    print(f"params_b={params / 1_000_000_000:.4f}")
    if not args.min_params <= params <= args.max_params:
        raise SystemExit(
            f"parameter count outside expected range: {params} "
            f"not in [{args.min_params}, {args.max_params}]"
        )


if __name__ == "__main__":
    main()
