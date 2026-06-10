from __future__ import annotations

from kobun_llm.tokenizer import BYTE_FALLBACK_TOKENIZER_TYPE, tokenizer_from_text


def main() -> None:
    train_text = "\u3042\u3044\u3046\u3048\u304a/\n"
    heldout_text = "\u3042\U00020bb7/\u3092\u304b\u3057\n"
    tokenizer = tokenizer_from_text(train_text, tokenizer_type=BYTE_FALLBACK_TOKENIZER_TYPE)

    encoded_train = tokenizer.encode(train_text)
    decoded_train = tokenizer.decode(encoded_train)
    if decoded_train != train_text:
        raise SystemExit("train text does not roundtrip through byte fallback tokenizer")
    if any(str(tokenizer.itos[token_id]).startswith("<0x") for token_id in encoded_train):
        raise SystemExit("train-derived chars should remain direct tokens")

    encoded_heldout = tokenizer.encode(heldout_text)
    decoded_heldout = tokenizer.decode(encoded_heldout)
    if decoded_heldout != heldout_text:
        raise SystemExit("heldout-only chars do not roundtrip through UTF-8 byte fallback")
    if "\U00020bb7" in tokenizer.stoi:
        raise SystemExit("heldout-only char leaked into direct tokenizer vocab")
    byte_tokens = [tokenizer.itos[token_id] for token_id in encoded_heldout if str(tokenizer.itos[token_id]).startswith("<0x")]
    if not byte_tokens:
        raise SystemExit("heldout-only chars did not use byte fallback tokens")

    payload = tokenizer.to_dict()
    restored = tokenizer.from_dict(payload)
    if restored.decode(restored.encode(heldout_text)) != heldout_text:
        raise SystemExit("serialized byte fallback tokenizer does not preserve lossless behavior")
    print(
        "byte_fallback_tokenizer_ok=true "
        f"vocab_size={tokenizer.vocab_size} "
        f"byte_tokens_used={len(byte_tokens)}"
    )


if __name__ == "__main__":
    main()
