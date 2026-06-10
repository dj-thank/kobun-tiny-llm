from __future__ import annotations

from dataclasses import dataclass
from typing import Self


BYTE_FALLBACK_TOKENIZER_TYPE = "byte_fallback_char_v1"
BYTE_TOKEN_PREFIX = "<0x"
BYTE_TOKEN_SUFFIX = ">"


def byte_token(value: int) -> str:
    if not 0 <= value <= 255:
        raise ValueError(f"byte value out of range: {value}")
    return f"{BYTE_TOKEN_PREFIX}{value:02X}{BYTE_TOKEN_SUFFIX}"


def byte_token_value(token: str) -> int | None:
    if not (token.startswith(BYTE_TOKEN_PREFIX) and token.endswith(BYTE_TOKEN_SUFFIX)):
        return None
    raw = token[len(BYTE_TOKEN_PREFIX) : -len(BYTE_TOKEN_SUFFIX)]
    if len(raw) != 2:
        return None
    try:
        return int(raw, 16)
    except ValueError:
        return None


@dataclass(frozen=True)
class CharTokenizer:
    stoi: dict[str, int]
    itos: dict[int, str]
    unk_token: str | None = "<unk>"
    tokenizer_type: str = "char"

    @classmethod
    def from_text(cls, text: str, add_unk: bool = True) -> Self:
        chars = sorted(set(text))
        stoi = {ch: i for i, ch in enumerate(chars)}
        unk_token = "<unk>" if add_unk else None
        if unk_token is not None and unk_token not in stoi:
            stoi[unk_token] = len(stoi)
        itos = {i: ch for ch, i in stoi.items()}
        return cls(stoi=stoi, itos=itos, unk_token=unk_token)

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def missing_chars(self, text: str) -> list[str]:
        return sorted(set(text) - set(self.stoi))

    def encode(self, text: str) -> list[int]:
        missing = self.missing_chars(text)
        unk_id = self.stoi.get(self.unk_token) if self.unk_token is not None else None
        if missing and unk_id is None:
            shown = "".join(missing[:20])
            raise ValueError(f"Prompt contains characters outside tokenizer vocabulary: {shown!r}")
        return [self.stoi.get(ch, unk_id) for ch in text]  # type: ignore[list-item]

    def decode(self, ids: list[int]) -> str:
        chars = []
        for token_id in ids:
            ch = self.itos[int(token_id)]
            chars.append("�" if self.unk_token is not None and ch == self.unk_token else ch)
        return "".join(chars)

    def to_dict(self) -> dict[str, object]:
        return {"stoi": self.stoi, "unk_token": self.unk_token, "tokenizer_type": self.tokenizer_type}

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "CharTokenizer":
        if payload.get("tokenizer_type") == BYTE_FALLBACK_TOKENIZER_TYPE or payload.get("byte_fallback") is True:
            return ByteFallbackCharTokenizer.from_dict(payload)
        stoi = {str(k): int(v) for k, v in dict(payload["stoi"]).items()}
        itos = {i: ch for ch, i in stoi.items()}
        raw_unk = payload.get("unk_token")
        unk_token = str(raw_unk) if isinstance(raw_unk, str) and str(raw_unk) in stoi else None
        if unk_token is None and "<unk>" in stoi:
            unk_token = "<unk>"
        return cls(stoi=stoi, itos=itos, unk_token=unk_token)


@dataclass(frozen=True)
class ByteFallbackCharTokenizer(CharTokenizer):
    tokenizer_type: str = BYTE_FALLBACK_TOKENIZER_TYPE

    @classmethod
    def from_text(cls, text: str, add_unk: bool = True) -> "ByteFallbackCharTokenizer":
        direct_chars = sorted(set(text))
        stoi = {ch: i for i, ch in enumerate(direct_chars)}
        for value in range(256):
            token = byte_token(value)
            if token not in stoi:
                stoi[token] = len(stoi)
        unk_token = "<unk>" if add_unk else None
        if unk_token is not None and unk_token not in stoi:
            stoi[unk_token] = len(stoi)
        itos = {i: ch for ch, i in stoi.items()}
        return cls(stoi=stoi, itos=itos, unk_token=unk_token)

    @property
    def direct_chars(self) -> set[str]:
        return {token for token in self.stoi if len(token) == 1}

    def missing_chars(self, text: str) -> list[str]:
        return []

    def direct_missing_chars(self, text: str) -> list[str]:
        return sorted(set(text) - self.direct_chars)

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for ch in text:
            token_id = self.stoi.get(ch)
            if token_id is not None:
                ids.append(token_id)
                continue
            for value in ch.encode("utf-8"):
                ids.append(self.stoi[byte_token(value)])
        return ids

    def decode(self, ids: list[int]) -> str:
        chars: list[str] = []
        pending = bytearray()

        def flush_pending() -> None:
            nonlocal pending
            if pending:
                chars.append(pending.decode("utf-8", errors="replace"))
                pending = bytearray()

        for token_id in ids:
            token = self.itos[int(token_id)]
            value = byte_token_value(token)
            if value is not None:
                pending.append(value)
                continue
            flush_pending()
            chars.append("�" if self.unk_token is not None and token == self.unk_token else token)
        flush_pending()
        return "".join(chars)

    def to_dict(self) -> dict[str, object]:
        payload = super().to_dict()
        payload["tokenizer_type"] = BYTE_FALLBACK_TOKENIZER_TYPE
        payload["byte_fallback"] = True
        payload["byte_token_prefix"] = BYTE_TOKEN_PREFIX
        payload["byte_token_suffix"] = BYTE_TOKEN_SUFFIX
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "ByteFallbackCharTokenizer":
        stoi = {str(k): int(v) for k, v in dict(payload["stoi"]).items()}
        for value in range(256):
            token = byte_token(value)
            if token not in stoi:
                raise ValueError(f"byte fallback tokenizer is missing byte token: {token}")
        itos = {i: ch for ch, i in stoi.items()}
        raw_unk = payload.get("unk_token")
        unk_token = str(raw_unk) if isinstance(raw_unk, str) and str(raw_unk) in stoi else None
        if unk_token is None and "<unk>" in stoi:
            unk_token = "<unk>"
        return cls(stoi=stoi, itos=itos, unk_token=unk_token)


def tokenizer_from_text(text: str, tokenizer_type: str = "char", add_unk: bool = True) -> CharTokenizer:
    if tokenizer_type == "char":
        return CharTokenizer.from_text(text, add_unk=add_unk)
    if tokenizer_type == BYTE_FALLBACK_TOKENIZER_TYPE:
        return ByteFallbackCharTokenizer.from_text(text, add_unk=add_unk)
    raise ValueError(f"unknown tokenizer_type: {tokenizer_type}")


def tokenizer_vocab_source_text(training_text: str, tokenizer_extra_text: str, tokenizer_type: str) -> str:
    if tokenizer_type == BYTE_FALLBACK_TOKENIZER_TYPE and tokenizer_extra_text:
        return tokenizer_extra_text
    return training_text + tokenizer_extra_text
