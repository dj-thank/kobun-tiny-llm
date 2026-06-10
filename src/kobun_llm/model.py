from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 128
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.1
    intermediate_size: int | None = None
    num_key_value_heads: int | None = None
    norm_type: str = "layernorm"
    mlp_type: str = "gelu"
    use_rope: bool = False
    rope_theta: float = 10000.0
    tie_word_embeddings: bool = False
    attention_bias: bool = True
    mlp_bias: bool = True
    qk_norm: bool = False


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_float = x.float()
        normed = x_float * torch.rsqrt(x_float.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (self.weight * normed).type_as(x)


def make_norm(config: GPTConfig) -> nn.Module:
    if config.norm_type == "rmsnorm":
        return RMSNorm(config.n_embd)
    if config.norm_type == "layernorm":
        return nn.LayerNorm(config.n_embd)
    raise ValueError(f"Unknown norm_type: {config.norm_type}")


def build_rope_cache(block_size: int, dim: int, theta: float) -> tuple[torch.Tensor, torch.Tensor]:
    if dim % 2 != 0:
        raise ValueError("RoPE requires an even head dimension")
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    positions = torch.arange(block_size, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)
    cos = freqs.cos().view(1, 1, block_size, dim // 2)
    sin = freqs.sin().view(1, 1, block_size, dim // 2)
    return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    _, _, time, _ = x.shape
    cos = cos[:, :, :time, :].to(dtype=x.dtype)
    sin = sin[:, :, :time, :].to(dtype=x.dtype)
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    rotated = torch.stack((x_even * cos - x_odd * sin, x_even * sin + x_odd * cos), dim=-1)
    return rotated.flatten(-2)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        if config.n_embd % config.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        self.n_head = config.n_head
        self.n_kv_head = config.num_key_value_heads or config.n_head
        if config.n_head % self.n_kv_head != 0:
            raise ValueError("n_head must be divisible by num_key_value_heads")
        self.head_dim = config.n_embd // config.n_head
        self.use_legacy_qkv = config.num_key_value_heads is None and config.attention_bias
        self.use_rope = config.use_rope
        self.rope_theta = config.rope_theta
        self.q_norm = RMSNorm(self.head_dim) if config.qk_norm else None
        self.k_norm = RMSNorm(self.head_dim) if config.qk_norm else None
        if self.use_rope:
            rope_cos, rope_sin = build_rope_cache(config.block_size, self.head_dim, config.rope_theta)
            self.register_buffer("rope_cos", rope_cos, persistent=False)
            self.register_buffer("rope_sin", rope_sin, persistent=False)
        else:
            self.rope_cos = None
            self.rope_sin = None
        if self.use_legacy_qkv:
            self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd)
        else:
            kv_channels = self.n_kv_head * self.head_dim
            self.q_proj = nn.Linear(config.n_embd, config.n_head * self.head_dim, bias=config.attention_bias)
            self.k_proj = nn.Linear(config.n_embd, kv_channels, bias=config.attention_bias)
            self.v_proj = nn.Linear(config.n_embd, kv_channels, bias=config.attention_bias)
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=config.attention_bias)
        self.attn_drop = nn.Dropout(config.dropout)
        self.resid_drop = nn.Dropout(config.dropout)
        mask = torch.tril(torch.ones(config.block_size, config.block_size))
        self.register_buffer("mask", mask.view(1, 1, config.block_size, config.block_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, time, channels = x.shape
        if self.use_legacy_qkv:
            qkv = self.qkv(x)
            q, k, v = qkv.split(channels, dim=2)
            q = q.view(batch, time, self.n_head, self.head_dim).transpose(1, 2)
            k = k.view(batch, time, self.n_head, self.head_dim).transpose(1, 2)
            v = v.view(batch, time, self.n_head, self.head_dim).transpose(1, 2)
        else:
            q = self.q_proj(x).view(batch, time, self.n_head, self.head_dim).transpose(1, 2)
            k = self.k_proj(x).view(batch, time, self.n_kv_head, self.head_dim).transpose(1, 2)
            v = self.v_proj(x).view(batch, time, self.n_kv_head, self.head_dim).transpose(1, 2)
            if self.n_kv_head != self.n_head:
                repeat = self.n_head // self.n_kv_head
                k = k.repeat_interleave(repeat, dim=1)
                v = v.repeat_interleave(repeat, dim=1)
        if self.q_norm is not None and self.k_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)
        if self.use_rope:
            assert self.rope_cos is not None and self.rope_sin is not None
            q = apply_rope(q, self.rope_cos, self.rope_sin)
            k = apply_rope(k, self.rope_cos, self.rope_sin)

        att = (q @ k.transpose(-2, -1)) * (self.head_dim**-0.5)
        att = att.masked_fill(self.mask[:, :, :time, :time] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(batch, time, channels)
        return self.resid_drop(self.proj(y))


class FeedForward(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        hidden = config.intermediate_size or 4 * config.n_embd
        self.mlp_type = config.mlp_type
        if self.mlp_type == "swiglu":
            self.gate_proj = nn.Linear(config.n_embd, hidden, bias=config.mlp_bias)
            self.up_proj = nn.Linear(config.n_embd, hidden, bias=config.mlp_bias)
            self.down_proj = nn.Linear(hidden, config.n_embd, bias=config.mlp_bias)
            self.drop = nn.Dropout(config.dropout)
        elif self.mlp_type == "gelu":
            self.net = nn.Sequential(
                nn.Linear(config.n_embd, hidden),
                nn.GELU(),
                nn.Linear(hidden, config.n_embd),
                nn.Dropout(config.dropout),
            )
        else:
            raise ValueError(f"Unknown mlp_type: {self.mlp_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mlp_type == "swiglu":
            return self.drop(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))
        return self.net(x)


class Block(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln1 = make_norm(config)
        self.attn = CausalSelfAttention(config)
        self.ln2 = make_norm(config)
        self.ff = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        self.token_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = None if config.use_rope else nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.ln_f = make_norm(config)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.apply(self._init_weights)
        if config.tie_word_embeddings:
            self.tie_weights()

    def tie_weights(self) -> None:
        self.head.weight = self.token_emb.weight

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        _, time = idx.shape
        if time > self.config.block_size:
            raise ValueError("Input sequence is longer than block_size")
        x = self.token_emb(idx)
        if self.pos_emb is not None:
            positions = torch.arange(0, time, device=idx.device).unsqueeze(0)
            x = x + self.pos_emb(positions)
        x = self.drop(x)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        presence_penalty: float = 0.0,
        logits_processor: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    ) -> torch.Tensor:
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            if presence_penalty > 0:
                for batch_index in range(idx.size(0)):
                    used = torch.unique(idx[batch_index])
                    logits[batch_index, used] -= presence_penalty
            if logits_processor is not None:
                logits = logits_processor(idx, logits)
            if top_k is not None:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[:, [-1]]] = -float("inf")
            if top_p is not None and 0 < top_p < 1:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                sorted_probs = F.softmax(sorted_logits, dim=-1)
                cumulative = torch.cumsum(sorted_probs, dim=-1)
                remove = cumulative > top_p
                remove[..., 1:] = remove[..., :-1].clone()
                remove[..., 0] = False
                sorted_logits = sorted_logits.masked_fill(remove, -float("inf"))
                logits = torch.full_like(logits, -float("inf"))
                logits.scatter_(dim=-1, index=sorted_indices, src=sorted_logits)
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_id), dim=1)
        return idx
