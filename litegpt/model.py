from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


_ROPE: dict[tuple[str, int, int, str], tuple[torch.Tensor, torch.Tensor]] = {}


def _rope_table(seq: int, half: int, device: torch.device, dtype: torch.dtype):
    key = (str(device), int(seq), int(half), str(dtype))
    cached = _ROPE.get(key)
    if cached is not None:
        return cached

    # compute in fp32 then cast
    idx = torch.arange(half, device=device, dtype=torch.float32)
    pos = torch.arange(seq, device=device, dtype=torch.float32).unsqueeze(1)
    inv = 10000.0 ** (-2.0 * idx / (2.0 * half))
    ang = pos * inv
    cos = torch.cos(ang).to(dtype)
    sin = torch.sin(ang).to(dtype)
    _ROPE[key] = (cos, sin)
    return cos, sin


def _apply_rope(q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # q,k: (B,H,T,D)
    b, h, t, d = q.shape
    half = d // 2
    cos, sin = _rope_table(t, half, q.device, q.dtype)
    cos = cos.view(1, 1, t, half)
    sin = sin.view(1, 1, t, half)

    def rot(x: torch.Tensor) -> torch.Tensor:
        a = x[..., :half]
        b2 = x[..., half : 2 * half]
        y = torch.cat([a * cos - b2 * sin, a * sin + b2 * cos], dim=-1)
        if d > 2 * half:
            y = torch.cat([y, x[..., 2 * half :]], dim=-1)
        return y

    return rot(q), rot(k)


class ScaleNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.g = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).sqrt()
        return x / (rms + self.eps) * self.g


class CausalAttn(nn.Module):
    def __init__(self, dim: int, heads: int, drop: float):
        super().__init__()
        if dim % heads != 0:
            raise ValueError("width must be divisible by heads")
        self.h = heads
        self.dh = dim // heads
        self.to_qkv = nn.Linear(dim, 3 * dim)
        self.to_out = nn.Linear(dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        qkv = self.to_qkv(x)
        q, k, v = qkv.split(c, dim=-1)
        q = q.view(b, t, self.h, self.dh).transpose(1, 2)
        k = k.view(b, t, self.h, self.dh).transpose(1, 2)
        v = v.view(b, t, self.h, self.dh).transpose(1, 2)

        q, k = _apply_rope(q, k)

        s = (q @ k.transpose(-2, -1)) * (self.dh ** -0.5)
        s = s.masked_fill(mask == 0, float("-inf"))
        p = F.softmax(s, dim=-1)
        p = self.drop(p)
        y = p @ v
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        y = self.to_out(y)
        y = self.drop(y)
        return y


class SwiGLU(nn.Module):
    def __init__(self, dim: int, drop: float):
        super().__init__()
        self.w1 = nn.Linear(dim, 4 * dim)
        self.w2 = nn.Linear(dim, 4 * dim)
        self.w3 = nn.Linear(4 * dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = F.silu(self.w1(x))
        b = self.w2(x)
        return self.drop(self.w3(a * b))


class Layer(nn.Module):
    def __init__(self, dim: int, heads: int, drop: float):
        super().__init__()
        self.n1 = ScaleNorm(dim)
        self.attn = CausalAttn(dim, heads, drop)
        self.n2 = ScaleNorm(dim)
        self.ff = SwiGLU(dim, drop)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.n1(x), mask)
        x = x + self.ff(self.n2(x))
        return x


class LiteGPT(nn.Module):
    def __init__(self, vocab: int, *, seq_len: int, layers: int, heads: int, width: int, dropout: float):
        super().__init__()
        self.seq_len = seq_len
        self.emb = nn.Embedding(vocab, width)
        self.blocks = nn.ModuleList([Layer(width, heads, dropout) for _ in range(layers)])
        self.out_norm = ScaleNorm(width)
        self.lm = nn.Linear(width, vocab, bias=False)
        self.lm.weight = self.emb.weight

        # Cache causal mask to avoid per-forward allocation.
        m = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.uint8))
        self.register_buffer("causal_mask", m.view(1, 1, seq_len, seq_len), persistent=False)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        b, t = idx.shape
        if t > self.seq_len:
            idx = idx[:, -self.seq_len :]
            t = idx.shape[1]

        x = self.emb(idx)
        m = self.causal_mask[:, :, :t, :t]
        for blk in self.blocks:
            x = blk(x, m)
        x = self.out_norm(x)
        return self.lm(x)
