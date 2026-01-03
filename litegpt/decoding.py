from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from collections import defaultdict

import torch


_PUN = set(list(",，。．、：:；;！!？?…"))


def _clean_prefix(s: str) -> str:
    i = 0
    while i < len(s) and (s[i].isspace() or s[i] in _PUN):
        i += 1
    return s[i:]


def _scrub_text(s: str) -> str:
    # Remove common replacement glyphs / control chars that can appear with byte-level tokenization.
    s = s.replace("\ufffd", "")
    s = "".join(ch for ch in s if ch.isprintable() or ch in "\n\t")
    return s


@dataclass
class Sampler:
    max_new: int = 96
    min_new: int = 6
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    repeat_penalty: float = 1.0
    stop: Optional[list[str]] = None


@torch.no_grad()
def reply(model, codec, *, prompt: str, device: torch.device, prefix_template: str, s: Sampler) -> str:
    model.eval()
    norm = prompt.replace(" ", "").replace("\u3000", "")
    prefix = prefix_template.format(q=norm)
    start = codec.encode(prefix, add_special=True)

    x = torch.tensor(start, dtype=torch.long, device=device).unsqueeze(0)
    recent: list[int] = []
    # For trigram blocking: map (prev2, prev1) -> {next_tokens seen before}.
    trigram_next: dict[tuple[int, int], set[int]] = defaultdict(set)

    for step in range(int(s.max_new)):
        logits = model(x)[:, -1, :]
        temp = float(s.temperature)
        greedy = temp <= 0.0
        if not greedy:
            logits = logits / max(1e-6, temp)

        # avoid obvious junk
        for tid in (codec.pad, codec.bos, codec.unk):
            if tid is not None and tid >= 0:
                logits[0, tid] = -float("inf")

        if step < int(s.min_new):
            logits[0, codec.eos] = -float("inf")

        if float(s.repeat_penalty) > 1.0 and recent:
            for tid in recent[-16:]:
                logits[0, tid] = logits[0, tid] / float(s.repeat_penalty)

        # Simple 3-gram blocking to reduce degenerate repetition.
        # Efficient version: only ban tokens that actually occurred after this bigram.
        if len(recent) >= 2:
            a, b = recent[-2], recent[-1]
            banned = trigram_next.get((a, b))
            if banned:
                idx = torch.tensor(list(banned), dtype=torch.long, device=logits.device)
                logits[0, idx] = -float("inf")

        if greedy:
            next_id = torch.argmax(logits, dim=-1, keepdim=True)
            probs = None
        else:
            probs = torch.softmax(logits, dim=-1)

        if probs is not None and int(s.top_k) > 0:
            v, i = torch.topk(probs, int(s.top_k))
            p = torch.zeros_like(probs).scatter_(1, i, v)
            z = p.sum(dim=-1, keepdim=True)
            probs = torch.where(z > 0, p / z, probs)

        if probs is not None and float(s.top_p) < 1.0:
            srt, idx = torch.sort(probs, descending=True)
            c = torch.cumsum(srt, dim=-1)
            keep = c <= float(s.top_p)
            srt = srt * keep
            p = torch.zeros_like(probs).scatter_(1, idx, srt)
            z = p.sum(dim=-1, keepdim=True)
            probs = torch.where(z > 0, p / z, probs)

        if probs is not None:
            probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
            if probs.sum() <= 0:
                next_id = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                next_id = torch.multinomial(probs, 1)

        x = torch.cat([x, next_id], dim=1)
        tid = int(next_id.item())
        if len(recent) >= 2:
            trigram_next[(recent[-2], recent[-1])].add(tid)
        recent.append(tid)

        if tid == codec.eos:
            break

        if s.stop:
            out_ids = x[0].tolist()[len(start) :]
            text = codec.decode(out_ids)
            if any(text.endswith(ss) for ss in s.stop):
                break

    out_ids = x[0].tolist()[len(start) :]
    text = codec.decode(out_ids)
    text = _scrub_text(text)
    return _clean_prefix(text)
