from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List, Tuple, Optional

import torch
from torch.utils.data import Dataset

from .tokenizer import TextCodec


@dataclass(frozen=True)
class Row:
    x: List[int]
    y: List[int]


class JsonlInstruction(Dataset):
    def __init__(self, path: str, *, codec: TextCodec, seq_len: int, prefix_template: str):
        self._path = path
        self._codec = codec
        self._seq_len = int(seq_len)
        self._prefix_template = prefix_template

        # Byte offsets for each non-empty JSONL line. This keeps memory small and
        # allows random access + shuffle without loading/encoding everything.
        self._offsets: list[int] = []
        with open(path, "rb") as f:
            while True:
                off = f.tell()
                line = f.readline()
                if not line:
                    break
                if line.strip():
                    self._offsets.append(off)

        self._fp: Optional[object] = None

    def _file(self):
        if self._fp is None or getattr(self._fp, "closed", True):
            self._fp = open(self._path, "rb")
        return self._fp

    def __len__(self) -> int:
        return len(self._offsets)

    def __getitem__(self, idx: int) -> Tuple[List[int], List[int]]:
        f = self._file()
        f.seek(self._offsets[idx])
        line = f.readline()
        obj = json.loads(line.decode("utf-8"))
        # New schema: {"query": ..., "answer": ...}
        # Backward compatible with older corpora: {"prompt": ..., "completion": ...}
        q = str(obj.get("query", obj.get("prompt", "")))
        a = str(obj.get("answer", obj.get("completion", "")))

        prefix_text = self._prefix_template.format(q=q)
        prefix = self._codec.encode(prefix_text, add_special=True)
        answer = self._codec.encode(a, add_special=False)

        ids = prefix + answer + [self._codec.eos]
        if len(ids) > self._seq_len:
            ids = ids[: self._seq_len]

        # next-token targets
        tars = ids[1:] + [self._codec.eos]

        # mask the whole prefix except the last position so the first answer token participates
        ignore = min(max(0, len(prefix) - 1), len(tars))
        if ignore:
            tars[:ignore] = [-100] * ignore

        return ids, tars


def stack_pad(batch: List[Tuple[List[int], List[int]]], *, seq_len: int, pad_id: int):
    xs: list[list[int]] = []
    ys: list[list[int]] = []
    for x, y in batch:
        xs.append(x + [pad_id] * (seq_len - len(x)))
        ys.append(y + [-100] * (seq_len - len(y)))
    return torch.tensor(xs, dtype=torch.long), torch.tensor(ys, dtype=torch.long)
