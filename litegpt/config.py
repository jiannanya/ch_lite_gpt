from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


def read_yaml(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    obj = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError("Config YAML must be a mapping")
    return obj


def pick_device(want: str | None):
    import torch

    if want is None or want == "auto":
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if want == "cpu":
        return torch.device("cpu")
    raise RuntimeError(f"Unknown device: {want}. Use auto|cpu")


def threads_from(spec: str) -> int:
    import os

    if spec == "auto":
        return os.cpu_count() or 1
    try:
        n = int(spec)
        return max(1, n)
    except Exception:
        return os.cpu_count() or 1


@dataclass(frozen=True)
class Paths:
    train_jsonl: str
    valid_jsonl: str
    tokenizer_json: str
    out_dir: str
