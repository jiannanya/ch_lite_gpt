from __future__ import annotations

from pathlib import Path


def mkdir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)
