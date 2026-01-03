from __future__ import annotations

import argparse

import torch

from litegpt.config import read_yaml
from litegpt.model import LiteGPT
from litegpt.tokenizer import load_codec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hparams", type=str, default="hparams_100m.yaml")
    args = ap.parse_args()

    cfg = read_yaml(args.hparams)
    codec = load_codec(str(cfg["paths"].get("tokenizer_json")))

    net_cfg = cfg["net"]
    model = LiteGPT(
        codec.vocab,
        seq_len=int(cfg["text"]["seq_len"]),
        layers=int(net_cfg["layers"]),
        heads=int(net_cfg["heads"]),
        width=int(net_cfg["width"]),
        dropout=float(net_cfg.get("dropout", 0.0)),
    )

    n = sum(p.numel() for p in model.parameters())

    def fmt_bytes(b: float) -> str:
        for unit in ("B", "KB", "MB", "GB", "TB"):
            if b < 1024.0:
                return f"{b:.2f}{unit}"
            b /= 1024.0
        return f"{b:.2f}PB"

    print(f"params: {n:,}")
    print(f"weights fp32: {fmt_bytes(n * 4)}")
    print(f"weights bf16/fp16: {fmt_bytes(n * 2)}")
    print(f"weights int8 (rough): {fmt_bytes(n * 1)}")

    # Optimizer states are often much larger than weights (e.g., Adam keeps m/v).
    print(f"adam states fp32 (rough): {fmt_bytes(n * 8)}  # m + v")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
