from __future__ import annotations

import argparse

import warnings

import torch

from litegpt.config import pick_device
from litegpt.decoding import Sampler, reply
from litegpt.model import LiteGPT
from litegpt.tokenizer import load_codec


warnings.filterwarnings("ignore", message="TypedStorage is deprecated")


def load_ckpt(path: str):
    # our checkpoints store only tensors + plain dict
    return torch.load(path, map_location="cpu", weights_only=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="artifacts/last.pt")
    ap.add_argument("--prompt", type=str, required=True)
    ap.add_argument("--device", type=str, default="cpu", choices=["cpu", "auto"])
    ap.add_argument("--max_new", type=int, default=None)
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--top_k", type=int, default=None)
    ap.add_argument("--top_p", type=float, default=None)
    ap.add_argument("--repeat_penalty", type=float, default=None)
    ap.add_argument("--stop", nargs="*", default=None)
    args = ap.parse_args()

    obj = load_ckpt(args.ckpt)
    cfg = obj["cfg"]

    codec = load_codec(str(cfg["paths"]["tokenizer_json"]))

    net_cfg = cfg["net"]
    model = LiteGPT(
        codec.vocab,
        seq_len=int(cfg["text"]["seq_len"]),
        layers=int(net_cfg["layers"]),
        heads=int(net_cfg["heads"]),
        width=int(net_cfg["width"]),
        dropout=float(net_cfg.get("dropout", 0.0)),
    )

    sd = obj["weights"]
    is_int8 = any("_packed_params" in k for k in sd.keys())
    if is_int8:
        dev = torch.device("cpu")
        model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    else:
        dev = pick_device(args.device)

    model.load_state_dict(sd)
    model.to(dev)

    dec = cfg.get("decode", {})
    s = Sampler(
        max_new=int(args.max_new if args.max_new is not None else dec.get("max_new", 96)),
        min_new=int(dec.get("min_new", 6)),
        temperature=float(args.temperature if args.temperature is not None else dec.get("temperature", 0.0)),
        top_k=int(args.top_k if args.top_k is not None else dec.get("top_k", 0)),
        top_p=float(args.top_p if args.top_p is not None else dec.get("top_p", 1.0)),
        repeat_penalty=float(args.repeat_penalty if args.repeat_penalty is not None else dec.get("repeat_penalty", 1.05)),
        stop=list(args.stop) if args.stop is not None else list(dec.get("stop", [])),
    )

    prefix = str(cfg["text"]["prefix_pattern"])
    out = reply(model, codec, prompt=args.prompt, device=dev, prefix_template=prefix, s=s)
    print(out)


if __name__ == "__main__":
    main()
