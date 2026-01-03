from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .batching import JsonlInstruction, stack_pad
from .config import pick_device, read_yaml, threads_from
from .file_io import mkdir
from .model import LiteGPT
from .randomness import seed_all
from .tokenizer import load_codec


def _cosine_warmup(step: int, warmup: int, total: int) -> float:
    if step < warmup:
        return step / max(1, warmup)
    x = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * x))


def _accum_steps(batch_tokens: int, micro: int, seq_len: int) -> int:
    # micro batch processes roughly micro*seq_len tokens
    denom = max(1, micro * seq_len)
    return max(1, int(math.ceil(batch_tokens / denom)))


@torch.no_grad()
def _eval(model: nn.Module, loader: DataLoader, loss_fn: nn.Module, device: torch.device) -> float:
    model.eval()
    total = 0.0
    n = 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        logits = model(xb)
        loss = loss_fn(logits.view(-1, logits.size(-1)), yb.view(-1))
        total += float(loss.item())
        n += 1
    model.train()
    return total / max(1, n)


def fit(hparams_path: str, device_arg: str | None) -> dict[str, Any]:
    cfg = read_yaml(hparams_path)

    seed_all(int(cfg.get("seed", 123)))
    torch.set_num_threads(threads_from(str(cfg.get("threads", "auto"))))

    paths = cfg["paths"]
    out_dir = Path(str(paths["out_dir"]))
    mkdir(out_dir)

    export_cfg = cfg.get("export", {})
    export_int8 = bool(export_cfg.get("int8", False))

    codec = load_codec(str(paths.get("tokenizer_json")))

    seq_len = int(cfg["text"]["seq_len"])
    prefix_tmpl = str(cfg["text"]["prefix_pattern"])

    train_ds = JsonlInstruction(str(paths["train_jsonl"]), codec=codec, seq_len=seq_len, prefix_template=prefix_tmpl)
    valid_ds = JsonlInstruction(str(paths["valid_jsonl"]), codec=codec, seq_len=seq_len, prefix_template=prefix_tmpl)

    micro = int(cfg["optim"]["micro_batch"])

    def coll(b):
        return stack_pad(b, seq_len=seq_len, pad_id=codec.pad)

    train_loader = DataLoader(train_ds, batch_size=micro, shuffle=True, num_workers=0, collate_fn=coll)
    valid_loader = DataLoader(valid_ds, batch_size=micro, shuffle=False, num_workers=0, collate_fn=coll)

    net_cfg = cfg["net"]
    model = LiteGPT(
        codec.vocab,
        seq_len=seq_len,
        layers=int(net_cfg["layers"]),
        heads=int(net_cfg["heads"]),
        width=int(net_cfg["width"]),
        dropout=float(net_cfg.get("dropout", 0.0)),
    )

    dev = pick_device(device_arg)
    model.to(dev)

    opt_cfg = cfg["optim"]
    lr = float(opt_cfg["lr"])
    wd = float(opt_cfg["weight_decay"])
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    total_steps = int(opt_cfg["total_steps"])
    warmup = int(opt_cfg["warmup_steps"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: _cosine_warmup(s, warmup, total_steps))

    max_seconds_raw = opt_cfg.get("max_seconds", None)
    max_seconds = None if max_seconds_raw in (None, "", "none", "None") else float(max_seconds_raw)

    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    batch_tokens = int(opt_cfg.get("batch_tokens", 2048))
    accum = _accum_steps(batch_tokens, micro, seq_len)

    clip = float(opt_cfg.get("grad_clip", 1.0))
    every = int(cfg["eval"]["every"])

    # Auto-stop when loss is sufficiently small.
    # User requirement: stop once loss <= 0.0001.
    stop_loss = 1e-4

    step = 0
    micro_step = 0
    t0 = time.time()
    last_log_t = t0

    def _save_last() -> None:
        torch.save({"weights": model.state_dict(), "cfg": cfg}, out_dir / "last.pt")

    def _export_int8() -> None:
        if not export_int8:
            return
        q = torch.quantization.quantize_dynamic(
            model.to("cpu"),
            {torch.nn.Linear},
            dtype=torch.qint8,
        )
        torch.save({"weights": q.state_dict(), "cfg": cfg}, out_dir / "int8.pt")

    train_iter = iter(train_loader)
    model.train()
    try:
        while step < total_steps:
            try:
                xb, yb = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                xb, yb = next(train_iter)

            xb = xb.to(dev)
            yb = yb.to(dev)

            logits = model(xb)
            loss = loss_fn(logits.view(-1, logits.size(-1)), yb.view(-1))
            loss.backward()
            micro_step += 1

            if micro_step >= accum:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                opt.step()
                opt.zero_grad(set_to_none=True)
                sched.step()

                step += 1
                micro_step = 0

                if step % 10 == 0:
                    cur_lr = sched.get_last_lr()[0]
                    now = time.time()
                    elapsed = now - t0
                    dt = now - last_log_t
                    last_log_t = now
                    steps_left = max(0, total_steps - step)
                    eta = (elapsed / max(1, step)) * steps_left
                    print(
                        f"step {step} loss {loss.item():.4f} lr {cur_lr:.6f} "
                        f"accum {accum} +{dt:.1f}s elapsed {elapsed:.1f}s eta {eta:.1f}s"
                    )

                if step % every == 0:
                    ev = _eval(model, valid_loader, loss_fn, dev)
                    dt = time.time() - t0
                    print(f"valid loss {ev:.4f} elapsed {dt:.1f}s")
                    _save_last()

                    if ev <= stop_loss:
                        print(f"early stop: valid loss {ev:.6f} <= {stop_loss}")
                        break

                if loss.item() <= stop_loss:
                    print(f"early stop: train loss {loss.item():.6f} <= {stop_loss}")
                    break

                if max_seconds is not None and (time.time() - t0) >= max_seconds:
                    break
    except KeyboardInterrupt:
        # Be resilient: still save a usable checkpoint on unexpected interruption.
        msg = "interrupted; saving checkpoint"
        if export_int8:
            msg += " and exporting int8"
        msg += "..."
        print(msg)
        _save_last()
        try:
            _export_int8()
        except Exception as e:
            print(f"int8 export failed: {e}")
        return {"out_dir": str(out_dir), "elapsed": float(time.time() - t0), "interrupted": True}

    _save_last()
    _export_int8()
    return {"out_dir": str(out_dir), "elapsed": float(time.time() - t0), "interrupted": False}
