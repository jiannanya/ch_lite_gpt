from __future__ import annotations

import argparse
import json
from pathlib import Path

from litegpt.config import read_yaml

import subprocess
import sys
from typing import Any
import locale
import re


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    rows: list[dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def _norm(s: str) -> str:
    # Lightweight normalization for strict-ish answers.
    return "".join(s.split()).strip()


_ANSWER_RE = re.compile(r"[\u4e00-\u9fffA-Za-z0-9_\-]+")


def _extract_answer(raw: str) -> str:
    """Extract a short answer span from model output.

    The verification dataset contains many prompts that request short answers
    (names, channels, single actions). Models may still emit trailing junk; we
    score using a conservative extracted span.
    """
    s = (raw or "").strip()
    if not s:
        return ""
    # Prefer the first line.
    s = s.splitlines()[0].strip()
    # Strip common wrappers.
    s = s.strip(" \t\r\n\"'“”‘’，。;；:：!?！？()（）[]【】{}<>《》")
    m = _ANSWER_RE.search(s)
    return m.group(0) if m else s


def _run(cmd: list[str]) -> None:
    print("\n$ " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hparams", type=str, default="hparams_100m.yaml")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--device", type=str, default="cpu", choices=["cpu", "auto"])
    ap.add_argument("--show_expected", action="store_true")
    ap.add_argument("--ckpt", type=str, default=None)
    ap.add_argument("--max_new", type=int, default=48)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--min_new", type=int, default=1)
    args = ap.parse_args()

    cfg = read_yaml(args.hparams)
    out_dir = str(cfg["paths"]["out_dir"])

    ckpt_path = Path(args.ckpt) if args.ckpt else (Path(out_dir) / "last.pt")
    if not ckpt_path.exists():
        raise SystemExit(
            f"Checkpoint not found: {ckpt_path}. Run training first (runner.train_cpu or runner.train_and_verify), "
            "or pass --ckpt to an existing checkpoint."
        )

    valid_path = str(cfg["paths"]["valid_jsonl"])
    rows = _read_jsonl(valid_path)
    if not rows:
        raise SystemExit(f"No rows found in valid_jsonl: {valid_path}")

    import random

    rng = random.Random(int(args.seed))
    n = max(1, min(int(args.n), len(rows)))
    idxs = rng.sample(range(len(rows)), k=n)

    exact = 0
    ran = 0
    enc = locale.getpreferredencoding(False) or "utf-8"
    for i in idxs:
        row = rows[i]
        prompt = str(row.get("query", row.get("prompt", ""))).strip()
        expected = str(row.get("answer", row.get("completion", ""))).strip()
        if not prompt or not expected:
            continue
        ran += 1

        print("\n" + "=" * 80)
        print(f"[sample {ran}/{n}] prompt:\n{prompt}\n")
        if args.show_expected:
            print(f"expected:\n{expected}\n")

        # We keep subprocess usage for isolation and consistent decoding defaults.
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "runner.query_model",
                "--ckpt",
                str(ckpt_path),
                "--prompt",
                prompt,
                "--device",
                args.device,
                "--max_new",
                str(int(args.max_new)),
                "--min_new",
                str(int(args.min_new)),
                "--temperature",
                str(float(args.temperature)),
                "--top_k",
                "0",
                "--top_p",
                "1.0",
                "--repeat_penalty",
                "1.0",
                "--stop",
                "\n",
                "。",
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        raw = proc.stdout or b""
        try:
            got = raw.decode(enc, errors="replace").strip()
        except Exception:
            got = raw.decode("utf-8", errors="replace").strip()
        print("model:\n" + got)

        extracted = _extract_answer(got)
        if extracted and extracted != got:
            print("extracted:\n" + extracted)

        if _norm(got) == _norm(expected) or (_norm(extracted) and _norm(extracted) == _norm(expected)):
            exact += 1
            print("match: EXACT")
        elif _norm(expected) and (_norm(expected) in _norm(got) or (_norm(extracted) and _norm(expected) in _norm(extracted))):
            print("match: CONTAINS")
        else:
            print("match: MISS")

    if ran == 0:
        raise SystemExit("No valid samples with both query/answer")
    print("\n" + "-" * 80)
    print(f"summary: exact={exact}/{ran} ({exact / ran * 100:.1f}%)")


if __name__ == "__main__":
    main()
