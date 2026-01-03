from __future__ import annotations

import argparse
import subprocess
import sys

from litegpt.trainloop import fit


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=str, default="cpu", choices=["cpu", "auto"])
    ap.add_argument("--hparams", type=str, default="hparams_100m.yaml")
    args = ap.parse_args()

    info = fit(args.hparams, args.device)
    print(f"done: out_dir={info['out_dir']} elapsed={info['elapsed']:.2f}s")

    if bool(info.get("interrupted", False)):
        print("train did not complete normally; verification will not run")
        raise SystemExit(2)

    subprocess.run(
        [sys.executable, "-m", "runner.verify", "--hparams", args.hparams],
        check=True,
    )


if __name__ == "__main__":
    main()
