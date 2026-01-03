from __future__ import annotations

import argparse
import json
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers

from litegpt.config import read_yaml
from litegpt.file_io import mkdir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hparams", type=str, default="hparams_100m.yaml")
    args = ap.parse_args()

    cfg = read_yaml(args.hparams)
    train_jsonl = str(cfg["paths"]["train_jsonl"])
    out_json = str(cfg["paths"]["tokenizer_json"])
    vocab = int(cfg["bpe"]["vocab_size"])

    prefix = str(cfg["text"]["prefix_pattern"])

    # Build BPE from texts that match the exact train/infer formatting.
    outp = Path(out_json)
    mkdir(outp.parent)

    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab,
        special_tokens=["<bos>", "<eos>", "<pad>", "<unk>"],
    )

    def iter_texts():
        with open(train_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                q = str(obj.get("prompt", ""))
                a = str(obj.get("completion", ""))
                yield prefix.format(q=q) + a

    tok.train_from_iterator(iter_texts(), trainer)

    tok.post_processor = processors.TemplateProcessing(
        single="<bos> $A <eos>",
        pair=None,
        special_tokens=[
            ("<bos>", tok.token_to_id("<bos>")),
            ("<eos>", tok.token_to_id("<eos>")),
        ],
    )

    tok.save(str(outp))
    print(f"tokenizer saved: {out_json}")


if __name__ == "__main__":
    main()
