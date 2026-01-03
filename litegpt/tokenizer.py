from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Protocol


class TextCodec(Protocol):
    vocab: int
    bos: int
    eos: int
    pad: int
    unk: int

    def encode(self, text: str, add_special: bool = False) -> List[int]:
        ...

    def decode(self, ids: List[int]) -> str:
        ...


@dataclass
class ByteCodec:
    vocab: int = 260
    bos: int = 256
    eos: int = 257
    pad: int = 258
    unk: int = 259

    def encode(self, text: str, add_special: bool = False) -> List[int]:
        raw = text.encode("utf-8")
        ids = list(raw)
        if add_special:
            return [self.bos] + ids + [self.eos]
        return ids

    def decode(self, ids: List[int]) -> str:
        keep = [i for i in ids if 0 <= i < 256]
        return bytes(keep).decode("utf-8", errors="ignore")


def load_codec(tokenizer_json: str | None) -> TextCodec:
    if tokenizer_json:
        p = Path(tokenizer_json)
        if p.exists():
            try:
                from tokenizers import Tokenizer

                core = Tokenizer.from_file(str(p))

                class HFCodec:
                    def __init__(self, t: Tokenizer):
                        self._t = t
                        self.vocab = t.get_vocab_size()
                        self.bos = t.token_to_id("<bos>")
                        self.eos = t.token_to_id("<eos>")
                        self.pad = t.token_to_id("<pad>")
                        self.unk = t.token_to_id("<unk>")
                        if None in (self.bos, self.eos, self.pad, self.unk):
                            raise ValueError("Tokenizer missing special tokens")

                    def encode(self, text: str, add_special: bool = False) -> List[int]:
                        return self._t.encode(text, add_special_tokens=add_special).ids

                    def decode(self, ids: List[int]) -> str:
                        return self._t.decode(ids)

                return HFCodec(core)
            except Exception:
                pass

    return ByteCodec()
