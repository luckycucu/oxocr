"""Indra 识别器词表（推理子集，移植自 indra_ocr/vocab.py）。

class 0 是 CTC blank，class i+1 是 chars[i]。token 为单字符或一个 LaTeX
命令（``\\frac``）。cjk 词表含韩文 jamo（U+1100…），decode 时 NFC 合成音节。
训练侧工具（build_* / encode / canonical / indic_*）不接。
"""

from __future__ import annotations

import json
import unicodedata

__all__ = ["Vocab"]


class Vocab:
    def __init__(self, chars: list[str]):
        self.chars = list(chars)
        self.index = {c: i + 1 for i, c in enumerate(self.chars)}
        self.has_commands = any(len(c) > 1 for c in self.chars)
        self.jamo = "\u1100" in self.index

    @property
    def n_classes(self) -> int:
        return len(self.chars) + 1  # + blank

    def decode(self, ids: list[int]) -> str:
        toks = [self.chars[i - 1] for i in ids if 0 < i <= len(self.chars)]
        if self.jamo:
            return unicodedata.normalize("NFC", "".join(toks))
        if not self.has_commands:
            return "".join(toks)
        out = []
        for k, t in enumerate(toks):
            out.append(t)
            # "\\alpha" + "x" 不能读回 "\\alphax"
            nxt = toks[k + 1] if k + 1 < len(toks) else ""
            if len(t) > 2 and t[0] == "\\" and t[1:].isalpha() and nxt[:1].isalpha():
                out.append(" ")
        return "".join(out)

    @classmethod
    def from_json(cls, s: str) -> Vocab:
        d = json.loads(s)
        if d.get("blank", 0) != 0:
            raise ValueError("only blank=0 vocabularies are supported")
        return cls(d["chars"])
