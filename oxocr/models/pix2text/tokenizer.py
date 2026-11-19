"""Pix2Text MFR 的 ByteLevel-BPE 解码（仅解码方向，纯 stdlib）。

tokenizer.json 是 XLMRobertaTokenizerFast 的 tokenizers 导出：词片用 GPT-2
字节字母表（bytes_to_unicode）表示。解码 = 拼词片 → 过滤特殊 token →
字节逆映射 → UTF-8，与 examples/OCR-buddy-main/src/offscreen/formula.ts
的 decodeIds 逐行对齐（含未知字符丢弃语义），不引入 tokenizers 依赖。
"""
from __future__ import annotations

import json
from pathlib import Path

__all__ = ["byte_to_unicode_inverse", "MfrTokenizer"]


def byte_to_unicode_inverse() -> dict[str, int]:
    """GPT-2 bytes_to_unicode 的逆映射：字母表字符 → 原字节值（Ġ → 32 等）。"""
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs, strict=True)}


class MfrTokenizer:
    """只做 id[] → 文本；编码方向用不到（自回归解码产出 token id）。"""

    def __init__(self, path: str | Path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        vocab: dict[str, int] = data["model"]["vocab"]
        self._id_to_piece: list[str] = [""] * (max(vocab.values(), default=-1) + 1)
        for piece, idx in vocab.items():
            self._id_to_piece[idx] = piece
        self._special_ids = frozenset(
            t["id"] for t in data.get("added_tokens", ()) if t.get("special")
        )
        self._byte_inv = byte_to_unicode_inverse()

    @property
    def vocab_size(self) -> int:
        return len(self._id_to_piece)

    def decode(self, ids: list[int]) -> str:
        pieces = "".join(
            self._id_to_piece[i]
            for i in ids
            if 0 <= i < len(self._id_to_piece) and i not in self._special_ids
        )
        raw = bytes(self._byte_inv[ch] for ch in pieces if ch in self._byte_inv)
        return raw.decode("utf-8", errors="replace").strip()
