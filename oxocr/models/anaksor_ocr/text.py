"""高棉语文本归一化（移植 anaksor-ocr text.py，Apache-2.0）。

同一渲染字簇可有多种码点序列（元音/shifters/coeng 顺序、分裂元音、不可见 ZWSP、
渲染相同的 coeng-DA 与 coeng-TA……）。训练标签与预测都过 normalize() 得到唯一规范拼写，
因此归一化是识别输出契约的一部分，在模型内部完成。
"""

from __future__ import annotations

import re
import unicodedata

__all__ = ["normalize", "normalize_khmer", "COENG", "MARKS"]

# ── 码点 ─────────────────────────────────────────────────────────────────────
COENG = "្"
RO = "រ"
ROBAT = "៌"
SHIFTERS = {"៉", "៊"}  # muusikatoan, triisap
DEP_VOWELS = {chr(c) for c in range(0x17B6, 0x17C6)}  # ា .. ៅ
SIGNS = {
    chr(c)
    for c in (
        0x17C6,
        0x17C7,
        0x17C8,
        0x17CB,
        0x17CD,
        0x17CE,
        0x17CF,
        0x17D0,
        0x17D1,
        0x17D3,
        0x17DD,
    )
}
CONSONANTS = {chr(c) for c in range(0x1780, 0x17A3)}
INDEP_VOWELS = {chr(c) for c in range(0x17A3, 0x17B4)}
BASES = CONSONANTS | INDEP_VOWELS
MARKS = {ROBAT} | SHIFTERS | DEP_VOWELS | SIGNS

# 渲染相同的分裂元音 → 单码点
_VOWEL_COMPOSE = {
    frozenset(("េ", "ី")): "ើ",  # េ + ី → ើ
    frozenset(("េ", "ា")): "ោ",  # េ + ា → ោ
}

# 不可见 / 格式字符——图像中永不可见
_INVISIBLE = dict.fromkeys(map(ord, "​‌‍⁠﻿­឴឵"), None)

_CHAR_MAP = {
    "ឣ": "អ",  # 废弃 ឣ → អ
    "ឤ": "អា",  # 废弃 ឤ → អា
    "៘": "។ល។",  # 废弃 ៘ → ។ល។
    " ": " ",
    " ": " ",
    " ": " ",
    " ": " ",
    " ": " ",
    "　": " ",
    "\t": " ",
    "\n": " ",
    "\r": " ",
    "‘": "'",
    "’": "'",
    "‚": "'",
    "′": "'",
    "´": "'",
    "`": "'",
    "“": '"',
    "”": '"',
    "„": '"',
    "″": '"',
    "‐": "-",
    "‑": "-",
    "‒": "-",
    "−": "-",
    "·": "•",
    "●": "•",  # 中点 / 黑圆 → 项目符号
    "…": "...",
}
_CHAR_TABLE = str.maketrans(_CHAR_MAP)
_WS = re.compile(r" {2,}")


def _is_khmer(c: str) -> bool:
    return "ក" <= c <= "៿"


def _normalize_cluster(base: str, marks: list[str], unify_da: bool) -> str:
    """重排一个基字符后的 marks：robat、coeng（Ro 最后）、shifter、元音、符号。"""
    robat: list[str] = []
    coengs: list[str] = []
    coeng_ro: list[str] = []
    shifters: list[str] = []
    vowels: list[str] = []
    signs: list[str] = []
    for m in marks:
        if m.startswith(COENG):
            sub = m[1]
            if unify_da and sub == "ដ":  # coeng DA 与 coeng TA 渲染相同
                sub = "ត"
            (coeng_ro if sub == RO else coengs).append(COENG + sub)
        elif m == ROBAT:
            robat.append(m)
        elif m in SHIFTERS:
            shifters.append(m)
        elif m in DEP_VOWELS:
            vowels.append(m)
        else:
            signs.append(m)

    def dedupe(xs):
        out = []
        for x in xs:
            if x not in out:
                out.append(x)
        return out

    vowels = dedupe(vowels)
    if len(vowels) >= 2:
        for pair, comp in _VOWEL_COMPOSE.items():
            if pair <= set(vowels):
                i = min(vowels.index(v) for v in pair)
                vowels = [v for v in vowels if v not in pair]
                vowels.insert(i, comp)
    return (
        base
        + "".join(robat[:1])
        + "".join(dedupe(coengs))
        + "".join(coeng_ro[:1])
        + "".join(shifters[:1])
        + "".join(vowels)
        + "".join(dedupe(signs))
    )


def normalize_khmer(text: str, unify_da: bool = True) -> str:
    """高棉字簇的规范顺序；非高棉文本原样通过。"""
    out, i, n = [], 0, len(text)
    while i < n:
        c = text[i]
        if c not in BASES:
            # 无基字符的 mark（或游离 coeng）——保留 marks，丢弃游离 coeng
            if c == COENG and not (i + 1 < n and text[i + 1] in BASES):
                i += 1
                continue
            out.append(c)
            i += 1
            continue
        i += 1
        marks = []
        while i < n:
            d = text[i]
            if d == COENG:
                if i + 1 < n and text[i + 1] in BASES:
                    marks.append(d + text[i + 1])
                    i += 2
                else:  # 游离 coeng——不可见，丢弃
                    i += 1
                continue
            if d in MARKS:
                marks.append(d)
                i += 1
                continue
            break
        out.append(_normalize_cluster(c, marks, unify_da) if marks else c)
    return "".join(out)


def normalize(text: str, unify_da: bool = True) -> str:
    """标签与预测共用的完整归一化。"""
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    # 全角 ASCII → 半角
    text = "".join(chr(ord(c) - 0xFEE0) if "！" <= c <= "～" else c for c in text)
    text = text.translate(_INVISIBLE).translate(_CHAR_TABLE)
    if any(_is_khmer(c) for c in text):
        text = normalize_khmer(text, unify_da=unify_da)
    return _WS.sub(" ", text).strip()
