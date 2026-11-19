"""泰语文本归一化与拼写校验（移植 OpenThai2.0 runtime/thai_norm.py + line_engine 脚本修正）。

canonical 是识别输出契约的一部分：NFC（禁 NFKC，会拆 sara am）、PUA legacy 字形映射、
marks 重排去重、sara am 组合、空白折叠。fix_scripts 修正泰文旁形近拉丁字符；
invalid_share 度量泰文拼写合法率（印章/签名噪声过滤用）。
"""

from __future__ import annotations

import re
import unicodedata

__all__ = ["canonical", "fix_scripts", "invalid_share", "is_thai"]

# Legacy Thai glyph variants (shifted/lowered marks, tail-less ฐ ญ) -> standard code points.
_PUA = {
    0xF700: 0x0E10, 0xF701: 0x0E34, 0xF702: 0x0E35, 0xF703: 0x0E36, 0xF704: 0x0E37,
    0xF705: 0x0E48, 0xF706: 0x0E49, 0xF707: 0x0E4A, 0xF708: 0x0E4B, 0xF709: 0x0E4C,
    0xF70A: 0x0E48, 0xF70B: 0x0E49, 0xF70C: 0x0E4A, 0xF70D: 0x0E4B, 0xF70E: 0x0E4C,
    0xF70F: 0x0E0D, 0xF710: 0x0E31, 0xF711: 0x0E4D, 0xF712: 0x0E47, 0xF713: 0x0E48,
    0xF714: 0x0E49, 0xF715: 0x0E4A, 0xF716: 0x0E4B, 0xF717: 0x0E4C, 0xF718: 0x0E38,
    0xF719: 0x0E39, 0xF71A: 0x0E3A,
}
_DROP = dict.fromkeys(map(ord, "​‌‍⁠­﻿"))
_TRANSLATE = {**_PUA, **_DROP, 0x00A0: 0x20}

_VOWEL_MARKS = set("ัิีึืฺุู็")
_TONE_MARKS = set("่้๊๋์๎")
_NIKHAHIT = "ํ"
_MARKS = _VOWEL_MARKS | _TONE_MARKS | {_NIKHAHIT}
_MARK_RUN = re.compile("[" + "".join(sorted(_MARKS)) + "]{2,}")
_SARA_AM = re.compile("([่-๋]?)ํ([่-๋]?)า")
_SPACES = re.compile(r"[ \t\f\v]+")


def _order(m: re.Match) -> str:
    run = m.group(0)
    out: list[str] = []
    for ch in sorted(run, key=lambda c: 0 if c in _VOWEL_MARKS else 1 if c in _TONE_MARKS else 2):
        if not out or out[-1] != ch:  # a doubled mark renders as one; keep one
            out.append(ch)
    return "".join(out)


def canonical(text: str) -> str:
    text = unicodedata.normalize("NFC", (text or "").translate(_TRANSLATE))
    text = _MARK_RUN.sub(_order, text)
    text = _SARA_AM.sub(lambda m: m.group(1) + m.group(2) + "ำ", text)
    text = text.replace("เเ", "แ").replace("ำำ", "ำ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(_SPACES.sub(" ", line).strip() for line in text.split("\n")).strip()


# Script fix-ups for look-alikes a line recognizer cannot tell apart without
# context: a lone Latin W glued to Thai letters (Prompt/Kanit Bold พ is a W
# with a loop), and a Latin O or an Arabic 0 inside a Thai-digit number.
_LONE_W = re.compile(r"(?<![A-Za-z])[Ww](?=\.?[ก-ฮเ-ไ])")
_ZERO_IN_THAI = re.compile(r"(?<=[๐-๙])[Oo0]|[Oo0](?=[๐-๙])")


def fix_scripts(text: str) -> str:
    text = _LONE_W.sub("พ", text)
    return _ZERO_IN_THAI.sub("๐", text)


_MARKS_SPELL = set("ัิีึืฺุู็่้๊๋์๎ํ")
_LEADING = set("เแโใไ")


def is_thai(ch: str) -> bool:
    return "฀" <= ch <= "๿"


def _cons(ch: str) -> bool:
    return "ก" <= ch <= "ฮ"


def invalid_share(text: str) -> float:
    """Share of Thai characters that break Thai spelling: a mark with no consonant
    under it, or a leading vowel (เ แ โ ใ ไ) not followed by a consonant."""
    thai = bad = 0
    for i, ch in enumerate(text):
        if not is_thai(ch):
            continue
        thai += 1
        prev = text[i - 1] if i else " "
        if ch in _MARKS_SPELL and not (_cons(prev) or (prev in _MARKS_SPELL and i >= 2 and _cons(text[i - 2]))):
            bad += 1
        elif ch in _LEADING and not (i + 1 < len(text) and _cons(text[i + 1])):
            bad += 1
    return bad / thai if thai else 0.0