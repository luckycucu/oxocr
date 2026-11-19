"""行内连接策略（JoinPolicy）。

smart_space：safe/risky 词表 + 字母边界 + 符号边界 + 几何间隙兜底（移植参考实现）。
layout_space：在 smart_space 判定需要空格后按几何间隙计算空格数量。
"""
from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from oxocr.types import TextItem

if TYPE_CHECKING:
    from oxocr.postprocess.base import PostContext

__all__ = [
    "SmartSpaceJoiner",
    "LayoutSpaceJoiner",
    "SingleSpaceJoiner",
    "NoSpaceJoiner",
    "needs_space_between",
]

_SAFE_WORDS = frozenset({
    'the', 'and', 'for', 'not', 'but', 'yet', 'nor',
    'with', 'from', 'this', 'that', 'they', 'them', 'then',
    'than', 'when', 'where', 'which', 'while', 'what', 'who',
    'has', 'had', 'was', 'were', 'are', 'been', 'being',
    'can', 'may', 'will', 'shall', 'would', 'could', 'should', 'must',
    'each', 'every', 'other', 'some', 'such', 'very', 'just', 'also',
    'now', 'here', 'there', 'how', 'why', 'let', 'get', 'got',
    'new', 'old', 'big', 'end', 'top', 'use', 'way', 'day',
    'def', 'class', 'import', 'return', 'self', 'true', 'false', 'none',
    'async', 'await', 'try', 'except', 'finally', 'raise', 'pass',
    'break', 'continue', 'elif', 'else', 'case',
    'var', 'const', 'func', 'struct', 'enum', 'type', 'interface',
    'module', 'method', 'actual', 'behavior', 'layout', 'mode',
    'core', 'understand', 'assemble', 'look', 'build', 'data',
    'int', 'str', 'list', 'dict', 'set', 'map', 'filter', 'lambda',
})

_RISKY_WORDS = frozenset({
    'a', 'an', 'in', 'on', 'at', 'to', 'of', 'by', 'up', 'or',
    'is', 'it', 'be', 'he', 'we', 'my', 'me', 'us', 'do', 'if', 'so',
    'no', 'as', 'am',
})

_GAP_SPACE_RATIO = 0.3
_SPECIAL_CHARS = set('/_-.,:;()[]{}@#$%^&*+=|\\~`\'"')

_CJK_RANGES = (
    (0x4E00, 0x9FFF),
    (0x3400, 0x4DBF),
    (0xF900, 0xFAFF),
)
_CJK_SPACE_COEFF = 0.8
_LATIN_SPACE_COEFF = 1.0


def contains_chinese(text: str) -> bool:
    return any(lo <= ord(ch) <= hi for ch in text for lo, hi in _CJK_RANGES)


def needs_space_between(prev_text: str, curr_text: str, gap: float, avg_char_w: float) -> bool:
    if not prev_text or not curr_text:
        return False
    last_char = prev_text[-1]
    first_char = curr_text[0]

    # 规则1: 字母-字母边界
    if last_char.isalpha() and first_char.isalpha():
        prev_lower = prev_text.lower()
        curr_lower = curr_text.lower()
        if prev_lower in _SAFE_WORDS and curr_lower in _SAFE_WORDS:
            return True
        if prev_lower in _RISKY_WORDS or curr_lower in _RISKY_WORDS:
            if gap >= avg_char_w * _GAP_SPACE_RATIO:
                return True
            combined = prev_text + curr_text
            if combined.isalpha() and len(prev_text) <= 5 and len(curr_text) <= 3:
                return False
            return True
        if gap < avg_char_w * _GAP_SPACE_RATIO:
            combined = prev_text + curr_text
            if combined.isalpha() and len(prev_text) <= 5 and len(curr_text) <= 3:
                return False
            return True
        return True

    # 规则2: 特殊符号边界
    if last_char in _SPECIAL_CHARS or first_char in _SPECIAL_CHARS:
        if last_char in ')]}' and first_char.isalpha():
            return True
        return False

    # 规则3: 几何距离兜底
    return gap > avg_char_w * 0.5


def _sorted_items(items: list[TextItem]) -> list[TextItem]:
    return sorted(items, key=lambda it: it.box.x_min)


def _pairs(items: list[TextItem]):
    for i in range(1, len(items)):
        prev, curr = items[i - 1], items[i]
        gap = curr.box.x_min - prev.box.x_max
        avg_char_w = (
            prev.box.width / max(len(prev.text), 1) + curr.box.width / max(len(curr.text), 1)
        ) / 2.0
        yield prev, curr, gap, avg_char_w


class SmartSpaceJoiner:
    name: ClassVar[str] = "smart_space"

    def join(self, items: list[TextItem], ctx: PostContext) -> str:
        if not items:
            return ""
        items = _sorted_items(items)
        parts = [items[0].text]
        for prev, curr, gap, avg_char_w in _pairs(items):
            if needs_space_between(prev.text, curr.text, gap, avg_char_w):
                parts.append(" " + curr.text)
            else:
                parts.append(curr.text)
        return "".join(parts)


class LayoutSpaceJoiner:
    name: ClassVar[str] = "layout_space"

    def join(self, items: list[TextItem], ctx: PostContext) -> str:
        if not items:
            return ""
        items = _sorted_items(items)
        parts = [items[0].text]
        for prev, curr, gap, avg_char_w in _pairs(items):
            if needs_space_between(prev.text, curr.text, gap, avg_char_w):
                coeff = (
                    _CJK_SPACE_COEFF
                    if (contains_chinese(prev.text) or contains_chinese(curr.text))
                    else _LATIN_SPACE_COEFF
                )
                num_spaces = max(1, int(gap / max(avg_char_w * coeff, 1e-6)))
                parts.append(" " * num_spaces + curr.text)
            else:
                parts.append(curr.text)
        return "".join(parts)


class SingleSpaceJoiner:
    name: ClassVar[str] = "single_space"

    def join(self, items: list[TextItem], ctx: PostContext) -> str:
        return " ".join(it.text for it in _sorted_items(items))


class NoSpaceJoiner:
    name: ClassVar[str] = "no_space"

    def join(self, items: list[TextItem], ctx: PostContext) -> str:
        return "".join(it.text for it in _sorted_items(items))
