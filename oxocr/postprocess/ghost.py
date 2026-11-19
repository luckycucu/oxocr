"""幽灵 token 过滤：修复 box 重叠导致的重复字符。

修正参考实现：限定在**同一行内**执行（分组后调用），避免跨行误吸收。
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, ClassVar

from oxocr.types import TextItem, TextLine

if TYPE_CHECKING:
    from oxocr.postprocess.base import PostContext

__all__ = ["GhostFilter", "remove_ghost_tokens"]

_GHOST_NOISE_PREFIXES = {"<": "k", "1": "l", "5": "s", "v": "w"}


def remove_ghost_tokens(items: list[TextItem], overlap_ratio: float = 1.1) -> list[TextItem]:
    if len(items) <= 1:
        return items
    ordered = sorted(items, key=lambda it: it.box.x_min)
    result: list[TextItem] = []
    for i, curr in enumerate(ordered):
        if i == 0:
            result.append(curr)
            continue
        prev = result[-1]
        prev_text, curr_text = prev.text, curr.text
        if not prev_text or not curr_text.strip():
            result.append(curr)
            continue

        prev_char_w = prev.box.width / max(len(prev_text), 1)
        curr_char_w = curr.box.width / max(len(curr_text), 1)
        min_char_w = min(prev_char_w, curr_char_w)
        actual_overlap = prev.box.x_max - curr.box.x_min
        if actual_overlap < min_char_w * overlap_ratio:
            result.append(curr)
            continue

        # 情况1: 粘连幽灵前缀（单字母 + 空格/结尾）
        first_letter = curr_text[0]
        trimmed = None
        if first_letter.isalpha() and (len(curr_text) == 1 or curr_text[1] == " "):
            if prev_text.endswith(first_letter):
                trimmed = curr_text[1:]
        # 情况2: 纯噪声符号剥离
        if trimmed is None and first_letter in _GHOST_NOISE_PREFIXES:
            if (
                prev_text.endswith(_GHOST_NOISE_PREFIXES[first_letter])
                and len(curr_text) > 1
                and curr_text[1] == " "
            ):
                trimmed = curr_text[1:]

        if trimmed is not None:
            result.append(
                TextItem(text=trimmed, score=curr.score, box=curr.box, angle=curr.angle,
                         chars=curr.chars, meta=dict(curr.meta))
            )
        else:
            result.append(curr)
    return result


class GhostFilter:
    name: ClassVar[str] = "ghost_filter"
    default_options: ClassVar[Mapping[str, Any]] = {"enabled": True, "overlap_ratio": 1.1}

    def on_lines(self, lines: list[TextLine], ctx: PostContext) -> list[TextLine]:
        opts = getattr(self, "options", self.default_options)
        ratio = float(opts.get("overlap_ratio", 1.1))
        for line in lines:
            line.items = remove_ghost_tokens(line.items, ratio)
        return lines
