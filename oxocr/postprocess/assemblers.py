"""四种排版组装器（模式本身也是插件，可扩展第 5 种）。

para 段落判据修正（参考实现 layout.py:440 的宽度递增规则会把自然变宽的行
全部拆成独立段落）：改用「行距 + 左边界对齐 + 末行短」组合——
仅当上一行明显短于本段最宽行（末行短）时才视为段落结束。
"""
from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from oxocr.postprocess.grouping import enclosing_box
from oxocr.types import TextBlock, TextLine

if TYPE_CHECKING:
    from oxocr.postprocess.base import PostContext

__all__ = ["RawAssembler", "LineAssembler", "ParaAssembler", "LayoutAssembler"]


def _is_mostly_english(text: str) -> bool:
    if not text.strip():
        return False
    ascii_chars = sum(1 for c in text if ord(c) < 128)
    return ascii_chars / max(len(text), 1) > 0.6


def _fill_line_text(lines: list[TextLine], ctx: PostContext, joiner) -> None:
    for line in lines:
        if not line.text:
            line.text = joiner.join(line.items, ctx)


def _line_block(line: TextLine) -> TextBlock:
    return TextBlock(
        kind="line", lines=[line], text=line.text, box=line.box, score=line.score
    )


class RawAssembler:
    name: ClassVar[str] = "raw"
    default_join: ClassVar[str] = "no_space"

    def assemble(self, lines: list[TextLine], ctx: PostContext, joiner) -> list[TextBlock]:
        items = [it for line in lines for it in line.items]
        items.sort(key=lambda it: (it.box.y_center, it.box.x_min))
        blocks = []
        for item in items:
            line = TextLine(items=[item], box=item.box, text=item.text, score=item.score)
            blocks.append(
                TextBlock(
                    kind="raw", lines=[line], text=item.text, box=item.box, score=item.score
                )
            )
        return blocks


class LineAssembler:
    name: ClassVar[str] = "line"
    default_join: ClassVar[str] = "smart_space"

    def assemble(self, lines: list[TextLine], ctx: PostContext, joiner) -> list[TextBlock]:
        _fill_line_text(lines, ctx, joiner)
        return [_line_block(line) for line in lines]


class LayoutAssembler:
    name: ClassVar[str] = "layout"
    default_join: ClassVar[str] = "layout_space"

    def assemble(self, lines: list[TextLine], ctx: PostContext, joiner) -> list[TextBlock]:
        _fill_line_text(lines, ctx, joiner)
        return [_line_block(line) for line in lines]


class ParaAssembler:
    name: ClassVar[str] = "para"
    default_join: ClassVar[str] = "smart_space"

    def assemble(self, lines: list[TextLine], ctx: PostContext, joiner) -> list[TextBlock]:
        if not lines:
            return []
        _fill_line_text(lines, ctx, joiner)

        paragraphs: list[list[TextLine]] = [[lines[0]]]
        para_max_width = lines[0].box.width
        for curr in lines[1:]:
            prev = paragraphs[-1][-1]
            line_gap = curr.box.y_min - prev.box.y_max
            avg_h = (prev.height + curr.height) / 2.0
            left_diff = abs(curr.box.x_min - prev.box.x_min)

            is_new = line_gap > avg_h * 0.5 or left_diff > avg_h * 1.5
            # 末行短：上一行明显短于本段最宽行 → 上一行是段落结尾
            if not is_new and prev.box.width < para_max_width * 0.9:
                is_new = True
            if not is_new and _is_mostly_english(prev.text):
                prev_stripped = prev.text.rstrip()
                curr_stripped = curr.text.lstrip()
                if prev_stripped and curr_stripped:
                    if prev_stripped[-1] in ".!?。！？" and curr_stripped[0].isupper():
                        is_new = True

            if is_new:
                paragraphs.append([curr])
                para_max_width = curr.box.width
            else:
                paragraphs[-1].append(curr)
                para_max_width = max(para_max_width, curr.box.width)

        blocks = []
        for para_lines in paragraphs:
            texts = [line.text for line in para_lines]
            separator = " " if _is_mostly_english("".join(texts)) else ""
            blocks.append(
                TextBlock(
                    kind="paragraph",
                    lines=para_lines,
                    text=separator.join(texts),
                    box=enclosing_box([line.box for line in para_lines]),
                    score=round(sum(line.score for line in para_lines) / len(para_lines), 5),
                    meta={"line_count": len(para_lines)},
                )
            )
        return blocks
