from __future__ import annotations

from oxocr.config import PostprocessConfig
from oxocr.postprocess.assemblers import (
    LayoutAssembler,
    LineAssembler,
    ParaAssembler,
    RawAssembler,
)
from oxocr.postprocess.base import PostContext
from oxocr.types import Box, TextItem, TextLine


class _Joiner:
    name = "test"

    def join(self, items, ctx):
        return " ".join(it.text for it in sorted(items, key=lambda i: i.box.x_min))


def _line(text, x0, y, width, height=14, x1=None):
    box = Box.from_rect(x0, y, (x1 - x0) if x1 is not None else width, height)
    item = TextItem(text=text, score=1.0, box=box)
    return TextLine(items=[item], box=box, text=text, score=1.0)


def _ctx():
    return PostContext(document=None, config=PostprocessConfig(), meta={})


def test_raw_mode_flattens_items_no_merge():
    l1 = _line("a", 0, 0, 20)
    l2 = _line("b", 0, 100, 20)
    blocks = RawAssembler().assemble([l1, l2], _ctx(), _Joiner())
    assert [b.kind for b in blocks] == ["raw", "raw"]
    assert [b.text for b in blocks] == ["a", "b"]


def test_line_mode_merges_same_row():
    box = Box.from_rect(0, 0, 60, 14)
    items = [_line("x", 0, 0, 20).items[0], _line("y", 40, 1, 20).items[0]]
    line = TextLine(items=items, box=box, text="", score=1.0)
    blocks = LineAssembler().assemble([line], _ctx(), _Joiner())
    assert blocks[0].kind == "line"
    assert blocks[0].text == "x y"


def test_para_keeps_increasing_widths_in_one_paragraph():
    # 方案移植修正 #8：行宽自然递增不应拆成多段
    lines = [
        _line("line one content here", 0, 0, 0, x1=200),
        _line("line two content here and more", 0, 20, 0, x1=240),
        _line("line three content here and even more text", 0, 40, 0, x1=280),
    ]
    blocks = ParaAssembler().assemble(lines, _ctx(), _Joiner())
    assert len(blocks) == 1
    assert blocks[0].meta["line_count"] == 3


def test_para_breaks_on_last_short_line():
    lines = [
        _line("first full width line", 0, 0, 0, x1=300),
        _line("short tail", 0, 20, 0, x1=90),
        _line("next paragraph full line", 0, 44, 0, x1=300),
    ]
    blocks = ParaAssembler().assemble(lines, _ctx(), _Joiner())
    assert len(blocks) == 2


def test_para_breaks_on_large_line_gap():
    lines = [
        _line("a b c", 0, 0, 0, x1=200),
        _line("d e f", 0, 120, 0, x1=200),
    ]
    blocks = ParaAssembler().assemble(lines, _ctx(), _Joiner())
    assert len(blocks) == 2


def test_layout_mode_uses_geometry():
    box = Box.from_rect(0, 0, 100, 14)
    items = [_line("a", 0, 0, 20).items[0], _line("b", 80, 0, 20).items[0]]
    line = TextLine(items=items, box=box, text="", score=1.0)
    blocks = LayoutAssembler().assemble([line], _ctx(), _Joiner())
    assert blocks[0].kind == "line"
