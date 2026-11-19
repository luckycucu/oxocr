from __future__ import annotations

from oxocr.postprocess.grouping import group_lines
from oxocr.types import Box, TextItem


def _item(text, x0, y0, w=30, h=12):
    return TextItem(text=text, score=1.0, box=Box.from_rect(x0, y0, w, h))


def test_same_row_grouped_left_to_right():
    items = [_item("b", 40, 0), _item("a", 0, 2)]
    lines = group_lines(items)
    assert len(lines) == 1
    assert [it.text for it in lines[0].items] == ["a", "b"]
    assert lines[0].box.x_min == 0 and lines[0].box.x_max == 70


def test_two_rows_separated():
    items = [_item("top", 0, 0), _item("bottom", 0, 100)]
    lines = group_lines(items)
    assert len(lines) == 2


def test_line_score_is_mean():
    a = _item("a", 0, 0)
    a.score = 0.8
    b = _item("b", 40, 0)
    b.score = 1.0
    lines = group_lines([a, b])
    assert lines[0].score == round((0.8 + 1.0) / 2, 5)


def test_empty_input():
    assert group_lines([]) == []
