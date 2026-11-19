from __future__ import annotations

from oxocr.postprocess.joiners import (
    LayoutSpaceJoiner,
    NoSpaceJoiner,
    SingleSpaceJoiner,
    SmartSpaceJoiner,
    needs_space_between,
)
from oxocr.types import Box, TextItem


def _item(text, x0, x1):
    return TextItem(text=text, score=1.0, box=Box.from_rect(x0, 0, x1 - x0, 10))


def test_needs_space_two_safe_words_true():
    assert needs_space_between("with", "from", gap=0.0, avg_char_w=10) is True


def test_needs_space_risky_close_gap_false():
    # "a" risky，间距极小且组合是短词 → 不补空格（合并成一个词）
    assert needs_space_between("c", "at", gap=0.0, avg_char_w=10) is False


def test_needs_space_risky_large_gap_true():
    assert needs_space_between("a", "cat", gap=8.0, avg_char_w=10) is True


def test_needs_space_symbol_bracket_then_alpha():
    assert needs_space_between("x)", "foo", gap=0.0, avg_char_w=10) is True
    assert needs_space_between("foo", "(bar", gap=0.0, avg_char_w=10) is False


def test_needs_space_geometric_fallback():
    assert needs_space_between("中文", "文字", gap=8.0, avg_char_w=10) is True
    assert needs_space_between("中文", "文字", gap=1.0, avg_char_w=10) is False


def test_smart_joiner_inserts_space():
    j = SmartSpaceJoiner()
    items = [_item("the", 0, 30), _item("cat", 40, 70)]
    assert j.join(items, None) == "the cat"


def test_single_and_no_space_joiners():
    items = [_item("a", 0, 10), _item("b", 20, 30)]
    assert SingleSpaceJoiner().join(items, None) == "a b"
    assert NoSpaceJoiner().join(items, None) == "ab"


def test_layout_joiner_space_count_scales_with_gap():
    j = LayoutSpaceJoiner()
    small = [_item("the", 0, 30), _item("cat", 33, 60)]   # gap ~3
    big = [_item("the", 0, 30), _item("cat", 60, 90)]      # gap ~30
    n_small = j.join(small, None).count(" ")
    n_big = j.join(big, None).count(" ")
    assert n_big > n_small
