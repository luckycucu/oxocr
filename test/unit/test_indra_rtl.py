"""Indra rtl.flip 单测（无权重）：自逆、基行方向、镜像括号、组合 mark。"""
from __future__ import annotations

from oxocr.models.indra_ocr.rtl import clusters, flip, has_rtl, is_rtl, rtl_base


def test_is_rtl_has_rtl_rtl_base():
    assert is_rtl("א") and is_rtl("م")
    assert not is_rtl("a") and not is_rtl("1")
    assert has_rtl("abc אבג") and not has_rtl("abc 123")
    assert rtl_base("אבגד abc")  # 4 RTL > 3 LTR
    assert not rtl_base("abc אבג")  # L=3 R=3，不 >
    assert rtl_base("123 אבג")  # 数字（EN）不计入 LTR 字母


def test_flip_pure_rtl_reverses():
    assert flip("שלום עולם") == "םלוע םולש"
    assert flip("مرحبا") == "ابحرم"


def test_flip_self_inverse():
    for text in (
        "שלום עולם",
        "abc 123 אבג",
        "אבגד abc 123",
        "(אבג) [שלום]",
        "בְּרֵאשִׁית בָּרָא",
        "mixed עברית and English 42",
    ):
        assert flip(flip(text)) == text, text


def test_flip_ltr_base_reverses_rtl_runs_in_place():
    assert flip("abc אבג def") == "abc גבא def"
    assert flip("hello עולם world") == "hello םלוע world"


def test_flip_rtl_base_keeps_ltr_runs_whole():
    # RTL 基行：LTR run "abc" 整体保序，单元序反转
    assert flip("אבגד abc") == "abc דגבא"
    # 数字是 LTR 单元
    assert flip("אבגד 123") == "123 דגבא"


def test_flip_mirrors_brackets():
    assert flip("(אבג)") == "(גבא)"
    assert flip("[שלום]") == "[םולש]"
    assert flip("(אבג)", mirror=False) == ")גבא("


def test_clusters_marks_stay_with_base():
    # 希伯来 niqqud / 阿拉伯 harakat 随基字成簇
    cls = clusters("בְּ")
    assert cls == ["בְּ"]
    text = "בְּרֵאשִׁית"
    flipped = flip(text)
    assert flipped == "".join(reversed(clusters(text)))
    assert flip(flipped) == text


def test_base_rtl_override():
    # 显式段落方向覆盖字母计数
    text = "abc אבג"  # 字母计数 LTR 基
    assert flip(text, base_rtl=True) == "גבא abc"
    assert flip(text, base_rtl=False) == "abc גבא"


def test_no_rtl_passthrough():
    assert flip("hello 123") == "hello 123"
    assert flip("") == ""
