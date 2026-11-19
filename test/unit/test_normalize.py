from __future__ import annotations

from oxocr.postprocess.normalize import normalize_text


def test_collapse_spaces():
    assert normalize_text("a    b   c") == "a b c"


def test_tabs_and_unicode_space():
    assert normalize_text("a\tb c　d") == "a b c d"


def test_strip_edges():
    assert normalize_text("  hi  ") == "hi"


def test_fullwidth_to_halfwidth():
    assert normalize_text("ＡＢＣ１２３", fullwidth_to_halfwidth=True) == "ABC123"


def test_fullwidth_kept_when_disabled():
    assert normalize_text("ＡＢＣ", fullwidth_to_halfwidth=False) == "ＡＢＣ"


def test_collapse_disabled_preserves_layout():
    assert normalize_text("a    b", collapse_spaces=False) == "a    b"
