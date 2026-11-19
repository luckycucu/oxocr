from __future__ import annotations

from oxocr.postprocess.ghost import remove_ghost_tokens
from oxocr.types import Box, TextItem


def _item(text, x0, x1, y=0):
    return TextItem(text=text, score=1.0, box=Box.from_rect(x0, y, x1 - x0, 12))


def test_absorb_repeated_tail_letter_ghost():
    # "let" + "t me"（重叠）→ 吸收成 " me"，避免 "let t me"
    items = [_item("let", 0, 30), _item("t me", 18, 58)]
    out = remove_ghost_tokens(items)
    assert [it.text for it in out] == ["let", " me"]


def test_absorb_noise_symbol_ghost():
    # "look" + "< at"（'<' 是 'k' 的噪声），prev 以 k 结尾 → 剥离成 " at"
    items = [_item("look", 0, 40), _item("< at", 27, 68)]
    out = remove_ghost_tokens(items)
    assert [it.text for it in out] == ["look", " at"]


def test_no_absorption_without_overlap():
    items = [_item("let", 0, 30), _item("out", 60, 95)]
    out = remove_ghost_tokens(items)
    assert [it.text for it in out] == ["let", "out"]


def test_single_item_untouched():
    items = [_item("solo", 0, 20)]
    assert remove_ghost_tokens(items) == items
