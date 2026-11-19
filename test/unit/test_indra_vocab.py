"""Indra Vocab 解码单测（无权重）：jamo NFC、LaTeX 命令补空格、blank/越界过滤。"""
from __future__ import annotations

import pytest
from oxocr.models.indra_ocr.vocab import Vocab


def test_decode_plain_chars_and_blank_filter():
    v = Vocab(list("abc "))
    assert v.n_classes == 5
    assert v.decode([1, 2, 3]) == "abc"
    assert v.decode([0, 1, 0, 2, 99, -1, 3]) == "abc"  # blank/越界丢弃
    assert v.decode([]) == ""


def test_decode_jamo_composes_nfc():
    # ᄀ ᅡ ᆨ ᅳ ᆫ → 가 근（NFC 合成音节）
    chars = ["ᄀ", "ᅡ", "ᆨ", "ᅳ", "ᆫ", "A"]
    v = Vocab(chars)
    assert v.jamo
    assert v.decode([1, 2]) == "가"
    assert v.decode([1, 4, 5]) == "근"
    assert v.decode([1, 2, 6]) == "가A"
    # 无初声的 jamo 序列无法合成，保持原样（不丢字符）
    assert v.decode([2, 3]) == "ᅡᆨ"


def test_decode_latex_command_spacing():
    v = Vocab(["\\alpha", "\\frac", "x", "y", "{", "}", "1", "2"])
    assert v.has_commands
    # 命令后接字母补一个空格；命令后接非字母不补
    assert v.decode([1, 3]) == "\\alpha x"
    assert v.decode([1, 5]) == "\\alpha{"
    assert v.decode([2, 5, 7, 8, 6]) == "\\frac{12}"
    assert v.decode([3, 1, 4]) == "x\\alpha y"
    assert v.decode([1]) == "\\alpha"  # 末尾命令不补空格


def test_from_json_roundtrip_and_blank_guard():
    v = Vocab(list("ab"))
    import json

    s = json.dumps({"blank": 0, "chars": v.chars}, ensure_ascii=False)
    v2 = Vocab.from_json(s)
    assert v2.chars == v.chars
    with pytest.raises(ValueError):
        Vocab.from_json(json.dumps({"blank": 1, "chars": ["a"]}))


def test_real_meta_vocabs(indra_models_dir):
    """真实 meta.json 四词表：规模、jamo 标志、LaTeX 命令（仅权重在时）。"""
    import json

    meta = json.loads((indra_models_dir / "indra-ocr" / "meta.json").read_text(encoding="utf-8"))
    base = Vocab.from_json(meta["vocab"])
    cjk = Vocab.from_json(meta["vocab_cjk"])
    rtl = Vocab.from_json(meta["vocab_rtl"])
    world = Vocab.from_json(meta["vocab_world"])
    assert (len(base.chars), len(cjk.chars), len(rtl.chars), len(world.chars)) == (
        1697, 11749, 1355, 2935,
    )
    assert base.has_commands and not cjk.has_commands
    assert cjk.jamo and not base.jamo
    assert "\\frac" in base.index
    assert base.decode([base.index["\\alpha"], base.index["x"]]) == "\\alpha x"
