"""MfrTokenizer 解码契约（构造最小词表，不依赖权重）。"""
from __future__ import annotations

import json

import pytest
from oxocr.models.pix2text.tokenizer import MfrTokenizer, byte_to_unicode_inverse

_MINIMAL_VOCAB = {
    "<pad>": 0,
    "<s>": 1,
    "</s>": 2,
    "<unk>": 3,
    "<mask>": 4,
    "a": 5,
    "b": 6,
    "frac": 7,
    "Ġx": 8,  # Ġ = 字节 32（空格）的字母表字符
    "Î»": 9,  # 0xCE 0xBB → UTF-8 的 λ
}

_MINIMAL_ADDED = [
    {"id": i, "content": c, "special": True}
    for i, c in enumerate(("<pad>", "<s>", "</s>", "<unk>", "<mask>"))
]


@pytest.fixture
def tokenizer(tmp_path) -> MfrTokenizer:
    path = tmp_path / "tokenizer.json"
    path.write_text(
        json.dumps({"model": {"vocab": _MINIMAL_VOCAB}, "added_tokens": _MINIMAL_ADDED}),
        encoding="utf-8",
    )
    return MfrTokenizer(path)


def test_basic_piece_concat(tokenizer):
    assert tokenizer.decode([5, 6, 7]) == "abfrac"


def test_special_tokens_dropped(tokenizer):
    # 2=</s>、4=<mask> 都是 special，夹在中间也不出现在输出
    assert tokenizer.decode([5, 2, 6, 4, 7]) == "abfrac"


def test_byte_level_piece_maps_to_space(tokenizer):
    # Ġ → 字节 32："Ġx" 词片解码为 " x"，拼接后经字节逆映射得到空格
    assert tokenizer.decode([5, 8, 6]) == "a xb"


def test_multibyte_utf8(tokenizer):
    assert tokenizer.decode([9]) == "λ"


def test_out_of_range_id_dropped(tokenizer):
    assert tokenizer.decode([5, 99, 6]) == "ab"


def test_decode_strips_whitespace(tokenizer):
    assert tokenizer.decode([8, 5]) == "xa"  # " x" + "a" → strip → "xa"


def test_byte_inverse_table():
    inv = byte_to_unicode_inverse()
    # 可打印 ASCII 直出；空格与控制字节映射到 256+n 段
    assert inv["!"] == 33
    assert inv["~"] == 126
    assert inv["Ġ"] == 32
    assert inv[chr(288)] == 32
    # 高位字节（0xA1-0xFF）直出
    assert inv[chr(206)] == 206
    # 字节集合恰好覆盖 0-255
    assert sorted(inv.values()) == list(range(256))
