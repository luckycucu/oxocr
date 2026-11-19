"""高棉语 normalize 的 golden 单测（纯文本，无权重、无 onnxruntime）。

覆盖：字簇 marks 重排、分裂元音组合、coeng-DA→TA、coeng-Ro 最后、
不可见字符去除、全角转半角、字符映射、空白折叠、非高棉原样。
"""

from __future__ import annotations

import pytest
from oxocr.models.anaksor_ocr.text import COENG, normalize, normalize_khmer

# ── 字簇 marks 重排 ──────────────────────────────────────────────────────────


def test_cluster_coeng_before_vowel():
    # ក + ា + ្ម → coeng 在元音前
    assert normalize("កា្ម") == "ក្មា"


def test_cluster_robat_first():
    assert normalize("កា៌") == "ក៌ា"


def test_cluster_shifter_after_coeng():
    assert normalize("កា៉") == "ក៉ា"


def test_cluster_sign_after_vowel():
    # ំ (0x17C6) 是符号，排在元音后
    assert normalize("កំា") == "កាំ"


def test_cluster_full_order():
    # robat → coeng → shifter → 元音 → 符号
    assert normalize("ក" + "ា" + "ំ" + "៉" + "្ម" + "៌") == "ក៌្ម៉ាំ"


def test_coeng_ro_last():
    assert normalize("ក" + COENG + "រ" + COENG + "ម") == "ក្ម្រ"


def test_duplicate_vowel_deduped():
    assert normalize("កាា") == "កា"


# ── 分裂元音组合 ─────────────────────────────────────────────────────────────


def test_split_vowel_e_i():
    assert normalize("កេី") == "កើ"


def test_split_vowel_e_a():
    assert normalize("កេា") == "កោ"


def test_split_vowel_order_insensitive():
    assert normalize("កីេ") == "កើ"


# ── coeng-DA 统一 ────────────────────────────────────────────────────────────


def test_coeng_da_to_ta():
    assert normalize("ក្ដ") == "ក្ត"


def test_coeng_da_kept_when_unify_off():
    assert normalize("ក្ដ", unify_da=False) == "ក្ដ"


# ── 游离 coeng ───────────────────────────────────────────────────────────────


def test_stray_coeng_dropped():
    assert normalize("ក្") == "ក"


def test_stray_coeng_mid_dropped():
    # coeng 后接元音 mark（非基字符）→ 游离 coeng 丢弃
    assert normalize("ក" + COENG + "ា") == "កា"


def test_coeng_before_base_kept():
    assert normalize("ក" + COENG + "ខ") == "ក្ខ"


# ── 不可见字符 / 全角 / 映射 / 空白 ──────────────────────────────────────────


def test_zwsp_removed():
    assert normalize("ក​ខ") == "កខ"


def test_invisible_format_chars_removed():
    assert normalize("a‌b‍c⁠d﻿e") == "abcde"


def test_fullwidth_to_halfwidth():
    assert normalize("ＡＢＣ１２３") == "ABC123"


def test_char_map_quotes_and_ellipsis():
    assert normalize("‘hi’…") == "'hi'..."


def test_whitespace_collapse():
    assert normalize("a  b\tc\nd") == "a b c d"


def test_nfc_latin():
    assert normalize("é") == "é"


def test_non_khmer_passthrough():
    assert normalize("Hello World 123") == "Hello World 123"


def test_empty():
    assert normalize("") == ""


# ── normalize_khmer 单独 ─────────────────────────────────────────────────────


def test_normalize_khmer_no_whitespace_collapse():
    # normalize_khmer 只做字簇重排，不折叠空白
    assert normalize_khmer("កា  ខ") == "កា  ខ"


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("ភ្នំពេញ", "ភ្នំពេញ"),  # 已规范
        ("បន្ទាយស្រី", "បន្ទាយស្រី"),
    ],
)
def test_already_normalized_idempotent(raw, expected):
    assert normalize(raw) == expected
    assert normalize(normalize(raw)) == expected
