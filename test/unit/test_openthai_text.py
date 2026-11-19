"""泰语归一化 / 形近修正 / 拼写合法率 golden（无权重）。"""

from __future__ import annotations

import pytest
from oxocr.models.openthai_ocr.text import canonical, fix_scripts, invalid_share, is_thai


@pytest.mark.parametrize(
    ("src", "dst"),
    [
        ("สวัสดี", "สวัสดี"),  # 合法文本原样
        ("Hello  World", "Hello World"),  # 空白折叠
        ("  leading and trailing  ", "leading and trailing"),
        ("a\t\tb", "a b"),
        ("line1\r\nline2\rline3", "line1\nline2\nline3"),
        ("๑๒๓๔๕", "๑๒๓๔๕"),  # 泰数字保持泰数字
        ("ำ", "ำ"),  # NFC 不拆 sara am（禁 NFKC）
        ("ນ້ຳ", "ນ້ຳ"),  # 老挝文不触发泰文规则
    ],
)
def test_canonical_basic(src, dst):
    assert canonical(src) == dst


def test_canonical_pua_legacy_glyphs():
    assert canonical("\uF700") == "\u0E10"  # tail-less ฐ
    assert canonical("\uF70F") == "\u0E0D"  # tail-less ญ
    assert canonical("ก\uF705") == "ก\u0E48"  # shifted tone
    assert canonical("ก\uF718") == "ก\u0E38"  # shifted vowel
    assert canonical("\uF71A") == "\u0E3A"  # shifted phinthu


def test_canonical_mark_order_and_dedup():
    # 声调在元音前 typed → 重排为 元音→声调
    assert canonical("ก่ิ") == "กิ่"
    # 重复 marks 折叠
    assert canonical("กิิ") == "กิ"
    assert canonical("ก่่") == "ก่"
    # nikhahit 排最后
    assert canonical("กํ่") == "ก่ํ"


@pytest.mark.parametrize("src", ["นํ้า", "น้ํา"])
def test_canonical_sara_am(src):
    assert canonical(src) == "น้ำ"


def test_canonical_doubles():
    assert canonical("เเ") == "แ"
    assert canonical("ำำ") == "ำ"


def test_canonical_invisibles():
    assert canonical("a​b") == "ab"  # ZWSP
    assert canonical("a­b") == "ab"  # soft hyphen
    assert canonical("﻿ab") == "ab"  # BOM
    assert canonical("a b") == "a b"  # NBSP → space
    assert canonical("") == ""
    assert canonical(None) == ""


@pytest.mark.parametrize(
    ("src", "dst"),
    [
        ("W.ศ.", "พ.ศ."),  # 泰文旁孤立 W → พ
        ("wก", "พก"),
        ("abcWก", "abcWก"),  # 前面是拉丁字母：不修
        ("WK", "WK"),  # 后面不是泰文：不修
        ("๗o.๘๒", "๗๐.๘๒"),  # 泰数字中的 o → ๐
        ("O๑", "๐๑"),
        ("๙0", "๙๐"),
        ("10O", "10O"),  # 纯拉丁数字：不修
        ("Hello", "Hello"),
    ],
)
def test_fix_scripts(src, dst):
    assert fix_scripts(src) == dst


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("สวัสดี", 0.0),
        ("น้ำ", 0.0),
        ("กิ่", 0.0),
        ("ประเทศไทย 2026", 0.0),
        ("abc", 0.0),  # 非泰文
        ("", 0.0),
    ],
)
def test_invalid_share_valid(text, expected):
    assert invalid_share(text) == pytest.approx(expected)


def test_invalid_share_broken():
    assert invalid_share("่") == 1.0  # 声调无辅音底座
    assert invalid_share("เ") == 1.0  # 前置元音无后接辅音
    assert invalid_share("ูี้ซ์") > 0.5  # 印章噪声样张
    assert invalid_share("ก่") == 0.0  # 声调有底座
    assert invalid_share("กิ่") == 0.0  # 元音+声调都有底座


def test_is_thai():
    assert is_thai("ก") and is_thai("๙")
    assert not is_thai("a") and not is_thai("中")