"""anaksor 识别器纯逻辑单测：CTC 解码、字符位置、宽高比桶、批缩减、metadata 校验。

不依赖权重：FakeSession 模拟 ONNX 输出，monkeypatch 模拟 metadata。
"""

from __future__ import annotations

import numpy as np
import pytest
from oxocr.errors import InferenceError
from oxocr.models.anaksor_ocr.recognizer import (
    SvtrRecognizer,
    _batch_for,
    ctc_greedy,
    ctc_positions,
    read_dict,
)

CLASSES = ["blank", "a", "b", "c"]


def _probs(rows: list[list[int]], probs: list[list[float]]) -> np.ndarray:
    """rows: 每帧类别 id；probs: 每帧最高概率。其余类别填小值。"""
    b, t = len(rows), len(rows[0])
    out = np.full((b, t, len(CLASSES)), 0.01, dtype=np.float32)
    for i, (row, p) in enumerate(zip(rows, probs, strict=True)):
        for j, (idx, prob) in enumerate(zip(row, p, strict=True)):
            out[i, j, idx] = prob
    return out


# ── ctc_greedy ───────────────────────────────────────────────────────────────


def test_ctc_greedy_collapse_and_blank():
    probs = _probs([[0, 1, 1, 0, 2, 0, 3, 3]], [[0.9, 0.8, 0.7, 0.9, 0.6, 0.9, 0.5, 0.4]])
    ((text, score),) = ctc_greedy(probs, CLASSES)
    assert text == "abc"
    assert score == pytest.approx((0.8 + 0.6 + 0.5) / 3)  # 保留帧算术平均


def test_ctc_greedy_all_blank():
    probs = _probs([[0, 0, 0]], [[0.9, 0.9, 0.9]])
    ((text, score),) = ctc_greedy(probs, CLASSES)
    assert text == "" and score == 0.0


def test_ctc_greedy_batch():
    probs = _probs([[1, 0, 2], [0, 3, 3]], [[0.8, 0.9, 0.7], [0.9, 0.6, 0.5]])
    out = ctc_greedy(probs, CLASSES)
    assert out[0][0] == "ab"
    assert out[1][0] == "c"


# ── ctc_positions ────────────────────────────────────────────────────────────


def test_ctc_positions_center_and_max():
    # a 占帧 1..2（中心 (1+2+1)/2/4 = 0.5），b 占帧 3（中心 0.875）
    probs = _probs([[0, 1, 1, 2]], [[0.9, 0.8, 0.6, 0.7]])
    (seq,) = ctc_positions(probs, CLASSES)
    assert [c for c, _, _ in seq] == ["a", "b"]
    assert seq[0][1] == pytest.approx(0.5)
    assert seq[0][2] == pytest.approx(0.8)  # 帧段最高概率
    assert seq[1][1] == pytest.approx(0.875)


def test_ctc_positions_empty():
    probs = _probs([[0, 0]], [[0.9, 0.9]])
    (seq,) = ctc_positions(probs, CLASSES)
    assert seq == []


# ── shape_for / _batch_for ───────────────────────────────────────────────────


@pytest.mark.parametrize(
    "w,h,expected",
    [
        (64, 64, (64, 64)),
        (96, 48, (96, 48)),
        (144, 48, (144, 48)),
        (192, 48, (192, 48)),
        (240, 48, (240, 48)),  # ratio 5
        (960, 48, (960, 48)),  # ratio 20
        (2000, 48, (960, 48)),  # ratio > 20 截断
        (10, 100, (64, 64)),  # ratio < 1 截断为 1
    ],
)
def test_shape_for(w, h, expected):
    assert SvtrRecognizer.shape_for(w, h) == expected


def test_batch_for_full_up_to_ratio4():
    assert _batch_for(1, 32) == 32
    assert _batch_for(4, 32) == 32


def test_batch_for_shrinks_with_ratio_squared():
    assert _batch_for(20, 32) == max(1, int(32 * 16 / 400))
    assert _batch_for(8, 32) == int(32 * 16 / 64)


# ── read_dict ────────────────────────────────────────────────────────────────


def test_read_dict_appends_space(tmp_path):
    p = tmp_path / "dict.txt"
    p.write_text("ក\nខ\na\n", encoding="utf-8")
    chars = read_dict(p)
    assert chars == ["ក", "ខ", "a", " "]


# ── 分组批（FakeSession）─────────────────────────────────────────────────────


class FakeSession:
    def __init__(self):
        self.calls: list[tuple[int, int]] = []

    def __call__(self, input_data: np.ndarray) -> np.ndarray:
        b, _, h, w = input_data.shape
        self.calls.append((b, w))
        t = (w - 1) // 4 + 1
        out = np.full((b, t, len(CLASSES)), 0.01, dtype=np.float32)
        out[:, :, 0] = 0.2
        for idx, frame in ((1, 0), (2, t // 2), (3, t - 1)):
            out[:, frame, idx] = 0.8
        return out


def _make_recognizer(batch_num: int = 32) -> SvtrRecognizer:
    rec = SvtrRecognizer.__new__(SvtrRecognizer)
    rec.batch_num = batch_num
    rec.classes = CLASSES
    rec.session = FakeSession()
    return rec


def test_recognizer_groups_by_shape():
    rec = _make_recognizer()
    crops = [
        np.zeros((48, 96, 3), dtype=np.uint8),  # (96,48)
        np.zeros((64, 64, 3), dtype=np.uint8),  # (64,64)
        np.zeros((48, 200, 3), dtype=np.uint8),  # (192,48)
        np.zeros((48, 96, 3), dtype=np.uint8),  # (96,48) 同桶
    ]
    results = rec(crops)
    assert len(rec.session.calls) == 3  # 三个形状桶
    assert all(r["text"] == "abc" for r in results)
    assert all(r["score"] == pytest.approx(0.8) for r in results)
    assert all(len(r["chars"]) == 3 for r in results)


def test_recognizer_batch_shrinks_for_wide():
    rec = _make_recognizer(batch_num=32)
    crops = [np.zeros((48, 960, 3), dtype=np.uint8) for _ in range(5)]  # ratio 20 → 批 1
    rec(crops)
    assert len(rec.session.calls) == 5
    assert all(b == 1 for b, _ in rec.session.calls)


def test_recognizer_empty():
    assert _make_recognizer()([]) == []


# ── metadata 校验（monkeypatch OrtSession）──────────────────────────────────


class _FakeOrt:
    def __init__(self, meta):
        self._meta = meta

    def metadata(self):
        return self._meta


@pytest.fixture
def dict_file(tmp_path):
    p = tmp_path / "dict.txt"
    p.write_text("ក\nខ\na\n", encoding="utf-8")  # 3 字符 + 空格 + blank = 5 类
    return p


def _patch_ort(monkeypatch, meta):
    monkeypatch.setattr(
        "oxocr.models.anaksor_ocr.recognizer.OrtSession", lambda *a, **k: _FakeOrt(meta)
    )


def test_meta_kind_mismatch(monkeypatch, dict_file):
    _patch_ort(monkeypatch, {"anaksor_kind": "det"})
    with pytest.raises(InferenceError, match="det"):
        SvtrRecognizer("m.onnx", dict_file)


def test_meta_head_mismatch(monkeypatch, dict_file):
    _patch_ort(monkeypatch, {"anaksor_kind": "rec", "anaksor_head": "smtr"})
    with pytest.raises(InferenceError, match="smtr"):
        SvtrRecognizer("m.onnx", dict_file)


def test_meta_classes_mismatch(monkeypatch, dict_file):
    _patch_ort(
        monkeypatch, {"anaksor_kind": "rec", "anaksor_head": "ctc", "anaksor_classes": "999"}
    )
    with pytest.raises(InferenceError, match="999"):
        SvtrRecognizer("m.onnx", dict_file)


def test_meta_ok(monkeypatch, dict_file):
    _patch_ort(monkeypatch, {"anaksor_kind": "rec", "anaksor_head": "ctc", "anaksor_classes": "5"})
    rec = SvtrRecognizer("m.onnx", dict_file)
    assert len(rec.classes) == 5
    assert rec.classes[0] == "blank"
