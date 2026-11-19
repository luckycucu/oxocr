"""读光识别器纯逻辑单测：CTC 解码、重叠合并、几何平均 score、组块语义推断。

不依赖权重与 onnxruntime：用 FakeSession 模拟模型输出。
"""
from __future__ import annotations

import numpy as np
import pytest
from oxocr.errors import InferenceError
from oxocr.models.duguang_ocr.recognizer import TextRecognizer, merge_strings_with_overlap


def _make_recognizer(mapping: dict[int, str] | None = None) -> TextRecognizer:
    rec = TextRecognizer.__new__(TextRecognizer)
    rec.batch_num = 6
    rec.label_mapping = mapping or {2: "a", 3: "b", 4: "c", 5: "d", 6: "e"}
    return rec


class FakeSession:
    def __init__(self, output: np.ndarray):
        self.output = output

    def __call__(self, input_data: np.ndarray) -> np.ndarray:
        return self.output


# ── merge_strings_with_overlap ──


def test_merge_no_overlap():
    assert merge_strings_with_overlap(["hello", "world"]) == "helloworld"


def test_merge_full_overlap():
    assert merge_strings_with_overlap(["abc", "abc"]) == "abc"


def test_merge_partial_overlap():
    assert merge_strings_with_overlap(["abcdef", "defghi"]) == "abcdefghi"


def test_merge_empty_list():
    assert merge_strings_with_overlap([]) == ""


def test_merge_single_string():
    assert merge_strings_with_overlap(["hello"]) == "hello"


def test_merge_three_chunks():
    assert merge_strings_with_overlap(["abcde", "cdefg", "efghi"]) == "abcdefghi"


def test_merge_empty_strings():
    assert merge_strings_with_overlap(["", "abc", ""]) == "abc"


# ── CTC decode ──


def test_ctc_decode_basic():
    rec = _make_recognizer()
    ids = np.array([0, 2, 2, 0, 3, 0, 4, 4, 4])
    probs = np.array([0.9, 0.8, 0.7, 0.9, 0.6, 0.9, 0.5, 0.4, 0.3])
    text, kept_probs = rec._ctc_decode(ids, probs)
    assert text == "abc"
    assert len(kept_probs) == 3
    assert kept_probs[0] == pytest.approx(0.8)
    assert kept_probs[1] == pytest.approx(0.6)
    assert kept_probs[2] == pytest.approx(0.5)


def test_ctc_decode_skip_blank():
    rec = _make_recognizer()
    ids = np.array([0, 0, 0])
    probs = np.array([0.9, 0.9, 0.9])
    text, kept_probs = rec._ctc_decode(ids, probs)
    assert text == ""
    assert kept_probs == []


def test_ctc_decode_skip_index_1():
    rec = _make_recognizer()
    ids = np.array([2, 1, 3, 1, 4])
    probs = np.array([0.8, 0.9, 0.7, 0.9, 0.6])
    text, kept_probs = rec._ctc_decode(ids, probs)
    assert text == "abc"
    assert len(kept_probs) == 3


def test_ctc_decode_collapse_repeats():
    rec = _make_recognizer()
    ids = np.array([2, 2, 2, 3, 3, 4])
    probs = np.array([0.8, 0.7, 0.6, 0.5, 0.4, 0.3])
    text, kept_probs = rec._ctc_decode(ids, probs)
    assert text == "abc"
    assert kept_probs == [0.8, 0.5, 0.3]


def test_ctc_decode_unknown_index_skipped():
    rec = _make_recognizer({2: "a", 3: "b"})
    ids = np.array([2, 99, 3])
    probs = np.array([0.8, 0.9, 0.7])
    text, kept_probs = rec._ctc_decode(ids, probs)
    assert text == "ab"
    assert len(kept_probs) == 2


# ── geometric mean score ──


def test_geometric_mean_score_direct():
    rec = _make_recognizer()
    n_crops = 2
    output = np.zeros((n_crops, 5, 7), dtype=np.float32)
    output[0, 0, 2] = 10.0
    output[0, 1, 3] = 10.0
    output[1, 0, 4] = 10.0
    rec.session = FakeSession(output)
    crops = [np.zeros((32, 100, 3), dtype=np.uint8)] * n_crops
    results = rec(crops)
    assert len(results) == 2
    assert results[0][0] == "ab"
    assert results[1][0] == "c"
    assert 0 < results[0][1] <= 1.0
    assert 0 < results[1][1] <= 1.0


def test_geometric_mean_score_overlap():
    rec = _make_recognizer()
    n_crops = 1
    input_rows = n_crops * 3
    output = np.zeros((input_rows, 5, 7), dtype=np.float32)
    output[0, 0, 2] = 10.0
    output[1, 0, 3] = 10.0
    output[2, 0, 4] = 10.0
    rec.session = FakeSession(output)
    crops = [np.zeros((32, 100, 3), dtype=np.uint8)]
    results = rec(crops)
    assert len(results) == 1
    assert results[0][0] == "abc"
    assert 0 < results[0][1] <= 1.0


# ── chunk semantics inference ──


def test_chunk_semantics_direct_decode():
    """output_rows * 3 == input_rows → direct decode (large/max)."""
    rec = _make_recognizer()
    output = np.zeros((2, 5, 7), dtype=np.float32)
    output[0, 0, 2] = 10.0
    output[1, 0, 3] = 10.0
    rec.session = FakeSession(output)
    crops = [np.zeros((32, 100, 3), dtype=np.uint8)] * 2
    results = rec(crops)
    assert results[0][0] == "a"
    assert results[1][0] == "b"


def test_chunk_semantics_overlap_merge():
    """output_rows == input_rows → overlap merge (small)."""
    rec = _make_recognizer()
    output = np.zeros((6, 5, 7), dtype=np.float32)
    output[0, 0, 2] = 10.0
    output[1, 0, 2] = 10.0
    output[2, 0, 3] = 10.0
    output[3, 0, 3] = 10.0
    output[4, 0, 4] = 10.0
    output[5, 0, 4] = 10.0
    rec.session = FakeSession(output)
    crops = [np.zeros((32, 100, 3), dtype=np.uint8)] * 2
    results = rec(crops)
    assert results[0][0] == "ab"
    assert results[1][0] == "bc"


def test_chunk_semantics_mismatch_raises():
    """output_rows not matching either pattern → InferenceError."""
    rec = _make_recognizer()
    output = np.zeros((5, 5, 7), dtype=np.float32)
    rec.session = FakeSession(output)
    crops = [np.zeros((32, 100, 3), dtype=np.uint8)] * 2
    with pytest.raises(InferenceError, match="不匹配"):
        rec(crops)


def test_empty_input_returns_empty():
    rec = _make_recognizer()
    assert rec([]) == []


def test_geometric_mean_not_product():
    """score 是几何平均而非连乘：长行不会被系统性压低。"""
    rec = _make_recognizer({i: chr(ord("a") + i - 2) for i in range(2, 12)})
    output = np.zeros((1, 20, 12), dtype=np.float32)
    for j in range(10):
        output[0, j, j + 2] = 10.0
    rec.session = FakeSession(output)
    crops = [np.zeros((32, 100, 3), dtype=np.uint8)]
    results = rec(crops)
    assert results[0][0] == "abcdefghij"
    score = results[0][1]
    assert score > 0.9
