"""Indra NgramLM / beam_search 单测（合成 npz，无权重）。"""
from __future__ import annotations

import math

import numpy as np
import pytest
from oxocr.models.indra_ocr.lm import NgramLM, _pack, beam_search


@pytest.fixture
def lm(tmp_path):
    """order=2, vocab_size=3, bits=11 的小模型。

    unigram: token0 → -0.7, token1 → -1.2；bigram: (0,)→1 → -0.2；
    回退权重: 空历史 0.3，历史 (0,) 0.1。
    """
    path = tmp_path / "lm.npz"
    np.savez(
        path,
        order=np.array(2),
        vocab_size=np.array(3),
        k0=np.array([1, 2], np.uint64),
        v0=np.array([-0.7, -1.2]),
        k1=np.array([(1 << 11) | 2], np.uint64),
        v1=np.array([-0.2]),
        hk0=np.array([0], np.uint64),
        hb0=np.array([0.3]),
        hk1=np.array([1], np.uint64),
        hb1=np.array([0.1]),
    )
    return NgramLM(path)


def test_logp_hits_and_backoff(lm):
    assert lm.logp((), 0) == pytest.approx(-0.7)
    assert lm.logp((), 1) == pytest.approx(-1.2)
    # unigram 未见的 token2：空历史回退 + 均匀底
    assert lm.logp((), 2) == pytest.approx(0.3 - math.log(3))
    # bigram 命中
    assert lm.logp((0,), 1) == pytest.approx(-0.2)
    # bigram 未见 token0：回退 0.1 + unigram -0.7
    assert lm.logp((0,), 0) == pytest.approx(0.1 - 0.7)
    # 历史 (1,) 模型未见：m=0，直接 unigram
    assert lm.logp((1,), 1) == pytest.approx(-1.2)
    # 长历史截断到 order-1
    assert lm.logp((2, 0), 1) == pytest.approx(-0.2)


def test_logp_keys_matches_logp(lm):
    pairs = [
        ((), 0), ((), 1), ((), 2),
        ((0,), 0), ((0,), 1), ((0,), 2),
        ((1,), 0), ((1,), 1),
        ((2, 0), 1), ((0, 1), 2),
    ]
    keys = [_pack(h + (t,), lm.bits) for h, t in pairs]
    got = lm.logp_keys(keys)
    for (h, t), v in zip(pairs, got, strict=True):
        assert v == pytest.approx(lm.logp(h, t)), (h, t)
    # 缓存命中路径
    assert lm.logp_keys(keys) == got


def test_max_logp_bound(lm):
    bound = lm.max_logp
    for h in ((), (0,), (1,), (2,)):
        for t in range(3):
            assert lm.logp(h, t) <= bound + 1e-9


def test_beam_search_collapses_repeat(lm):
    # 两步都强 favor token1、无 blank → CTC 折叠为单个 token
    topi = np.array([[1, 2], [1, 2]], np.int64)
    topv = np.array([[-0.05, -4.0], [-0.05, -4.0]])
    blank = np.array([-8.0, -8.0])
    ids, steps = beam_search(blank, topv, topi, lm, alpha=0.5, beta=1.5, beam=8)
    assert ids == [1]
    assert steps == [0]


def test_beam_search_blank_splits_tokens(lm):
    # 三步 1-blank-1：blank 分隔使 token1 发两次（beta 加大奖励多 token；
    # token2 概率低于 cand_floor 不参与）
    topi = np.array([[1, 2], [1, 2], [1, 2]], np.int64)
    topv = np.array([[-0.05, -9.0]] * 3)
    blank = np.array([-8.0, -0.5, -8.0])
    ids, steps = beam_search(blank, topv, topi, lm, alpha=0.5, beta=5.0, beam=8)
    assert ids == [1, 1]
    assert steps == [0, 2]


def test_beam_search_blank_skip(lm):
    # blank 概率 > 0.999 的步直接跳过
    topi = np.array([[1, 2], [1, 2]], np.int64)
    topv = np.array([[-0.05, -4.0], [-0.05, -4.0]])
    blank = np.array([-8.0, -0.0005])
    ids, _ = beam_search(blank, topv, topi, lm, alpha=0.5, beta=1.5, beam=8)
    assert ids == [1]


def test_beam_search_cand_floor(lm):
    # 低于 cand_floor 的类不尝试：token2 概率 e^-9 远低于 3e-3
    topi = np.array([[2, 1], [2, 1]], np.int64)
    topv = np.array([[-9.0, -0.05], [-9.0, -0.05]])
    blank = np.array([-8.0, -8.0])
    ids, _ = beam_search(blank, topv, topi, lm, alpha=0.5, beta=1.5, beam=8, cand_floor=3e-3)
    assert ids == [1]
