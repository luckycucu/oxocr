"""OpenThai 泰语 CTC 行识别（ONNX 后端，输入 [N,3,48,W]）。

color 归一（默认 flatten：逐列背景减除，任意极性/渐变背景可读）→ 宽高比排序分批、
批内零填充（归一化后 0 = 中灰）→ CTC 贪心解码（score = 保留帧算术平均）。
输出不做泰语归一化：canonical/fix_scripts 由管线在每行最终文本上调用一次。
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from oxocr.errors import InferenceError
from oxocr.models.openthai_ocr.geometry import flatten, to_gray
from oxocr.runtime import imageops
from oxocr.runtime.session import OrtSession

__all__ = ["CtcRecognizer", "ctc_greedy", "read_dict", "prep_crop", "COLOR_MODES"]

_HEIGHT = 48  # 权重输入固定高度，不暴露为参数
_MAX_W = 3200  # 批宽上限（内存护栏），不暴露为参数
COLOR_MODES = ("bgr", "gray", "gray_inv", "stretch", "flatten")


def read_dict(path: str | Path) -> list[str]:
    """字典每行一个字符（UTF-8）；CTC 类表 = [blank] + chars + [space]。"""
    chars = [c for c in Path(path).read_text(encoding="utf-8").split("\n") if c]
    return ["", *chars, " "]


def ctc_greedy(probs: np.ndarray, classes: list[str]) -> list[tuple[str, float]]:
    """(B, T, C) 概率的 best-path CTC 解码：重复折叠、blank 去除。
    score = 保留帧概率的算术平均（空结果 0）。"""
    ids, best = probs.argmax(2), probs.max(2)
    out = []
    for row, p in zip(ids, best, strict=True):
        keep = row != 0
        keep[1:] &= row[1:] != row[:-1]
        text = "".join(classes[i] if i < len(classes) else "" for i in row[keep])
        out.append((text, float(np.mean(p[keep])) if keep.any() else 0.0))
    return out


def prep_crop(crop: np.ndarray, mode: str) -> np.ndarray:
    """识别前的颜色归一：训练只见过灰度行，彩色/暗底/渐变 crop 须先归一。"""
    if mode == "bgr" or crop.ndim != 3 or not crop.size:
        return crop
    g = to_gray(crop)
    if mode == "gray":
        return np.repeat(g[:, :, None], 3, axis=2)
    if mode == "flatten":
        return np.repeat(flatten(g)[:, :, None], 3, axis=2)
    # Levels from a 256-bin histogram: linear time.
    c = np.cumsum(np.bincount(g.ravel(), minlength=256))
    lo, med, hi = (int(np.searchsorted(c, q * c[-1])) for q in (0.02, 0.5, 0.98))
    lut = np.arange(256, dtype=np.float32)
    if mode == "stretch" and hi - lo >= 16:
        lut = np.clip((lut - lo) * (255.0 / (hi - lo)), 0, 255)
    if lut[med] < 128:  # the background dominates a line crop: dark background = light text
        lut = 255 - lut
    out = lut[g].astype(np.uint8)
    return np.repeat(out[:, :, None], 3, axis=2)


class CtcRecognizer:
    def __init__(
        self,
        model_path: str | Path,
        dict_path: str | Path,
        batch_num: int = 16,
        num_threads: int = 0,
        providers: Sequence[str] = ("CPUExecutionProvider",),
    ):
        self.session = OrtSession(model_path, num_threads=num_threads, providers=providers)
        self.batch_num = int(batch_num)
        self.classes = read_dict(dict_path)
        out_shape = self.session.session.get_outputs()[0].shape
        n_classes = out_shape[-1]
        if isinstance(n_classes, int) and n_classes != len(self.classes):
            raise InferenceError(
                f"{Path(model_path).name} 有 {n_classes} 类，"
                f"字典 {Path(dict_path).name} 给出 {len(self.classes)} 类；请使用匹配的字典"
            )

    def __call__(self, crops: list[np.ndarray], color: str = "flatten") -> list[tuple[str, float]]:
        if color not in COLOR_MODES:
            raise InferenceError(f"rec_color 必须是 {COLOR_MODES} 之一，得到 {color!r}")
        if not crops:
            return []
        if color == "flatten":  # cheaper at the recognizer's height
            crops = [
                prep_crop(
                    imageops.resize(
                        cr,
                        (max(1, int(math.ceil(_HEIGHT * cr.shape[1] / max(cr.shape[0], 1)))), _HEIGHT),
                    ),
                    color,
                )
                for cr in crops
            ]
        else:
            crops = [prep_crop(cr, color) for cr in crops]
        ratios = [cr.shape[1] / max(cr.shape[0], 1) for cr in crops]
        out: list[tuple[str, float]] = [("", 0.0)] * len(crops)
        order = sorted(range(len(crops)), key=lambda i: ratios[i])
        for b in range(0, len(order), self.batch_num):
            idx = order[b : b + self.batch_num]
            width = min(_MAX_W, max(16, int(math.ceil(_HEIGHT * max(ratios[i] for i in idx)))))
            batch = np.zeros((len(idx), 3, _HEIGHT, width), np.float32)  # 0 = mid-grey after normalising
            for n, i in enumerate(idx):
                rw = min(width, max(1, int(math.ceil(_HEIGHT * ratios[i]))))
                im = imageops.resize(crops[i], (rw, _HEIGHT)).astype(np.float32) / 255.0
                batch[n, :, :, :rw] = ((im - 0.5) / 0.5).transpose(2, 0, 1)
            probs = self.session(batch)
            for n, (text, score) in enumerate(ctc_greedy(probs, self.classes)):
                out[idx[n]] = (text, score)
        return out