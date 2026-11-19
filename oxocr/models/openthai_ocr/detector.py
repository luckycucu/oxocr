"""OpenThai PP-OCRv6-tiny DB 行检测（ONNX 后端）。

DB 家族第四变体：白边 pad + 双向 limit resize（小图也放大，上限 4×）、
BGR /255 后 ImageNet mean/std、无 dilate、fast 得分（最小外接矩形内均值）、
unclip 1.4。坐标缩放回原图、减 pad、clip。
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pyclipper

from oxocr.runtime import imageops
from oxocr.runtime.session import OrtSession
from oxocr.types import polygon_area, polygon_perimeter

__all__ = ["DbDetector"]

_MAX_CANDIDATES = 3000  # 权重绑定的导出常量，不暴露为参数
_MIN_SIZE = 3


class DbDetector:
    def __init__(
        self,
        model_path: str | Path,
        num_threads: int = 0,
        providers: Sequence[str] = ("CPUExecutionProvider",),
        limit_side_len: int = 1280,
        thresh: float = 0.2,
        box_thresh: float = 0.4,
        unclip_ratio: float = 1.4,
    ):
        self.session = OrtSession(model_path, num_threads=num_threads, providers=providers)
        self.limit_side_len = int(limit_side_len)
        self.thresh = float(thresh)
        self.box_thresh = float(box_thresh)
        self.unclip_ratio = float(unclip_ratio)
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def _preprocess(self, img: np.ndarray, limit_side_len: int) -> tuple[np.ndarray, int, float, float]:
        """白边 pad → 双向 limit resize（32 倍数）→ ImageNet BGR。
        返回 (blob, pad, sx, sy)：sx/sy 为 pad 后原图到概率图的坐标比例。"""
        h0, w0 = img.shape[:2]
        pad = max(8, int(0.05 * min(h0, w0)))
        padded = np.pad(img, ((pad, pad), (pad, pad), (0, 0)), mode="constant", constant_values=255)
        h, w = padded.shape[:2]
        s = min(limit_side_len / max(h, w), 4.0) if limit_side_len else 1.0
        rh = max(32, int(round(h * s / 32)) * 32)
        rw = max(32, int(round(w * s / 32)) * 32)
        resized = imageops.resize(padded, (rw, rh))
        x = resized.astype(np.float32) / 255.0
        x = (x - self.mean) / self.std
        blob = np.expand_dims(x.transpose((2, 0, 1)), axis=0).astype(np.float32)
        return blob, pad, w / rw, h / rh

    @staticmethod
    def _min_area_quad(points: np.ndarray) -> tuple[np.ndarray, float]:
        rect = imageops.min_area_rect(points)
        box = imageops.order_points_clockwise(imageops.box_points(rect))
        return box, min(rect[1])

    @staticmethod
    def _unclip(quad: np.ndarray, ratio: float) -> np.ndarray | None:
        length = polygon_perimeter(quad)
        if length == 0:
            return None
        distance = polygon_area(quad) * ratio / length
        offset = pyclipper.PyclipperOffset()
        offset.AddPath([tuple(p) for p in quad.tolist()], pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        grown = offset.Execute(distance)
        return np.array(grown[0], dtype=np.float32) if grown else None

    def __call__(
        self,
        img: np.ndarray,
        *,
        thresh: float | None = None,
        box_thresh: float | None = None,
        unclip_ratio: float | None = None,
        limit_side_len: int | None = None,
    ) -> tuple[list[np.ndarray], list[float]]:
        thresh = self.thresh if thresh is None else float(thresh)
        box_thresh = self.box_thresh if box_thresh is None else float(box_thresh)
        unclip_ratio = self.unclip_ratio if unclip_ratio is None else float(unclip_ratio)
        limit_side_len = self.limit_side_len if limit_side_len is None else int(limit_side_len)

        h0, w0 = img.shape[:2]
        blob, pad, sx, sy = self._preprocess(img, limit_side_len)
        prob = self.session(blob)[0, 0]
        mask = (prob > thresh).astype(np.uint8)

        boxes, scores = [], []
        for contour in imageops.find_contours(mask)[:_MAX_CANDIDATES]:
            quad, short = self._min_area_quad(contour)
            if short < _MIN_SIZE:
                continue
            score = imageops.box_score_fast(prob, quad)
            if score < box_thresh:
                continue
            grown = self._unclip(quad, unclip_ratio)
            if grown is None:
                continue
            quad, short = self._min_area_quad(grown.reshape(-1, 1, 2))
            if short < _MIN_SIZE + 2:
                continue
            quad[:, 0] = np.clip(quad[:, 0] * sx - pad, 0, w0 - 1)
            quad[:, 1] = np.clip(quad[:, 1] * sy - pad, 0, h0 - 1)
            if quad[:, 0].max() - quad[:, 0].min() >= 2 and quad[:, 1].max() - quad[:, 1].min() >= 2:
                boxes.append(quad.astype(np.float32))
                scores.append(float(score))
        return boxes, scores