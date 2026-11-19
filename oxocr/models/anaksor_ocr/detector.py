"""anaksor DBNet++ 行检测（ONNX 后端）。

DB 家族第三个变体：max-limit + min-side resize（长边 >limit 缩小 / <min 放大）、
BGR /255 后 ImageNet mean/std、无 dilate、slow 得分（contour 多边形内均值）。
几何原语全部复用 imageops，unclip 用 pyclipper + 鞋带公式。
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pyclipper

from oxocr.errors import InferenceError
from oxocr.runtime import imageops
from oxocr.runtime.session import OrtSession
from oxocr.types import polygon_area, polygon_perimeter

__all__ = ["DbDetector"]

_MAX_CANDIDATES = 1000  # 权重绑定的导出常量，不暴露为参数


class DbDetector:
    def __init__(
        self,
        model_path: str | Path,
        num_threads: int = 0,
        providers: Sequence[str] = ("CPUExecutionProvider",),
        limit_side_len: int = 1280,
        min_side_len: int = 736,
        thresh: float = 0.3,
        box_thresh: float = 0.5,
        unclip_ratio: float = 1.5,
    ):
        self.session = OrtSession(model_path, num_threads=num_threads, providers=providers)
        meta = self.session.metadata()
        if meta.get("anaksor_kind") not in (None, "det"):
            raise InferenceError(
                f"{Path(model_path).name} 是 {meta['anaksor_kind']} 模型，不是 det 检测权重"
            )
        if meta.get("anaksor_head") not in (None, "db"):
            raise InferenceError(
                f"{Path(model_path).name} 的 head 为 {meta['anaksor_head']}，期望 db"
            )
        self.limit_side_len = int(limit_side_len)
        self.min_side_len = int(min_side_len)
        self.thresh = float(thresh)
        self.box_thresh = float(box_thresh)
        self.unclip_ratio = float(unclip_ratio)
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def _resize(self, img: np.ndarray, limit_side_len: int, min_side_len: int) -> np.ndarray:
        h, w = img.shape[:2]
        long_side = max(h, w)
        scale = 1.0
        if long_side > limit_side_len:
            scale = limit_side_len / long_side
        elif long_side < min_side_len:
            scale = min_side_len / long_side
        nh = max(32, int(round(h * scale / 32)) * 32)
        nw = max(32, int(round(w * scale / 32)) * 32)
        return imageops.resize(img, (nw, nh))

    def _preprocess(self, img: np.ndarray, limit_side_len: int, min_side_len: int) -> np.ndarray:
        resized = self._resize(img, limit_side_len, min_side_len)
        bgr = resized.astype(np.float32) / 255.0
        normed = (bgr - self.mean) / self.std
        return np.expand_dims(normed.transpose((2, 0, 1)), axis=0).astype(np.float32)

    @staticmethod
    def _min_area_quad(points: np.ndarray) -> tuple[np.ndarray, float]:
        rect = imageops.min_area_rect(points)
        pts = sorted(list(imageops.box_points(rect)), key=lambda x: x[0])
        if pts[1][1] > pts[0][1]:
            idx1, idx4 = 0, 1
        else:
            idx1, idx4 = 1, 0
        if pts[3][1] > pts[2][1]:
            idx2, idx3 = 2, 3
        else:
            idx2, idx3 = 3, 2
        box = np.array([pts[idx1], pts[idx2], pts[idx3], pts[idx4]])
        return box, min(rect[1])

    @staticmethod
    def _unclip(quad: np.ndarray, ratio: float) -> np.ndarray | None:
        distance = polygon_area(quad) * ratio / max(polygon_perimeter(quad), 1e-6)
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(quad, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        grown = offset.Execute(distance)
        if len(grown) != 1:  # 消失或分裂：不是文本行
            return None
        return np.array(grown[0], dtype=np.float32)

    def __call__(
        self,
        img: np.ndarray,
        *,
        thresh: float | None = None,
        box_thresh: float | None = None,
        unclip_ratio: float | None = None,
        limit_side_len: int | None = None,
        min_side_len: int | None = None,
    ) -> tuple[np.ndarray | None, list[float]]:
        thresh = self.thresh if thresh is None else float(thresh)
        box_thresh = self.box_thresh if box_thresh is None else float(box_thresh)
        unclip_ratio = self.unclip_ratio if unclip_ratio is None else float(unclip_ratio)
        limit_side_len = self.limit_side_len if limit_side_len is None else int(limit_side_len)
        min_side_len = self.min_side_len if min_side_len is None else int(min_side_len)

        ori_h, ori_w = img.shape[:2]
        blob = self._preprocess(img, limit_side_len, min_side_len)
        prob = self.session(blob)[0, 0]
        mask = (prob > thresh).astype(np.uint8)
        height, width = mask.shape

        boxes, scores = [], []
        for contour in imageops.find_contours(mask)[:_MAX_CANDIDATES]:
            quad, short = self._min_area_quad(contour)
            if short < 3:
                continue
            score = imageops.box_score_fast(prob, contour)  # slow：contour 内均值
            if score < box_thresh:
                continue
            grown = self._unclip(quad, unclip_ratio)
            if grown is None:
                continue
            quad, short = self._min_area_quad(grown)
            if short < 5:
                continue
            quad[:, 0] = np.clip(np.round(quad[:, 0] / width * ori_w), 0, ori_w)
            quad[:, 1] = np.clip(np.round(quad[:, 1] / height * ori_h), 0, ori_h)
            boxes.append(quad.astype(np.int32))
            scores.append(score)
        if not boxes:
            return None, []
        arr, ordered = imageops.sort_boxes(np.array(boxes, dtype=np.int32), scores)
        assert ordered is not None
        return arr, ordered
