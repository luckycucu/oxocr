"""DB 文本检测（ONNX 后端）。

修正参考实现：limit_side_len / thresh / box_thresh / unclip_ratio 真正下传，
可在构造时给默认、也可在每次调用时覆盖。
"""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pyclipper

from oxocr.runtime import imageops
from oxocr.runtime.session import OrtSession
from oxocr.types import polygon_area, polygon_perimeter

__all__ = ["TextDetector"]


class TextDetector:
    def __init__(
        self,
        model_path: str | Path,
        num_threads: int = 0,
        providers: Sequence[str] = ("CPUExecutionProvider",),
        limit_side_len: int = 736,
        thresh: float = 0.3,
        box_thresh: float = 0.5,
        unclip_ratio: float = 1.6,
    ):
        self.session = OrtSession(model_path, num_threads=num_threads, providers=providers)
        self.limit_side_len = int(limit_side_len)
        self.thresh = float(thresh)
        self.box_thresh = float(box_thresh)
        self.unclip_ratio = float(unclip_ratio)
        self.mean = np.array([0.5, 0.5, 0.5])
        self.std = np.array([0.5, 0.5, 0.5])
        self.scale = 1 / 255.0

    def _resize(self, img: np.ndarray, limit_side_len: int) -> np.ndarray | None:
        h, w = img.shape[:2]
        if min(h, w) < limit_side_len:
            ratio = float(limit_side_len) / (h if h < w else w)
        else:
            ratio = 1.0
        resize_h = int(round(h * ratio / 32) * 32)
        resize_w = int(round(w * ratio / 32) * 32)
        if resize_w <= 0 or resize_h <= 0:
            return None
        return imageops.resize(img, (resize_w, resize_h))

    def _preprocess(self, img: np.ndarray, limit_side_len: int) -> np.ndarray:
        resized = self._resize(img, limit_side_len)
        if resized is None:
            return np.empty((1, 3, 0, 0), dtype=np.float32)
        normed = (resized.astype("float32") * self.scale - self.mean) / self.std
        return np.expand_dims(normed.transpose((2, 0, 1)), axis=0).astype(np.float32)

    @staticmethod
    def _get_mini_boxes(contour: np.ndarray) -> tuple[np.ndarray, float]:
        bounding_box = imageops.min_area_rect(contour)
        points = sorted(list(imageops.box_points(bounding_box)), key=lambda x: x[0])
        if points[1][1] > points[0][1]:
            idx1, idx4 = 0, 1
        else:
            idx1, idx4 = 1, 0
        if points[3][1] > points[2][1]:
            idx2, idx3 = 2, 3
        else:
            idx2, idx3 = 3, 2
        box = np.array([points[idx1], points[idx2], points[idx3], points[idx4]])
        return box, min(bounding_box[1])

    def _unclip(self, box: np.ndarray, ratio: float) -> np.ndarray:
        # 鞋带公式替代 shapely 的面积/周长
        distance = polygon_area(box) * ratio / max(polygon_perimeter(box), 1e-6)
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        return np.array(offset.Execute(distance)).reshape((-1, 1, 2))

    def _boxes_from_bitmap(
        self,
        pred: np.ndarray,
        bitmap: np.ndarray,
        dest_w: int,
        dest_h: int,
        box_thresh: float,
        unclip_ratio: float,
    ) -> tuple[np.ndarray, list[float]]:
        contours = imageops.find_contours(bitmap)
        num_contours = min(len(contours), 1000)
        boxes, scores = [], []
        for i in range(num_contours):
            pts, sside = self._get_mini_boxes(contours[i])
            if sside < 3:
                continue
            score = imageops.box_score_fast(pred, pts.reshape(-1, 2))
            if score < box_thresh:
                continue
            box = self._unclip(pts, unclip_ratio)
            box, sside = self._get_mini_boxes(box)
            if sside < 5:
                continue
            box[:, 0] = np.clip(np.round(box[:, 0] / bitmap.shape[1] * dest_w), 0, dest_w)
            box[:, 1] = np.clip(np.round(box[:, 1] / bitmap.shape[0] * dest_h), 0, dest_h)
            boxes.append(box.astype(np.int32))
            scores.append(score)
        return np.array(boxes, dtype=np.int32), scores

    @staticmethod
    def _filter_boxes(
        boxes: np.ndarray, scores: list[float], img_h: int, img_w: int
    ) -> tuple[np.ndarray, list[float]]:
        new_boxes, new_scores = [], []
        for box, score in zip(boxes, scores, strict=True):
            x_sorted = box[np.argsort(box[:, 0]), :]
            left = x_sorted[:2][np.argsort(x_sorted[:2, 1])]
            right = x_sorted[2:][np.argsort(x_sorted[2:, 1])]
            box_ordered = np.array([left[0], right[0], right[1], left[1]], dtype="float32")
            box_ordered[:, 0] = np.clip(box_ordered[:, 0], 0, img_w - 1)
            box_ordered[:, 1] = np.clip(box_ordered[:, 1], 0, img_h - 1)
            rw = int(np.linalg.norm(box_ordered[0] - box_ordered[1]))
            rh = int(np.linalg.norm(box_ordered[0] - box_ordered[3]))
            if rw <= 3 or rh <= 3:
                continue
            new_boxes.append(box_ordered)
            new_scores.append(score)
        return np.array(new_boxes), new_scores

    @staticmethod
    def _sorted_boxes(dt_boxes: np.ndarray) -> np.ndarray:
        if len(dt_boxes) == 0:
            return dt_boxes
        y_coords = dt_boxes[:, 0, 1]
        y_order = np.argsort(y_coords, kind="stable")
        boxes_y_sorted = dt_boxes[y_order]
        y_sorted = y_coords[y_order]
        dy = np.diff(y_sorted)
        line_increments = (dy >= 10).astype(np.int32)
        line_ids = np.concatenate([[0], np.cumsum(line_increments)])
        x_coords = boxes_y_sorted[:, 0, 0]
        final_order = np.lexsort((x_coords, line_ids))
        return boxes_y_sorted[final_order]

    def __call__(
        self,
        img: np.ndarray,
        *,
        limit_side_len: int | None = None,
        thresh: float | None = None,
        box_thresh: float | None = None,
        unclip_ratio: float | None = None,
    ) -> tuple[np.ndarray | None, list[float]]:
        limit_side_len = self.limit_side_len if limit_side_len is None else int(limit_side_len)
        thresh = self.thresh if thresh is None else float(thresh)
        box_thresh = self.box_thresh if box_thresh is None else float(box_thresh)
        unclip_ratio = self.unclip_ratio if unclip_ratio is None else float(unclip_ratio)

        ori_h, ori_w = img.shape[:2]
        blob = self._preprocess(img, limit_side_len)
        if blob.shape[2] == 0 or blob.shape[3] == 0:
            return None, []

        preds = self.session(blob)
        pred = preds[0, 0]
        segmentation = pred > thresh
        mask = imageops.dilate(segmentation.astype(np.uint8), kernel_size=2)

        boxes, scores = self._boxes_from_bitmap(pred, mask, ori_w, ori_h, box_thresh, unclip_ratio)
        if len(boxes) == 0:
            return None, []
        boxes, scores = self._filter_boxes(boxes, scores, ori_h, ori_w)
        if len(boxes) == 0:
            return None, []
        boxes = self._sorted_boxes(boxes)
        return boxes, scores
