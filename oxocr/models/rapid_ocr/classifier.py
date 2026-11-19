"""0/180° 方向分类（ONNX 后端）。"""
from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from oxocr.runtime import imageops
from oxocr.runtime.session import OrtSession

__all__ = ["TextClassifier"]


class TextClassifier:
    def __init__(
        self,
        model_path: str | Path,
        cls_thresh: float = 0.9,
        cls_image_shape: Sequence[int] = (3, 48, 192),
        label_list: Sequence[str] = ("0", "180"),
        batch_num: int = 6,
        num_threads: int = 0,
        providers: Sequence[str] = ("CPUExecutionProvider",),
    ):
        self.session = OrtSession(model_path, num_threads=num_threads, providers=providers)
        self.cls_thresh = float(cls_thresh)
        self.cls_image_shape = list(cls_image_shape)
        self.label_list = list(label_list)
        self.batch_num = int(batch_num)

    def _resize_norm_img(self, img: np.ndarray) -> np.ndarray:
        img_c, img_h, img_w = self.cls_image_shape
        h, w = img.shape[:2]
        ratio = w / float(h)
        resized_w = img_w if math.ceil(img_h * ratio) > img_w else int(math.ceil(img_h * ratio))
        resized = imageops.resize(img, (resized_w, img_h), "bilinear")
        resized = resized.astype("float32").transpose((2, 0, 1)) / 255
        resized = (resized - 0.5) / 0.5
        padded = np.zeros((img_c, img_h, img_w), dtype=np.float32)
        padded[:, :, :resized_w] = resized
        return padded

    def __call__(self, img_list: list[np.ndarray]) -> tuple[list[np.ndarray], list[float]]:
        """返回 (校正后图像列表, 每条的旋转角 0/180)。"""
        if not img_list:
            return [], []
        results = list(img_list)
        angles = [0.0] * len(results)
        width_list = [img.shape[1] / float(img.shape[0]) for img in results]
        indices = np.argsort(np.array(width_list))
        for beg in range(0, len(results), self.batch_num):
            end = min(len(results), beg + self.batch_num)
            batch = np.concatenate(
                [self._resize_norm_img(results[idx])[np.newaxis, :] for idx in indices[beg:end]]
            ).astype(np.float32)
            preds = self.session(batch)
            for j, idx in enumerate(indices[beg:end]):
                label_idx = int(preds[j].argmax())
                label = self.label_list[label_idx]
                score = float(preds[j, label_idx])
                if "180" in label and score > self.cls_thresh:
                    results[idx] = np.rot90(results[idx], 2)
                    angles[idx] = 180.0
        return results, angles
