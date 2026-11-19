"""LineNet 行检测（移植自 indra_ocr/det_post.py + pipeline.detect）。

与 DB 家族完全不同：core 带 + 上下距离回归，无 unclip/轮廓多边形，
输出轴对齐行框。det.onnx 输入 [1,1,H,W]（32 倍数白补边 ink map），
输出 [1,3,h,w]（stride 2：core logit / top dist / bottom dist，无 picture 通道）。
训练侧 encode_targets 不接。
"""

from __future__ import annotations

import math

import numpy as np

from oxocr.runtime.imageops import connected_components, resize
from oxocr.runtime.session import OrtSession

__all__ = ["LineDetector", "page_to_input", "decode"]

CORE_X_SHRINK = 0.15  # core 每端比行短的行高比例
OUT_STRIDE = 2
INPUT_MULTIPLE = 32
ADAPTIVE_MIN_H = 9.0  # 行高中位数（输入 px）低于此触发二次检测
ADAPTIVE_TARGET_H = 12.0
ADAPTIVE_MAX_SIDE = 2048


def page_to_input(gray: np.ndarray, long_side: int) -> tuple[np.ndarray, float]:
    """长边缩到 long_side，H/W 各取 32 倍数白补边。返回 (float32 ink map, scale)。"""
    h, w = gray.shape
    scale = long_side / max(h, w)
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    if (nh, nw) != (h, w):
        interp = "area" if scale < 1 else "bilinear"
        gray = resize(gray, (nw, nh), interpolation=interp)
    ph = math.ceil(nh / INPUT_MULTIPLE) * INPUT_MULTIPLE
    pw = math.ceil(nw / INPUT_MULTIPLE) * INPUT_MULTIPLE
    out = np.zeros((ph, pw), np.float32)  # ink map：0 = 白
    out[:nh, :nw] = 1.0 - gray.astype(np.float32) / 255.0
    return out, scale


def decode(out: np.ndarray, thresh: float = 0.5, min_area: int = 2) -> np.ndarray:
    """解码一页。out: [3, h, w]（core logit, top dist, bottom dist）。

    返回 [N, 5] 输入像素行框：x0, y0, x1, y1, score。
    """
    prob = 1.0 / (1.0 + np.exp(-out[0]))
    mask = (prob > thresh).astype(np.uint8)
    n, lab, stats = connected_components(mask, connectivity=4)
    if n <= 1:
        return np.zeros((0, 5), np.float32)
    s = OUT_STRIDE
    flat = lab.ravel()
    keep = flat > 0
    idx = flat[keep]
    ys = (np.indices(lab.shape)[0].ravel()[keep].astype(np.float32) + 0.5) * s
    top = ys - out[1].ravel()[keep]
    bot = ys + out[2].ravel()[keep]
    sc = prob.ravel()[keep]
    cnt = np.bincount(idx, minlength=n).astype(np.float32)
    # 均值足够稳健，且远比中位数便宜
    y0 = np.bincount(idx, weights=top, minlength=n) / np.maximum(cnt, 1)
    y1 = np.bincount(idx, weights=bot, minlength=n) / np.maximum(cnt, 1)
    score = np.bincount(idx, weights=sc, minlength=n) / np.maximum(cnt, 1)
    boxes = []
    for i in range(1, n):
        if stats[i, 4] < min_area:
            continue
        lh = max(1.0, y1[i] - y0[i])
        ex = min(CORE_X_SHRINK * lh, 0.5 * lh)
        x0 = stats[i, 0] * s - ex
        x1 = (stats[i, 0] + stats[i, 2]) * s + ex
        boxes.append((x0, y0[i], x1, y1[i], score[i]))
    return np.asarray(boxes, np.float32).reshape(-1, 5)


class LineDetector:
    def __init__(self, model_path, num_threads: int = 0, providers=("CPUExecutionProvider",)):
        self.session = OrtSession(model_path, num_threads=num_threads, providers=providers)

    def _detect_at(self, gray: np.ndarray, long_side: int, thresh: float):
        x, s = page_to_input(gray, long_side)
        out = self.session(x[None, None])[0]
        return decode(out, thresh=thresh), s

    def __call__(
        self, gray: np.ndarray, *, long_side: int, thresh: float, adaptive: bool = True
    ) -> np.ndarray:
        """行框 [N, 5]（原图像素 x0,y0,x1,y1,score）；行高过小时放大重检。"""
        boxes, s = self._detect_at(gray, long_side, thresh)
        if adaptive and len(boxes) >= 8:
            med_h = float(np.median(boxes[:, 3] - boxes[:, 1]))
            if med_h < ADAPTIVE_MIN_H:
                ls = (
                    int(min(ADAPTIVE_MAX_SIDE, long_side * ADAPTIVE_TARGET_H / max(med_h, 1.0)))
                    // INPUT_MULTIPLE
                    * INPUT_MULTIPLE
                )
                if ls > long_side:
                    boxes, s = self._detect_at(gray, ls, thresh)
        if len(boxes):
            boxes[:, :4] /= s
            h, w = gray.shape
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w)
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h)
        return boxes
