"""行裁剪与切分几何（移植 OpenThai2.0 runtime/line_engine.py 的几何部分）。

cv2 替代：透视裁剪复用 imageops + np.pad(edge) 预填充（等价 BORDER_REPLICATE），
灰度化用 numpy 加权，medianBlur 用一维滑窗中值（精确中值，edge 边界等价 cv2 replicate）。
"""

from __future__ import annotations

import math

import numpy as np

from oxocr.runtime import imageops

__all__ = [
    "crop_line",
    "line_height",
    "page_skew",
    "split_one",
    "vertical",
    "to_gray",
    "flatten",
    "median_blur_1d",
]

_BGR_WEIGHTS = np.array([0.114, 0.587, 0.299], dtype=np.float32)


def to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    return (img.astype(np.float32) @ _BGR_WEIGHTS).astype(np.uint8)


def median_blur_1d(row: np.ndarray, win: int) -> np.ndarray:
    """一维中值滤波（奇数窗），edge 边界，等价 cv2.medianBlur 在 3×W 条带上的中行。"""
    pad = win // 2
    padded = np.pad(row, pad, mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, win)
    return np.median(windows, axis=1).astype(row.dtype)


def flatten(g: np.ndarray) -> np.ndarray:
    """Dark text on white from any line crop: each column's background is the
    median of the crop's top and bottom margin rows, smoothed along the line,
    and text is whatever differs from it, light or dark."""
    h, w = g.shape
    k = max(2, h // 6)
    bg = np.median(np.concatenate([g[:k], g[-k:]], 0), axis=0).astype(np.uint8)
    win = max(3, min(h | 1, 255))
    bg = median_blur_1d(bg, win).astype(np.float32)
    d = np.abs(g.astype(np.float32) - bg[None, :])
    hist = np.cumsum(np.bincount(d.astype(np.uint8).ravel(), minlength=256))
    hi = max(float(np.searchsorted(hist, 0.98 * hist[-1])), 24.0)
    return (255 - np.clip(d * (255.0 / hi), 0, 255)).astype(np.uint8)


def crop_line(img: np.ndarray, box: np.ndarray) -> np.ndarray:
    """Perspective-rectify one quad into a line image（越界 edge 填充等价 REPLICATE）。"""
    box = np.asarray(box, dtype=np.float32).reshape(4, 2)
    w = int(max(np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[2] - box[3])))
    h = int(max(np.linalg.norm(box[0] - box[3]), np.linalg.norm(box[1] - box[2])))
    w, h = max(w, 1), max(h, 1)
    ih, iw = img.shape[:2]
    slack = 2  # bicubic 插值邻域
    left = int(np.ceil(max(0.0, -float(box[:, 0].min())))) + slack
    top = int(np.ceil(max(0.0, -float(box[:, 1].min())))) + slack
    right = int(np.ceil(max(0.0, float(box[:, 0].max()) - iw))) + slack
    bottom = int(np.ceil(max(0.0, float(box[:, 1].max()) - ih))) + slack
    padded = np.pad(img, ((top, bottom), (left, right), (0, 0)), mode="edge")
    src = box + np.array([left, top], dtype=np.float32)
    dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    matrix = imageops.get_perspective_transform(src, dst)
    return imageops.warp_perspective(padded, matrix, (w, h))


def line_height(boxes: list[np.ndarray]) -> float:
    """Typical text-line height: the median over line-shaped boxes (at least
    twice as wide as tall), so detached-mark fragments don't skew it."""
    hs = [float(max(np.linalg.norm(b[0] - b[3]), 1.0)) for b in boxes]
    lines = [h for b, h in zip(boxes, hs) if np.linalg.norm(b[0] - b[1]) >= 2 * h]
    return float(np.median(lines if len(lines) >= 3 else hs)) if hs else 1.0


def page_skew(boxes: list[np.ndarray]) -> float:
    """Median text-line angle in degrees, from boxes at least 3x wider than tall."""
    angles = []
    for b in boxes:
        dx, dy = b[1][0] - b[0][0], b[1][1] - b[0][1]
        h = np.linalg.norm(b[0] - b[3])
        if np.hypot(dx, dy) >= 3 * max(h, 1.0):
            angles.append(math.degrees(math.atan2(dy, dx)))
    return float(np.median(angles)) if angles else 0.0


def vertical(box: np.ndarray, ratio: float) -> int:
    """+1 for a line at least `ratio` times taller than wide, -1 for one that
    much wider than tall, 0 otherwise."""
    w, h = np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[0] - box[3])
    return 1 if h >= ratio * max(w, 1.0) else -1 if w >= ratio * max(h, 1.0) else 0


def split_one(img: np.ndarray, b: np.ndarray, ratio: float) -> list[np.ndarray]:
    """One line box -> its pieces between blank stretches >= ratio x its height,
    with such a stretch trimmed off either end (the detector can run a line's
    box across an empty gutter up to the next column's first letter)."""
    if not ratio:
        return [b]
    c = crop_line(img, b)
    h, w = c.shape[:2]
    if w < 4 * h or h < 4:
        return [b]
    g = to_gray(c)
    ink = (flatten(g) < 128).any(axis=0)
    segs, start, blank = [], None, 0
    for x, on in enumerate(ink):
        if on:
            if start is None:
                start = x
            elif blank >= ratio * h:
                segs.append((start, x - blank))
                start = x
            blank = 0
        elif start is not None:
            blank += 1
    if start is not None:
        segs.append((start, w - blank))
    if not segs or (len(segs) == 1 and segs[0][0] < ratio * h and w - segs[0][1] < ratio * h):
        return [b]
    top, bot = b[1] - b[0], b[2] - b[3]
    pad = 0.15 * h
    out = []
    for x0, x1 in segs:
        t0, t1 = max(0.0, (x0 - pad) / w), min(1.0, (x1 + pad) / w)
        out.append(np.array([b[0] + t0 * top, b[0] + t1 * top, b[3] + t1 * bot, b[3] + t0 * bot], np.float32))
    return out