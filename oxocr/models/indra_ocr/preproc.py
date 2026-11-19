"""行裁剪与批编排（移植自 indra_ocr/rec_preproc.py，cv2 → imageops）。

识别器输入由行框三步制成：
  1. 上下扩 PAD_Y x 行高、左右扩 PAD_X x 行高；
  2. 从页面裁出该区域（页外补纸白）；
  3. 保比 resize 到 LINE_HEIGHT 高。
结果是 ink map（0 = 纸，1 = 墨）[LINE_HEIGHT, W]。暗行（深底亮字）反色读取。
"""

from __future__ import annotations

import math

import numpy as np

from oxocr.runtime.imageops import resize

__all__ = [
    "LINE_HEIGHT",
    "PAD_Y",
    "PAD_X",
    "WIDTH_MULTIPLE",
    "to_gray",
    "is_dark_line",
    "crop_line",
    "whole_image_ink",
    "pad_batch",
    "width_batches",
]

LINE_HEIGHT = 32
PAD_Y = 0.2
PAD_X = 0.3
WIDTH_MULTIPLE = 8  # 宽度补到该倍数（时间步 stride）


def to_gray(image: np.ndarray) -> np.ndarray:
    """BGR uint8 → uint8 灰度（BT.601 加权，与核心前处理同一约定）。"""
    if image.ndim == 2:
        return image
    return np.dot(image[..., :3], [0.114, 0.587, 0.299]).astype(np.uint8)


def _percentiles(b: np.ndarray, qs=(0.02, 0.5, 0.98)) -> list[float]:
    """uint8 数组的 np.percentile（线性插值），走直方图，不排序像素。"""
    n = b.size
    cum = np.cumsum(np.bincount(b.ravel(), minlength=256))
    out = []
    for q in qs:
        x = q * (n - 1)
        i = math.floor(x)
        g = x - i
        lo = float(np.searchsorted(cum, i, side="right"))
        hi = float(np.searchsorted(cum, min(i + 1, n - 1), side="right"))
        d = hi - lo
        out.append(hi - d * (1 - g) if g >= 0.5 else lo + d * g)
    return out


def is_dark_line(gray: np.ndarray, box) -> bool:
    """深底亮字（横幅、深色表头、幻灯片）：行框中位数靠近暗端、亮尾更长。
    识别器读浅底深字，故暗行反色读取。"""
    H, W = gray.shape
    x0, y0, x1, y1 = (float(v) for v in box[:4])
    b = gray[
        max(0, int(y0)) : min(H, int(y1) + 1), max(0, int(x0)) : min(W, int(x1) + 1)
    ]
    if not b.size:
        return False
    lo, med, hi = (
        _percentiles(b) if b.dtype == np.uint8 else np.percentile(b, (2, 50, 98))
    )
    return bool(hi - med > 2 * (med - lo) + 8)


def crop_line(
    gray: np.ndarray,
    box,
    height: int = LINE_HEIGHT,
    pad_y: float = PAD_Y,
    pad_x: float = PAD_X,
    max_width: int = 4096,
    window: int = 1,
    dark: bool | None = None,
) -> np.ndarray:
    """从 uint8 灰度页裁一行，返回 float32 ink map [window * height, W]。

    高窗（window > 1）保持文字尺度并带上行上下的页面；其中带即 window=1 裁剪。
    dark: 已知的 ``is_dark_line`` 结果（高窗与行带两份裁剪共享判定）。
    """
    pad_y = (window * (1 + 2 * pad_y) - 1) / 2
    height = height * window

    x0, y0, x1, y1 = (float(v) for v in box[:4])
    h = max(1.0, y1 - y0)
    X0, X1 = x0 - pad_x * h, x1 + pad_x * h
    Y0, Y1 = y0 - pad_y * h, y1 + pad_y * h
    H, W = gray.shape
    ix0, iy0 = int(math.floor(X0)), int(math.floor(Y0))
    ix1, iy1 = int(math.ceil(X1)), int(math.ceil(Y1))
    cw, ch = max(1, ix1 - ix0), max(1, iy1 - iy0)
    sx0, sy0, sx1, sy1 = max(0, ix0), max(0, iy0), min(W, ix1), min(H, iy1)
    if dark is None:
        dark = is_dark_line(gray, box)
    crop = np.full((ch, cw), 0 if dark else 255, np.uint8)
    if sx1 > sx0 and sy1 > sy0:
        crop[sy0 - iy0 : sy1 - iy0, sx0 - ix0 : sx1 - ix0] = gray[sy0:sy1, sx0:sx1]
    out_w = max(WIDTH_MULTIPLE, min(max_width, round(cw * height / ch)))
    interp = "area" if ch > height else "bilinear"
    resized = resize(crop, (out_w, height), interpolation=interp)
    ink = resized.astype(np.float32) / 255.0
    return ink if dark else 1.0 - ink


def _content_bbox(gray: np.ndarray) -> tuple[int, int, int, int]:
    """剥掉纯色外边（核心 pad_border 补边与自然白边），返回内容 (y0,y1,x0,x1)。

    逐圈 trim 全等于当前角色的边行/边列；角色随剥边变化（黑补边→白自然边）。
    扫描边不纯色时提前停止，保留少量边距，不影响识别。
    """
    H, W = gray.shape
    y0, y1, x0, x1 = 0, H, 0, W
    while y1 - y0 > 1 and x1 - x0 > 1:
        c = gray[y0, x0]
        changed = False
        while y1 - y0 > 1 and np.all(gray[y0, x0:x1] == c):
            y0 += 1
            changed = True
        while y1 - y0 > 1 and np.all(gray[y1 - 1, x0:x1] == c):
            y1 -= 1
            changed = True
        while x1 - x0 > 1 and np.all(gray[y0:y1, x0] == c):
            x0 += 1
            changed = True
        while x1 - x0 > 1 and np.all(gray[y0:y1, x1 - 1] == c):
            x1 -= 1
            changed = True
        if not changed:
            break
    return y0, y1, x0, x1


def whole_image_ink(gray: np.ndarray, height: int, max_width: int = 4096) -> np.ndarray:
    """整图去纯色外边后保比缩放到固定高（公式编码器静态 H），返回 ink map。

    不裁行、不加扩边：整图即单公式；去外边是为在 64px 高度下保住公式占比
    （核心 pad_border 会补边，小图补边后公式占比被压垮）。
    """
    y0, y1, x0, x1 = _content_bbox(gray)
    g = gray[y0:y1, x0:x1]
    H, W = g.shape
    out_w = max(WIDTH_MULTIPLE, min(max_width, round(W * height / H)))
    interp = "area" if H > height else "bilinear"
    resized = resize(g, (out_w, height), interpolation=interp)
    return 1.0 - resized.astype(np.float32) / 255.0


def pad_batch(lines: list[np.ndarray], multiple: int = WIDTH_MULTIPLE):
    """ink map 右补零到同一宽度。返回 (array [B, 1, H, W], widths [B])。"""
    H = lines[0].shape[0]
    Wm = max(x.shape[1] for x in lines)
    Wm = int(math.ceil(Wm / multiple) * multiple)
    out = np.zeros((len(lines), 1, H, Wm), np.float32)
    widths = np.zeros(len(lines), np.int64)
    for i, x in enumerate(lines):
        out[i, 0, :, : x.shape[1]] = x
        widths[i] = x.shape[1]
    return out, widths


def width_batches(
    widths: list[int],
    max_batch: int = 64,
    max_ratio: float = 1.3,
    min_width: int = 64,
    max_area: int = 64 * 640,
) -> list[list[int]]:
    """按宽度相近分组，减少 padding 浪费。

    窄行优先。批在 ``max_batch`` 行、下一行宽于批首行 ``max_ratio`` 倍
    （低于 ``min_width`` 的按 ``min_width`` 计，短行不碎批）、或 行数 x 宽
    超过 ``max_area`` 像素列时关闭：ONNX Runtime 保留最大分配，大扫描页
    64 行 x 4096px 曾占 5 GB。
    """
    order = sorted(range(len(widths)), key=lambda i: widths[i])
    batches: list[list[int]] = []
    for i in order:
        if (
            batches
            and len(batches[-1]) < max_batch
            and max(widths[i], min_width)
            <= max_ratio * max(widths[batches[-1][0]], min_width)
            and (len(batches[-1]) + 1) * max(widths[i], min_width) <= max_area
        ):
            batches[-1].append(i)
        else:
            batches.append([i])
    return batches
