"""LaTeX-OCR 预处理：对比度拉伸 + 阈值裁剪 + 暗底反色 + 32 对齐 + resizer 迭代缩放。

逐行对齐 examples/RapidLaTeXOCR-main 的 PreProcess / loop_image_resizer，仅做两处等价替换
（已对 4 张官方测试图逐字节验证）：cv2.findNonZero/boundingRect → np.nonzero 求 bbox；
cv2 RGB 往返 → pad 后已是灰度，直接 L 通道归一化。运行期不依赖 cv2。
"""
from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

__all__ = ["pad", "minmax_size", "to_tensor", "resize_loop"]

_MAX_DIMS = (672, 192)
_MIN_DIMS = (32, 32)
_MEAN = np.float32(0.7931 * 255.0)
_STD = np.float32(0.1738 * 255.0)
_DIVABLE = 32
_RESIZER_MAX_ITERS = 10  # 宽度桶可能振荡（官方 1.png 触顶），必须保留上限


def pad(img: Image.Image, divable: int = _DIVABLE) -> Image.Image:
    """对比度拉伸 → 阈值找文字 → 暗底反色 → 裁剪 bbox → 补到 divable 倍数白底。"""
    data = np.array(img.convert("LA"))
    if data[..., -1].var() == 0:
        data = data[..., 0].astype(np.uint8)
    else:
        data = (255 - data[..., -1]).astype(np.uint8)

    lo, hi = int(data.min()), int(data.max())
    if hi == lo:  # 均匀图：参考会 nan，此处置 0 兜底
        stretched = np.zeros(data.shape, dtype=np.float64)
    else:
        stretched = (data - lo) / (hi - lo) * 255.0

    if stretched.mean() > 128:
        gray = 255 * (stretched < 128).astype(np.uint8)
    else:
        gray = 255 * (stretched > 128).astype(np.uint8)
        stretched = 255 - stretched

    ys, xs = np.nonzero(gray)
    if len(xs) == 0:  # 无文字：参考 cv2.boundingRect(None) 会崩，取整图兜底
        a, b, w, h = 0, 0, img.size[0], img.size[1]
    else:
        a, b = int(xs.min()), int(ys.min())
        w, h = int(xs.max()) - a + 1, int(ys.max()) - b + 1

    rect = stretched[b : b + h, a : a + w].astype(np.uint8)
    im = Image.fromarray(rect).convert("L")
    pw = divable * (w // divable + (1 if w % divable else 0))
    ph = divable * (h // divable + (1 if h % divable else 0))
    padded = Image.new("L", (pw, ph), 255)
    padded.paste(im, (0, 0))
    return padded


def minmax_size(img: Image.Image) -> Image.Image:
    """超 max_dims 等比缩小（BILINEAR），不足 min_dims 白底补边。"""
    ratios = [a / b for a, b in zip(img.size, _MAX_DIMS, strict=True)]
    if any(r > 1 for r in ratios):
        size = np.maximum(np.array(img.size) // max(ratios), 1)
        img = img.resize(tuple(size.astype(int)), Image.Resampling.BILINEAR)
    padded_size = (max(img.size[0], _MIN_DIMS[0]), max(img.size[1], _MIN_DIMS[1]))
    if padded_size != img.size:
        padded = Image.new("L", padded_size, 255)
        padded.paste(img, img.getbbox())
        img = padded
    return img


def to_tensor(pad_img: Image.Image) -> np.ndarray:
    """L 灰度 → (v-mean)/std → [1,1,H,W] float32（省参考的 RGB 往返）。"""
    gray = np.asarray(pad_img, dtype=np.float32)
    return ((gray - _MEAN) / _STD)[None, None, ...]


def resize_loop(pil_rgb: Image.Image, resizer_sess: Any) -> np.ndarray:
    """迭代调用 image_resizer 预测宽度桶，直到宽度收敛或触顶，返回送入 encoder 的张量。"""
    input_image = minmax_size(pad(pil_rgb))
    r, w, h = 1.0, input_image.size[0], input_image.size[1]
    final = to_tensor(input_image)
    for _ in range(_RESIZER_MAX_ITERS):
        h = int(h * r)
        resample = Image.Resampling.BILINEAR if r > 1 else Image.Resampling.LANCZOS
        pad_img = pad(minmax_size(input_image.resize((w, h), resample)))
        final = to_tensor(pad_img)
        res = resizer_sess.run(None, {"input": final})[0]
        w = (int(np.argmax(res[0])) + 1) * 32
        if w == pad_img.size[0]:
            break
        r = w / pad_img.size[0]
    return final
