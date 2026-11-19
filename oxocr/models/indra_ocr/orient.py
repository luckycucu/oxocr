"""页面方向：倾斜估计 + 四分转 + 竖排（移植自 indra_ocr/pipeline.py 方向部分）。

deskew：行墨剖面锐度最大的角度（±45° 粗搜 2° → 0.5°/0.1° 精搜），无需权重。
auto_orient：orient.onnx 768² ink map 五分类（0/90/180/270/竖排），
argmax 概率 > ORIENT_MIN 才转。竖排（class 4）仅 cjk 模型处理。
所有几何改动的 box 逆映射回送入图坐标（unrotate_box / unturn_box）。
"""

from __future__ import annotations

import math

import numpy as np

from oxocr.runtime.imageops import resize, rotate
from oxocr.runtime.session import OrtSession

__all__ = [
    "ORIENT_MIN",
    "VERTICAL",
    "DESKEW_MIN",
    "preview",
    "estimate_skew",
    "rotate_page",
    "unrotate_box",
    "unturn_box",
    "orient_input",
    "orientation",
    "upright",
]

ORIENT_SIDE = 768
ORIENT_MIN = 0.95  # 低于此概率不转：真实页面几乎都正，转错代价大
VERTICAL = 4  # orient.onnx 竖排类
DESKEW_MAX = 45.0
DESKEW_MIN = 0.3  # 更小的倾斜不纠（搜索步长 0.1）
PREVIEW_SIDE = 1600


def _ink_thumb(gray: np.ndarray, side: int) -> np.ndarray:
    """uint8 ink map（255 = 墨）：页面缩到 side 像素。"""
    h, w = gray.shape
    s = side / max(h, w)
    small = resize(gray, (max(1, round(w * s)), max(1, round(h * s))), interpolation="area")
    return 255 - small


def _row_sharpness(ink: np.ndarray, a: float) -> float:
    """页面转 a 度后水平墨剖面逐行变化的锐度：文本行水平时最高。"""
    r = rotate(ink, a, fill=0)
    p = r.sum(1, dtype=np.int64).astype(np.float64)
    return float(((p[1:] - p[:-1]) ** 2).sum())


def estimate_skew(gray: np.ndarray, side: int = 800) -> float:
    """纠正文本行所需的逆时针角度：半尺寸缩略图 ±45° 粗搜 2°，取最优两角与 0
    在全缩略图上 0.5°、0.1° 精搜。空白页返回 0。"""
    fine = _ink_thumb(gray, side)
    if fine.mean() < 0.003 * 255:
        return 0.0
    coarse = _ink_thumb(gray, side // 2)
    grid = np.arange(-DESKEW_MAX, DESKEW_MAX + 1e-6, 2.0)
    scores = [_row_sharpness(coarse, float(a)) for a in grid]
    starts = {0.0, *(float(grid[i]) for i in np.argsort(scores)[-2:])}
    near = [a for a0 in starts for a in np.arange(a0 - 2.0, a0 + 2.0 + 1e-6, 0.5)]
    # 相近起点的 0.5° 区间重叠：每个角度只评一次
    seen: dict[float, float] = dict.fromkeys(near, 0.0)
    seen.update(zip(seen, [_row_sharpness(fine, float(a)) for a in seen], strict=True))
    a1 = max(near, key=seen.__getitem__)
    last = np.arange(a1 - 0.5, a1 + 0.5 + 1e-6, 0.1)
    vals = [_row_sharpness(fine, float(a)) for a in last]
    return float(max(zip(last, vals, strict=True), key=lambda av: av[1])[0])


def preview(gray: np.ndarray, side: int = PREVIEW_SIDE) -> np.ndarray:
    """页面一次性缩到至多 side 像素（面积均值）：倾斜搜索与方向模型读它。"""
    h, w = gray.shape
    s = side / max(h, w)
    if s >= 1:
        return gray
    return resize(gray, (max(1, round(w * s)), max(1, round(h * s))), interpolation="area")


def rotate_page(gray: np.ndarray, degrees: float) -> np.ndarray:
    """绕中心逆时针转 degrees（同尺寸，露出区域填纸白）。"""
    return rotate(gray, degrees, fill=255)


def _rot_matrix(center: tuple[float, float], degrees: float) -> np.ndarray:
    """与 cv2.getRotationMatrix2D 同一约定的 2x3 旋转矩阵。"""
    a = math.radians(degrees)
    c, s = math.cos(a), math.sin(a)
    cx, cy = center
    return np.array([[c, s, (1 - c) * cx - s * cy], [-s, c, s * cx + (1 - c) * cy]])


def unrotate_box(box, degrees: float, wh: tuple[int, int]) -> list[float]:
    """rotate_page(.., degrees) 后页面上的 box 映射回旋转前：四角逆旋转取包围盒并 clip。"""
    w, h = wh
    M = _rot_matrix((w / 2, h / 2), -degrees)
    x0, y0, x1, y1 = box[:4]
    c = M @ np.array([[x0, x1, x0, x1], [y0, y0, y1, y1], [1, 1, 1, 1]], float)
    return [
        round(float(np.clip(c[0].min(), 0, w)), 1),
        round(float(np.clip(c[1].min(), 0, h)), 1),
        round(float(np.clip(c[0].max(), 0, w)), 1),
        round(float(np.clip(c[1].max(), 0, h)), 1),
    ]


def unturn_box(box, k: int, hw: tuple[int, int]) -> list[float]:
    """四分转（rot90(gray, -k)，竖排 k=4 为 rot90(gray, 1)）的逆映射。

    hw 为转后（upright）页面的 (H, W)。返回原图坐标 [x0, y0, x1, y1]。
    """
    H, W = hw
    x0, y0, x1, y1 = (float(v) for v in box[:4])
    if k == 1:
        return [y0, W - x1, y1, W - x0]
    if k == 2:
        return [W - x1, H - y1, W - x0, H - y0]
    if k in (3, VERTICAL):
        return [H - y1, x0, H - y0, x1]
    return [x0, y0, x1, y1]


def orient_input(gray: np.ndarray, side: int = ORIENT_SIDE) -> np.ndarray:
    """[side, side] float32 ink map（1 = 墨）：长边缩到 side，补零成方阵。"""
    h, w = gray.shape
    s = side / max(h, w)
    small = resize(gray, (max(1, round(w * s)), max(1, round(h * s))), interpolation="area")
    out = np.zeros((side, side), np.float32)
    out[: small.shape[0], : small.shape[1]] = 1.0 - small.astype(np.float32) / 255.0
    return out


def orientation(session: OrtSession, gray: np.ndarray) -> int:
    """orient.onnx 判定的逆时针四分转数；无模型或概率不足返回 0；竖排返回 VERTICAL。"""
    p = session(orient_input(gray)[None, None])[0]
    k = int(p.argmax())
    return k if p[k] > ORIENT_MIN else 0


def upright(
    gray: np.ndarray,
    orient_session: OrtSession | None,
    *,
    deskew: bool = True,
    auto_orient: bool = True,
    allow_vertical: bool = False,
) -> tuple[np.ndarray, int, float]:
    """页面转正（k 个四分转）并纠倾斜。返回 (page, k, skew)。

    半转保持行方向，角度沿用；四分转后行曾是列，在转后页面上重测角度。
    allow_vertical=False 时忽略竖排类（仅 cjk 模型处理竖排）。
    """
    small = preview(gray)
    a = estimate_skew(small) if deskew else 0.0
    k = 0
    if auto_orient and orient_session is not None:
        thumb = rotate_page(small, a) if abs(a) >= DESKEW_MIN else small
        k = orientation(orient_session, thumb)
        if k == VERTICAL and not allow_vertical:
            k = 0
    if k == VERTICAL:
        # 逆时针四分转：列成行，最右列在上，各列从左到右读
        gray = np.ascontiguousarray(np.rot90(gray, 1))
        small = np.ascontiguousarray(np.rot90(small, 1))
    elif k:
        gray = np.ascontiguousarray(np.rot90(gray, -k))
        small = np.ascontiguousarray(np.rot90(small, -k))
    if k in (1, 3, VERTICAL) and deskew:
        a = estimate_skew(small)
    if abs(a) >= DESKEW_MIN:
        gray = rotate_page(gray, a)
    else:
        a = 0.0
    return gray, k, a
