"""cv2 替代层（唯一实现）。

核心不依赖 OpenCV：resize/warp 走 Pillow，形态学/几何走 numpy，
多边形填充与轮廓提取走 upolygon。参考实现中的重复函数在此收敛。
"""

from __future__ import annotations

import math

import numpy as np
import upolygon
from PIL import Image

__all__ = [
    "resize",
    "dilate",
    "convex_hull",
    "min_area_rect",
    "box_points",
    "box_score_fast",
    "find_contours",
    "order_points_clockwise",
    "get_perspective_transform",
    "warp_perspective",
    "get_rotate_crop_image",
    "otsu_threshold",
    "sort_boxes",
    "connected_components",
    "rotate",
]

_RESAMPLE = {
    "nearest": Image.Resampling.NEAREST,
    "bilinear": Image.Resampling.BILINEAR,
    "bicubic": Image.Resampling.BICUBIC,
    "area": Image.Resampling.BOX,
}


def resize(img: np.ndarray, dsize: tuple[int, int], interpolation: str = "bilinear") -> np.ndarray:
    """dsize = (w, h)，与 cv2.resize 一致。"""
    pil_img = Image.fromarray(img)
    resized = pil_img.resize(
        dsize, resample=_RESAMPLE.get(interpolation, Image.Resampling.BILINEAR)
    )
    return np.array(resized)


def dilate(mask: np.ndarray, kernel_size: int = 2, anchor: int | None = None) -> np.ndarray:
    """矩形核形态学膨胀，等价 cv2.dilate(mask, np.ones((k, k)))。

    anchor 默认 k//2（cv2 默认中心锚点）；边界外按 0 处理。
    用滑窗 maximum 实现，修正参考实现不正确的切片分支。
    """
    k = int(kernel_size)
    if k < 1:
        raise ValueError("kernel_size 必须 >= 1")
    a = k // 2 if anchor is None else int(anchor)
    if not 0 <= a < k:
        raise ValueError("anchor 必须在 [0, kernel_size) 内")
    h, w = mask.shape[:2]
    padded = np.pad(mask, ((a, k - 1 - a), (a, k - 1 - a)), mode="constant", constant_values=0)
    out = np.zeros((h, w), dtype=mask.dtype)
    for dy in range(k):
        for dx in range(k):
            np.maximum(out, padded[dy : dy + h, dx : dx + w], out=out)
    return out


def _cross(o, a, b) -> float:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def convex_hull(points: np.ndarray) -> np.ndarray:
    """Andrew 单调链凸包，逆时针（数学坐标），不含共线冗余点。"""
    pts = np.unique(np.asarray(points, dtype=np.float64).reshape(-1, 2), axis=0)
    if len(pts) <= 2:
        return pts
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]
    lower: list = []
    for p in pts:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list = []
    for p in reversed(pts):
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1])


def min_area_rect(contour: np.ndarray) -> tuple[tuple[float, float], tuple[float, float], float]:
    """最小外接旋转矩形，返回 ((cx, cy), (w, h), angle_deg)。

    候选角度必为凸包边方向（旋转卡壳的核心引理）；这里对所有候选角
    一次性批量旋转求包围盒，避免参考实现的逐角度 Python 循环。
    """
    points = np.asarray(contour, dtype=np.float64).reshape(-1, 2)
    hull = convex_hull(points)
    if len(hull) < 3:
        center = hull.mean(axis=0) if len(hull) else np.zeros(2)
        return (float(center[0]), float(center[1])), (1.0, 1.0), 0.0

    rolled = np.roll(hull, -1, axis=0)
    edges = rolled - hull
    angles = np.arctan2(edges[:, 1], edges[:, 0])
    cos_a, sin_a = np.cos(angles), np.sin(angles)
    # 绕 -angle 旋转，使该边水平：xs 沿边方向、ys 垂直边方向
    hx, hy = hull[:, 0][None, :], hull[:, 1][None, :]
    xs = hx * cos_a[:, None] + hy * sin_a[:, None]
    ys = -hx * sin_a[:, None] + hy * cos_a[:, None]
    x_min, x_max = xs.min(axis=1), xs.max(axis=1)
    y_min, y_max = ys.min(axis=1), ys.max(axis=1)
    widths, heights = x_max - x_min, y_max - y_min
    areas = widths * heights
    best = int(np.argmin(areas))
    angle = float(angles[best])
    cx_rot = (x_min[best] + x_max[best]) / 2.0
    cy_rot = (y_min[best] + y_max[best]) / 2.0
    # 逆旋转（+angle）回到原坐标系
    c, s = math.cos(angle), math.sin(angle)
    cx = cx_rot * c - cy_rot * s
    cy = cx_rot * s + cy_rot * c
    return (
        (float(cx), float(cy)),
        (float(widths[best]), float(heights[best])),
        float(math.degrees(angle)),
    )


def box_points(rect: tuple) -> np.ndarray:
    """min_area_rect 的矩形 → 4 角点（与参考实现同一角度约定）。"""
    (cx, cy), (w, h), angle = rect
    angle_rad = math.radians(angle)
    cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
    dx, dy = w / 2.0, h / 2.0
    corners = np.array([[-dx, -dy], [dx, -dy], [dx, dy], [-dx, dy]])
    rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    return corners @ rot.T + np.array([cx, cy])


def box_score_fast(bitmap: np.ndarray, box: np.ndarray) -> float:
    """bitmap 在 box 多边形内的均值（单一实现，检测器共用）。"""
    h, w = bitmap.shape[:2]
    xmin = int(np.clip(np.floor(box[:, 0].min()), 0, w - 1))
    xmax = int(np.clip(np.ceil(box[:, 0].max()), 0, w - 1))
    ymin = int(np.clip(np.floor(box[:, 1].min()), 0, h - 1))
    ymax = int(np.clip(np.ceil(box[:, 1].max()), 0, h - 1))
    if xmax <= xmin or ymax <= ymin:
        return 0.0
    mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
    box_f = box.copy()
    box_f[:, 0] -= xmin
    box_f[:, 1] -= ymin
    upolygon.draw_polygon(mask, [box_f.astype(int).flatten().tolist()], 1)
    region = bitmap[ymin : ymax + 1, xmin : xmax + 1]
    mask_bool = mask.astype(bool)
    if mask_bool.sum() == 0:
        return 0.0
    return float(region[mask_bool].mean())


def find_contours(binary: np.ndarray) -> list[np.ndarray]:
    """二值图外轮廓列表，每个轮廓为 (N, 2) int32。"""
    _, ext_paths, _ = upolygon.find_contours(binary.astype(bool))
    return [np.array(path, dtype=np.int32).reshape(-1, 2) for path in ext_paths]


def order_points_clockwise(pts: np.ndarray) -> np.ndarray:
    """4 点排序为 tl/tr/br/bl。"""
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    x_sorted = pts[np.argsort(pts[:, 0]), :]
    left_most = x_sorted[:2, :]
    right_most = x_sorted[2:, :]
    left_most = left_most[np.argsort(left_most[:, 1]), :]
    tl, bl = left_most
    right_most = right_most[np.argsort(right_most[:, 1]), :]
    tr, br = right_most
    return np.array([tl, tr, br, bl], dtype=np.float32)


def get_perspective_transform(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """4 点对应求 3x3 透视矩阵（src → dst）。"""
    src = np.asarray(src, dtype=np.float64).reshape(4, 2)
    dst = np.asarray(dst, dtype=np.float64).reshape(4, 2)
    rows = []
    for i in range(4):
        x, y = src[i]
        big_x, big_y = dst[i]
        rows.append([x, y, 1, 0, 0, 0, -big_x * x, -big_x * y])
        rows.append([0, 0, 0, x, y, 1, -big_y * x, -big_y * y])
    coeffs = np.linalg.solve(np.array(rows), dst.flatten())
    return np.vstack([coeffs[:3], coeffs[3:6], [coeffs[6], coeffs[7], 1.0]])


def warp_perspective(
    img: np.ndarray,
    matrix: np.ndarray,
    dsize: tuple[int, int],
    *,
    interpolation: str = "bicubic",
    fillcolor: int | tuple | None = None,
) -> np.ndarray:
    """PIL 透视变换（内部用逆矩阵），dsize = (w, h)；fillcolor 填充边界外区域。"""
    inv = np.linalg.inv(np.asarray(matrix, dtype=np.float64))
    coeffs = inv.flatten()[:8].tolist()
    pil_img = Image.fromarray(img)
    result = pil_img.transform(
        dsize,
        Image.Transform.PERSPECTIVE,
        coeffs,
        _RESAMPLE.get(interpolation, Image.Resampling.BICUBIC),
        fillcolor=fillcolor,
    )
    return np.array(result)


def get_rotate_crop_image(img: np.ndarray, points: np.ndarray) -> np.ndarray:
    """按四边形透视裁出文本行；高宽比 >= 1.5 时旋转 90°（竖排行）。"""
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    crop_w = int(max(np.linalg.norm(pts[0] - pts[1]), np.linalg.norm(pts[2] - pts[3])))
    crop_h = int(max(np.linalg.norm(pts[0] - pts[3]), np.linalg.norm(pts[1] - pts[2])))
    if crop_w <= 0 or crop_h <= 0:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    pts_std = np.array([[0, 0], [crop_w, 0], [crop_w, crop_h], [0, crop_h]], dtype=np.float32)
    matrix = get_perspective_transform(pts, pts_std)
    dst = warp_perspective(img, matrix, (crop_w, crop_h))
    if dst.shape[0] / dst.shape[1] >= 1.5:
        dst = np.rot90(dst)
    return dst


def otsu_threshold(gray: np.ndarray) -> int:
    """Otsu 全局二值化阈值（最大类间方差），等价 cv2.threshold 的 THRESH_OTSU。"""
    hist = np.bincount(np.asarray(gray, dtype=np.uint8).ravel(), minlength=256)
    total = int(hist.sum())
    if total == 0:
        return 0
    omega = np.cumsum(hist) / total
    mu = np.cumsum(hist * np.arange(256)) / total
    denom = omega * (1.0 - omega)
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_b = (mu[-1] * omega - mu) ** 2 / denom
    sigma_b[~np.isfinite(sigma_b)] = -1.0
    return int(np.argmax(sigma_b))


def sort_boxes(
    boxes: np.ndarray, scores: list[float] | None = None
) -> tuple[np.ndarray, list[float] | None]:
    """检测框阅读序排序：按顶边 y 聚类（间隔 >= 10px 分行），行内按 x。

    scores 提供时同步重排（与 boxes 对齐）。
    """
    if len(boxes) == 0:
        return boxes, scores
    y_coords = boxes[:, 0, 1]
    y_order = np.argsort(y_coords, kind="stable")
    boxes_y_sorted = boxes[y_order]
    dy = np.diff(y_coords[y_order])
    line_increments = (dy >= 10).astype(np.int32)
    line_ids = np.concatenate([[0], np.cumsum(line_increments)])
    final_order = np.lexsort((boxes_y_sorted[:, 0, 0], line_ids))
    sorted_scores = None if scores is None else [scores[i] for i in y_order[final_order]]
    return boxes_y_sorted[final_order], sorted_scores


def connected_components(
    mask: np.ndarray, connectivity: int = 8
) -> tuple[int, np.ndarray, np.ndarray]:
    """二值图连通域标记，等价 cv2.connectedComponentsWithStats（不含 centroids）。

    返回 (n, labels, stats)：n 含背景 0；labels int32 (H, W)；stats [n, 5]
    (left, top, width, height, area)，第 0 行为背景。编号按首次出现的光栅序
    （分区与 cv2 一致；cv2 的编号次序因其 union 策略可能不同）。
    行游程 union-find：开销正比于游程数而非像素数。
    """
    if connectivity not in (4, 8):
        raise ValueError("connectivity 必须为 4 或 8")
    binary = np.asarray(mask) != 0
    if binary.ndim != 2:
        raise ValueError("mask 必须是二维数组")
    h, w = binary.shape
    labels = np.zeros((h, w), np.int32)
    bg_area = int(h * w - int(binary.sum()))
    if not binary.any():
        return 1, labels, np.array([[0, 0, w, h, bg_area]], np.int32)

    padded = np.zeros((h, w + 2), np.int8)
    padded[:, 1:-1] = binary
    d = np.diff(padded, axis=1)
    r_s, c_s = np.where(d == 1)  # 游程起点（行, 列）
    _, c_e = np.where(d == -1)  # 游程终点（不含）
    nruns = len(r_s)
    parent = np.arange(nruns)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = int(parent[x])
        return x

    row_start = np.searchsorted(r_s, np.arange(h + 1))
    ext = 1 if connectivity == 8 else 0
    for r in range(h - 1):
        a0, a1 = int(row_start[r]), int(row_start[r + 1])
        b0, b1 = int(row_start[r + 1]), int(row_start[r + 2])
        i, j = a0, b0
        while i < a1 and j < b1:
            if c_e[i] + ext <= c_s[j]:
                i += 1
            elif c_e[j] + ext <= c_s[i]:
                j += 1
            else:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[max(ri, rj)] = min(ri, rj)
                if c_e[i] <= c_e[j]:
                    i += 1
                else:
                    j += 1

    seen: dict[int, int] = {}
    run_labels = np.zeros(nruns, np.int32)
    for i in range(nruns):
        root = find(i)
        if root not in seen:
            seen[root] = len(seen) + 1
        run_labels[i] = seen[root]
    n = len(seen) + 1
    for i in range(nruns):
        labels[r_s[i], c_s[i] : c_e[i]] = run_labels[i]

    left = np.full(n, w, np.int64)
    top = np.full(n, h, np.int64)
    right = np.zeros(n, np.int64)
    bottom = np.zeros(n, np.int64)
    np.minimum.at(left, run_labels, c_s)
    np.minimum.at(top, run_labels, r_s)
    np.maximum.at(right, run_labels, c_e)
    np.maximum.at(bottom, run_labels, r_s + 1)
    areas = np.bincount(run_labels, weights=c_e - c_s, minlength=n)
    stats = np.zeros((n, 5), np.int32)
    stats[0] = [0, 0, w, h, bg_area]
    stats[1:, 0] = left[1:]
    stats[1:, 1] = top[1:]
    stats[1:, 2] = right[1:] - left[1:]
    stats[1:, 3] = bottom[1:] - top[1:]
    stats[1:, 4] = areas[1:]
    return n, labels, stats


def rotate(
    img: np.ndarray,
    degrees: float,
    *,
    fill: int | tuple = 255,
    interpolation: str = "bilinear",
    expand: bool = False,
) -> np.ndarray:
    """绕中心逆时针旋转，等价 cv2.getRotationMatrix2D + warpAffine。

    正角度为逆时针（与 cv2 约定一致）；旋转露出的区域填 fill
    （灰度图默认纸白 255，ink map 传 0，彩色图可传逐通道元组）。
    expand=False 画布尺寸不变；expand=True 画布扩大容纳整图
    （新尺寸 round(H*|sin|+W*|cos|)，与 cv2 手动扩画布公式一致）。
    """
    resample = _RESAMPLE.get(interpolation, Image.Resampling.BILINEAR)
    pil_img = Image.fromarray(img)
    if not expand:
        rotated = pil_img.rotate(degrees, resample=resample, expand=False, fillcolor=fill)
        return np.array(rotated)
    h, w = img.shape[:2]
    rad = math.radians(degrees)
    c, s = math.cos(rad), math.sin(rad)
    # cv2.getRotationMatrix2D((w/2, h/2), degrees, 1.0)
    m = np.array([
        [c, s, (1 - c) * w / 2 - s * h / 2],
        [-s, c, s * w / 2 + (1 - c) * h / 2],
    ])
    nw = int(round(h * abs(s) + w * abs(c)))
    nh = int(round(h * abs(c) + w * abs(s)))
    m[0, 2] += nw / 2 - w / 2
    m[1, 2] += nh / 2 - h / 2
    inv = np.linalg.inv(np.vstack([m, [0, 0, 1]]))
    coeffs = inv.flatten()[:6].tolist()
    rotated = pil_img.transform((nw, nh), Image.Transform.AFFINE, coeffs, resample, fillcolor=fill)
    return np.array(rotated)
