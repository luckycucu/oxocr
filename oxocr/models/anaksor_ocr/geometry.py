"""四边形几何：角点排序、扩边、透视裁剪、包含框剔除（移植 anaksor-ocr geometry.py）。

cv2 替代：透视变换复用 imageops，BORDER_REPLICATE 用 np.pad(edge) 预填充等价实现，
intersectConvexConvex 用 Sutherland–Hodgman 凸多边形裁剪（两框皆凸四边形）。
"""

from __future__ import annotations

import numpy as np

from oxocr.runtime import imageops
from oxocr.types import polygon_area

__all__ = ["order_quad", "quad_size", "expand_quad", "warp_crop", "drop_contained"]


def order_quad(pts) -> np.ndarray:
    """4 点排序为 tl/tr/br/bl（绕质心角度排序，旋转框安全）。

    多于 4 点先取最小外接矩形。与 imageops.order_points_clockwise（按 x 排序）
    不同：45° 等旋转框按 x 排序会错序。
    """
    pts = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
    if len(pts) != 4:
        pts = imageops.box_points(imageops.min_area_rect(pts))
    c = pts.mean(0)
    ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    pts = pts[np.argsort(ang)]  # 图像坐标下顺时针，起点约 tl
    return np.roll(pts, -int(np.argmin(pts.sum(1))), axis=0)


def quad_size(q) -> tuple[float, float]:
    """(沿上下边的宽, 沿左右边的高)。"""
    q = np.asarray(q, dtype=np.float32)
    w = (np.linalg.norm(q[1] - q[0]) + np.linalg.norm(q[2] - q[3])) / 2
    h = (np.linalg.norm(q[3] - q[0]) + np.linalg.norm(q[2] - q[1])) / 2
    return float(w), float(h)


def expand_quad(q, pad_x: float, pad_y: float) -> np.ndarray:
    """沿四边形自身轴向外扩 pad_x / pad_y 像素（tl/tr/br/bl 顺序）。"""
    q = np.asarray(q, dtype=np.float32)
    ux = q[1] - q[0]
    ux = ux / (np.linalg.norm(ux) + 1e-6)
    uy = q[3] - q[0]
    uy = uy / (np.linalg.norm(uy) + 1e-6)
    return np.stack(
        [
            q[0] - ux * pad_x - uy * pad_y,
            q[1] + ux * pad_x - uy * pad_y,
            q[2] + ux * pad_x + uy * pad_y,
            q[3] - ux * pad_x + uy * pad_y,
        ]
    ).astype(np.float32)


def warp_crop(img: np.ndarray, q) -> np.ndarray:
    """把四边形内容透视 warp 为正向矩形；越界部分等价 cv2.BORDER_REPLICATE。

    PIL 透视变换无 replicate 边界：先按 quad 越界量 + 插值邻域做 edge 预填充，
    warp 采样点全部落在填充后的图内，与 cv2 逐点等价。
    """
    q = np.asarray(q, dtype=np.float32).reshape(4, 2)
    w, h = quad_size(q)
    w, h = max(int(round(w)), 2), max(int(round(h)), 2)
    ih, iw = img.shape[:2]
    slack = 2  # bicubic 插值邻域
    left = int(np.ceil(max(0.0, -float(q[:, 0].min())))) + slack
    top = int(np.ceil(max(0.0, -float(q[:, 1].min())))) + slack
    right = int(np.ceil(max(0.0, float(q[:, 0].max()) - iw))) + slack
    bottom = int(np.ceil(max(0.0, float(q[:, 1].max()) - ih))) + slack
    padded = np.pad(img, ((top, bottom), (left, right), (0, 0)), mode="edge")
    src = q + np.array([left, top], dtype=np.float32)
    dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    matrix = imageops.get_perspective_transform(src, dst)
    return imageops.warp_perspective(padded, matrix, (w, h))


def _signed_area(pts: np.ndarray) -> float:
    x, y = pts[:, 0], pts[:, 1]
    return float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0


def _convex_intersection_area(a: np.ndarray, b: np.ndarray) -> float:
    """Sutherland–Hodgman：凸四边形 a 被凸四边形 b 裁剪后的面积（替代 cv2.intersectConvexConvex）。"""
    out = [np.asarray(p, dtype=np.float64) for p in a]
    sign = 1.0 if _signed_area(b) >= 0 else -1.0
    n = len(b)
    for i in range(n):
        e0 = np.asarray(b[i], dtype=np.float64)
        e1 = np.asarray(b[(i + 1) % n], dtype=np.float64)
        edge = e1 - e0
        inp, out = out, []
        if not inp:
            break

        def side(p, e0=e0, edge=edge):
            return sign * (edge[0] * (p[1] - e0[1]) - edge[1] * (p[0] - e0[0]))

        s = inp[-1]
        ss = side(s)
        for p in inp:
            sp = side(p)
            if sp >= 0:
                if ss < 0:
                    out.append(_line_intersect(s, p, e0, edge))
                out.append(p)
            elif ss >= 0:
                out.append(_line_intersect(s, p, e0, edge))
            s, ss = p, sp
    if len(out) < 3:
        return 0.0
    return polygon_area(np.array(out))


def _line_intersect(s: np.ndarray, p: np.ndarray, e0: np.ndarray, edge: np.ndarray) -> np.ndarray:
    """线段 s→p 与直线 e0+t*edge 的交点（调用方保证跨越）。"""
    d = p - s
    denom = d[0] * edge[1] - d[1] * edge[0]
    if abs(denom) < 1e-12:
        return p
    t = ((e0[0] - s[0]) * edge[1] - (e0[1] - s[1]) * edge[0]) / denom
    return s + t * d


def drop_contained(quads, min_inside: float = 0.8) -> list[int]:
    """保留框的索引：面积 >= min_inside 落在更大框内的小框，若是碎片（短边比
    r<0.4，如分离的元音/下标）或重复（r>0.75，同高）则剔除；
    两行合并框内的单行（约半高，0.4<=r<=0.75）保留。
    """
    n = len(quads)
    if n < 2:
        return list(range(n))
    q = [np.asarray(x, dtype=np.float32).reshape(4, 2) for x in quads]
    area = np.array([polygon_area(x) for x in q])
    side = np.array([min(quad_size(x)) for x in q])
    lo = np.array([x.min(0) for x in q])
    hi = np.array([x.max(0) for x in q])
    drop = set()
    for i in np.argsort(area):
        if area[i] <= 0:
            continue
        cand = np.where((area > area[i]) & np.all(lo <= hi[i], 1) & np.all(hi >= lo[i], 1))[0]
        for j in cand:
            if j in drop:
                continue
            inter = _convex_intersection_area(q[i], q[j])
            r = side[i] / max(side[j], 1e-6)
            if inter >= min_inside * area[i] and (r < 0.4 or r > 0.75):
                drop.add(int(i))
                break
    return [i for i in range(n) if i not in drop]
