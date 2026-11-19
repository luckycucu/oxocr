"""SegLink 文本检测（读光 max，ONNX 后端）。

NHWC 前处理 + 18 输出解码（纯 numpy 移植自 utils_seglink.py）。
去掉 batch=1 的 padding 脚手架，单图直出；输出名从 session 动态取。
SegLink 不产生检测置信度，返回 scores=None。
"""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from oxocr.runtime import imageops
from oxocr.runtime.session import OrtSession

__all__ = ["SegLinkDetector"]

_MEAN = np.array([123.68, 116.78, 103.94], dtype=np.float32)
_OFFSET_VARIANCE = 0.1
_ANCHOR_SIZES = np.array([6.0, 11.84210526, 23.68421053, 45.0, 90.0, 150.0])
_N_DET_LAYERS = 6
_N_LOCAL_LINKS = 8
_N_CROSS_LINKS = 4
_OFFSET_DIM = 6
_POS_LABEL = 1


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def _get_coord(offsets, map_size, offsets_defaults):
    for l_idx in range(_N_DET_LAYERS):
        upper = offsets_defaults[l_idx + 1][0] if l_idx + 1 < _N_DET_LAYERS else float("inf")
        if offsets < upper:
            base = offsets_defaults[l_idx][0]
            x = (offsets - base) % map_size[l_idx][1]
            y = (offsets - base) // map_size[l_idx][1]
            return l_idx, x, y
    raise ValueError(f"offsets {offsets} 超出所有层")


def _is_valid_coord(l_idx, x, y, map_size):
    w = map_size[l_idx][1]
    h = map_size[l_idx][0]
    return 0 <= x < w and 0 <= y < h


def _get_neighbours(l_idx, x, y, map_size, offsets_defaults):
    if l_idx == 0:
        coord = [
            (0, x - 1, y - 1), (0, x, y - 1), (0, x + 1, y - 1),
            (0, x - 1, y), (0, x + 1, y),
            (0, x - 1, y + 1), (0, x, y + 1), (0, x + 1, y + 1),
        ]
    else:
        coord = [
            (l_idx, x - 1, y - 1), (l_idx, x, y - 1), (l_idx, x + 1, y - 1),
            (l_idx, x - 1, y), (l_idx, x + 1, y),
            (l_idx, x - 1, y + 1), (l_idx, x, y + 1), (l_idx, x + 1, y + 1),
            (l_idx - 1, 2 * x, 2 * y), (l_idx - 1, 2 * x + 1, 2 * y),
            (l_idx - 1, 2 * x, 2 * y + 1), (l_idx - 1, 2 * x + 1, 2 * y + 1),
        ]
    neighbours = []
    link_idx = 0
    for nl_idx, nx, ny in coord:
        if _is_valid_coord(nl_idx, nx, ny, map_size):
            node_off = offsets_defaults[nl_idx][0] + map_size[nl_idx][1] * ny + nx
            if l_idx == 0:
                link_off = (
                    offsets_defaults[l_idx][1]
                    + (map_size[l_idx][1] * y + x) * _N_LOCAL_LINKS
                    + link_idx
                )
            else:
                off_tmp = (map_size[l_idx][1] * y + x) * (_N_LOCAL_LINKS + _N_CROSS_LINKS)
                link_off = offsets_defaults[l_idx][1] + off_tmp + link_idx
            neighbours.append([node_off, link_off, link_idx])
        link_idx += 1
    return neighbours


def _decode_image_by_join(node_scores, link_scores, node_threshold, link_threshold,
                          map_size, offsets_defaults):
    node_mask = node_scores[:, _POS_LABEL] >= node_threshold
    link_mask = link_scores[:, _POS_LABEL] >= link_threshold
    group_mask = np.zeros_like(node_mask, np.int32) - 1
    offsets_pos = np.where(node_mask)[0]

    def find_root(point):
        root = point
        update = False
        while group_mask[root] != -1:
            root = group_mask[root]
            update = True
        if update:
            group_mask[point] = root
        return root

    def join(p1, p2):
        r1, r2 = find_root(p1), find_root(p2)
        if r1 != r2:
            group_mask[r1] = r2

    for offsets in offsets_pos:
        l_idx, x, y = _get_coord(offsets, map_size, offsets_defaults)
        neighbours = _get_neighbours(l_idx, x, y, map_size, offsets_defaults)
        for noffsets in neighbours:
            if link_mask[noffsets[1]] and node_mask[noffsets[0]]:
                join(offsets, noffsets[0])

    root_map: dict[int, int] = {}
    mask = np.zeros_like(node_mask, dtype=np.int32)
    for point in offsets_pos:
        root = find_root(point)
        if root not in root_map:
            root_map[root] = len(root_map) + 1
        mask[point] = root_map[root]
    return mask


def _decode_image(node_scores, link_scores, reg, image_size, anchor_sizes,
                  node_thresh, link_thresh):
    map_size = []
    offsets_defaults = []
    off_node = 0
    off_link = 0
    for i in range(_N_DET_LAYERS):
        offsets_defaults.append([off_node, off_link])
        ms = image_size // (2 ** (2 + i))
        map_size.append(ms)
        off_node += int(ms[0] * ms[1])
        if i == 0:
            off_link += int(ms[0] * ms[1]) * _N_LOCAL_LINKS
        else:
            off_link += int(ms[0] * ms[1]) * (_N_LOCAL_LINKS + _N_CROSS_LINKS)

    group_all = _decode_image_by_join(
        node_scores, link_scores, node_thresh, link_thresh, map_size, offsets_defaults
    )
    group_all -= 1
    valid = np.where(group_all >= 0)[0]
    count = len(valid)
    segments = np.zeros((count, _OFFSET_DIM), dtype=np.float32)
    for i, offsets in enumerate(valid):
        l_idx, x, y = _get_coord(offsets, map_size, offsets_defaults)
        rs = anchor_sizes[l_idx]
        eps = 1e-6
        segments[i, 0] = reg[offsets, 0] * rs + (2 ** (2 + l_idx)) * (x + 0.5)
        segments[i, 1] = reg[offsets, 1] * rs + (2 ** (2 + l_idx)) * (y + 0.5)
        segments[i, 2] = np.exp(reg[offsets, 2]) * rs - eps
        segments[i, 3] = np.exp(reg[offsets, 3]) * rs - eps
        segments[i, 4] = reg[offsets, 4]
        segments[i, 5] = reg[offsets, 5]
    group_indices = group_all[valid]
    return segments, group_indices


def _combine_segs(segs: np.ndarray) -> np.ndarray:
    if len(segs) == 1:
        cx, cy, w, h = segs[0, :4]
        theta = np.arctan2(segs[0, 4], segs[0, 5])
        return np.array([cx, cy, w, h, theta])

    cxs, cys = segs[:, 0], segs[:, 1]
    theta_coss, theta_sins = segs[:, 4], segs[:, 5]
    bar_theta = np.arctan2(theta_sins.sum(), theta_coss.sum())
    k = np.tan(bar_theta)
    b = np.mean(cys - k * cxs)
    proj_xs = (k * cys + cxs - k * b) / (k ** 2 + 1)
    proj_ys = (k * k * cys + k * cxs + b) / (k ** 2 + 1)
    proj = np.stack((proj_xs, proj_ys), -1)

    max_dist = -1.0
    idx1 = idx2 = 0
    for i in range(len(proj)):
        for j in range(i + 1, len(proj)):
            dist = float(np.sqrt(np.sum((proj[i] - proj[j]) ** 2)))
            if dist > max_dist:
                idx1, idx2, max_dist = i, j, dist
    seg1, seg2 = segs[idx1], segs[idx2]
    bcx, bcy = (seg1[:2] + seg2[:2]) / 2.0
    bh = float(np.mean(segs[:, 3]))
    bw = max_dist + (seg1[2] + seg2[2]) / 2.0
    return np.array([bcx, bcy, bw, bh, bar_theta])


def _combine_segments(segments: np.ndarray, group_indices: np.ndarray) -> np.ndarray:
    if len(segments) == 0:
        return np.empty((0, 5), dtype=np.float32)
    num_groups = int(group_indices.max()) + 1
    rboxes = []
    for i in range(num_groups):
        group_segs = segments[group_indices == i]
        if len(group_segs) > 0:
            rboxes.append(_combine_segs(group_segs))
    return np.array(rboxes, dtype=np.float32) if rboxes else np.empty((0, 5), dtype=np.float32)


def _rboxes_to_polygons(rboxes: np.ndarray) -> np.ndarray:
    theta = rboxes[:, 4:5]
    cxcy = rboxes[:, :2]
    half_w = rboxes[:, 2:3] / 2.0
    half_h = rboxes[:, 3:4] / 2.0
    v1 = np.hstack([np.cos(theta) * half_w, np.sin(theta) * half_w])
    v2 = np.hstack([-np.sin(theta) * half_h, np.cos(theta) * half_h])
    p1 = cxcy - v1 - v2
    p2 = cxcy + v1 - v2
    p3 = cxcy + v1 + v2
    p4 = cxcy - v1 + v2
    return np.hstack([p1, p2, p3, p4])


def _point_dist(x1, y1, x2, y2):
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def _cal_width(box):
    pd1 = _point_dist(box[0], box[1], box[2], box[3])
    pd2 = _point_dist(box[4], box[5], box[6], box[7])
    return (pd1 + pd2) / 2


def _polygon2rbox(polygon):
    x1, x2, x3, x4 = polygon[0], polygon[2], polygon[4], polygon[6]
    y1, y2, y3, y4 = polygon[1], polygon[3], polygon[5], polygon[7]
    c_x = (x1 + x2 + x3 + x4) / 4
    c_y = (y1 + y2 + y3 + y4) / 4
    w1 = _point_dist(x1, y1, x2, y2)
    w2 = _point_dist(x3, y3, x4, y4)
    h1 = _point_line_dist(c_x, c_y, x1, y1, x2, y2)
    h2 = _point_line_dist(c_x, c_y, x3, y3, x4, y4)
    w = (w1 + w2) / 2
    theta1 = np.arctan2(y2 - y1, x2 - x1)
    theta2 = np.arctan2(y3 - y4, x3 - x4)
    theta = (theta1 + theta2) / 2.0
    return [c_x, c_y, w, h1 + h2, theta]


def _point_in_rbox(c, rbox):
    cx0, cy0 = c[0], c[1]
    cx1, cy1 = rbox[0], rbox[1]
    w, h, theta = rbox[2], rbox[3], rbox[4]
    dist_x = abs((cx1 - cx0) * np.cos(theta) + (cy1 - cy0) * np.sin(theta))
    dist_y = abs(-(cx1 - cx0) * np.sin(theta) + (cy1 - cy0) * np.cos(theta))
    return dist_x < w / 2.0 and dist_y < h / 2.0


def _point_line_dist(px, py, x1, y1, x2, y2):
    eps = 1e-6
    dx, dy = x2 - x1, y2 - y1
    div = np.sqrt(dx * dx + dy * dy) + eps
    return abs(px * dy - py * dx + x2 * y1 - y2 * x1) / div


def _nms(boxes):
    boxes = sorted(boxes, key=lambda x: -x[8])
    flag = [True] * len(boxes)
    for i, a in enumerate(boxes):
        if not flag[i]:
            continue
        for j, b in enumerate(boxes):
            if j <= i or not flag[j]:
                continue
            rbox_a = _polygon2rbox(a[:8])
            rbox_b = _polygon2rbox(b[:8])
            if _point_in_rbox(rbox_a[:2], rbox_b) or _point_in_rbox(rbox_b[:2], rbox_a):
                if a[8] > b[8]:
                    flag[j] = False
    return [box for i, box in enumerate(boxes) if flag[i]]


class SegLinkDetector:
    def __init__(
        self,
        model_path: str | Path,
        image_size: int = 512,
        num_threads: int = 0,
        providers: Sequence[str] = ("CPUExecutionProvider",),
        node_thresh: float = 0.4,
        link_thresh: float = 0.6,
    ):
        self.session = OrtSession(model_path, num_threads=num_threads, providers=providers)
        self.image_size = int(image_size)
        self.node_thresh = float(node_thresh)
        self.link_thresh = float(link_thresh)

    def _preprocess(self, img: np.ndarray) -> tuple[np.ndarray, int, int]:
        h, w = img.shape[:2]
        s = self.image_size
        side = max(h, w)
        padded = np.zeros((side, side, 3), dtype=np.float32)
        padded[:h, :w, :] = img
        resized = imageops.resize(padded.astype(np.uint8), (s, s)).astype(np.float32)
        resized = resized - _MEAN
        blob = np.expand_dims(resized, axis=0)
        return blob, h, w

    def _decode(self, all_maps: list[np.ndarray]) -> np.ndarray:
        image_size = np.array([self.image_size, self.image_size])
        all_nodes, all_links, all_reg = [], [], []
        for i in range(_N_DET_LAYERS):
            cls_maps = all_maps[i * 3][0]
            lnk_maps = all_maps[i * 3 + 1][0]
            reg_maps = all_maps[i * 3 + 2][0] * _OFFSET_VARIANCE

            cls_prob = _softmax(cls_maps.reshape(-1, 2))
            lnk_reshaped = lnk_maps.reshape(-1, 4)
            lnk_pos = _softmax(lnk_reshaped[:, :2])
            lnk_mut = _softmax(lnk_reshaped[:, 2:])
            lnk_prob = np.concatenate([lnk_pos, lnk_mut], axis=1)

            all_nodes.append(cls_prob)
            all_links.append(lnk_prob)
            all_reg.append(reg_maps.reshape(-1, _OFFSET_DIM))

        nodes_flat = np.concatenate(all_nodes, axis=0)
        links_flat = np.concatenate(all_links, axis=0)
        reg_flat = np.concatenate(all_reg, axis=0)

        segments, group_indices = _decode_image(
            nodes_flat, links_flat, reg_flat, image_size, _ANCHOR_SIZES,
            self.node_thresh, self.link_thresh,
        )
        return _combine_segments(segments, group_indices)

    def __call__(
        self,
        img: np.ndarray,
        *,
        node_thresh: float | None = None,
        link_thresh: float | None = None,
    ) -> tuple[np.ndarray | None, None]:
        self.node_thresh = self.node_thresh if node_thresh is None else float(node_thresh)
        self.link_thresh = self.link_thresh if link_thresh is None else float(link_thresh)

        ori_h, ori_w = img.shape[:2]
        blob, _, _ = self._preprocess(img)
        input_name = self.session.session.get_inputs()[0].name
        all_maps = self.session.session.run(None, {input_name: blob})
        rboxes = self._decode(all_maps)
        if len(rboxes) == 0:
            return None, None

        polygons = _rboxes_to_polygons(rboxes)
        scale = float(max(ori_h, ori_w)) / self.image_size
        polygons[:, ::2] = np.clip(np.round(polygons[:, ::2] * scale), 0, ori_w - 1)
        polygons[:, 1::2] = np.clip(np.round(polygons[:, 1::2] * scale), 0, ori_h - 1)
        polygons = polygons.astype(np.int32)

        dt_n9 = [o.tolist() + [_cal_width(o)] for o in polygons]
        dt_nms = _nms(dt_n9)
        if not dt_nms:
            return None, None
        boxes = np.array([o[:8] for o in dt_nms], dtype=np.float32).reshape(-1, 4, 2)
        boxes = self._sorted_boxes(boxes)
        return boxes, None

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
