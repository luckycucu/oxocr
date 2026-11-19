"""OpenThaiModel：det → 整页方向/deskew → 空白切分 → 批识别 → 过滤 → 阅读序 → TextItem。

模型只管「像素 → 带 box 的文本条目」：内部 rot90/deskew 用 tf 矩阵累积并自行逆映射，
box 使用送入图坐标；核心前处理的逆映射仍由核心负责。整页方向（侧向/倒置投票、
deskew）是模型基准成绩的来源，默认开启（orientation=False 可关）。
TextItem.angle 恒 0，页级 rotation/text_angle 写 ModelOutput.meta。
"""

from __future__ import annotations

import math
import time

import numpy as np

from oxocr.errors import InferenceError
from oxocr.models.base import Capabilities, InferRequest, ModelOutput, OcrModel
from oxocr.models.openthai_ocr.detector import DbDetector
from oxocr.models.openthai_ocr.geometry import crop_line, flatten, page_skew, split_one, to_gray, vertical
from oxocr.models.openthai_ocr.order import fragments, reading_order
from oxocr.models.openthai_ocr.recognizer import CtcRecognizer
from oxocr.models.openthai_ocr.text import canonical, fix_scripts, invalid_share
from oxocr.runtime import imageops
from oxocr.types import Box, TextItem

__all__ = ["OpenThaiModel"]

_ORI_SAMPLE = 8  # 方向投票取样行数（策略绑定常量，不暴露）
_FLIP_BELOW = 0.8  # 行置信度中值低于此才试倒置
_FLIP_MARGIN = 0.15  # 倒置须明显更优才转正


def _rot90_matrix(k: int, shape: tuple[int, int]) -> np.ndarray:
    """3x3 map from an image of `shape` to np.rot90(image, k) coordinates."""
    h, w = shape
    if k == 1:  # counter-clockwise
        return np.array([[0, 1, 0], [-1, 0, w - 1], [0, 0, 1]], np.float64)
    if k == -1:  # clockwise
        return np.array([[0, -1, h - 1], [1, 0, 0], [0, 0, 1]], np.float64)
    return np.array([[-1, 0, w - 1], [0, -1, h - 1], [0, 0, 1]], np.float64)  # 180


def _apply(m: np.ndarray, box: np.ndarray) -> np.ndarray:
    pts = np.hstack([np.asarray(box, np.float64), np.ones((len(box), 1))]) @ m.T
    return pts[:, :2]


def _trim_line_margin(img: np.ndarray) -> tuple[np.ndarray, int, int]:
    """use_det=False 的行裁剪图可能带核心 pad_border 补的白边：按墨迹 bbox
    修掉空白边，恢复行裁剪的紧宽高比（否则整图缩到 H=48 时文字被压扁）。
    返回 (裁剪图, x偏移, y偏移)，偏移由管线压入 tf 统一逆映射。"""
    ink = flatten(to_gray(img)) < 128
    ys, xs = np.where(ink)
    if len(xs) == 0:
        return img, 0, 0
    pad = 2
    y0, y1 = max(0, int(ys.min()) - pad), int(ys.max()) + 1 + pad
    x0, x1 = max(0, int(xs.min()) - pad), int(xs.max()) + 1 + pad
    return img[y0:y1, x0:x1], x0, y0


def _deskew(img: np.ndarray, angle: float) -> tuple[np.ndarray, np.ndarray]:
    """Rotate so lines leaning by `angle` degrees run level; canvas grown to
    fit, new area filled with the page's own border colour."""
    h, w = img.shape[:2]
    rad = math.radians(angle)
    c, s = math.cos(rad), math.sin(rad)
    m = np.array([
        [c, s, (1 - c) * w / 2 - s * h / 2],
        [-s, c, s * w / 2 + (1 - c) * h / 2],
    ])
    nw = int(round(h * abs(s) + w * abs(c)))
    nh = int(round(h * abs(c) + w * abs(s)))
    m[0, 2] += nw / 2 - w / 2
    m[1, 2] += nh / 2 - h / 2
    edge = np.concatenate([img[0], img[-1], img[:, 0], img[:, -1]], 0)
    fill = tuple(int(v) for v in np.median(edge, axis=0)) if img.ndim == 3 else int(np.median(edge))
    out = imageops.rotate(img, angle, fill=fill, interpolation="bicubic", expand=True)
    return out, np.vstack([m, [0, 0, 1]])


class OpenThaiModel(OcrModel):
    family = "openthai_ocr"
    capabilities = Capabilities(backend="onnxruntime")

    def __init__(self, spec, files, options):
        super().__init__(spec, files, options)
        self.detector: DbDetector | None = None
        self.recognizer: CtcRecognizer | None = None

    def load(self) -> None:
        p = self.params()
        threads = self.options.num_threads
        providers = self.options.providers
        if self.options.use_det:
            self.detector = DbDetector(
                self.files["det"],
                num_threads=threads,
                providers=providers,
                limit_side_len=p["det_limit_side_len"],
                thresh=p["det_thresh"],
                box_thresh=p["det_box_thresh"],
                unclip_ratio=p["det_unclip_ratio"],
            )
        if self.options.use_rec:
            self.recognizer = CtcRecognizer(
                self.files["rec"],
                self.files["dict"],
                batch_num=p["rec_batch_num"],
                num_threads=threads,
                providers=providers,
            )
        self._loaded = True

    def unload(self) -> None:
        self.detector = None
        self.recognizer = None
        self._loaded = False

    def _detect(self, image: np.ndarray, p: dict) -> tuple[list[np.ndarray], list[float]]:
        assert self.detector is not None
        return self.detector(
            image,
            thresh=p["det_thresh"],
            box_thresh=p["det_box_thresh"],
            unclip_ratio=p["det_unclip_ratio"],
            limit_side_len=p["det_limit_side_len"],
        )

    def _read_boxes(self, img: np.ndarray, boxes: list[np.ndarray], p: dict) -> list[tuple[str, float]]:
        """Recognise every box. A box taller than wide is either a vertical run
        of text or one tall character, so it is also read turned both ways and
        the most confident reading kept."""
        assert self.recognizer is not None
        raw = [crop_line(img, b) for b in boxes]
        tall = [n for n, c in enumerate(raw) if c.shape[0] >= 1.5 * c.shape[1]]
        crops = raw + [np.ascontiguousarray(np.rot90(raw[n], k)) for k in (1, -1) for n in tall]
        res = self.recognizer(crops, color=p["rec_color"])
        out = res[: len(raw)]
        for j, n in enumerate(tall):
            for alt in (res[len(raw) + j], res[len(raw) + len(tall) + j]):
                if alt[1] > out[n][1]:
                    out[n] = alt
        return out

    def _page(self, img: np.ndarray, boxes: list[np.ndarray], p: dict) -> dict:
        """Recognition, then reading order (the column test looks at the text).

        Each detected line is first cut at wide blank stretches (split_one).
        The pieces replace the line only if every piece survives drop_below
        AND together they read at least 90% of the characters the whole line
        does; otherwise the line is read whole."""
        groups = [split_one(img, b, p["split_gap"]) for b in boxes]
        changed = [len(g) > 1 or g[0] is not b for b, g in zip(boxes, groups)]
        pieces = [piece for g in groups for piece in g]
        whole = [b for b, ch in zip(boxes, changed) if ch]
        res = self._read_boxes(img, pieces + whole, p)
        final: list[np.ndarray] = []
        out: list[tuple[str, float]] = []
        k, e = 0, len(pieces)
        for b, g, ch in zip(boxes, groups, changed):
            r = res[k : k + len(g)]
            k += len(g)
            if ch:
                alt = res[e]
                e += 1
                n_pieces = sum(len(x[0].replace(" ", "")) for x in r)
                if (min(x[1] for x in r) < p["drop_below"]
                        or n_pieces < 0.9 * len(alt[0].replace(" ", ""))):
                    final.append(b)
                    out.append(alt)
                    continue
            final += g
            out += r
        angle = page_skew(final)
        rows = reading_order(final, angle, texts=[x[0] for x in out])
        return {"rows": rows, "angle": angle, "boxes": final,
                "rec": {i: (x[0], x[1]) for i, x in enumerate(out)}}

    def _sideways(self, img: np.ndarray, boxes: list[np.ndarray], p: dict) -> int:
        """np.rot90 k that turns a sideways page upright, 0 if it is not
        sideways: most text lines run vertically. The direction is whichever
        way the longest vertical lines read with more confidence."""
        assert self.recognizer is not None
        votes = [vertical(b, 2.0) for b in boxes]
        vert = [b for b, v in zip(boxes, votes) if v == 1]
        if len(vert) < 3 or len(vert) <= votes.count(-1):
            return 0
        vert.sort(key=lambda b: -np.linalg.norm(b[0] - b[3]))
        raw = [crop_line(img, b) for b in vert[:_ORI_SAMPLE]]
        res = self.recognizer(
            [np.ascontiguousarray(np.rot90(c, k)) for k in (1, -1) for c in raw],
            color=p["rec_color"],
        )
        ccw = sum(c for _, c in res[: len(raw)])
        cw = sum(c for _, c in res[len(raw) :])
        return 1 if ccw > cw else -1

    def _upside_down(self, img: np.ndarray, page: dict, p: dict) -> bool:
        """Only for a page that reads badly: do its longest lines read better
        turned 180 degrees?"""
        assert self.recognizer is not None
        confs = [c for t, c in page["rec"].values() if len(t) >= 2]
        if len(confs) < 3 or float(np.median(confs)) >= _FLIP_BELOW:
            return False
        boxes = page["boxes"]
        idx = sorted(
            (i for i in page["rec"] if vertical(boxes[i], 2.0) == -1),
            key=lambda i: -np.linalg.norm(boxes[i][0] - boxes[i][1]),
        )[:_ORI_SAMPLE]
        if len(idx) < 3:
            return False
        up = float(np.mean([page["rec"][i][1] for i in idx]))
        flipped = self.recognizer(
            [np.ascontiguousarray(crop_line(img, boxes[i])[::-1, ::-1]) for i in idx],
            color=p["rec_color"],
        )
        return float(np.mean([c for _, c in flipped])) > up + _FLIP_MARGIN

    def infer(self, image: np.ndarray, request: InferRequest) -> ModelOutput:
        if not self._loaded:
            raise InferenceError(f"模型 {self.spec.name} 尚未 load()")
        p = self.params()
        p.update(request.params)
        elapse: dict[str, float] = {}
        h0, w0 = image.shape[:2]
        tf = np.eye(3)  # 送入图 → 工作图
        rotation = 0
        straightened = 0.0
        det_scores: list[float] | None = None

        if request.use_det and self.detector is not None:
            t0 = time.perf_counter()
            boxes, det_scores = self._detect(image, p)
            if not boxes and h0 <= 200 and w0 >= 1.5 * h0:
                # Nothing found in a small, line-shaped image: read it as one line.
                boxes = [np.array([[0, 0], [w0 - 1, 0], [w0 - 1, h0 - 1], [0, h0 - 1]], np.float32)]
                det_scores = None
            if p["orientation"] and self.recognizer is not None:
                k = self._sideways(image, boxes, p)
                if k:
                    tf = _rot90_matrix(k, image.shape[:2]) @ tf
                    image = np.ascontiguousarray(np.rot90(image, k))
                    boxes, det_scores = self._detect(image, p)
                    rotation = 90 if k == -1 else 270
                skew = page_skew(boxes)
                if abs(skew) >= p["deskew_above"]:
                    image, m = _deskew(image, skew)
                    tf = m @ tf
                    boxes, det_scores = self._detect(image, p)
                    straightened = skew
            elapse["det"] = time.perf_counter() - t0
        else:
            image, tx, ty = _trim_line_margin(image)
            th, tw = image.shape[:2]
            boxes = [np.array([[0, 0], [tw, 0], [tw, th], [0, th]], np.float32)]
            tf = np.array([[1, 0, -tx], [0, 1, -ty], [0, 0, 1]], np.float64) @ tf

        if not request.use_rec or self.recognizer is None:
            items = []
            for ri, row in enumerate(reading_order(boxes)):
                for i in row:
                    score = 1.0 if det_scores is None else float(det_scores[i])
                    items.append(TextItem(
                        text="", score=score,
                        box=Box.from_points(_apply(np.linalg.inv(tf), boxes[i])),
                        meta={"row": ri},
                    ))
            return ModelOutput(items=items, elapse=elapse,
                               meta={"rotation": rotation, "text_angle": round(page_skew(boxes) + straightened, 2)})

        t0 = time.perf_counter()
        page = self._page(image, boxes, p)
        if p["orientation"] and not rotation and self.detector is not None \
                and self._upside_down(image, page, p):
            tf = _rot90_matrix(2, image.shape[:2]) @ tf
            image = np.ascontiguousarray(image[::-1, ::-1])
            rotation = 180
            boxes, det_scores = self._detect(image, p)
            page = self._page(image, boxes, p)
        elapse["rec"] = time.perf_counter() - t0

        back = np.linalg.inv(tf)
        final = page["boxes"]
        frag = fragments(final, page["rec"])
        drop = max(float(request.text_score), float(p["drop_below"]))
        items: list[TextItem] = []
        dropped: list[dict] = []
        for ri, row in enumerate(page["rows"]):
            for i in row:
                text, conf = page["rec"][i]
                text = fix_scripts(canonical(text))
                if i in frag:
                    continue
                reason = None
                if not text:
                    reason = "empty"
                elif conf < drop:
                    reason = "low_confidence"
                elif len(text.replace(" ", "")) < p["min_chars"]:
                    reason = "too_short"
                elif conf < p["junk_below"] and invalid_share(text) > p["junk_invalid"]:
                    reason = "invalid_thai"
                box = _apply(back, final[i])
                if reason:
                    dropped.append({
                        "text": text, "confidence": round(float(conf), 4), "reason": reason, "row": ri,
                        "box": [[round(float(x), 1), round(float(y), 1)] for x, y in box],
                    })
                    continue
                items.append(TextItem(text=text, score=float(conf),
                                      box=Box.from_points(box), meta={"row": ri}))
        return ModelOutput(
            items=items, elapse=elapse,
            meta={"rotation": rotation,
                  "text_angle": round(page["angle"] + straightened, 2),
                  "dropped": dropped},
        )