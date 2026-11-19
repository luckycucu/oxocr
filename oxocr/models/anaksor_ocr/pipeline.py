"""AnaksorModel：det → 方向变体裁剪 → 窗口批识别 → 字簇拼接 → TextItem。

模型只管「像素 → 带 box 的文本条目」：box 使用送入图坐标（原始 det quad，未扩边），
前处理与坐标逆映射由核心负责。方向变体（竖框 up/cw/ccw、横框可选 180°）取优，
旋转 -0.05 惩罚仅用于方向比较；TextItem.angle 恒 0，方向写 meta["rotation"]。
"""

from __future__ import annotations

import time

import numpy as np

from oxocr.errors import InferenceError
from oxocr.models.anaksor_ocr.detector import DbDetector
from oxocr.models.anaksor_ocr.geometry import (
    drop_contained,
    expand_quad,
    order_quad,
    quad_size,
    warp_crop,
)
from oxocr.models.anaksor_ocr.longline import line_windows, stitch
from oxocr.models.anaksor_ocr.recognizer import SvtrRecognizer
from oxocr.models.anaksor_ocr.text import normalize
from oxocr.models.base import Capabilities, InferRequest, ModelOutput, OcrModel
from oxocr.types import Box, TextItem

__all__ = ["AnaksorModel"]


class AnaksorModel(OcrModel):
    family = "anaksor_ocr"
    capabilities = Capabilities(backend="onnxruntime")

    def __init__(self, spec, files, options):
        super().__init__(spec, files, options)
        self.detector: DbDetector | None = None
        self.recognizer: SvtrRecognizer | None = None

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
                min_side_len=p["det_min_side_len"],
                thresh=p["det_thresh"],
                box_thresh=p["det_box_thresh"],
                unclip_ratio=p["det_unclip_ratio"],
            )
        if self.options.use_rec:
            self.recognizer = SvtrRecognizer(
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

    def _recognize_quads(
        self, image: np.ndarray, quads: list[np.ndarray], p: dict, *, expand: bool = True
    ) -> dict:
        """{行号: (text, score, rotation)}，每框取最优方向。

        expand=False（use_det=False / recognize_crop）：用户给的已是行裁剪图，
        不再扩边 warp，整图直接作为一个 crop（同 duguang 语义）。
        """
        assert self.recognizer is not None
        jobs = []  # (行号, 方向, 窗口, stitches)
        for li, q in enumerate(quads):
            if expand:
                w, h = quad_size(q)
                vertical = h > 1.5 * w
                short = min(w, h)
                if vertical:
                    qq = expand_quad(q, short * p["crop_pad_y"], short * p["crop_pad_x"])
                else:
                    qq = expand_quad(q, short * p["crop_pad_x"], short * p["crop_pad_y"])
                crop = warp_crop(image, qq)
            else:
                crop = image
                vertical = crop.shape[0] > 1.5 * crop.shape[1]
            if crop.shape[0] < 6 or crop.shape[1] < 6:
                continue
            variants = [("up", crop)]
            if vertical and p["try_vertical"]:
                # 带上下标的高棉字簇本身竖长（如 ស្ត្រី），up 始终候选
                variants += [("cw", np.rot90(crop, -1)), ("ccw", np.rot90(crop, 1))]
            elif p["try_180"]:
                variants.append(("180", np.rot90(crop, 2)))
            for name, v in variants:
                wins, stitches = line_windows(v, self.recognizer.max_ratio)
                jobs.append((li, name, wins, stitches))
        pieces = [piece for _, _, wins, _ in jobs for piece, _, _ in wins]
        recs = iter(self.recognizer(pieces) if pieces else [])
        out: dict[int, tuple[tuple[str, float, str], float]] = {}
        for li, name, wins, stitches in jobs:
            rs = [next(recs) for _ in wins]
            if len(wins) == 1:
                text, score = normalize(rs[0]["text"]), float(rs[0]["score"])
            else:
                text, score = stitch(wins, stitches, rs)
            key = score if name == "up" else score - 0.05  # 旋转须明显更优才胜出
            if li not in out or key > out[li][1]:
                out[li] = ((text, score, name), key)
        return {li: v for li, (v, _) in out.items()}

    def infer(self, image: np.ndarray, request: InferRequest) -> ModelOutput:
        if not self._loaded:
            raise InferenceError(f"模型 {self.spec.name} 尚未 load()")
        p = self.params()
        p.update(request.params)
        elapse: dict[str, float] = {}
        h, w = image.shape[:2]

        det_scores: list[float] | None = None
        if request.use_det and self.detector is not None:
            t0 = time.perf_counter()
            boxes, det_scores = self.detector(
                image,
                thresh=p["det_thresh"],
                box_thresh=p["det_box_thresh"],
                unclip_ratio=p["det_unclip_ratio"],
                limit_side_len=p["det_limit_side_len"],
                min_side_len=p["det_min_side_len"],
            )
            elapse["det"] = time.perf_counter() - t0
            if boxes is None or len(boxes) == 0:
                return ModelOutput(items=[], elapse=elapse)
            quads = [order_quad(b) for b in boxes]
            keep = [i for i, q in enumerate(quads) if min(quad_size(q)) >= 5]
            keep = [keep[k] for k in drop_contained([quads[i] for i in keep])]
            quads = [quads[i] for i in keep]
            det_scores = [det_scores[i] for i in keep]
        else:
            quads = [np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)]

        items: list[TextItem] = []
        if request.use_rec and self.recognizer is not None:
            t0 = time.perf_counter()
            has_det = request.use_det and self.detector is not None
            recs = self._recognize_quads(image, quads, p, expand=has_det)
            elapse["rec"] = time.perf_counter() - t0
            for li in sorted(recs):
                text, score, rotation = recs[li]
                if not text.strip() or score < request.text_score:
                    continue
                meta: dict = {"rotation": rotation}
                if det_scores is not None:
                    meta["det_score"] = float(det_scores[li])
                items.append(
                    TextItem(
                        text=text,
                        score=score,
                        box=Box.from_points(quads[li]),
                        meta=meta,
                    )
                )
        elif not request.use_rec:
            for i, q in enumerate(quads):
                score = 1.0 if det_scores is None else float(det_scores[i])
                items.append(TextItem(text="", score=score, box=Box.from_points(q)))
        return ModelOutput(items=items, elapse=elapse)
