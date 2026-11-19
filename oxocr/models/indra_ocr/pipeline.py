"""IndraModel：upright → LineNet 检测 → 单识别器 CTC（文本）；math 整图直识。

模型只管「像素 → 带 box 的文本条目」：box 使用送入图坐标，模型内部
deskew/orient 的几何改动自行逆映射（unrotate_box/unturn_box），核心前处理
的逆映射仍由核心负责。文本模型输出自然语言（后处理默认开）；math 模型是
整图公式识别器（无检测，整图保比缩放到编码器高 → 注意力解码，整图框，
同 pix2text MFR，后处理默认关）。不做排版、不读配置文件、不打印（铁律）。
"""

from __future__ import annotations

import json
import time
import unicodedata
from pathlib import Path

import numpy as np

from oxocr.errors import InferenceError
from oxocr.models.base import Capabilities, InferRequest, ModelOutput, OcrModel
from oxocr.models.indra_ocr.detector import LineDetector
from oxocr.models.indra_ocr.mathrec import MathReader
from oxocr.models.indra_ocr.orient import VERTICAL, unrotate_box, unturn_box, upright
from oxocr.models.indra_ocr.preproc import to_gray, whole_image_ink
from oxocr.models.indra_ocr.recognizer import LineRecognizer
from oxocr.models.indra_ocr.rtl import flip, has_rtl, is_rtl
from oxocr.runtime.session import OrtSession
from oxocr.types import Box, TextItem

__all__ = ["IndraModel"]


def _rect_box(b) -> Box:
    x0, y0, x1, y1 = (float(v) for v in b[:4])
    return Box.from_rect(x0, y0, x1 - x0, y1 - y0)


class IndraModel(OcrModel):
    family = "indra_ocr"
    capabilities = Capabilities(backend="onnxruntime")

    def __init__(self, spec, files, options):
        super().__init__(spec, files, options)
        roles = {f.role: f.filename for f in spec.files}
        self.math_mode = "enc" in roles
        # 组别由 rec 文件名推导：rec.onnx → base，rec_cjk.onnx → cjk …
        stem = Path(roles.get("rec", "rec")).stem
        self.group = stem[len("rec") :].lstrip("_") or "base"
        self.detector: LineDetector | None = None
        self.recognizer: LineRecognizer | None = None
        self.math: MathReader | None = None
        self.orient: OrtSession | None = None

    def load(self) -> None:
        p = self.params()
        threads = self.options.num_threads
        providers = self.options.providers
        if self.options.use_det and "det" in self.files:
            self.detector = LineDetector(
                self.files["det"], num_threads=threads, providers=providers
            )
        if "orient" in self.files:
            self.orient = OrtSession(
                self.files["orient"], num_threads=threads, providers=providers
            )
        if self.math_mode:
            if self.options.use_rec:
                self.math = MathReader(self.files, num_threads=threads, providers=providers)
        elif self.options.use_rec and "rec" in self.files:
            meta = json.loads(Path(self.files["meta"]).read_text(encoding="utf-8"))
            self.recognizer = LineRecognizer(
                self.files["rec"],
                meta,
                self.group,
                lm_path=self.files.get("lm"),
                batch_num=p["rec_batch_num"],
                num_threads=threads,
                providers=providers,
            )
        self._loaded = True

    def unload(self) -> None:
        self.detector = None
        self.recognizer = None
        self.math = None
        self.orient = None
        self._loaded = False

    def _rtl_decode(self, recs: list[tuple[str, float]]) -> list[tuple[str, float]]:
        """视觉序 → 逻辑序：页向由全页字母 R/AL vs L 计数判定（同示例 assemble）。"""
        letters = [c for text, _ in recs for c in text if c.isalpha()]
        n_rtl = sum(is_rtl(c) for c in letters)
        page_rtl = n_rtl > len(letters) - n_rtl
        out = []
        for text, conf in recs:
            t = unicodedata.normalize("NFC", " ".join(text.split()))
            if has_rtl(t):
                t = flip(t, base_rtl=page_rtl or None)
            out.append((t, conf))
        return out

    def _infer_math(self, image: np.ndarray, request: InferRequest) -> ModelOutput:
        """整图公式识别（同 MFR）：整图保比缩放 → 注意力解码 → 整图框单条目。"""
        if not request.use_rec:
            return ModelOutput(items=[], elapse={})
        assert self.math is not None
        p = self.params()
        p.update(request.params)
        min_conf = float(p["math_min_conf"])
        gray = to_gray(image)
        h, w = gray.shape
        t0 = time.perf_counter()
        crop = whole_image_ink(gray, self.math.height)
        text, conf = self.math.read([crop], batch=1, min_conf=min_conf)[0]
        elapse = {"rec": time.perf_counter() - t0}
        # 整图直识不拒收：仅空文本或低于可调 math_min_conf（默认 0）才无条目；
        # 低置信读法也输出，score=conf 供调用方自行过滤
        if not text.strip() or conf < min_conf:
            return ModelOutput(items=[], elapse=elapse)
        return ModelOutput(
            items=[
                TextItem(
                    text=text, score=conf, box=Box.from_rect(0.0, 0.0, float(w), float(h))
                )
            ],
            elapse=elapse,
        )

    def infer(self, image: np.ndarray, request: InferRequest) -> ModelOutput:
        if not self._loaded:
            raise InferenceError(f"模型 {self.spec.name} 尚未 load()")
        if self.math_mode:
            return self._infer_math(image, request)
        p = self.params()
        p.update(request.params)
        gray = to_gray(image)
        h0, w0 = gray.shape
        elapse: dict[str, float] = {}
        meta: dict = {}

        k, skew = 0, 0.0
        det_scores: list[float] | None = None
        if request.use_det and self.detector is not None:
            t0 = time.perf_counter()
            if p["deskew"] or (p["auto_orient"] and self.orient is not None):
                t1 = time.perf_counter()
                gray, k, skew = upright(
                    gray,
                    self.orient,
                    deskew=bool(p["deskew"]),
                    auto_orient=bool(p["auto_orient"]),
                    allow_vertical=self.group == "cjk",
                )
                meta["orient"] = {
                    "rotation": 0 if k == VERTICAL else 90 * k,
                    "skew": round(skew, 2),
                    "vertical": k == VERTICAL,
                    "seconds": round(time.perf_counter() - t1, 4),
                }
            boxes = self.detector(
                gray,
                long_side=int(p["det_long_side"]),
                thresh=float(p["det_thresh"]),
                adaptive=bool(p["adaptive"]),
            )
            elapse["det"] = time.perf_counter() - t0
            if len(boxes) == 0:
                return ModelOutput(items=[], elapse=elapse, meta=meta)
            det_scores = boxes[:, 4].tolist()
            boxes = boxes[:, :4]
        else:
            boxes = np.array([[0, 0, w0, h0]], np.float32)

        items: list[TextItem] = []
        if request.use_rec and self.recognizer is not None:
            t0 = time.perf_counter()
            recs = self.recognizer(gray, boxes, use_lm=bool(p["use_lm"]))
            elapse["rec"] = time.perf_counter() - t0
            if self.group == "rtl":
                recs = self._rtl_decode(recs)
            for i, (text, conf) in enumerate(recs):
                if not text.strip() or conf < request.text_score:
                    continue
                m = {"det_score": det_scores[i]} if det_scores is not None else {}
                items.append(TextItem(text=text, score=conf, box=_rect_box(boxes[i]), meta=m))
        elif not request.use_rec:
            for i, b in enumerate(boxes):
                score = 1.0 if det_scores is None else det_scores[i]
                items.append(TextItem(text="", score=score, box=_rect_box(b)))

        if items and (skew or k):
            # 逆 deskew/orient 回送入图坐标（核心再逆前处理）
            hw = (gray.shape[0], gray.shape[1])
            for it in items:
                x0, y0, w, h = it.box.as_rect()
                b = [x0, y0, x0 + w, y0 + h]
                if skew:
                    b = unrotate_box(b, skew, (hw[1], hw[0]))
                if k:
                    b = unturn_box(b, k, hw)
                it.box = _rect_box(b)
        return ModelOutput(items=items, elapse=elapse, meta=meta)
