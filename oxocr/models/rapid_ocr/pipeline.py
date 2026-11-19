"""PpOcrModel：det → cls → rec 编排，v3~v6 共用一个实现类。

模型只管「像素 → 带 box 的文本条目」：box 使用送入图坐标，
前处理与坐标逆映射由核心负责。
"""
from __future__ import annotations

import time

import numpy as np

from oxocr.errors import InferenceError
from oxocr.models.base import Capabilities, InferRequest, ModelOutput, OcrModel
from oxocr.models.rapid_ocr.classifier import TextClassifier
from oxocr.models.rapid_ocr.detector import TextDetector
from oxocr.models.rapid_ocr.recognizer import TextRecognizer
from oxocr.models.rapid_ocr.specs import cls_variant
from oxocr.runtime.imageops import get_rotate_crop_image
from oxocr.types import Box, TextItem

__all__ = ["PpOcrModel"]


class PpOcrModel(OcrModel):
    family = "ppocr"
    capabilities = Capabilities(angle_classification=True, backend="onnxruntime")

    def __init__(self, spec, files, options):
        super().__init__(spec, files, options)
        self.detector: TextDetector | None = None
        self.classifier: TextClassifier | None = None
        self.recognizer: TextRecognizer | None = None

    def load(self) -> None:
        p = self.params()
        threads = self.options.num_threads
        providers = self.options.providers
        if self.options.use_det:
            self.detector = TextDetector(
                self.files["det"],
                num_threads=threads,
                providers=providers,
                limit_side_len=p["det_limit_side_len"],
                thresh=p["det_thresh"],
                box_thresh=p["det_box_thresh"],
                unclip_ratio=p["det_unclip_ratio"],
            )
        if self.options.use_cls and self.capabilities.angle_classification and "cls" in self.files:
            # 形状取自权重本身：textline_ori 的 80×160 是静态维度，不能沿用 48×192
            variant = cls_variant(p["cls_model"], f"模型 {self.spec.name}")
            self.classifier = TextClassifier(
                self.files["cls"],
                cls_thresh=p["cls_thresh"],
                cls_image_shape=variant.image_shape,
                batch_num=p["cls_batch_num"],
                num_threads=threads,
                providers=providers,
            )
        if self.options.use_rec:
            self.recognizer = TextRecognizer(
                self.files["rec"],
                keys=self.files.get("keys"),
                rec_image_shape=p["rec_image_shape"],
                rec_batch_num=p["rec_batch_num"],
                num_threads=threads,
                providers=providers,
            )
        self._loaded = True

    def unload(self) -> None:
        self.detector = None
        self.classifier = None
        self.recognizer = None
        self._loaded = False

    def infer(self, image: np.ndarray, request: InferRequest) -> ModelOutput:
        if not self._loaded:
            raise InferenceError(f"模型 {self.spec.name} 尚未 load()")
        p = self.params()
        p.update(request.params)
        elapse: dict[str, float] = {}
        h, w = image.shape[:2]

        det_boxes: np.ndarray | None = None
        det_scores: list[float] = []
        if request.use_det and self.detector is not None:
            t0 = time.perf_counter()
            det_boxes, det_scores = self.detector(
                image,
                limit_side_len=p["det_limit_side_len"],
                thresh=p["det_thresh"],
                box_thresh=p["det_box_thresh"],
                unclip_ratio=p["det_unclip_ratio"],
            )
            elapse["det"] = time.perf_counter() - t0
            if det_boxes is None or len(det_boxes) == 0:
                return ModelOutput(items=[], elapse=elapse)
            crops = [get_rotate_crop_image(image, box) for box in det_boxes]
        else:
            crops = [image]
            det_boxes = np.array([[[0, 0], [w, 0], [w, h], [0, h]]], dtype=np.float32)
            det_scores = [1.0]

        angles = [0.0] * len(crops)
        if request.use_cls and self.classifier is not None and crops:
            t0 = time.perf_counter()
            crops, angles = self.classifier(crops)
            elapse["cls"] = time.perf_counter() - t0

        items: list[TextItem] = []
        if request.use_rec and self.recognizer is not None and crops:
            t0 = time.perf_counter()
            rec_results = self.recognizer(crops)
            elapse["rec"] = time.perf_counter() - t0
            for i, (text, score) in enumerate(rec_results):
                if not text.strip() or score < request.text_score:
                    continue
                items.append(
                    TextItem(
                        text=text,
                        score=score,
                        box=Box.from_points(det_boxes[i]),
                        angle=angles[i],
                        meta={"det_score": float(det_scores[i])},
                    )
                )
        elif not request.use_rec:
            # 仅检测：保留 box，文本为空
            for i in range(len(det_boxes)):
                items.append(
                    TextItem(
                        text="",
                        score=float(det_scores[i]),
                        box=Box.from_points(det_boxes[i]),
                        angle=angles[i],
                    )
                )
        return ModelOutput(items=items, elapse=elapse)
