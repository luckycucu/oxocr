"""DuguangModel：det → 裁剪 → rec 编排，三档共用一个实现类。

模型只管「像素 → 带 box 的文本条目」：box 使用送入图坐标，
前处理与坐标逆映射由核心负责。
"""
from __future__ import annotations

import time

import numpy as np

from oxocr.errors import InferenceError
from oxocr.models.base import Capabilities, InferRequest, ModelOutput, OcrModel
from oxocr.models.duguang_ocr.detector import DbDetector
from oxocr.models.duguang_ocr.recognizer import TextRecognizer
from oxocr.models.duguang_ocr.seglink import SegLinkDetector
from oxocr.runtime.imageops import get_rotate_crop_image
from oxocr.types import Box, TextItem

__all__ = ["DuguangModel"]


class DuguangModel(OcrModel):
    family = "duguang_ocr"
    capabilities = Capabilities(backend="onnxruntime")

    def __init__(self, spec, files, options):
        super().__init__(spec, files, options)
        self.detector: DbDetector | SegLinkDetector | None = None
        self.recognizer: TextRecognizer | None = None

    def _validate_det_size(self, det: DbDetector | SegLinkDetector) -> None:
        inputs = det.session.session.get_inputs()
        shape = inputs[0].shape
        if isinstance(det, SegLinkDetector):
            hw = shape[1:3]
        else:
            hw = shape[2:4]
        for dim in hw:
            if isinstance(dim, int) and dim != det.image_size:
                raise InferenceError(
                    f"模型 {self.spec.name} 的 det 权重期望输入尺寸 {dim}，"
                    f"但 det_image_size={det.image_size}；请使用匹配的权重文件"
                )

    def load(self) -> None:
        p = self.params()
        threads = self.options.num_threads
        providers = self.options.providers
        if self.options.use_det:
            if self.spec.name == "duguang_ocr_max":
                self.detector = SegLinkDetector(
                    self.files["det"],
                    image_size=p["det_image_size"],
                    num_threads=threads,
                    providers=providers,
                    node_thresh=p["det_node_thresh"],
                    link_thresh=p["det_link_thresh"],
                )
            else:
                self.detector = DbDetector(
                    self.files["det"],
                    image_size=p["det_image_size"],
                    num_threads=threads,
                    providers=providers,
                    thresh=p["det_thresh"],
                    box_thresh=p["det_box_thresh"],
                    unclip_ratio=p["det_unclip_ratio"],
                )
            self._validate_det_size(self.detector)
        if self.options.use_rec:
            self.recognizer = TextRecognizer(
                self.files["rec"],
                self.files["vocab"],
                batch_num=p["rec_batch_num"],
                num_threads=threads,
                providers=providers,
            )
        self._loaded = True

    def unload(self) -> None:
        self.detector = None
        self.recognizer = None
        self._loaded = False

    def infer(self, image: np.ndarray, request: InferRequest) -> ModelOutput:
        if not self._loaded:
            raise InferenceError(f"模型 {self.spec.name} 尚未 load()")
        p = self.params()
        p.update(request.params)
        elapse: dict[str, float] = {}
        h, w = image.shape[:2]
        is_seglink = self.spec.name == "duguang_ocr_max"

        det_boxes: np.ndarray | None = None
        det_scores: list[float] | None = None
        if request.use_det and self.detector is not None:
            t0 = time.perf_counter()
            if isinstance(self.detector, SegLinkDetector):
                det_boxes, det_scores = self.detector(
                    image,
                    node_thresh=p["det_node_thresh"],
                    link_thresh=p["det_link_thresh"],
                )
            else:
                det_boxes, det_scores = self.detector(
                    image,
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
            det_scores = None if is_seglink else [1.0]

        items: list[TextItem] = []
        if request.use_rec and self.recognizer is not None and crops:
            t0 = time.perf_counter()
            rec_results = self.recognizer(crops)
            elapse["rec"] = time.perf_counter() - t0
            for i, (text, score) in enumerate(rec_results):
                if not text.strip() or score < request.text_score:
                    continue
                meta = {}
                if det_scores is not None:
                    meta["det_score"] = float(det_scores[i])
                items.append(
                    TextItem(
                        text=text,
                        score=score,
                        box=Box.from_points(det_boxes[i]),
                        meta=meta,
                    )
                )
        elif not request.use_rec:
            for i in range(len(det_boxes)):
                score = 1.0 if det_scores is None else float(det_scores[i])
                items.append(
                    TextItem(
                        text="",
                        score=score,
                        box=Box.from_points(det_boxes[i]),
                    )
                )
        return ModelOutput(items=items, elapse=elapse)
