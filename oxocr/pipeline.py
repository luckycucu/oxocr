"""编排：load → preprocess → execute → coord-map → postprocess → render。

调用方（门面 / OcrEngine）在调用前已把 mode / output / 快捷覆盖解析进 config，
本模块只负责按 config 组装各阶段并执行。Executor 由调用方创建并持有（常驻复用）。
"""
from __future__ import annotations

import time
from collections.abc import Sequence

import numpy as np

from oxocr.config import OcrConfig
from oxocr.errors import InvalidImageError
from oxocr.models.base import InferRequest, ModelOutput
from oxocr.postprocess.base import PostprocessPipeline
from oxocr.preprocess.base import Affine, PreContext, PreprocessPipeline
from oxocr.runtime.images import load_image
from oxocr.types import DebugInfo, Document, Elapse, OcrResult, TextItem


def build_preprocess_pipeline(config: OcrConfig, instances: Sequence | None) -> PreprocessPipeline:
    steps = list(instances) if instances is not None else list(config.preprocess.steps)
    return PreprocessPipeline(steps, options=config.preprocess.options, on_error=config.preprocess.on_error)


def build_postprocess_pipeline(config: OcrConfig, modifiers: Sequence | None) -> PostprocessPipeline:
    return PostprocessPipeline(
        config.postprocess, modifiers=list(modifiers) if modifiers is not None else None
    )


def _identity_ctx(image: np.ndarray) -> PreContext:
    h, w = image.shape[:2]
    return PreContext(image=image, original_size=(w, h), transform=Affine.identity())


def _build_request(config: OcrConfig, spec_defaults: dict) -> InferRequest:
    mo = config.model_options
    params = dict(spec_defaults)
    params.update(mo.params)
    return InferRequest(
        text_score=config.infer.text_score,
        use_det=mo.use_det,
        use_cls=mo.use_cls,
        use_rec=mo.use_rec,
        params=params,
    )


def _inverse_map(items: list[TextItem], transform: Affine) -> None:
    inv = transform.inverse
    for item in items:
        item.box = inv.apply_to_box(item.box)
        if item.chars:
            for cb in item.chars:
                cb.box = inv.apply_to_box(cb.box)


class OcrPipeline:
    def __init__(self, config: OcrConfig, executor, spec_defaults: dict, *,
                 pre_instances=None, post_instances=None, post_default_off: bool = False,
                 pre_default_off: bool = False, pad_default_off: bool = False):
        self.config = config
        self.executor = executor
        self.spec_defaults = spec_defaults
        self.pre_instances = pre_instances
        self.post_instances = post_instances
        self.post_default_off = post_default_off
        self.pre_default_off = pre_default_off
        self.pad_default_off = pad_default_off

    def run(self, image_input) -> OcrResult:
        cfg = self.config
        if self.post_default_off:
            # own_postprocess 模型（公式识别等）：默认跳过后处理，输出原样进结果；
            # 调用处显式传 postprocess 时门面不会置此标志
            cfg = cfg.merge({"postprocess": {"enabled": False}})
        if self.pre_default_off:
            # own_preprocess 模型（如 rapid_latex_ocr 自带对比度拉伸/裁剪/反色）：默认跳过
            # 核心前处理，避免 pad_border 补边等改动干扰模型内部预处理；显式传 preprocess 时不置此标志
            cfg = cfg.merge({"preprocess": {"enabled": False}})
        elif self.pad_default_off:
            # 无检测模型（公式识别等）：pad_border 补边只为文字检测服务，无检测则无必要，
            # 默认关闭该步（保留 dark_to_light）；显式传 preprocess 时不置此标志
            cfg = cfg.merge({"preprocess": {"options": {"pad_border": {"enabled": False}}}})
        t_total = time.perf_counter()

        original = load_image(image_input)
        if original.size == 0 or original.shape[0] == 0 or original.shape[1] == 0:
            raise InvalidImageError("空图像无法识别")
        h, w = original.shape[:2]

        pre_pipeline = build_preprocess_pipeline(cfg, self.pre_instances)
        elapse_pre = 0.0
        if cfg.preprocess.enabled:
            t0 = time.perf_counter()
            processed, ctx = pre_pipeline.run(original, debug=cfg.debug)
            elapse_pre = time.perf_counter() - t0
        else:
            processed, ctx = original, _identity_ctx(original)

        request = _build_request(cfg, self.spec_defaults)
        output: ModelOutput = self.executor.run(processed, request)

        _inverse_map(output.items, ctx.transform)

        post_pipeline = build_postprocess_pipeline(cfg, self.post_instances)
        t0 = time.perf_counter()
        document: Document = post_pipeline.run(output.items, (w, h))
        post_pipeline.meta.update(
            {"model": cfg.model, "items": output.items}
        )
        elapse_post = time.perf_counter() - t0

        elapse = Elapse(
            load=float(output.elapse.get("load", 0.0)),
            preprocess=elapse_pre,
            det=float(output.elapse.get("det", 0.0)),
            cls=float(output.elapse.get("cls", 0.0)),
            rec=float(output.elapse.get("rec", 0.0)),
            postprocess=elapse_post,
            total=time.perf_counter() - t_total,
        )
        # 先定稿 elapse 再渲染，使 output="json" 的文本也带耗时信息
        post_pipeline.meta["elapse"] = elapse
        text = post_pipeline.render(document)
        if not isinstance(text, str):
            import json as _json

            text = _json.dumps(text, ensure_ascii=False)

        debug = None
        if cfg.debug:
            debug = DebugInfo(
                processed_image=processed,
                transform=ctx.transform.matrix,
                theme=ctx.meta.get("theme"),
                raw_output={"items_count": len(output.items), "elapse": output.elapse},
                stage_log=pre_pipeline.stages + post_pipeline.stages,
            )

        return OcrResult(
            items=output.items,
            document=document,
            image_size=(w, h),
            model=cfg.model,
            elapse=elapse,
            config=cfg,
            debug=debug,
            _text=text,
            _image=original,
            _post=post_pipeline,
        )
