"""公共门面：ocr() / readtext() / OcrEngine 与模型管理入口。

参数优先级（api.md §1.3）：调用处 overrides/mode/preprocess/postprocess
> config > ModelSpec.defaults > 全局默认。spec.defaults 在请求参数层合并，
OcrSpec 数据不写入 config。
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from oxocr.config import OcrConfig, apply_overrides
from oxocr.models.base import ModelSpec, OcrModel
from oxocr.pipeline import OcrPipeline
from oxocr.registry.registry import (
    register_model_impl,
    register_spec,
    resolve_files,
    resolve_model_dir,
    resolve_spec,
)
from oxocr.runtime.executor import InProcessExecutor, SubprocessExecutor

__all__ = [
    "ocr",
    "readtext",
    "OcrEngine",
    "list_models",
    "get_model_spec",
    "download_model",
    "load_model",
    "unload_model",
    "unload_all",
    "loaded_models",
]

# 重新导出注册 API（api.md §4.3）
__all__ += ["register_spec", "register_model_impl"]


def _resolve_stage(value: Any) -> tuple[dict | None, list | None]:
    """把 preprocess/postprocess 参数拆成 (config 补丁, 实例注入列表)。"""
    if value is None:
        return None, None
    if isinstance(value, bool):
        return {"enabled": value}, None
    if isinstance(value, Mapping):
        return dict(value), None
    if isinstance(value, (list, tuple)):
        if all(isinstance(x, str) for x in value):
            return {"steps": list(value)}, None
        return {"enabled": True}, list(value)
    raise TypeError(f"preprocess/postprocess 参数类型非法: {type(value).__name__}")


def _build_config(
    config: OcrConfig | Mapping | str | Path | None,
    *,
    model_name: str | None,
    mode: str | None,
    output: str | None,
    preprocess: Any,
    postprocess: Any,
    model_dir: str | Path | None,
    backend: str | None,
    resident: bool | None,
    debug: bool | None,
    overrides: Mapping[str, Any],
) -> tuple[OcrConfig, list | None, list | None, bool, bool]:
    # 后处理是否被「显式表态」：调用处传了 postprocess=...，或 Mapping 配置里
    # 带 postprocess 键。OcrConfig 实例无法区分默认值与显式开启，视为未表态
    # （需要恢复后处理时传 postprocess=True）。供 own_postprocess 门控使用。
    post_explicit = postprocess is not None or (
        isinstance(config, Mapping) and "postprocess" in config
    )
    # 前处理同理：供 own_preprocess 门控使用（恢复核心前处理时传 preprocess=True）。
    pre_explicit = preprocess is not None or (
        isinstance(config, Mapping) and "preprocess" in config
    )
    base = OcrConfig.from_dict(config) if config is not None else OcrConfig()
    cfg = base.merge({})  # 独立副本：避免就地修改污染调用方/引擎的配置
    if model_name is not None:
        cfg.model = model_name
    if model_dir is not None:
        cfg.model_dir = Path(model_dir)
    if backend is not None:
        cfg.backend = backend
    if resident is not None:
        cfg.resident = resident
    if debug is not None:
        cfg.debug = debug

    patches: dict[str, Any] = {}
    if mode is not None:
        patches.setdefault("postprocess", {})["mode"] = mode
    if output is not None:
        patches.setdefault("postprocess", {})["output"] = output
    pre_patch, pre_inst = _resolve_stage(preprocess)
    if pre_patch:
        patches.setdefault("preprocess", {}).update(pre_patch)
    post_patch, post_inst = _resolve_stage(postprocess)
    if post_patch:
        patches.setdefault("postprocess", {}).update(post_patch)
    if patches:
        cfg = cfg.merge(patches)
    cfg = apply_overrides(cfg, overrides)
    return cfg, pre_inst, post_inst, post_explicit, pre_explicit


def _post_default_off(spec: ModelSpec, post_explicit: bool) -> bool:
    """own_postprocess 模型（公式识别等）且用户未显式表态时，默认跳过后处理。"""
    return bool(spec.capabilities.own_postprocess) and not post_explicit


def _pre_default_off(spec: ModelSpec, pre_explicit: bool) -> bool:
    """own_preprocess 模型（自带前处理）且用户未显式表态时，默认跳过核心前处理。"""
    return bool(spec.capabilities.own_preprocess) and not pre_explicit


def _pad_default_off(spec: ModelSpec, pre_explicit: bool) -> bool:
    """无检测模型（公式识别等）且用户未显式表态时，默认关闭 pad_border 补边。

    补边只为提高文字检测效果服务；无检测步骤的模型整图直识，补边无必要
    （indra math 甚至要在内部把补边裁掉）。保留 dark_to_light 等其余前处理。
    """
    return not spec.capabilities.detection and not pre_explicit


def _make_executor(spec: ModelSpec, cfg: OcrConfig):
    model_dir = resolve_model_dir(cfg.model_dir)
    files = resolve_files(spec, model_dir, cfg.model_options)
    if cfg.backend == "subprocess":
        return SubprocessExecutor(spec, files, cfg.model_options, cfg.subprocess), cfg, files, model_dir
    return (
        InProcessExecutor(spec, files, cfg.model_options, model_dir, resident=cfg.resident),
        cfg,
        files,
        model_dir,
    )


class OcrEngine:
    """复用引擎：批量场景共享同一常驻会话。

    默认与门面、其他同 key 引擎共享进程级常驻会话；无需 with/close 也不泄漏。
    """

    def __init__(
        self,
        model: str | ModelSpec = "ppocrv6_small",
        *,
        config: OcrConfig | Mapping | str | Path | None = None,
        preprocess: Any = None,
        postprocess: Any = None,
        backend: str | None = None,
        resident: bool | None = None,
        model_dir: str | Path | None = None,
        **overrides: Any,
    ):
        self._base_spec = model if isinstance(model, ModelSpec) else None
        model_name = model.name if isinstance(model, ModelSpec) else model
        (
            self._config, self._pre_inst, self._post_inst,
            self._post_explicit, self._pre_explicit,
        ) = _build_config(
            config,
            model_name=model_name,
            mode=None,
            output=None,
            preprocess=preprocess,
            postprocess=postprocess,
            model_dir=model_dir,
            backend=backend,
            resident=resident,
            debug=None,
            overrides=overrides,
        )
        self._spec = self._base_spec or resolve_spec(self._config.model)
        self._executor = None
        self._executor_sig = None

    @property
    def config(self) -> OcrConfig:
        return self._config

    @property
    def model_name(self) -> str:
        return self._config.model

    @property
    def is_loaded(self) -> bool:
        return self._executor is not None and self._executor.is_loaded

    def _ensure_executor_for(self, spec: ModelSpec, cfg: OcrConfig):
        sig = (
            spec.name,
            cfg.backend,
            cfg.resident,
            str(resolve_model_dir(cfg.model_dir)),
            cfg.model_options.fingerprint(),
        )
        if self._executor is None or self._executor_sig != sig or self._spec is not spec:
            if self._executor is not None:
                self._executor.close()
            self._spec = spec
            self._executor, _, _, _ = _make_executor(spec, cfg)
            self._executor_sig = sig
        return self._executor

    def _effective(self, kwargs: dict):
        """把 per-call kwargs 折进 config，并解析对应 spec（支持 model 覆盖）。"""
        model = kwargs.pop("model", None)
        spec_override = model if isinstance(model, ModelSpec) else None
        model_name = model.name if isinstance(model, ModelSpec) else model
        cfg, pre_inst, post_inst, call_explicit, call_pre_explicit = _build_config(
            self._config,
            model_name=model_name,
            mode=kwargs.pop("mode", None),
            output=kwargs.pop("output", None),
            preprocess=kwargs.pop("preprocess", None),
            postprocess=kwargs.pop("postprocess", None),
            model_dir=kwargs.pop("model_dir", None),
            backend=kwargs.pop("backend", None),
            resident=kwargs.pop("resident", None),
            debug=kwargs.pop("debug", None),
            overrides=kwargs,
        )
        if spec_override is not None:
            spec = spec_override
        elif cfg.model == self._config.model:
            spec = self._spec
        else:
            spec = resolve_spec(cfg.model)
        return (
            cfg, spec, pre_inst, post_inst,
            call_explicit or self._post_explicit,
            call_pre_explicit or self._pre_explicit,
        )

    def ocr(self, image, **kwargs) -> OcrResultType:
        cfg, spec, pre_inst, post_inst, post_explicit, pre_explicit = self._effective(kwargs)
        executor = self._ensure_executor_for(spec, cfg)
        pipeline = OcrPipeline(
            cfg, executor, spec.defaults, pre_instances=pre_inst, post_instances=post_inst,
            post_default_off=_post_default_off(spec, post_explicit),
            pre_default_off=_pre_default_off(spec, pre_explicit),
            pad_default_off=_pad_default_off(spec, pre_explicit),
        )
        return pipeline.run(image)

    def readtext(self, image, **kwargs) -> str:
        return self.ocr(image, **kwargs).text

    def batch(self, images: Iterable, **kwargs) -> list:
        return [self.ocr(img, **kwargs) for img in images]

    def detect_only(self, image, **kwargs) -> list:
        cfg, spec, pre_inst, post_inst, post_explicit, pre_explicit = self._effective(kwargs)
        cfg.model_options.use_rec = False
        cfg.model_options.use_cls = False
        cfg.postprocess.mode = "raw"
        executor = self._ensure_executor_for(spec, cfg)
        pipeline = OcrPipeline(
            cfg, executor, spec.defaults, pre_instances=pre_inst, post_instances=post_inst,
            post_default_off=_post_default_off(spec, post_explicit),
            pre_default_off=_pre_default_off(spec, pre_explicit),
            pad_default_off=_pad_default_off(spec, pre_explicit),
        )
        return [item.box for item in pipeline.run(image).items]

    def recognize_crop(self, crop, **kwargs):
        cfg, spec, pre_inst, post_inst, post_explicit, pre_explicit = self._effective(kwargs)
        cfg.model_options.use_det = False
        executor = self._ensure_executor_for(spec, cfg)
        pipeline = OcrPipeline(
            cfg, executor, spec.defaults, pre_instances=pre_inst, post_instances=post_inst,
            post_default_off=_post_default_off(spec, post_explicit),
            pre_default_off=_pre_default_off(spec, pre_explicit),
            pad_default_off=_pad_default_off(spec, pre_explicit),
        )
        result = pipeline.run(crop)
        return result.items[0] if result.items else None

    def load(self) -> None:
        executor = self._ensure_executor_for(self._spec, self._config)
        executor.warm()

    def close(self) -> None:
        if self._executor is not None:
            self._executor.close()

    def __enter__(self) -> OcrEngine:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


from oxocr.types import OcrResult as OcrResultType  # noqa: E402  仅类型标注使用


def ocr(
    image,
    *,
    model: str | ModelSpec = "ppocrv6_small",
    mode: str | None = None,
    output: str | None = None,
    config: OcrConfig | Mapping | str | Path | None = None,
    preprocess: Any = None,
    postprocess: Any = None,
    model_dir: str | Path | None = None,
    backend: str | None = None,
    resident: bool | None = None,
    debug: bool = False,
    **overrides: Any,
) -> OcrResultType:
    spec = model if isinstance(model, ModelSpec) else None
    model_name = model.name if isinstance(model, ModelSpec) else model
    cfg, pre_inst, post_inst, post_explicit, pre_explicit = _build_config(
        config,
        model_name=model_name,
        mode=mode,
        output=output,
        preprocess=preprocess,
        postprocess=postprocess,
        model_dir=model_dir,
        backend=backend,
        resident=resident,
        debug=debug,
        overrides=overrides,
    )
    if spec is None:
        spec = resolve_spec(cfg.model)
    executor, _, _, _ = _make_executor(spec, cfg)
    try:
        pipeline = OcrPipeline(
            cfg, executor, spec.defaults, pre_instances=pre_inst, post_instances=post_inst,
            post_default_off=_post_default_off(spec, post_explicit),
            pre_default_off=_pre_default_off(spec, pre_explicit),
            pad_default_off=_pad_default_off(spec, pre_explicit),
        )
        return pipeline.run(image)
    finally:
        executor.close()


def readtext(image, **kwargs) -> str:
    return ocr(image, **kwargs).text


def list_models(
    *, family: str | None = None, installed_only: bool = False,
    model_dir: str | Path | None = None,
) -> list:
    from oxocr.registry.registry import list_models as _list

    return _list(family=family, installed_only=installed_only, model_dir=model_dir)


def get_model_spec(name: str) -> ModelSpec:
    return resolve_spec(name)


def download_model(
    name: str, *, model_dir=None, force=False, progress=None, source: str | None = None
) -> dict[str, Path]:
    from oxocr.registry.download import download_model as _dl

    return _dl(name, model_dir=model_dir, force=force, progress=progress, source=source)


def load_model(name: str, *, model_dir=None, **options: Any) -> OcrModel:
    from oxocr.registry.registry import load_model as _load

    return _load(name, model_dir=model_dir, **options)


def unload_model(name: str, *, model_dir=None, **options: Any) -> int:
    from oxocr.config import ModelOptions
    from oxocr.runtime.pool import pool_instance

    opts = ModelOptions.from_dict(options) if options else None
    return pool_instance().unload(name, model_dir=resolve_model_dir(model_dir), options=opts)


def unload_all() -> int:
    from oxocr.runtime.pool import pool_instance

    return pool_instance().unload_all()


def loaded_models() -> list:
    from oxocr.runtime.pool import pool_instance

    return pool_instance().snapshot()
