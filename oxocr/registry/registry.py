"""模型注册 / 解析 / entry-point 发现。

解析顺序：内置 catalog → 运行时注册 → entry points（惰性扫描）。
目录解析：显式 model_dir 参数 > OXOCR_MODEL_DIR > 缓存目录 ~/.oxocr/models。
"""
from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import Any

from oxocr.config import ModelOptions
from oxocr.errors import (
    MissingDependencyError,
    ModelFileMissingError,
    ModelNotFoundError,
    UnknownParameterError,
)
from oxocr.models.base import ModelInfo, ModelSpec, OcrModel
from oxocr.models.duguang_ocr.specs import det_file_for
from oxocr.models.rapid_ocr.specs import cls_file_for
from oxocr.registry.catalog import BUILTIN_SPECS
from oxocr.registry.download import cache_dir

__all__ = [
    "register_spec",
    "register_model_impl",
    "resolve_spec",
    "resolve_model_dir",
    "resolve_files",
    "load_impl",
    "list_models",
    "load_model",
]

_RUNTIME_SPECS: dict[str, ModelSpec] = {}
_RUNTIME_IMPLS: dict[str, str | type[OcrModel]] = {}
_EP_SCANNED = False
_EP_SPECS: dict[str, ModelSpec] = {}

# resolve_files 的必需性分级：det/rec/cls 按 use_* 开关，keys/lm/orient 永远可选
_GATED_ROLES = frozenset({"det", "rec", "cls", "keys", "lm", "orient"})


def register_spec(spec: ModelSpec) -> None:
    _RUNTIME_SPECS[spec.name] = spec


def register_model_impl(family: str, impl: str | type[OcrModel]) -> None:
    _RUNTIME_IMPLS[family] = impl


def _scan_entry_points() -> None:
    global _EP_SCANNED
    if _EP_SCANNED:
        return
    _EP_SCANNED = True
    from importlib.metadata import entry_points

    try:
        eps = entry_points(group="oxocr.models")
    except Exception:
        return
    for ep in eps:
        try:
            cls = ep.load()
            for spec in cls.get_specs():
                _EP_SPECS.setdefault(spec.name, spec)
        except Exception:
            # 第三方 entry point 损坏不拖垮内置解析
            continue


def _known_names() -> list[str]:
    _scan_entry_points()
    return sorted({*BUILTIN_SPECS, *_RUNTIME_SPECS, *_EP_SPECS})


def resolve_spec(name: str) -> ModelSpec:
    if name in BUILTIN_SPECS:
        return BUILTIN_SPECS[name]
    if name in _RUNTIME_SPECS:
        return _RUNTIME_SPECS[name]
    _scan_entry_points()
    if name in _EP_SPECS:
        return _EP_SPECS[name]
    raise ModelNotFoundError(name, _known_names())


def import_string(path: str) -> Any:
    module_name, _, attr = path.partition(":")
    module = importlib.import_module(module_name)
    return getattr(module, attr) if attr else module


def load_impl(spec: ModelSpec) -> type[OcrModel]:
    override = _RUNTIME_IMPLS.get(spec.family)
    target = override if override is not None else spec.impl
    cls = import_string(target) if isinstance(target, str) else target
    if not isinstance(cls, type) or not issubclass(cls, OcrModel):
        raise TypeError(f"{spec.name} 的实现 {target!r} 不是 OcrModel 子类")
    for extra in spec.requires:
        try:
            importlib.import_module(extra)
        except ImportError as e:
            raise MissingDependencyError(extra, extra) from e
    return cls


def resolve_model_dir(model_dir: str | Path | None = None) -> Path:
    if model_dir is not None:
        return Path(model_dir)
    env = os.environ.get("OXOCR_MODEL_DIR")
    if env:
        return Path(env)
    return cache_dir() / "models"


def resolve_files(
    spec: ModelSpec, model_dir: Path, options: ModelOptions
) -> dict[str, Path]:
    """role → 已确认存在的本地路径。缺文件抛 ModelFileMissingError（附期望路径）。"""
    by_role = {f.role: f for f in spec.files}
    # cls 权重由 params['cls_model'] 决定，spec.files 里只是默认那一个
    if "cls" in by_role and "cls_model" in spec.defaults:
        by_role["cls"] = cls_file_for(spec, options.params)
    # det 权重由 params['det_image_size'] 决定（读光家族，与 cls 同型）
    if "det" in by_role and "det_image_size" in spec.defaults:
        by_role["det"] = det_file_for(spec, options.params)
    required: list[str] = []
    if options.use_det and "det" in by_role:
        required.append("det")
    if options.use_rec and "rec" in by_role:
        required.append("rec")
    if options.use_cls and spec.capabilities.angle_classification and "cls" in by_role:
        required.append("cls")

    resolved: dict[str, Path] = {}
    missing: dict[str, str] = {}
    for role, file in by_role.items():
        path = model_dir / file.filename
        if path.is_file():
            resolved[role] = path
        # det/rec/cls 按开关决定是否必需，keys/lm/orient 永远可选，其余（encoder/
        # decoder/tokenizer/config 等）是模型本体文件，恒为必需
        elif role in required or role not in _GATED_ROLES:
            missing[role] = str(path)
    if missing:
        raise ModelFileMissingError(spec.name, missing)
    return resolved


def _installed_paths(spec: ModelSpec, model_dir: Path) -> tuple[bool, dict[str, str]]:
    paths = {f.role: str(model_dir / f.filename) for f in spec.files}
    required = [f for f in spec.files if f.role not in ("keys", "lm", "orient")]
    installed = all(Path(paths[f.role]).is_file() for f in required)
    return installed, paths


def list_models(
    *, family: str | None = None, installed_only: bool = False,
    model_dir: str | Path | None = None,
) -> list[ModelInfo]:
    """列出可用模型。oxocr 自身代码路径为纯 stdlib，不导入 numpy/onnxruntime。"""
    _scan_entry_points()
    specs = {**BUILTIN_SPECS, **_EP_SPECS, **_RUNTIME_SPECS}
    model_dir = resolve_model_dir(model_dir)
    out: list[ModelInfo] = []
    for spec in specs.values():
        if family is not None and spec.family != family:
            continue
        installed, paths = _installed_paths(spec, model_dir)
        if installed_only and not installed:
            continue
        out.append(
            ModelInfo(
                name=spec.name,
                family=spec.family,
                name_chinese=spec.name_chinese or spec.name,
                languages=spec.languages,
                installed=installed,
                downloadable=(
                    bool(spec.modelscope_repo or spec.huggingface_repo)
                    or all(f.url for f in spec.files)
                ),
                requires=spec.requires,
                paths=paths,
            )
        )
    return sorted(out, key=lambda m: m.name)


def load_model(name: str, *, model_dir: str | Path | None = None, **options: Any) -> OcrModel:
    """绕过引擎与常驻池直接拿私有模型实例（已加载）；调用方负责 unload。"""
    spec = resolve_spec(name)
    option_fields = {"use_det", "use_cls", "use_rec", "num_threads", "providers", "params"}
    mo_data: dict[str, Any] = {}
    params: dict[str, Any] = dict(options.get("params") or {})
    for key, value in options.items():
        if key in option_fields:
            if key != "params":
                mo_data[key] = value
        elif key in spec.defaults:
            params[key] = value
        else:
            raise UnknownParameterError(
                key,
                allowed=sorted(option_fields | set(spec.defaults)),
                context=f"load_model({name!r})",
            )
    mo_data["params"] = params
    mo = ModelOptions.from_dict(mo_data)
    unknown = [k for k in mo.params if k not in spec.defaults]
    if unknown:
        raise UnknownParameterError(
            unknown[0], allowed=sorted(spec.defaults), context=f"模型 {name} 的 params"
        )
    directory = resolve_model_dir(model_dir)
    files = resolve_files(spec, directory, mo)
    impl = load_impl(spec)
    model = impl(spec, files, mo)
    model.ensure_loaded()
    return model
