"""oxocr 配置体系：dataclass + from_dict/to_dict/from_json，非法值钳位并 warning。

环境变量（构造默认值时读取）：
    OXOCR_MODEL_DIR / OXOCR_DEFAULT_MODEL / OXOCR_NUM_THREADS / OXOCR_RESIDENT
"""
from __future__ import annotations

import json
import logging
import os
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Literal

from oxocr.errors import UnknownParameterError

logger = logging.getLogger("oxocr.config")

LayoutMode = Literal["raw", "line", "para", "layout"]
OutputFormat = Literal["text", "json"]
Backend = Literal["inprocess", "subprocess"]

LAYOUT_MODES = ("raw", "line", "para", "layout")
OUTPUT_FORMATS = ("text", "json")
BACKENDS = ("inprocess", "subprocess")

DEFAULT_MODEL = "ppocrv6_small"

__all__ = [
    "LayoutMode",
    "OutputFormat",
    "Backend",
    "OcrConfig",
    "ModelOptions",
    "InferOptions",
    "PreprocessConfig",
    "PostprocessConfig",
    "SubprocessOptions",
    "apply_overrides",
]


def _warn(msg: str) -> None:
    warnings.warn(msg, stacklevel=3)


def _as_bool(value: Any, name: str, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        low = value.strip().lower()
        if low in ("1", "true", "yes", "on"):
            return True
        if low in ("0", "false", "no", "off"):
            return False
    if isinstance(value, (int, float)):
        return bool(value)
    _warn(f"{name}={value!r} 非法，回退为 {default}")
    return default


def _as_int(value: Any, name: str, default: int, minimum: int | None = None) -> int:
    try:
        out = int(value)
    except (TypeError, ValueError):
        _warn(f"{name}={value!r} 非法，回退为 {default}")
        return default
    if minimum is not None and out < minimum:
        _warn(f"{name}={out} 越界，钳位为 {minimum}")
        out = minimum
    return out


def _as_float(
    value: Any,
    name: str,
    default: float,
    lo: float | None = None,
    hi: float | None = None,
) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        _warn(f"{name}={value!r} 非法，回退为 {default}")
        return default
    if lo is not None and out < lo:
        _warn(f"{name}={out} 越界，钳位为 {lo}")
        out = lo
    if hi is not None and out > hi:
        _warn(f"{name}={out} 越界，钳位为 {hi}")
        out = hi
    return out


def _as_choice(value: Any, name: str, choices: Sequence[str], default: str) -> str:
    if isinstance(value, str) and value in choices:
        return value
    _warn(f"{name}={value!r} 非法，回退为 {default!r}（可选: {', '.join(choices)}）")
    return default


def _as_str_list(value: Any, name: str, default: Sequence[str]) -> tuple[str, ...]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple)):
        _warn(f"{name}={value!r} 非法，回退默认列表")
        return tuple(default)
    out = []
    for item in value:
        if isinstance(item, str) and item:
            out.append(item)
        else:
            _warn(f"{name} 中的 {item!r} 非法，已跳过")
    return tuple(out)


def _filter_keys(cls, data: Mapping[str, Any], section: str) -> dict[str, Any]:
    allowed = {f.name for f in fields(cls)}
    out = {}
    for key, value in data.items():
        if key in allowed:
            out[key] = value
        else:
            _warn(f"配置 {section} 中的未知键 {key!r} 已忽略")
    return out


@dataclass
class ModelOptions:
    use_det: bool = True
    use_cls: bool = False
    use_rec: bool = True
    num_threads: int = 0  # 0 = CPU 核心数一半
    providers: Sequence[str] = ("CPUExecutionProvider",)
    params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ModelOptions:
        data = _filter_keys(cls, data, "model_options")
        return cls(
            use_det=_as_bool(data.get("use_det", True), "model_options.use_det", True),
            use_cls=_as_bool(data.get("use_cls", False), "model_options.use_cls", False),
            use_rec=_as_bool(data.get("use_rec", True), "model_options.use_rec", True),
            num_threads=_as_int(
                data.get(
                    "num_threads", os.environ.get("OXOCR_NUM_THREADS", 0)
                ),
                "model_options.num_threads",
                0,
                minimum=0,
            ),
            providers=_as_str_list(
                data.get("providers", ("CPUExecutionProvider",)),
                "model_options.providers",
                ("CPUExecutionProvider",),
            ),
            params=dict(data.get("params") or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "use_det": self.use_det,
            "use_cls": self.use_cls,
            "use_rec": self.use_rec,
            "num_threads": self.num_threads,
            "providers": list(self.providers),
            "params": dict(self.params),
        }

    def fingerprint(self) -> str:
        """稳定指纹：参与常驻池键；推理级参数（text_score 等）不参与。"""
        import hashlib

        payload = json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=True, default=str)
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


@dataclass
class InferOptions:
    text_score: float = 0.5

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> InferOptions:
        data = _filter_keys(cls, data, "infer")
        return cls(
            text_score=_as_float(
                data.get("text_score", 0.5), "infer.text_score", 0.5, lo=0.0, hi=1.0
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {"text_score": self.text_score}


@dataclass
class PreprocessConfig:
    enabled: bool = True
    steps: Sequence[str] = ("dark_to_light", "pad_border")
    options: dict[str, dict] = field(default_factory=dict)
    on_error: Literal["skip", "raise"] = "skip"

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PreprocessConfig:
        data = _filter_keys(cls, data, "preprocess")
        options = data.get("options") or {}
        if not isinstance(options, dict):
            _warn("preprocess.options 非法，回退为空")
            options = {}
        return cls(
            enabled=_as_bool(data.get("enabled", True), "preprocess.enabled", True),
            steps=_as_str_list(
                data.get("steps", ("dark_to_light", "pad_border")),
                "preprocess.steps",
                ("dark_to_light", "pad_border"),
            ),
            options={str(k): dict(v or {}) for k, v in options.items()},
            on_error=_as_choice(data.get("on_error", "skip"), "preprocess.on_error", ("skip", "raise"), "skip"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "steps": list(self.steps),
            "options": {k: dict(v) for k, v in self.options.items()},
            "on_error": self.on_error,
        }


@dataclass
class PostprocessConfig:
    enabled: bool = True
    mode: LayoutMode = "para"
    join: str | None = None
    steps: Sequence[str] = ("ghost_filter", "corrections", "normalize_space")
    options: dict[str, dict] = field(default_factory=dict)
    output: OutputFormat = "text"
    paragraph_separator: str = "\n\n"
    line_separator: str = "\n"

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PostprocessConfig:
        data = _filter_keys(cls, data, "postprocess")
        options = data.get("options") or {}
        if not isinstance(options, dict):
            _warn("postprocess.options 非法，回退为空")
            options = {}
        join = data.get("join")
        if join is not None and not isinstance(join, str):
            _warn(f"postprocess.join={join!r} 非法，回退为 None")
            join = None
        return cls(
            enabled=_as_bool(data.get("enabled", True), "postprocess.enabled", True),
            mode=_as_choice(data.get("mode", "para"), "postprocess.mode", LAYOUT_MODES, "para"),
            join=join,
            steps=_as_str_list(
                data.get("steps", ("ghost_filter", "corrections", "normalize_space")),
                "postprocess.steps",
                ("ghost_filter", "corrections", "normalize_space"),
            ),
            options={str(k): dict(v or {}) for k, v in options.items()},
            output=_as_choice(data.get("output", "text"), "postprocess.output", OUTPUT_FORMATS, "text"),
            paragraph_separator=str(data.get("paragraph_separator", "\n\n")),
            line_separator=str(data.get("line_separator", "\n")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "mode": self.mode,
            "join": self.join,
            "steps": list(self.steps),
            "options": {k: dict(v) for k, v in self.options.items()},
            "output": self.output,
            "paragraph_separator": self.paragraph_separator,
            "line_separator": self.line_separator,
        }


@dataclass
class SubprocessOptions:
    timeout: float = 300.0
    python: str | None = None
    frozen_flag: str = "--oxocr-worker"

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SubprocessOptions:
        data = _filter_keys(cls, data, "subprocess")
        python = data.get("python")
        if python is not None:
            python = str(python)
        return cls(
            timeout=_as_float(data.get("timeout", 300.0), "subprocess.timeout", 300.0, lo=0.1),
            python=python,
            frozen_flag=str(data.get("frozen_flag", "--oxocr-worker")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"timeout": self.timeout, "python": self.python, "frozen_flag": self.frozen_flag}


def _env_model_dir() -> Path | None:
    value = os.environ.get("OXOCR_MODEL_DIR")
    return Path(value) if value else None


def _env_resident() -> bool:
    return os.environ.get("OXOCR_RESIDENT", "1") != "0"


@dataclass
class OcrConfig:
    model: str = field(default_factory=lambda: os.environ.get("OXOCR_DEFAULT_MODEL", DEFAULT_MODEL))
    model_dir: Path | None = field(default_factory=_env_model_dir)
    model_options: ModelOptions = field(default_factory=ModelOptions)
    infer: InferOptions = field(default_factory=InferOptions)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    postprocess: PostprocessConfig = field(default_factory=PostprocessConfig)
    backend: Backend = "inprocess"
    resident: bool = field(default_factory=_env_resident)
    subprocess: SubprocessOptions = field(default_factory=SubprocessOptions)
    debug: bool = False
    on_error: Literal["raise", "skip"] = "raise"

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> OcrConfig:
        if isinstance(data, OcrConfig):
            return data
        if not isinstance(data, Mapping):
            raise TypeError(f"config 需要映射，收到 {type(data).__name__}")
        data = _filter_keys(cls, data, "顶层")
        model = data.get("model") or os.environ.get("OXOCR_DEFAULT_MODEL", DEFAULT_MODEL)
        model_dir = data.get("model_dir")
        if model_dir is not None:
            model_dir = Path(model_dir)
        else:
            model_dir = _env_model_dir()
        return cls(
            model=str(model),
            model_dir=model_dir,
            model_options=ModelOptions.from_dict(data.get("model_options") or {}),
            infer=InferOptions.from_dict(data.get("infer") or {}),
            preprocess=PreprocessConfig.from_dict(data.get("preprocess") or {}),
            postprocess=PostprocessConfig.from_dict(data.get("postprocess") or {}),
            backend=_as_choice(data.get("backend", "inprocess"), "backend", BACKENDS, "inprocess"),
            resident=_as_bool(data.get("resident", _env_resident()), "resident", True),
            subprocess=SubprocessOptions.from_dict(data.get("subprocess") or {}),
            debug=_as_bool(data.get("debug", False), "debug", False),
            on_error=_as_choice(data.get("on_error", "raise"), "on_error", ("raise", "skip"), "raise"),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> OcrConfig:
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "model_dir": str(self.model_dir) if self.model_dir is not None else None,
            "model_options": self.model_options.to_dict(),
            "infer": self.infer.to_dict(),
            "preprocess": self.preprocess.to_dict(),
            "postprocess": self.postprocess.to_dict(),
            "backend": self.backend,
            "resident": self.resident,
            "subprocess": self.subprocess.to_dict(),
            "debug": self.debug,
            "on_error": self.on_error,
        }

    def merge(self, other: Mapping[str, Any] | OcrConfig) -> OcrConfig:
        """返回合并后的新对象，不改自身。other 优先。"""
        if isinstance(other, OcrConfig):
            other = other.to_dict()
        base = self.to_dict()
        merged = _deep_merge(base, dict(other))
        return OcrConfig.from_dict(merged)


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


# api.md §1.3 快捷覆盖键 → 配置落点
_SHORTCUTS: dict[str, tuple[str, ...]] = {
    "use_cls": ("model_options", "use_cls"),
    "text_score": ("infer", "text_score"),
    "num_threads": ("model_options", "num_threads"),
    "dark_convert": ("preprocess", "options", "dark_to_light", "enabled"),
    "force_pad": ("preprocess", "options", "pad_border", "force"),
    "det_limit_side_len": ("model_options", "params", "det_limit_side_len"),
    "correct": ("postprocess", "options", "corrections", "enabled"),
    "resident": ("resident",),
}


def apply_overrides(config: OcrConfig, overrides: Mapping[str, Any]) -> OcrConfig:
    """把调用处快捷 overrides 落到 config 对应位置；未知键抛 UnknownParameterError。"""
    if not overrides:
        return config
    unknown = [k for k in overrides if k not in _SHORTCUTS]
    if unknown:
        raise UnknownParameterError(
            unknown[0], allowed=sorted(_SHORTCUTS), context="调用处快捷参数"
        )
    patch: dict[str, Any] = {}
    for key, value in overrides.items():
        node = patch
        path = _SHORTCUTS[key]
        for part in path[:-1]:
            node = node.setdefault(part, {})
        node[path[-1]] = value
    return config.merge(patch)
