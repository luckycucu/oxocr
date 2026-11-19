"""模型扩展契约：ModelSpec 声明式清单 + OcrModel 抽象基类。

本模块仅依赖 stdlib（numpy 只在类型检查期导入），供 catalog / registry 轻量引用。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from oxocr.errors import UnknownParameterError
from oxocr.types import TextItem

if TYPE_CHECKING:
    import numpy as np

    from oxocr.config import ModelOptions

__all__ = [
    "ModelFile",
    "ModelSpec",
    "Capabilities",
    "ModelInfo",
    "InferRequest",
    "ModelOutput",
    "OcrModel",
    "MODELSCOPE_REPO",
    "HUGGINGFACE_REPO",
]

# oxocr 官方权重分发仓库；所有内置模型共用，仓库内路径与 ModelFile.filename 一致
MODELSCOPE_REPO = "mscoder/oxocr"
HUGGINGFACE_REPO = "Luoaho/oxocr"


@dataclass(frozen=True)
class ModelFile:
    role: str  # "det" | "rec" | "cls" | "keys" | "model"
    filename: str  # 相对 model_dir 的路径
    url: str | None = None  # None = 仅支持本地加载
    sha256: str | None = None
    size: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "filename": self.filename,
            "url": self.url,
            "sha256": self.sha256,
            "size": self.size,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ModelFile:
        return cls(
            role=str(data["role"]),
            filename=str(data["filename"]),
            url=data.get("url"),
            sha256=data.get("sha256"),
            size=data.get("size"),
        )


@dataclass(frozen=True)
class Capabilities:
    detection: bool = True
    recognition: bool = True
    angle_classification: bool = False
    char_boxes: bool = False
    backend: str = "onnxruntime"
    own_preprocess: bool = False
    own_postprocess: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "detection": self.detection,
            "recognition": self.recognition,
            "angle_classification": self.angle_classification,
            "char_boxes": self.char_boxes,
            "backend": self.backend,
            "own_preprocess": self.own_preprocess,
            "own_postprocess": self.own_postprocess,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Capabilities:
        return cls(**{k: data[k] for k in cls.__dataclass_fields__ if k in data})


@dataclass(frozen=True)
class ModelSpec:
    name: str
    family: str
    impl: str  # 惰性导入路径 "module:Class"
    files: tuple[ModelFile, ...]
    name_chinese: str = ""
    raw_author: str = ""
    raw_author_country: str | None = None
    raw_urls: tuple[str, ...] = ()
    raw_release_date: date | None = None
    raw_license: str = ""
    raw_model_ids: dict[str, str] = field(default_factory=dict)
    # hub 仓库 id（owner/name）；仓库内文件路径与 files.filename 一致
    modelscope_repo: str | None = None
    huggingface_repo: str | None = None
    current_author: str | None = None
    current_author_country: str | None = None
    current_urls: tuple[str, ...] | None = None
    current_release_date: date | None = None
    current_license: str | None = None
    model_type: str = "text"  # "text" | "latex"
    languages: tuple[str, ...] = ()
    languages_chinese: tuple[str, ...] = ()
    languages_english: tuple[str, ...] = ()
    size_bytes: int = 0
    star_rating: int = 0  # 0–5，0 表示未评级
    model_architecture: dict[str, str] = field(default_factory=dict)
    commercial_use: str = "unknown"  # "yes" | "no" | "unknown" | "restricted(...)"
    defaults: Mapping[str, Any] = field(default_factory=dict)
    requires: tuple[str, ...] = ()
    capabilities: Capabilities = field(default_factory=Capabilities)

    # current_* 为空时回退到 raw_*（见 docs/模型信息字段.md）
    @property
    def author(self) -> str:
        return self.current_author or self.raw_author

    @property
    def author_country(self) -> str | None:
        return self.current_author_country or self.raw_author_country

    @property
    def urls(self) -> tuple[str, ...]:
        return self.current_urls if self.current_urls is not None else self.raw_urls

    @property
    def release_date(self) -> date | None:
        return self.current_release_date or self.raw_release_date

    @property
    def license(self) -> str:
        return self.current_license or self.raw_license

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "family": self.family,
            "impl": self.impl,
            "files": [f.to_dict() for f in self.files],
            "name_chinese": self.name_chinese,
            "raw_author": self.raw_author,
            "raw_author_country": self.raw_author_country,
            "raw_urls": list(self.raw_urls),
            "raw_release_date": self.raw_release_date.isoformat() if self.raw_release_date else None,
            "raw_license": self.raw_license,
            "raw_model_ids": dict(self.raw_model_ids),
            "modelscope_repo": self.modelscope_repo,
            "huggingface_repo": self.huggingface_repo,
            "current_author": self.current_author,
            "current_author_country": self.current_author_country,
            "current_urls": list(self.current_urls) if self.current_urls is not None else None,
            "current_release_date": (
                self.current_release_date.isoformat() if self.current_release_date else None
            ),
            "current_license": self.current_license,
            "model_type": self.model_type,
            "languages": list(self.languages),
            "languages_chinese": list(self.languages_chinese),
            "languages_english": list(self.languages_english),
            "size_bytes": self.size_bytes,
            "star_rating": self.star_rating,
            "model_architecture": dict(self.model_architecture),
            "commercial_use": self.commercial_use,
            "defaults": dict(self.defaults),
            "requires": list(self.requires),
            "capabilities": self.capabilities.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ModelSpec:
        def _date(key: str) -> date | None:
            value = data.get(key)
            return date.fromisoformat(value) if value else None

        return cls(
            name=str(data["name"]),
            family=str(data["family"]),
            impl=str(data["impl"]),
            files=tuple(ModelFile.from_dict(f) for f in data.get("files", ())),
            name_chinese=str(data.get("name_chinese", "")),
            raw_author=str(data.get("raw_author", "")),
            raw_author_country=data.get("raw_author_country"),
            raw_urls=tuple(data.get("raw_urls", ())),
            raw_release_date=_date("raw_release_date"),
            raw_license=str(data.get("raw_license", "")),
            raw_model_ids=dict(data.get("raw_model_ids") or {}),
            modelscope_repo=data.get("modelscope_repo"),
            huggingface_repo=data.get("huggingface_repo"),
            current_author=data.get("current_author"),
            current_author_country=data.get("current_author_country"),
            current_urls=(
                tuple(data["current_urls"]) if data.get("current_urls") is not None else None
            ),
            current_release_date=_date("current_release_date"),
            current_license=data.get("current_license"),
            model_type=str(data.get("model_type", "text")),
            languages=tuple(data.get("languages", ())),
            languages_chinese=tuple(data.get("languages_chinese", ())),
            languages_english=tuple(data.get("languages_english", ())),
            size_bytes=int(data.get("size_bytes", 0)),
            star_rating=int(data.get("star_rating", 0)),
            model_architecture=dict(data.get("model_architecture") or {}),
            commercial_use=str(data.get("commercial_use", "unknown")),
            defaults=dict(data.get("defaults") or {}),
            requires=tuple(data.get("requires", ())),
            capabilities=Capabilities.from_dict(data.get("capabilities") or {}),
        )


@dataclass(frozen=True)
class ModelInfo:
    """list_models() 的返回，不含任何重对象。"""

    name: str
    family: str
    name_chinese: str
    languages: tuple[str, ...]
    installed: bool
    downloadable: bool
    requires: tuple[str, ...]
    paths: dict[str, str]


@dataclass
class InferRequest:
    text_score: float = 0.5
    use_det: bool = True
    use_cls: bool = False
    use_rec: bool = True
    params: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text_score": self.text_score,
            "use_det": self.use_det,
            "use_cls": self.use_cls,
            "use_rec": self.use_rec,
            "params": dict(self.params),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> InferRequest:
        return cls(
            text_score=float(data.get("text_score", 0.5)),
            use_det=bool(data.get("use_det", True)),
            use_cls=bool(data.get("use_cls", False)),
            use_rec=bool(data.get("use_rec", True)),
            params=dict(data.get("params") or {}),
        )


@dataclass
class ModelOutput:
    items: list[TextItem]
    elapse: dict[str, float] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "items": [it.to_dict() for it in self.items],
            "elapse": dict(self.elapse),
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ModelOutput:
        return cls(
            items=[TextItem.from_dict(it) for it in data.get("items", ())],
            elapse={k: float(v) for k, v in (data.get("elapse") or {}).items()},
            meta=dict(data.get("meta") or {}),
        )


class OcrModel(ABC):
    """模型唯一契约：像素 → 带 box 的文本条目。

    模型不知道前处理、不知道排版模式、不读配置文件；
    返回的 box 使用**送入图坐标**，逆映射由核心负责。
    """

    family: ClassVar[str] = ""
    capabilities: ClassVar[Capabilities] = Capabilities()

    def __init__(
        self,
        spec: ModelSpec,
        files: Mapping[str, Path],
        options: ModelOptions,
    ) -> None:
        self.spec = spec
        self.files = dict(files)
        self.options = options
        self._loaded = False

    @abstractmethod
    def load(self) -> None:
        """建立会话。构造时不得调用，由引擎惰性触发。"""

    @abstractmethod
    def unload(self) -> None:
        """释放会话与权重；调用后 is_loaded 为 False，可再次 load()。"""

    @abstractmethod
    def infer(self, image: np.ndarray, request: InferRequest) -> ModelOutput:
        """image: BGR uint8 (H, W, 3)，已完成前处理。不得打印到 stdout。"""

    @classmethod
    def get_specs(cls) -> Sequence[ModelSpec]:
        """entry-point 发现时调用；内置模型由 catalog 直接声明。"""
        return ()

    def validate_options(self, options: ModelOptions) -> None:
        """按 spec.defaults 校验 params 白名单。"""
        allowed = set(self.spec.defaults)
        for key in options.params:
            if key not in allowed:
                raise UnknownParameterError(
                    key, allowed=sorted(allowed), context=f"模型 {self.spec.name} 的 params"
                )

    def params(self) -> dict[str, Any]:
        """spec.defaults ← options.params 合并结果。"""
        merged = dict(self.spec.defaults)
        merged.update(self.options.params)
        return merged

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def ensure_loaded(self) -> None:
        if not self._loaded:
            self.load()

    def __enter__(self) -> OcrModel:
        self.ensure_loaded()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.unload()
