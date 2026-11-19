"""前处理协议：Preprocessor / PreContext / Affine / 注册表 / PreprocessPipeline。

铁律：任何几何改动必须同步更新 ctx.transform，推理后由核心统一逆映射，
保证结果 box 恒为原图坐标。
"""
from __future__ import annotations

import logging
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal, Protocol, runtime_checkable

import numpy as np

from oxocr.config import PreprocessConfig
from oxocr.errors import PluginNotFoundError, StageError
from oxocr.types import Box, StageLog

logger = logging.getLogger("oxocr.preprocess")

__all__ = [
    "Affine",
    "PreContext",
    "Preprocessor",
    "register_preprocessor",
    "resolve_preprocessor",
    "PreprocessPipeline",
]


@dataclass(frozen=True)
class Affine:
    """3x3 仿射矩阵，原图 → 处理图。"""

    matrix: np.ndarray

    @classmethod
    def identity(cls) -> Affine:
        return cls(np.eye(3, dtype=np.float64))

    @classmethod
    def translation(cls, dx: float, dy: float) -> Affine:
        m = np.eye(3, dtype=np.float64)
        m[0, 2] = dx
        m[1, 2] = dy
        return cls(m)

    @classmethod
    def scaling(cls, sx: float, sy: float) -> Affine:
        m = np.eye(3, dtype=np.float64)
        m[0, 0] = sx
        m[1, 1] = sy
        return cls(m)

    @classmethod
    def rotation(cls, angle_deg: float, center: tuple[float, float]) -> Affine:
        rad = math.radians(angle_deg)
        c, s = math.cos(rad), math.sin(rad)
        cx, cy = center
        m = np.eye(3, dtype=np.float64)
        m[0, 0], m[0, 1] = c, -s
        m[1, 0], m[1, 1] = s, c
        m[0, 2] = cx - c * cx + s * cy
        m[1, 2] = cy - s * cx - c * cy
        return cls(m)

    def then(self, other: Affine) -> Affine:
        """先 self 再 other。"""
        return Affine(other.matrix @ self.matrix)

    @property
    def inverse(self) -> Affine:
        return Affine(np.linalg.inv(self.matrix))

    def apply_to_points(self, pts: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
        return pts @ self.matrix[:2, :2].T + self.matrix[:2, 2]

    def apply_to_box(self, box: Box) -> Box:
        return box.transformed(self.matrix)


@dataclass
class PreContext:
    image: np.ndarray  # BGR uint8 (H, W, 3)
    original_size: tuple[int, int]  # (w, h)
    transform: Affine
    meta: dict[str, Any] = field(default_factory=dict)
    config: PreprocessConfig = field(default_factory=PreprocessConfig)
    debug: bool = False


@runtime_checkable
class Preprocessor(Protocol):
    name: ClassVar[str]
    default_options: ClassVar[Mapping[str, Any]]

    def apply(self, ctx: PreContext) -> PreContext | None:
        """就地修改 ctx 并返回它；返回 None 表示未改动。"""
        ...


_PREPROCESSORS: dict[str, type] = {}
_EP_SCANNED = False


def register_preprocessor(name: str):
    def decorator(cls):
        cls.name = name
        _PREPROCESSORS[name] = cls
        return cls

    return decorator


def _load_builtins() -> None:
    from oxocr.preprocess import dark_to_light, pad_border  # noqa: F401  触发注册


def resolve_preprocessor(name: str) -> Preprocessor:
    _load_builtins()
    if name in _PREPROCESSORS:
        return _PREPROCESSORS[name]()
    global _EP_SCANNED
    if not _EP_SCANNED:
        _EP_SCANNED = True
        from importlib.metadata import entry_points

        for ep in entry_points(group="oxocr.preprocessors"):
            try:
                cls = ep.load()
                _PREPROCESSORS.setdefault(ep.name, cls)
            except Exception:
                continue
    if name in _PREPROCESSORS:
        return _PREPROCESSORS[name]()
    raise PluginNotFoundError(name, "preprocess", sorted(_PREPROCESSORS))


def merged_options(plugin: Any, overrides: Mapping[str, Any] | None) -> dict[str, Any]:
    opts = dict(getattr(plugin, "default_options", {}) or {})
    opts.update(overrides or {})
    return opts


class PreprocessPipeline:
    def __init__(
        self,
        steps: Sequence[str | Preprocessor],
        options: Mapping[str, Mapping] | None = None,
        on_error: Literal["skip", "raise"] = "skip",
    ):
        self.steps = list(steps)
        self.options = dict(options or {})
        self.on_error = on_error
        self._stages: list[StageLog] = []

    @property
    def stages(self) -> list[StageLog]:
        return list(self._stages)

    def run(self, image: np.ndarray, *, debug: bool = False) -> tuple[np.ndarray, PreContext]:
        h, w = image.shape[:2]
        ctx = PreContext(
            image=image,
            original_size=(w, h),
            transform=Affine.identity(),
            config=PreprocessConfig(steps=[s if isinstance(s, str) else s.name for s in self.steps]),
            debug=debug,
        )
        self._stages = []
        for step in self.steps:
            plugin = resolve_preprocessor(step) if isinstance(step, str) else step
            name = getattr(plugin, "name", type(plugin).__name__)
            opts = merged_options(plugin, self.options.get(name))
            plugin.options = opts  # 管线注入合并后的选项
            if not opts.get("enabled", True):
                self._stages.append(StageLog(name=name, kind="preprocess", enabled=False))
                continue
            before_image = ctx.image
            before_shape = ctx.image.shape
            before_matrix = ctx.transform.matrix.copy()
            t0 = time.perf_counter()
            try:
                result = plugin.apply(ctx)
                if result is not None:
                    ctx = result
            except Exception as e:
                elapsed = time.perf_counter() - t0
                if self.on_error == "raise":
                    raise StageError(name, "preprocess", e) from e
                logger.warning("前处理插件 %s 失败，跳过: %s", name, e)
                self._stages.append(
                    StageLog(name=name, kind="preprocess", elapsed=elapsed, error=str(e))
                )
                continue
            elapsed = time.perf_counter() - t0
            changed = (
                ctx.image is not before_image
                or ctx.image.shape != before_shape
                or not np.array_equal(ctx.transform.matrix, before_matrix)
            )
            self._stages.append(
                StageLog(name=name, kind="preprocess", elapsed=elapsed, changed=changed)
            )
        return ctx.image, ctx
