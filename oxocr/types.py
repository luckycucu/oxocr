"""oxocr 公共数据类型。

约定：
- 所有 box 均为**原图像素坐标**（前处理几何改动由核心逆映射）。
- ndarray 图像为 BGR uint8 (H, W, 3)。
- 本模块惰性导入 numpy，保证 `import oxocr` / `list_models()` 接近纯 stdlib 开销。
"""
from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np

    from oxocr.config import OcrConfig
    from oxocr.postprocess.base import PostprocessPipeline

__all__ = [
    "Box",
    "CharBox",
    "TextItem",
    "TextLine",
    "TextBlock",
    "Document",
    "Elapse",
    "StageLog",
    "ThemeType",
    "ThemeResult",
    "DebugInfo",
    "OcrResult",
    "LoadedModelInfo",
]


def _np():
    import numpy as np

    return np


@dataclass(frozen=True)
class Box:
    """四边形文本框，(4, 2) float32，顺时针 tl/tr/br/bl。"""

    points: np.ndarray

    @classmethod
    def from_rect(cls, x: float, y: float, w: float, h: float) -> Box:
        np = _np()
        return cls(
            np.array(
                [[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float32
            )
        )

    @classmethod
    def from_points(cls, pts: Sequence | np.ndarray) -> Box:
        np = _np()
        arr = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
        if arr.shape[0] != 4:
            raise ValueError(f"Box 需要 4 个点，收到 {arr.shape[0]} 个")
        return cls(arr.copy())

    @property
    def x_min(self) -> float:
        return float(min(p[0] for p in self.points))

    @property
    def x_max(self) -> float:
        return float(max(p[0] for p in self.points))

    @property
    def y_min(self) -> float:
        return float(min(p[1] for p in self.points))

    @property
    def y_max(self) -> float:
        return float(max(p[1] for p in self.points))

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        return self.y_max - self.y_min

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x_min + self.x_max) / 2.0, (self.y_min + self.y_max) / 2.0)

    @property
    def y_center(self) -> float:
        return (self.y_min + self.y_max) / 2.0

    def as_rect(self) -> tuple[float, float, float, float]:
        return (self.x_min, self.y_min, self.width, self.height)

    def to_list(self) -> list[list[float]]:
        return [[float(x), float(y)] for x, y in self.points]

    def shifted(self, dx: float, dy: float) -> Box:
        np = _np()
        return Box(self.points + np.array([dx, dy], dtype=np.float32))

    def transformed(self, matrix: np.ndarray) -> Box:
        """套用 3x3 仿射矩阵（原图 → 处理图 或其逆）。"""
        np = _np()
        m = np.asarray(matrix, dtype=np.float64)
        pts = np.asarray(self.points, dtype=np.float64)
        out = pts @ m[:2, :2].T + m[:2, 2]
        return Box(out.astype(np.float32))

    def area(self) -> float:
        return polygon_area(self.points)

    def iou(self, other: Box) -> float:
        inter = polygon_intersection_area(self.points, other.points)
        union = self.area() + other.area() - inter
        return inter / union if union > 0 else 0.0

    def __array__(self, dtype=None, copy=None) -> np.ndarray:
        np = _np()
        arr = np.asarray(self.points)
        if dtype is not None:
            arr = arr.astype(dtype)
        return arr.copy() if copy else arr


def polygon_area(points: Sequence | np.ndarray) -> float:
    """鞋带公式求多边形面积（替代 shapely）。"""
    pts = [(float(p[0]), float(p[1])) for p in points]
    n = len(pts)
    if n < 3:
        return 0.0
    total = 0.0
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        total += x1 * y2 - x2 * y1
    return abs(total) / 2.0


def polygon_perimeter(points: Sequence | np.ndarray) -> float:
    pts = [(float(p[0]), float(p[1])) for p in points]
    n = len(pts)
    if n < 2:
        return 0.0
    total = 0.0
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        total += ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
    return total


def polygon_intersection_area(a: Sequence | np.ndarray, b: Sequence | np.ndarray) -> float:
    """Sutherland–Hodgman 裁剪（两个凸多边形）+ 鞋带面积。"""
    subject = [(float(p[0]), float(p[1])) for p in a]
    clip = [(float(p[0]), float(p[1])) for p in b]
    if len(subject) < 3 or len(clip) < 3:
        return 0.0

    def inside(p, edge_start, edge_end):
        return (
            (edge_end[0] - edge_start[0]) * (p[1] - edge_start[1])
            - (edge_end[1] - edge_start[1]) * (p[0] - edge_start[0])
        ) >= 0

    def intersection(p1, p2, edge_start, edge_end):
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = edge_start
        x4, y4 = edge_end
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-12:
            return p2
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))

    output = subject
    n = len(clip)
    # 裁剪多边形统一为鞋带符号为正的方向（inside 判定依赖该方向）
    signed = sum(
        clip[j][0] * clip[(j + 1) % n][1] - clip[(j + 1) % n][0] * clip[j][1]
        for j in range(n)
    )
    if signed < 0:
        clip = list(reversed(clip))
    for i in range(n):
        edge_start, edge_end = clip[i], clip[(i + 1) % n]
        input_list = output
        output = []
        if not input_list:
            break
        prev = input_list[-1]
        for curr in input_list:
            if inside(curr, edge_start, edge_end):
                if not inside(prev, edge_start, edge_end):
                    output.append(intersection(prev, curr, edge_start, edge_end))
                output.append(curr)
            elif inside(prev, edge_start, edge_end):
                output.append(intersection(prev, curr, edge_start, edge_end))
            prev = curr
    return polygon_area(output)


@dataclass
class CharBox:
    char: str
    score: float
    box: Box


@dataclass
class TextItem:
    text: str
    score: float
    box: Box
    angle: float = 0.0
    chars: list[CharBox] | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "text": self.text,
            "score": float(self.score),
            "box": self.box.to_list(),
            "angle": float(self.angle),
        }
        if self.chars is not None:
            data["chars"] = [
                {"char": c.char, "score": float(c.score), "box": c.box.to_list()}
                for c in self.chars
            ]
        if self.meta:
            data["meta"] = self.meta
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TextItem:
        chars = None
        if data.get("chars") is not None:
            chars = [
                CharBox(c["char"], float(c["score"]), Box.from_points(c["box"]))
                for c in data["chars"]
            ]
        return cls(
            text=str(data["text"]),
            score=float(data.get("score", 0.0)),
            box=Box.from_points(data["box"]),
            angle=float(data.get("angle", 0.0)),
            chars=chars,
            meta=dict(data.get("meta") or {}),
        )


@dataclass
class TextLine:
    items: list[TextItem]
    box: Box
    text: str = ""
    score: float = 0.0

    @property
    def y_center(self) -> float:
        return self.box.y_center

    @property
    def x_min(self) -> float:
        return self.box.x_min

    @property
    def x_max(self) -> float:
        return self.box.x_max

    @property
    def height(self) -> float:
        return self.box.height


@dataclass
class TextBlock:
    kind: str  # "raw" | "line" | "paragraph"
    lines: list[TextLine]
    text: str
    box: Box | None = None
    score: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "text": self.text,
            "box": self.box.to_list() if self.box is not None else None,
            "score": round(float(self.score), 5),
            "line_count": len(self.lines),
            "meta": self.meta,
        }


@dataclass
class Document:
    size: tuple[int, int]  # 原图 (w, h)
    lines: list[TextLine] = field(default_factory=list)
    blocks: list[TextBlock] = field(default_factory=list)


@dataclass
class Elapse:
    load: float = 0.0
    preprocess: float = 0.0
    det: float = 0.0
    cls: float = 0.0
    rec: float = 0.0
    postprocess: float = 0.0
    total: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "load": round(self.load, 4),
            "preprocess": round(self.preprocess, 4),
            "det": round(self.det, 4),
            "cls": round(self.cls, 4),
            "rec": round(self.rec, 4),
            "postprocess": round(self.postprocess, 4),
            "total": round(self.total, 4),
        }


@dataclass
class StageLog:
    name: str
    kind: str  # "preprocess" | "postprocess"
    enabled: bool = True
    elapsed: float = 0.0
    changed: bool = False
    error: str | None = None


class ThemeType(Enum):
    DARK = "深色主题 (Dark)"
    LIGHT = "浅色主题 (Light)"


@dataclass
class ThemeResult:
    theme: ThemeType
    confidence: float
    edge_brightness: float
    titlebar_brightness: float
    overall_brightness: float
    dark_pixel_ratio: float
    composite_score: float


@dataclass
class DebugInfo:
    processed_image: np.ndarray | None = None
    transform: np.ndarray | None = None  # 3x3，原图 → 送入图
    theme: ThemeResult | None = None
    raw_output: dict[str, Any] = field(default_factory=dict)
    stage_log: list[StageLog] = field(default_factory=list)


@dataclass
class LoadedModelInfo:
    name: str
    options_key: str
    loaded_at: float
    infer_count: int


@dataclass
class OcrResult:
    items: list[TextItem]
    document: Document
    image_size: tuple[int, int]
    model: str
    elapse: Elapse
    config: OcrConfig
    debug: DebugInfo | None = None
    _text: str = field(default="", repr=False, compare=False)
    _image: np.ndarray | None = field(default=None, repr=False, compare=False)
    _post: PostprocessPipeline | None = field(default=None, repr=False, compare=False)

    @property
    def text(self) -> str:
        return self._text

    @property
    def text_chars(self) -> int:
        return sum(len(it.text) for it in self.items)

    def as_text(
        self, mode: str | None = None, *, output: str = "text", **options
    ) -> str:
        """换排版/输出格式，只重跑组装与渲染，不重新推理。"""
        if self._post is None:
            sep = (
                self.config.postprocess.paragraph_separator
                if (mode or self.config.postprocess.mode) == "para"
                else self.config.postprocess.line_separator
            )
            return sep.join(b.text for b in self.document.blocks)
        rendered = self._post.render(self.document, mode=mode, output=output, **options)
        if isinstance(rendered, dict):
            return json.dumps(rendered, ensure_ascii=False)
        return rendered

    def as_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "image_size": list(self.image_size),
            "text": self._text,
            "elapse": self.elapse.as_dict(),
            "items": [it.to_dict() for it in self.items],
            "blocks": [b.to_dict() for b in self.document.blocks],
        }

    def as_json(self, *, indent: int | None = 2, ensure_ascii: bool = False) -> str:
        return json.dumps(self.as_dict(), indent=indent, ensure_ascii=ensure_ascii)

    def save(
        self,
        path: str | Path,
        *,
        format: str = "text",
        mode: str | None = None,
    ) -> None:
        if format == "json":
            content = self.as_json()
        else:
            content = self.as_text(mode)
        Path(path).write_text(content, encoding="utf-8")

    def crop(self, item: TextItem | int) -> np.ndarray:
        """从原图裁出条目区域（透视矫正）。"""
        if self._image is None:
            raise ValueError("结果未保留原图，无法裁剪")
        if isinstance(item, int):
            item = self.items[item]
        from oxocr.runtime.imageops import get_rotate_crop_image

        return get_rotate_crop_image(self._image, item.box.points)

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self) -> Iterator[TextItem]:
        return iter(self.items)

    def __bool__(self) -> bool:
        return any(it.text.strip() for it in self.items)
