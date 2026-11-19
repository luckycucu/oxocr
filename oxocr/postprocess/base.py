"""后处理协议与管线：分行 → 修饰(on_lines) → 组装 → 修饰(on_blocks) → 渲染。

run() 与 render() 分离：result.as_text(mode) 只重跑组装与渲染，不重新推理。
"""
from __future__ import annotations

import logging
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar, Protocol, runtime_checkable

from oxocr.config import PostprocessConfig
from oxocr.errors import PluginNotFoundError, StageError
from oxocr.postprocess.grouping import group_lines
from oxocr.types import Document, StageLog, TextBlock, TextItem, TextLine

logger = logging.getLogger("oxocr.postprocess")

__all__ = [
    "PostContext",
    "Postprocessor",
    "Assembler",
    "Joiner",
    "Renderer",
    "register_postprocessor",
    "register_assembler",
    "register_joiner",
    "register_renderer",
    "PostprocessPipeline",
]


@dataclass
class PostContext:
    document: Document
    config: PostprocessConfig
    meta: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Postprocessor(Protocol):
    name: ClassVar[str]
    default_options: ClassVar[Mapping[str, Any]]

    def on_lines(self, lines: list[TextLine], ctx: PostContext) -> list[TextLine]: ...

    def on_blocks(self, blocks: list[TextBlock], ctx: PostContext) -> list[TextBlock]: ...


@runtime_checkable
class Assembler(Protocol):
    name: ClassVar[str]
    default_join: ClassVar[str]

    def assemble(self, lines: list[TextLine], ctx: PostContext, joiner: Any) -> list[TextBlock]: ...


@runtime_checkable
class Joiner(Protocol):
    name: ClassVar[str]

    def join(self, items: list[TextItem], ctx: PostContext) -> str: ...


@runtime_checkable
class Renderer(Protocol):
    name: ClassVar[str]

    def render(self, blocks: list[TextBlock], ctx: PostContext) -> str | dict: ...


_POSTPROCESSORS: dict[str, type] = {}
_ASSEMBLERS: dict[str, type] = {}
_JOINERS: dict[str, type] = {}
_RENDERERS: dict[str, type] = {}
_EP_SCANNED: dict[str, bool] = {}


def _register(registry: dict[str, type], name: str):
    def decorator(cls):
        cls.name = name
        registry[name] = cls
        return cls

    return decorator


def register_postprocessor(name: str):
    return _register(_POSTPROCESSORS, name)


def register_assembler(name: str):
    return _register(_ASSEMBLERS, name)


def register_joiner(name: str):
    return _register(_JOINERS, name)


def register_renderer(name: str):
    return _register(_RENDERERS, name)


def _load_builtins() -> None:
    from oxocr.postprocess.assemblers import (
        LayoutAssembler,
        LineAssembler,
        ParaAssembler,
        RawAssembler,
    )
    from oxocr.postprocess.corrections import Corrections
    from oxocr.postprocess.ghost import GhostFilter
    from oxocr.postprocess.joiners import (
        LayoutSpaceJoiner,
        NoSpaceJoiner,
        SingleSpaceJoiner,
        SmartSpaceJoiner,
    )
    from oxocr.postprocess.normalize import NormalizeSpace
    from oxocr.postprocess.render import JsonRenderer, TextRenderer

    _POSTPROCESSORS.setdefault("ghost_filter", GhostFilter)
    _POSTPROCESSORS.setdefault("corrections", Corrections)
    _POSTPROCESSORS.setdefault("normalize_space", NormalizeSpace)
    _ASSEMBLERS.setdefault("raw", RawAssembler)
    _ASSEMBLERS.setdefault("line", LineAssembler)
    _ASSEMBLERS.setdefault("para", ParaAssembler)
    _ASSEMBLERS.setdefault("layout", LayoutAssembler)
    _JOINERS.setdefault("smart_space", SmartSpaceJoiner)
    _JOINERS.setdefault("layout_space", LayoutSpaceJoiner)
    _JOINERS.setdefault("single_space", SingleSpaceJoiner)
    _JOINERS.setdefault("no_space", NoSpaceJoiner)
    _RENDERERS.setdefault("text", TextRenderer)
    _RENDERERS.setdefault("json", JsonRenderer)


def _resolve(registry: dict[str, type], group: str, name: str):
    _load_builtins()
    if name in registry:
        return registry[name]()
    if not _EP_SCANNED.get(group):
        _EP_SCANNED[group] = True
        from importlib.metadata import entry_points

        for ep in entry_points(group=f"oxocr.{group}"):
            try:
                registry.setdefault(ep.name, ep.load())
            except Exception:
                continue
    if name in registry:
        return registry[name]()
    raise PluginNotFoundError(name, group, sorted(registry))


def resolve_postprocessor(name: str):
    return _resolve(_POSTPROCESSORS, "postprocessors", name)


def resolve_assembler(name: str):
    return _resolve(_ASSEMBLERS, "assemblers", name)


def resolve_joiner(name: str):
    return _resolve(_JOINERS, "joiners", name)


def resolve_renderer(name: str):
    return _resolve(_RENDERERS, "renderers", name)


def _merged_options(plugin: Any, overrides: Mapping[str, Any] | None) -> dict[str, Any]:
    opts = dict(getattr(plugin, "default_options", {}) or {})
    opts.update(overrides or {})
    return opts


class PostprocessPipeline:
    def __init__(
        self,
        config: PostprocessConfig,
        modifiers: Sequence[str | Postprocessor] | None = None,
        assembler: str | Assembler | None = None,
        joiner: str | Joiner | None = None,
        renderer: str | Renderer | None = None,
    ):
        self.config = config
        self.meta: dict[str, Any] = {}
        self._stages: list[StageLog] = []
        self._document_mode: str | None = None

        specs = list(modifiers) if modifiers is not None else list(config.steps)
        self.modifiers = [
            resolve_postprocessor(s) if isinstance(s, str) else s for s in specs
        ]
        if assembler is None:
            self.assembler = resolve_assembler(config.mode)
        else:
            self.assembler = resolve_assembler(assembler) if isinstance(assembler, str) else assembler
        self._injected_joiner = (
            (resolve_joiner(joiner) if isinstance(joiner, str) else joiner)
            if joiner is not None
            else None
        )
        if renderer is None:
            self.renderer = resolve_renderer(config.output)
        else:
            self.renderer = resolve_renderer(renderer) if isinstance(renderer, str) else renderer

    @property
    def stages(self) -> list[StageLog]:
        return list(self._stages)

    def _joiner_for(self, assembler) -> Any:
        if self._injected_joiner is not None:
            return self._injected_joiner
        name = self.config.join or getattr(assembler, "default_join", "smart_space")
        return resolve_joiner(name)

    def _plugin_options(self, plugin: Any) -> dict[str, Any]:
        name = getattr(plugin, "name", type(plugin).__name__)
        opts = _merged_options(plugin, self.config.options.get(name))
        plugin.options = opts
        return opts

    def _run_hook(self, plugin: Any, hook: str, value, ctx: PostContext, *, record: bool = True):
        name = getattr(plugin, "name", type(plugin).__name__)
        opts = self._plugin_options(plugin)
        if not opts.get("enabled", True):
            if record:
                self._stages.append(StageLog(name=name, kind="postprocess", enabled=False))
            return value
        fn = getattr(plugin, hook, None)
        if not callable(fn):
            return value
        if hook == "on_lines":
            before = (sum(len(line.items) for line in value),
                      sum(len(it.text) for line in value for it in line.items))
        else:
            before = (len(value), sum(len(b.text) for b in value))
        t0 = time.perf_counter()
        try:
            result = fn(value, ctx)
            if result is not None:
                value = result
        except Exception as e:
            if self.config.enabled and ctx.config.options.get(name, {}).get("on_error") == "raise":
                raise StageError(name, "postprocess", e) from e
            logger.warning("后处理插件 %s.%s 失败，跳过: %s", name, hook, e)
            if record:
                self._stages.append(
                    StageLog(name=name, kind="postprocess", elapsed=time.perf_counter() - t0, error=str(e))
                )
            return value
        elapsed = time.perf_counter() - t0
        if hook == "on_lines":
            after = (sum(len(line.items) for line in value),
                     sum(len(it.text) for line in value for it in line.items))
        else:
            after = (len(value), sum(len(b.text) for b in value))
        if record:
            self._stages.append(
                StageLog(name=name, kind="postprocess", elapsed=elapsed, changed=after != before)
            )
        return value

    def run(self, items: list[TextItem], image_size: tuple[int, int]) -> Document:
        self.meta = {"image_size": image_size}
        self._stages = []
        mode = getattr(self.assembler, "name", self.config.mode)
        raw_only = not self.config.enabled or mode == "raw"

        if raw_only:
            ordered = sorted(items, key=lambda it: (it.box.y_center, it.box.x_min))
            lines = [
                TextLine(items=[it], box=it.box, text=it.text, score=it.score) for it in ordered
            ]
            blocks = [
                TextBlock(kind="raw", lines=[line], text=line.text, box=line.box, score=line.score)
                for line in lines
            ]
            self._document_mode = "raw"
            return Document(size=image_size, lines=lines, blocks=blocks)

        lines = group_lines(items)
        document = Document(size=image_size, lines=lines, blocks=[])
        ctx = PostContext(document=document, config=self.config, meta=self.meta)
        for plugin in self.modifiers:
            lines = self._run_hook(plugin, "on_lines", lines, ctx)
        document.lines = lines
        blocks = self.assembler.assemble(lines, ctx, self._joiner_for(self.assembler))
        for plugin in self.modifiers:
            blocks = self._run_hook(plugin, "on_blocks", blocks, ctx)
        document.blocks = blocks
        self._document_mode = mode
        return document

    def render(
        self,
        document: Document,
        *,
        mode: str | None = None,
        output: str | None = None,
        **options: Any,
    ) -> str | dict:
        mode = mode or self._document_mode or self.config.mode
        output = output or self.config.output
        ctx = PostContext(document=document, config=self.config, meta=self.meta)

        if mode == self._document_mode:
            blocks = document.blocks
        else:
            assembler = resolve_assembler(mode)
            joiner = self._joiner_for(assembler)
            # 重置行文本以便新 JoinPolicy 重新连接；不改动原 document
            lines = [
                TextLine(items=line.items, box=line.box, text="", score=line.score)
                for line in document.lines
            ]
            blocks = assembler.assemble(lines, ctx, joiner)
            if mode != "raw":
                for plugin in self.modifiers:
                    blocks = self._run_hook(plugin, "on_blocks", blocks, ctx, record=False)

        renderer = self.renderer if output == getattr(self.renderer, "name", None) else resolve_renderer(output)
        renderer.options = _merged_options(renderer, self.config.options.get(getattr(renderer, "name", "")))
        renderer.options.update(options)
        return renderer.render(blocks, ctx)
