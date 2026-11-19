"""渲染器：text / json。"""
from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, ClassVar

from oxocr.types import TextBlock

if TYPE_CHECKING:
    from oxocr.postprocess.base import PostContext

__all__ = ["TextRenderer", "JsonRenderer"]


class TextRenderer:
    name: ClassVar[str] = "text"
    default_options: ClassVar[Mapping[str, Any]] = {}

    def render(self, blocks: list[TextBlock], ctx: PostContext) -> str:
        if not blocks:
            return ""
        separator = (
            ctx.config.paragraph_separator
            if blocks[0].kind == "paragraph"
            else ctx.config.line_separator
        )
        return separator.join(b.text for b in blocks)


class JsonRenderer:
    name: ClassVar[str] = "json"
    default_options: ClassVar[Mapping[str, Any]] = {}

    def render(self, blocks: list[TextBlock], ctx: PostContext) -> dict:
        items = ctx.meta.get("items") or []
        elapse = ctx.meta.get("elapse")
        return {
            "model": ctx.meta.get("model", ""),
            "image_size": list(ctx.document.size),
            "blocks": [b.to_dict() for b in blocks],
            "items": [it.to_dict() if hasattr(it, "to_dict") else it for it in items],
            "elapse": elapse.as_dict() if hasattr(elapse, "as_dict") else (elapse or {}),
        }
