"""空白与全角半角规范化（on_blocks，不影响几何）。"""
from __future__ import annotations

import re
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, ClassVar

from oxocr.types import TextBlock

if TYPE_CHECKING:
    from oxocr.postprocess.base import PostContext

__all__ = ["NormalizeSpace", "normalize_text"]

_WS_RE = re.compile(r"[\t\xa0\u3000]")
_MULTI_SPACE_RE = re.compile(r" {2,}")


def _fullwidth_to_halfwidth(text: str) -> str:
    out = []
    for ch in text:
        code = ord(ch)
        if code == 0x3000:
            out.append(" ")
        elif 0xFF01 <= code <= 0xFF5E:
            out.append(chr(code - 0xFEE0))
        else:
            out.append(ch)
    return "".join(out)


def normalize_text(text: str, *, fullwidth_to_halfwidth: bool = False, collapse_spaces: bool = True) -> str:
    if fullwidth_to_halfwidth:
        text = _fullwidth_to_halfwidth(text)
    text = _WS_RE.sub(" ", text)
    if collapse_spaces:
        text = _MULTI_SPACE_RE.sub(" ", text)
    return text.strip()


class NormalizeSpace:
    name: ClassVar[str] = "normalize_space"
    default_options: ClassVar[Mapping[str, Any]] = {
        "enabled": True,
        "fullwidth_to_halfwidth": False,
    }

    def on_blocks(self, blocks: list[TextBlock], ctx: PostContext) -> list[TextBlock]:
        opts = getattr(self, "options", self.default_options)
        # layout 模式的空格数量承载版式信息，不折叠
        collapse = ctx.config.mode != "layout"
        for block in blocks:
            block.text = normalize_text(
                block.text,
                fullwidth_to_halfwidth=bool(opts.get("fullwidth_to_halfwidth", False)),
                collapse_spaces=collapse,
            )
        return blocks
