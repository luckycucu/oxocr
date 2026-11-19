"""小图补边前处理。

修正参考实现：补边后把平移量压入 ctx.transform，推理后由核心逆映射，
结果 box 不再偏移 padding 量。
"""
from __future__ import annotations

import random
from collections.abc import Mapping
from typing import Any, ClassVar

import numpy as np

from oxocr.preprocess.base import Affine, PreContext, register_preprocessor
from oxocr.preprocess.theme import ThemeDetector
from oxocr.types import ThemeType

__all__ = ["PadBorder"]


@register_preprocessor("pad_border")
class PadBorder:
    name = "pad_border"
    default_options: ClassVar[Mapping[str, Any]] = {
        "enabled": True,
        "threshold": 150,
        "margin": 10,
        "force": False,
        "color": "auto",
    }

    def apply(self, ctx: PreContext) -> PreContext | None:
        opts = getattr(self, "options", self.default_options)
        img = ctx.image
        h, w = img.shape[:2]
        threshold = int(opts["threshold"])
        margin = int(opts["margin"])

        if opts["force"]:
            pad = random.randint(10, 50)
            color = random.choice((0, 255))
            top = bottom = left = right = pad
        else:
            if w >= threshold and h >= threshold:
                return None
            color = self._resolve_color(opts["color"], ctx)
            top = bottom = left = right = margin
            if h + top + bottom < threshold:
                extra = threshold - (h + top + bottom)
                top += extra // 2
                bottom += extra - extra // 2
            if w + left + right < threshold:
                extra = threshold - (w + left + right)
                left += extra // 2
                right += extra - extra // 2

        pad_width = [(top, bottom), (left, right)]
        if img.ndim == 3:
            pad_width.append((0, 0))
        ctx.image = np.pad(img, pad_width, mode="constant", constant_values=color)
        ctx.transform = ctx.transform.then(Affine.translation(left, top))
        ctx.meta["pad_border"] = {"top": top, "bottom": bottom, "left": left, "right": right}
        return ctx

    @staticmethod
    def _resolve_color(color: Any, ctx: PreContext) -> int | tuple:
        if color != "auto":
            return color
        theme = ctx.meta.get("theme")
        if theme is None:
            theme = ThemeDetector().detect(ctx.image)
            ctx.meta["theme"] = theme
        return 255 if theme.theme == ThemeType.DARK else 0
