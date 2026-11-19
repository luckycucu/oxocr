"""深色转浅色前处理：分段线性 LUT + Gamma + Unsharp（参数移植自参考实现）。

仅在主题判定为深色时转换；转换后重新判定主题写入 meta，
供 pad_border 决定填充颜色（补边颜色取决于最终送入模型的图像主题）。
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

import numpy as np
from PIL import Image, ImageFilter

from oxocr.preprocess.base import PreContext, register_preprocessor
from oxocr.preprocess.theme import ThemeDetector
from oxocr.types import ThemeType

__all__ = ["DarkToLight", "DarkToLightConverter"]


class DarkToLightConverter:
    def __init__(
        self,
        target_bg: int = 245,
        target_fg_max: int = 150,
        gamma: float = 0.85,
        sharpen_strength: float = 0.3,
        bg_percentile: float = 5.0,
        fg_percentile: float = 95.0,
    ):
        self.target_bg = int(target_bg)
        self.target_fg_max = int(target_fg_max)
        self.gamma = float(gamma)
        self.sharpen_strength = float(sharpen_strength)
        self.bg_percentile = float(bg_percentile)
        self.fg_percentile = float(fg_percentile)

    def convert(self, img: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        """返回 (转换后灰度图, 映射参数)。"""
        if img.ndim == 3:
            gray = np.dot(img[..., :3], [0.114, 0.587, 0.299]).astype(np.uint8)
        else:
            gray = img.copy()
        bg_level, fg_level = self._analyze_distribution(gray)
        lut = self._build_lut(bg_level, fg_level)
        mapped = lut[gray]
        corrected = self._apply_gamma(mapped)
        final = self._sharpen(corrected)
        return final, {
            "bg_level": bg_level,
            "fg_level": fg_level,
            "target_bg": self.target_bg,
            "target_fg_max": self.target_fg_max,
            "gamma": self.gamma,
        }

    def _analyze_distribution(self, gray: np.ndarray) -> tuple[int, int]:
        flat = gray.ravel()
        bg_level = int(np.percentile(flat, self.bg_percentile))
        fg_level = int(np.percentile(flat, self.fg_percentile))
        bg_level = max(bg_level, 0)
        fg_level = min(fg_level, 255)
        if fg_level - bg_level < 30:
            bg_level = max(bg_level - 15, 0)
            fg_level = min(fg_level + 15, 255)
        return bg_level, fg_level

    def _build_lut(self, bg_level: int, fg_level: int) -> np.ndarray:
        lut = np.zeros(256, dtype=np.uint8)
        for x in range(256):
            if x <= bg_level:
                y = self.target_bg if bg_level == 0 else self.target_bg + (x / bg_level) * (255 - self.target_bg)
            elif x <= fg_level:
                span = fg_level - bg_level
                y = self.target_fg_max if span == 0 else self.target_bg - ((x - bg_level) / span) * (self.target_bg - self.target_fg_max)
            else:
                span = 255 - fg_level
                y = 0 if span == 0 else self.target_fg_max * (1.0 - (x - fg_level) / span)
            lut[x] = int(np.clip(round(y), 0, 255))
        return lut

    def _apply_gamma(self, img: np.ndarray) -> np.ndarray:
        if abs(self.gamma - 1.0) < 1e-3:
            return img
        inv_gamma = 1.0 / self.gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255.0 for i in range(256)], dtype=np.uint8)
        return table[img]

    def _sharpen(self, img: np.ndarray) -> np.ndarray:
        if self.sharpen_strength <= 0:
            return img
        pil_img = Image.fromarray(img)
        blurred = np.array(pil_img.filter(ImageFilter.GaussianBlur(radius=1.5)))
        sharpened = img.astype(np.float64) * (1.0 + self.sharpen_strength) + blurred.astype(
            np.float64
        ) * (-self.sharpen_strength)
        return np.clip(sharpened, 0, 255).astype(np.uint8)


@register_preprocessor("dark_to_light")
class DarkToLight:
    name = "dark_to_light"
    default_options: ClassVar[Mapping[str, Any]] = {
        "enabled": True,
        "target_bg": 245,
        "target_fg_max": 150,
        "gamma": 0.85,
        "sharpen_strength": 0.3,
        "bg_percentile": 5.0,
        "fg_percentile": 95.0,
        "threshold": 128,
    }

    def apply(self, ctx: PreContext) -> PreContext | None:
        opts = getattr(self, "options", self.default_options)
        detector = ThemeDetector()
        result = detector.detect(ctx.image, threshold=opts["threshold"])
        ctx.meta["theme"] = result
        if result.theme != ThemeType.DARK:
            return None
        converter = DarkToLightConverter(
            target_bg=opts["target_bg"],
            target_fg_max=opts["target_fg_max"],
            gamma=opts["gamma"],
            sharpen_strength=opts["sharpen_strength"],
            bg_percentile=opts["bg_percentile"],
            fg_percentile=opts["fg_percentile"],
        )
        gray, params = converter.convert(ctx.image)
        ctx.image = np.stack([gray] * 3, axis=-1)
        ctx.meta["dark_converted"] = True
        ctx.meta["dark_to_light_params"] = params
        # 补边颜色按最终送入模型的图像主题取值，转换后重新判定
        ctx.meta["theme"] = detector.detect(ctx.image, threshold=opts["threshold"])
        return ctx
