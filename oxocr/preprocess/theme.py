"""桌面截图主题检测（多区域加权评分，移植自参考实现）。"""
from __future__ import annotations

import numpy as np

from oxocr.runtime import imageops
from oxocr.types import ThemeResult, ThemeType

__all__ = ["ThemeDetector"]


class ThemeDetector:
    EDGE_RATIO = 0.06
    TITLEBAR_RATIO = 0.08
    DARK_PIXEL_TH = 100

    W_EDGE = 0.40
    W_TITLE = 0.30
    W_OVERALL = 0.20
    W_RATIO = 0.10

    def detect(self, img: np.ndarray, threshold: float = 128) -> ThemeResult:
        img = self._resize(img, max_dim=1920)
        gray = np.dot(img[..., :3], [0.114, 0.587, 0.299]).astype(np.uint8)
        h, w = gray.shape[:2]

        edge_region = self._extract_edges(gray, h, w)
        titlebar_region = self._extract_titlebar(gray, h, w)

        edge_brightness = float(np.median(edge_region))
        titlebar_brightness = float(np.median(titlebar_region))
        overall_brightness = float(np.median(gray))
        dark_pixel_ratio = float(np.mean(gray < self.DARK_PIXEL_TH))

        composite_score = (
            self.W_EDGE * edge_brightness
            + self.W_TITLE * titlebar_brightness
            + self.W_OVERALL * overall_brightness
            + self.W_RATIO * (dark_pixel_ratio * 255)
        )
        is_dark = composite_score < threshold
        distance = abs(composite_score - threshold)
        confidence = min(distance / max(threshold, 1e-6), 1.0)
        return ThemeResult(
            theme=ThemeType.DARK if is_dark else ThemeType.LIGHT,
            confidence=round(confidence, 4),
            edge_brightness=round(edge_brightness, 2),
            titlebar_brightness=round(titlebar_brightness, 2),
            overall_brightness=round(overall_brightness, 2),
            dark_pixel_ratio=round(dark_pixel_ratio, 4),
            composite_score=round(composite_score, 2),
        )

    @staticmethod
    def _resize(img: np.ndarray, max_dim: int = 1920) -> np.ndarray:
        h, w = img.shape[:2]
        if max(h, w) <= max_dim:
            return img
        scale = max_dim / max(h, w)
        return imageops.resize(img, (int(w * scale), int(h * scale)), "area")

    def _extract_edges(self, gray: np.ndarray, h: int, w: int) -> np.ndarray:
        ew = max(int(w * self.EDGE_RATIO), 2)
        eh = max(int(h * self.EDGE_RATIO), 2)
        top = gray[:eh, :]
        bottom = gray[-eh:, :]
        left = gray[:, :ew]
        right = gray[:, -ew:]
        return np.concatenate([top.ravel(), bottom.ravel(), left.ravel(), right.ravel()])

    def _extract_titlebar(self, gray: np.ndarray, h: int, w: int) -> np.ndarray:
        th = max(int(h * self.TITLEBAR_RATIO), 2)
        return gray[:th, :].ravel()
