"""超宽行 crop 的重叠窗口与 CTC 位置拼接（移植 anaksor-ocr longline.py）。

窗口带约 1.5 倍行高的上下文越过拼接点，每个高棉字簇（基字符 + 下标/元音/符号）
整体取自离边缘最远的窗口。按空白列切分会把左置元音（如 េ）与辅音切开（约 8% 的切口）。
"""

from __future__ import annotations

import numpy as np

from oxocr.models.anaksor_ocr.text import COENG, MARKS, normalize
from oxocr.runtime import imageops

__all__ = ["ink_profile", "line_windows", "clusters", "stitch"]


def ink_profile(crop: np.ndarray) -> np.ndarray:
    """每列墨量（0 = 空白），兼容浅底深字与深底浅字。crop 为 BGR。"""
    g = crop @ np.array([0.114, 0.587, 0.299], dtype=np.float32) if crop.ndim == 3 else crop
    g = g.astype(np.uint8)
    thresh = imageops.otsu_threshold(g)
    bw = g > thresh
    border = np.concatenate([bw[0], bw[-1], bw[:, 0], bw[:, -1]])
    ink = bw if border.mean() < 0.5 else ~bw  # 深底：亮部是文字
    return ink.sum(0).astype(np.float32)


def line_windows(crop: np.ndarray, max_ratio: float, overlap: float = 3.0, fill: float = 0.9):
    """行 crop 的重叠窗口。

    返回 ([(piece, x0, x1)], stitches)：窗口 i 为 crop[:, x0:x1]；stitches[i] 是
    窗口 i 与 i+1 重叠区内切换归属的列（中部附近墨量最低列）。
    宽度不超过 max_ratio * 高度的 crop 为单窗口、无 stitches。
    """
    h, w = crop.shape[:2]
    if w <= max_ratio * h:
        return [(crop, 0, w)], []
    win = min(max_ratio * h, max(fill * max_ratio * h, 2.5 * overlap * h))
    ov = min(overlap * h, 0.4 * win)  # 小 max_ratio：保住每个窗口自己的部分
    n = int(np.ceil((w - ov) / (win - ov)))
    step = (w - ov) / n
    spans = [(int(round(i * step)), int(round(min(w, i * step + step + ov)))) for i in range(n)]
    spans[-1] = (spans[-1][0], w)
    prof = np.convolve(ink_profile(crop), np.ones(3) / 3, mode="same")
    stitches = []
    for (_, x1), (x0n, _) in zip(spans, spans[1:], strict=False):
        mid = (x0n + x1) / 2
        lo, hi = int(mid - ov / 6), int(mid + ov / 6) + 1
        seg = prof[lo:hi] + 0.01 * np.abs(np.arange(lo, hi) - mid)  # 平局取最靠中部
        stitches.append(lo + int(np.argmin(seg)))
    return [(crop[:, x0:x1], x0, x1) for x0, x1 in spans], stitches


def clusters(chars):
    """原始 CTC 字符 [(char, x, prob)] → 以基字符定位的高棉字簇。"""
    out = []
    for ch, x, p in chars:
        if out and (ch in MARKS or ch == COENG or out[-1]["chars"][-1][0] == COENG):
            out[-1]["chars"].append((ch, p))
        else:
            out.append({"x": x, "chars": [(ch, p)]})
    return out


def stitch(windows, stitches, results) -> tuple[str, float]:
    """由窗口识别结果拼出一行的文本与得分：窗口 i 贡献基字符落在
    stitches[i-1] 与 stitches[i] 之间的字簇。chars 为原始码点（未归一化），
    归一化只在拼接后的最终文本上做一次。
    """
    kept = []
    for i, ((_, x0, x1), r) in enumerate(zip(windows, results, strict=True)):
        lo = stitches[i - 1] if i > 0 else -np.inf
        hi = stitches[i] if i < len(stitches) else np.inf
        for c in clusters(r["chars"]):
            if lo <= x0 + c["x"] * (x1 - x0) < hi:
                kept += c["chars"]
    text = normalize("".join(ch for ch, _ in kept))
    score = float(np.mean([p for _, p in kept])) if kept else 0.0
    return text, score
