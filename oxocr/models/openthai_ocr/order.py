"""阅读顺序：行聚类 + XY-cut 分栏 + 碎框剔除（移植 OpenThai2.0 runtime/line_engine.py）。

只决定 items 顺序，不做文本拼接（排版归核心组装器）。纯 numpy/python，零额外依赖。
"""

from __future__ import annotations

import math
import re

import numpy as np

from oxocr.models.openthai_ocr.geometry import line_height, page_skew

__all__ = ["reading_order", "fragments"]

_DIGITS = set("0123456789๐๑๒๓๔๕๖๗๘๙")
_ENUM = re.compile(r"^\s*\(?([0-9๐-๙]{1,4})\s*[.)]")


def _enum(text: str) -> int | None:
    m = _ENUM.match(text)
    return int(m.group(1).translate(str.maketrans("๐๑๒๓๔๕๖๗๘๙", "0123456789"))) if m else None


def _gaps(iv: list[tuple[float, float]], min_gap: float) -> list[tuple[float, float]]:
    """Empty bands between intervals, at least min_gap wide."""
    iv = sorted(iv)
    out, reach = [], iv[0][1]
    for lo, hi in iv[1:]:
        if lo - reach >= min_gap:
            out.append((reach, lo))
        reach = max(reach, hi)
    return out


def _next_to(side: list[int], ext: list[tuple], hmed: float, gutter_on_right: bool) -> list[int]:
    """The column of `side` that touches the gutter (a side can hold several)."""
    if not side:
        return side
    g = _gaps([(ext[i][0], ext[i][1]) for i in side], 0.8 * hmed)
    if not g:
        return side
    if gutter_on_right:
        return [i for i in side if ext[i][0] >= g[-1][1]]
    return [i for i in side if ext[i][1] <= g[0][0]]


def _prose(col: list[int], ext: list[tuple], hmed: float, texts: list[str] | None = None) -> bool:
    if len(col) < 12:
        return False
    widths = [ext[i][1] - ext[i][0] for i in col]
    span = max(ext[i][1] for i in col) - min(ext[i][0] for i in col)
    if float(np.median(widths)) < 7 * hmed or float(np.mean([w >= 0.75 * span for w in widths])) < 0.5:
        return False
    if texts is not None:
        chars = [c for i in col for c in texts[i] if not c.isspace()]
        if chars and sum(c in _DIGITS for c in chars) > 0.3 * len(chars):
            return False
    return True


def _numbered_across(lcol: list[int], rcol: list[int], ext: list[tuple], hmed: float,
                     texts: list[str] | None) -> bool:
    """Most facing lines numbered n (left) and n+1 (right): a list that reads across."""
    if texts is None:
        return False
    pairs = 0
    for i in lcol:
        a = _enum(texts[i])
        if a is None:
            continue
        cy = (ext[i][2] + ext[i][3]) / 2
        if any(_enum(texts[j]) == a + 1 and abs((ext[j][2] + ext[j][3]) / 2 - cy) <= 0.5 * hmed for j in rcol):
            pairs += 1
    return pairs >= 0.5 * min(len(lcol), len(rcol))


def _blocks(idx: list[int], ext: list[tuple], hmed: float, texts: list[str] | None = None,
            depth: int = 0) -> list[list[int]]:
    """Recursive XY-cut into reading-order blocks.

    A vertical cut is taken only between two columns of PROSE (>=12 lines each,
    long wrapped lines, <=30% digits, not a list numbered across the row);
    tables and grids stay in row order. Otherwise the widest horizontal gap of
    at least 0.8 line heights splits top from bottom."""
    if len(idx) < 6 or depth > 12:
        return [idx]
    cuts = []
    for g0, g1 in _gaps([(ext[i][0], ext[i][1]) for i in idx], 0.8 * hmed):
        left = [i for i in idx if ext[i][1] <= g0]
        right = [i for i in idx if ext[i][0] >= g1]
        lcol, rcol = _next_to(left, ext, hmed, True), _next_to(right, ext, hmed, False)
        if (_prose(lcol, ext, hmed, texts) and _prose(rcol, ext, hmed, texts)
                and not _numbered_across(lcol, rcol, ext, hmed, texts)):
            cuts.append((g1 - g0, left, right))
    if cuts:
        _, left, right = max(cuts, key=lambda c: c[0])
        return _blocks(left, ext, hmed, texts, depth + 1) + _blocks(right, ext, hmed, texts, depth + 1)
    hg = _gaps([(ext[i][2], ext[i][3]) for i in idx], 0.8 * hmed)
    if not hg:
        return [idx]
    g0, g1 = max(hg, key=lambda g: g[1] - g[0])
    top = [i for i in idx if ext[i][3] <= g0]
    bottom = [i for i in idx if ext[i][2] >= g1]
    return _blocks(top, ext, hmed, texts, depth + 1) + _blocks(bottom, ext, hmed, texts, depth + 1)


def reading_order(boxes: list[np.ndarray], angle: float | None = None, columns: bool = True,
                  texts: list[str] | None = None) -> list[list[int]]:
    """Rows top to bottom, each left to right, in DESKEWED coordinates, block by
    block: prose text columns are read one after the other (see _blocks).

    Box corners are rotated by -skew first, then boxes grouped into rows by
    vertical overlap with the row's running centre."""
    if not boxes:
        return []
    a = math.radians(-(page_skew(boxes) if angle is None else angle))
    rot = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]], np.float32)
    pts = [b.astype(np.float32) @ rot.T for b in boxes]
    ext = [(float(p[:, 0].min()), float(p[:, 0].max()), float(p[:, 1].min()), float(p[:, 1].max())) for p in pts]
    hmed = line_height(boxes)
    idx = list(range(len(boxes)))
    blocks = _blocks(idx, ext, hmed, texts) if columns else [idx]
    cy = [(e[2] + e[3]) / 2 for e in ext]
    rx = [(e[0] + e[1]) / 2 for e in ext]
    hs = [float(max(np.linalg.norm(b[0] - b[3]), 1.0)) for b in boxes]
    tol = 0.5 * hmed
    rows: list[list[int]] = []
    for blk in blocks:
        # Rows from full-height lines only; small boxes join the nearest row
        # afterwards without moving it (a chain of mark fragments must not
        # merge two lines into one row).
        full = [i for i in blk if hs[i] >= 0.6 * hmed] or list(blk)
        small = [i for i in blk if i not in set(full)]
        brows: list[list[int]] = []
        for i in sorted(full, key=lambda k: cy[k]):
            if brows and abs(cy[i] - np.mean([cy[j] for j in brows[-1]])) <= tol:
                brows[-1].append(i)
            else:
                brows.append([i])
        centre = [float(np.mean([cy[j] for j in r])) for r in brows]
        for i in small:
            d = [abs(cy[i] - c) for c in centre]
            if d and min(d) <= tol + 0.5 * hs[i]:
                brows[int(np.argmin(d))].append(i)
            else:
                brows.append([i])
                centre.append(cy[i])
        order = sorted(range(len(brows)), key=lambda r: centre[r])
        rows += [sorted(brows[r], key=lambda k: rx[k]) for r in order]
    return rows


def fragments(boxes: list[np.ndarray], rec: dict[int, tuple[str, float]]) -> set[int]:
    """Boxes that are pieces of a neighbouring line (a stacked mark or a tail
    detected on its own): at most 3 characters, under 35% of the page's line
    height, and touching a full-height line above or below."""
    if len(boxes) < 3:
        return set()
    hs = [float(np.linalg.norm(b[0] - b[3])) for b in boxes]
    hmed = line_height(boxes)
    out = set()
    for i, b in enumerate(boxes):
        if hs[i] >= 0.35 * hmed or len(rec.get(i, ("", 0))[0].replace(" ", "")) > 3:
            continue
        x0, x1, y0, y1 = b[:, 0].min(), b[:, 0].max(), b[:, 1].min(), b[:, 1].max()
        for j, c in enumerate(boxes):
            if j == i or hs[j] < 0.6 * hmed:
                continue
            if c[:, 0].min() <= x1 and x0 <= c[:, 0].max() and \
                    min(abs(y0 - c[:, 1].max()), abs(c[:, 1].min() - y1)) <= 0.5 * hmed:
                out.add(i)
                break
    return out