"""几何分行：按 y 中心聚行，行内按 x 排序。"""
from __future__ import annotations

from oxocr.types import Box, TextItem, TextLine

__all__ = ["group_lines", "enclosing_box"]


def enclosing_box(boxes: list[Box]) -> Box:
    x_min = min(b.x_min for b in boxes)
    y_min = min(b.y_min for b in boxes)
    x_max = max(b.x_max for b in boxes)
    y_max = max(b.y_max for b in boxes)
    return Box.from_rect(x_min, y_min, x_max - x_min, y_max - y_min)


def group_lines(items: list[TextItem]) -> list[TextLine]:
    if not items:
        return []
    ordered = sorted(items, key=lambda it: (it.box.y_center, it.box.x_min))
    groups: list[list[TextItem]] = [[ordered[0]]]
    for item in ordered[1:]:
        prev = groups[-1][-1]
        h_thresh = max(prev.box.height, item.box.height) * 0.5
        if abs(item.box.y_center - prev.box.y_center) <= h_thresh:
            groups[-1].append(item)
        else:
            groups.append([item])

    lines: list[TextLine] = []
    for group in groups:
        group.sort(key=lambda it: it.box.x_min)
        box = enclosing_box([it.box for it in group])
        score = sum(it.score for it in group) / len(group)
        lines.append(TextLine(items=group, box=box, text="", score=round(score, 5)))
    return lines
