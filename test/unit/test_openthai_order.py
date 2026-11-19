"""阅读顺序：行聚类 / XY-cut 分栏 / 碎框剔除（合成框，无权重）。"""

from __future__ import annotations

import numpy as np
from oxocr.models.openthai_ocr.order import fragments, reading_order


def _box(x, y, w, h):
    return np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], np.float32)


def test_rows_top_to_bottom_left_to_right():
    boxes = [_box(200, 40, 100, 20), _box(0, 0, 100, 20),
             _box(200, 0, 100, 20), _box(0, 40, 100, 20)]
    rows = reading_order(boxes)
    assert rows == [[1, 2], [3, 0]]


def test_empty():
    assert reading_order([]) == []


def test_small_box_joins_nearest_row():
    boxes = [_box(0, 0, 200, 20), _box(0, 60, 200, 20), _box(50, 18, 10, 6)]
    rows = reading_order(boxes)
    assert rows[0] == [0, 2] or rows[0] == [2, 0]
    assert 2 in rows[0]  # 碎框挂到第一行，不独立成行
    assert rows[1] == [1]


def test_prose_columns_read_column_by_column():
    left = [_box(0, i * 30, 300, 20) for i in range(12)]
    right = [_box(400, i * 30, 300, 20) for i in range(12)]
    boxes = left + right
    rows = reading_order(boxes)
    flat = [i for r in rows for i in r]
    assert flat[:12] == list(range(12))  # 左栏先读完
    assert flat[12:] == list(range(12, 24))


def test_table_stays_in_row_order():
    # 短单元格（宽 < 7×行高）不是散文栏：按行走
    boxes = [_box(0, i * 30, 80, 20) for i in range(4)] + \
            [_box(200, i * 30, 80, 20) for i in range(4)]
    rows = reading_order(boxes)
    assert rows == [[0, 4], [1, 5], [2, 6], [3, 7]]


def test_numbered_list_across_not_split():
    left = [_box(0, i * 30, 300, 20) for i in range(12)]
    right = [_box(400, i * 30, 300, 20) for i in range(12)]
    texts = [f"{i + 1}. รายการซ้าย" for i in range(12)] + \
            [f"{i + 2}. รายการขวา" for i in range(12)]
    rows = reading_order(left + right, texts=texts)
    # 跨行编号列表：每行左右并读，不按栏切
    assert rows[0] == [0, 12]
    assert len(rows) == 12


def test_columns_disabled():
    left = [_box(0, i * 30, 300, 20) for i in range(12)]
    right = [_box(400, i * 30, 300, 20) for i in range(12)]
    rows = reading_order(left + right, columns=False)
    assert rows[0] == [0, 12]  # 行序


def test_fragments_dropped():
    boxes = [_box(0, 0, 200, 20), _box(0, 60, 200, 20), _box(50, 22, 10, 6)]
    rec = {0: ("บรรทัดแรก", 0.95), 1: ("บรรทัดที่สอง", 0.95), 2: ("่", 0.6)}
    assert fragments(boxes, rec) == {2}


def test_fragments_long_text_kept():
    boxes = [_box(0, 0, 200, 20), _box(0, 60, 200, 20), _box(50, 22, 10, 6)]
    rec = {0: ("บรรทัดแรก", 0.95), 1: ("บรรทัดที่สอง", 0.95), 2: ("ข้อความยาว", 0.9)}
    assert fragments(boxes, rec) == set()


def test_fragments_isolated_short_line_kept():
    # 独立短行（上下无整行贴着）不判碎框
    boxes = [_box(0, 0, 200, 20), _box(0, 60, 200, 20), _box(50, 200, 30, 6)]
    rec = {0: ("บรรทัดแรก", 0.95), 1: ("บรรทัดที่สอง", 0.95), 2: ("กข", 0.9)}
    assert fragments(boxes, rec) == set()


def test_fragments_too_few_boxes():
    assert fragments([_box(0, 0, 200, 20)], {}) == set()