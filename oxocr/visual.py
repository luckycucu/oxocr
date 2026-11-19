"""OCR 结果可视化（纯 Pillow + numpy，无 OpenCV、不内置字体）。

图像输入输出遵循包内约定：BGR uint8 (H, W, 3)；boxes 为原图像素坐标 (N, 4, 2)。
字体默认探测系统字体（优先中英文覆盖），可用 font_path 指定任意 ttf/ttc；
复杂文种（阿拉伯文/天城文等）在 libraqm 可用时自动启用 RAQM 排版。

用法::

    from oxocr.visual import draw_items

    result = oxocr.ocr("test.jpg")
    vis = draw_items("test.jpg", result.items)  # BGR ndarray，可交回 load_image/保存
"""
from __future__ import annotations

import math
import random
import string
from collections.abc import Sequence
from functools import lru_cache
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont, features

from oxocr.errors import OxocrError
from oxocr.runtime.imageops import get_perspective_transform, resize, warp_perspective
from oxocr.runtime.images import load_image
from oxocr.types import TextItem

__all__ = [
    "find_font",
    "draw_boxes",
    "draw_ocr",
    "draw_ocr_box_txt",
    "draw_items",
    "text_visual",
]

# ImageFont.truetype 按裸文件名即可在各平台字体目录检索，无需手工拼路径
_FONT_CANDIDATES = (
    "msyh.ttc",  # Windows 微软雅黑
    "simsun.ttc",
    "PingFang.ttc",  # macOS
    "Hiragino Sans GB.ttc",
    "NotoSansCJK-Regular.ttc",  # Linux
    "NotoSansCJKsc-Regular.otf",
    "wqy-microhei.ttc",
    "wqy-zenhei.ttc",
    "Arial Unicode.ttf",
    "arial.ttf",
    "DejaVuSans.ttf",
)


_HAVE_RAQM = features.check("raqm")


def _truetype(font_path: str, size: int) -> ImageFont.FreeTypeFont:
    if _HAVE_RAQM:
        return ImageFont.truetype(font_path, size, layout_engine=ImageFont.Layout.RAQM)
    return ImageFont.truetype(font_path, size)


@lru_cache(maxsize=8)
def find_font(font_path: str | None = None) -> str:
    """返回可加载的字体路径/名称；未指定时按候选链探测系统字体。"""
    if font_path is not None:
        _truetype(font_path, 20)
        return font_path
    for name in _FONT_CANDIDATES:
        try:
            _truetype(name, 20)
            return name
        except OSError:
            continue
    raise OxocrError("未找到可用系统字体，请通过 font_path 指定 ttf/ttc 字体文件")


def _to_rgb_pil(image: Any) -> Image.Image:
    return Image.fromarray(load_image(image)[:, :, ::-1])


def _to_bgr(pil: Image.Image | np.ndarray) -> np.ndarray:
    arr = np.asarray(pil)
    return arr[:, :, ::-1].copy()


def _boxes_list(boxes: Sequence[Any]) -> list[np.ndarray]:
    return [np.asarray(b, dtype=np.float64).reshape(-1, 2) for b in boxes]


def draw_boxes(
    image: Any,
    boxes: Sequence[Any],
    scores: Sequence[float] | None = None,
    drop_score: float = 0.5,
) -> np.ndarray:
    """在图像副本上画红色检测框，返回 BGR ndarray。"""
    pil = _to_rgb_pil(image)
    draw = ImageDraw.Draw(pil)
    if scores is None:
        scores = [1.0] * len(boxes)
    for box, score in zip(_boxes_list(boxes), scores):
        if score < drop_score or math.isnan(score):
            continue
        pts = [tuple(p) for p in box]
        draw.line(pts + [pts[0]], fill=(255, 0, 0), width=2)
    return _to_bgr(pil)


def _create_font(txt: str, sz: tuple[int, int], font_path: str | None) -> ImageFont.FreeTypeFont:
    """按框高取字号，超宽时缩到框宽内。sz = (w, h)。"""
    font_path = find_font(font_path)
    font_size = max(int(sz[1] * 0.75), 1)
    font = _truetype(font_path, font_size)
    length = font.getlength(txt)
    if length > sz[0]:
        font_size = max(int(font_size * sz[0] / length), 1)
        font = _truetype(font_path, font_size)
    return font


def _draw_box_txt_fine(
    img_size: tuple[int, int], box: np.ndarray, txt: str | None, font_path: str | None
) -> np.ndarray:
    """把文字渲染进小图再透视贴回框位置，边界外填白（供 bitwise_and 叠加）。"""
    box_width = max(int(math.dist(box[0], box[1])), 1)
    box_height = max(int(math.dist(box[0], box[3])), 1)
    vertical = box_height > 2 * box_width and box_height > 30
    if vertical:
        img_text = Image.new("RGB", (box_height, box_width), (255, 255, 255))
        if txt:
            font = _create_font(txt, (box_height, box_width), font_path)
            ImageDraw.Draw(img_text).text((0, 0), txt, fill=(0, 0, 0), font=font)
        img_text = img_text.transpose(Image.Transpose.ROTATE_270)
    else:
        img_text = Image.new("RGB", (box_width, box_height), (255, 255, 255))
        if txt:
            font = _create_font(txt, (box_width, box_height), font_path)
            ImageDraw.Draw(img_text).text((0, 0), txt, fill=(0, 0, 0), font=font)
    pts1 = np.float32([[0, 0], [box_width, 0], [box_width, box_height], [0, box_height]])
    matrix = get_perspective_transform(pts1, box.astype(np.float32))
    return warp_perspective(
        np.array(img_text),
        matrix,
        img_size,
        interpolation="nearest",
        fillcolor=(255, 255, 255),
    )


def draw_ocr_box_txt(
    image: Any,
    boxes: Sequence[Any],
    txts: Sequence[str] | None = None,
    scores: Sequence[float] | None = None,
    drop_score: float = 0.5,
    font_path: str | None = None,
) -> np.ndarray:
    """双联画：左侧原图叠加半透明彩色框，右侧白底按框位置透视绘制识别文字。

    返回 BGR ndarray，宽度为原图 2 倍。
    """
    pil = _to_rgb_pil(image)
    w, h = pil.size
    boxes = _boxes_list(boxes)
    if txts is None or len(txts) != len(boxes):
        txts = [None] * len(boxes)
    if scores is None:
        scores = [1.0] * len(boxes)

    img_left = pil.copy()
    img_right = np.full((h, w, 3), 255, dtype=np.uint8)
    rng = random.Random(0)
    draw_left = ImageDraw.Draw(img_left)
    for box, txt, score in zip(boxes, txts, scores):
        if score < drop_score or math.isnan(score):
            continue
        color = (rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255))
        draw_left.polygon([tuple(p) for p in box], fill=color)
        right_text = Image.fromarray(_draw_box_txt_fine((w, h), box, txt, font_path))
        pts = [tuple(p) for p in box]
        ImageDraw.Draw(right_text).line(pts + [pts[0]], fill=color, width=1)
        img_right = np.bitwise_and(img_right, np.array(right_text))

    img_left = Image.blend(pil, img_left, 0.5)
    img_show = Image.new("RGB", (w * 2, h), (255, 255, 255))
    img_show.paste(img_left, (0, 0))
    img_show.paste(Image.fromarray(img_right), (w, 0))
    return _to_bgr(img_show)


def _str_count(s: str) -> int:
    """中文计 1、英文/数字/空白计半个的显示宽度估算。"""
    en_dg = sum(1 for c in s if c in string.ascii_letters or c.isdigit() or c.isspace())
    return len(s) - math.ceil(en_dg / 2)


def text_visual(
    texts: Sequence[str],
    scores: Sequence[float] | None = None,
    img_h: int = 400,
    img_w: int = 600,
    threshold: float = 0.0,
    font_path: str | None = None,
) -> np.ndarray:
    """白底文字面板：编号 + 文本 + 分数，超长换行、超高横向分页。返回 RGB ndarray。"""
    if scores is None:
        scores = [1.0] * len(texts)
    if len(texts) != len(scores):
        raise ValueError("texts 与 scores 数量不一致")
    font_size = 20
    font = _truetype(find_font(font_path), font_size)
    gap = font_size + 5
    max_chars = max(img_w // font_size - 4, 1)

    def new_page() -> tuple[Image.Image, ImageDraw.ImageDraw]:
        img = Image.new("RGB", (img_w, img_h), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        draw.line([(img_w - 1, 0), (img_w - 1, img_h - 1)], fill=(0, 0, 0))
        return img, draw

    pages: list[np.ndarray] = []
    page, draw = new_page()
    count, index = 1, 0
    for idx, txt in enumerate(texts):
        index += 1
        if scores[idx] < threshold or math.isnan(scores[idx]):
            index -= 1
            continue
        first_line = True
        while _str_count(txt) >= max_chars:
            tmp = txt
            txt = tmp[:max_chars]
            new_txt = f"{index}: {txt}" if first_line else "    " + txt
            first_line = False
            draw.text((0, gap * count), new_txt, (0, 0, 0), font=font)
            txt = tmp[max_chars:]
            if count >= img_h // gap - 1:
                pages.append(np.array(page))
                page, draw = new_page()
                count = 0
            count += 1
        if first_line:
            new_txt = f"{index}: {txt}   {scores[idx]:.3f}"
        else:
            new_txt = f"  {txt}  {scores[idx]:.3f}"
        draw.text((0, gap * count), new_txt, (0, 0, 0), font=font)
        if count >= img_h // gap - 1 and idx + 1 < len(texts):
            pages.append(np.array(page))
            page, draw = new_page()
            count = 0
        count += 1
    pages.append(np.array(page))
    return np.concatenate(pages, axis=1)


def draw_ocr(
    image: Any,
    boxes: Sequence[Any],
    txts: Sequence[str] | None = None,
    scores: Sequence[float] | None = None,
    drop_score: float = 0.5,
    font_path: str | None = None,
) -> np.ndarray:
    """画框；提供 txts 时再拼接右侧文字面板（图缩放到最长边 600）。返回 BGR ndarray。"""
    vis = draw_boxes(image, boxes, scores, drop_score)
    if txts is None:
        return vis
    rgb = vis[:, :, ::-1]
    h, w = rgb.shape[:2]
    scale = 600 / max(h, w)
    img = resize(rgb, (max(int(w * scale), 1), max(int(h * scale), 1)))
    txt_img = text_visual(
        txts, scores, img_h=img.shape[0], img_w=600, threshold=drop_score, font_path=font_path
    )
    return _to_bgr(np.concatenate([img, txt_img], axis=1))


def draw_items(
    image: Any,
    items: Sequence[TextItem],
    drop_score: float = 0.5,
    font_path: str | None = None,
) -> np.ndarray:
    """从 oxocr 识别结果（OcrResult.items）直接生成双联可视化图。"""
    return draw_ocr_box_txt(
        image,
        [it.box for it in items],
        [it.text for it in items],
        [it.score for it in items],
        drop_score=drop_score,
        font_path=font_path,
    )