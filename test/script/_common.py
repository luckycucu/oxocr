"""test/script 离线测试脚本的公共工具。

提供离线环境设置、合成图生成、运行时图片参数解析与结果打印。
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = REPO_ROOT / "test" / "models"


def setup_offline_env() -> None:
    os.environ.setdefault("OXOCR_OFFLINE", "1")
    os.environ.setdefault("OXOCR_MODEL_DIR", str(MODELS_DIR))


def make_image(lines=("Hello oxocr offline", "second line here", "PP-OCR v6 tiny")):
    """生成白底黑字合成图，返回 BGR ndarray。"""
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (720, 200), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 44)
    except OSError:
        font = ImageFont.load_default()
    for i, line in enumerate(lines):
        draw.text((24, 24 + i * 56), line, fill=(0, 0, 0), font=font)
    return np.array(img)[:, :, ::-1].copy()


def make_formula_image():
    """生成 E=mc^2 合成公式图，返回 BGR ndarray。"""
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (480, 160), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arialbd.ttf", 56)
    except OSError:
        font = ImageFont.load_default()
    draw.text((40, 48), "E=mc^2", fill=(0, 0, 0), font=font)
    return np.array(img)[:, :, ::-1].copy()


def load_image(path):
    """读取图片文件为 BGR ndarray。"""
    import numpy as np
    from PIL import Image

    return np.array(Image.open(path).convert("RGB"))[:, :, ::-1].copy()


def parse_images(description: str):
    """解析命令行图片参数，返回 [(标题, BGR ndarray), ...]。

    用法：run_offline_xxx.py [IMAGE ...]，可传多张图片；
    不传时返回空列表，由脚本沿用默认合成图。
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "images", nargs="*", metavar="IMAGE",
        help="待识别的图片路径（可多张）；不指定则使用内置合成图",
    )
    args = parser.parse_args()
    images = []
    for p in args.images:
        if not Path(p).is_file():
            parser.error(f"图片不存在：{p}")
        images.append((p, load_image(p)))
    return images


def report(title: str, result) -> None:
    """统一打印一次 OCR 结果：全文、条目（分数/方向/坐标）、耗时。"""
    print(f"\n[{title}]  {result.text!r}")
    for it in result.items:
        rot = f" rot={it.meta['rotation']}" if "rotation" in it.meta else ""
        print(f"   {it.text!r:28s} score={it.score:.3f}{rot} rect={it.box.as_rect()}")
    print("耗时：", result.elapse.as_dict())