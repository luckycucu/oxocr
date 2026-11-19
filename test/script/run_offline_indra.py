"""离线测试：Indra OCR 五模型（latin/cjk/rtl/world/math）。

权重缺失自动报错提示。默认演示：latin/cjk/rtl 合成图 + world/math 真实样图
（test/data/indra/，缺失则跳过）。rtl 合成图先 flip 再渲染（PIL 无 raqm，
按码位绘制即视觉序），模型翻回逻辑序。

运行：
    PY=C:/ProgramData/miniconda3/envs/pyside6py310/python.exe
    OXOCR_OFFLINE=1 $PY test/script/run_offline_indra.py [IMAGE ...]

指定 IMAGE 时只用 indra_ocr_latin 识别所传图片（可多张）；不指定则演示五模型。
"""
from __future__ import annotations

import sys

from _common import (
    MODELS_DIR,
    REPO_ROOT,
    load_image,
    make_image,
    parse_images,
    report,
    setup_offline_env,
)

DATA = REPO_ROOT / "test" / "data" / "indra"


def _synth(lines, font="arialbd.ttf", size=40, width=640, height=200):
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (width, height), (250, 250, 250))
    draw = ImageDraw.Draw(img)
    try:
        f = ImageFont.truetype(font, size)
    except OSError:
        f = ImageFont.load_default()
    y = 20
    for line in lines:
        draw.text((20, y), line, fill=(12, 12, 12), font=f)
        y += size + 16
    return np.array(img)[:, :, ::-1].copy()


def main() -> int:
    # Windows 控制台默认 GBK，阿拉伯/泰文等打印会崩：统一 UTF-8 输出
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    images = parse_images("离线测试：Indra OCR 五模型（latin/cjk/rtl/world/math）")

    if not (MODELS_DIR / "indra-ocr" / "rec.onnx").is_file():
        print(f"未在 {MODELS_DIR}/indra-ocr 找到离线权重，请先放置模型文件。", file=sys.stderr)
        return 1

    setup_offline_env()
    import oxocr
    from oxocr.models.indra_ocr.rtl import flip

    if images:
        for title, image in images:
            report(title, oxocr.ocr(image, model="indra_ocr_latin", model_dir=MODELS_DIR))
        return 0

    demos = [
        ("latin 合成", "indra_ocr_latin", make_image(("Hello oxocr offline", "Indra Latin OCR"))),
        ("cjk 合成", "indra_ocr_cjk", _synth(["你好世界 OCR", "中文识别测试"], font="msyh.ttc")),
        ("rtl 合成（阿拉伯）", "indra_ocr_rtl", _synth([flip("مرحبا بالعالم")], font="arial.ttf", size=36)),
    ]
    for title, model, fname in [
        ("world 真实泰文", "indra_ocr_world", "world.png"),
        ("math 真实公式（整图直识）", "indra_ocr_math", "math2.png"),
    ]:
        if (DATA / fname).is_file():
            demos.append((title, model, load_image(DATA / fname)))
        else:
            print(f"\n（跳过 {title}：{DATA / fname} 不存在）")

    for title, model, image in demos:
        result = oxocr.ocr(image, model=model, model_dir=MODELS_DIR)
        report(title, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
