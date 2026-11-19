"""离线测试：openthai-ocr 泰语 OCR（TH+EN）。

权重缺失自动报错提示。默认使用固化合成泰文图 test/data/openthai/*.png。

运行：
    PY=C:/ProgramData/miniconda3/envs/pyside6py310/python.exe
    OXOCR_OFFLINE=1 $PY test/script/run_offline_openthai.py [IMAGE ...]

指定 IMAGE 时只识别所传图片（可多张）；不指定则照旧。
"""
from __future__ import annotations

import sys

from _common import (
    MODELS_DIR,
    REPO_ROOT,
    load_image,
    parse_images,
    report,
    setup_offline_env,
)

DATA_DIR = REPO_ROOT / "test" / "data" / "openthai"


def main() -> int:
    images = parse_images("离线测试：openthai-ocr 泰语 OCR（TH+EN）")

    if not (MODELS_DIR / "openthai2.0-ocr" / "rec.onnx").is_file():
        print(f"未在 {MODELS_DIR}/openthai2.0-ocr 找到离线权重，请先放置模型文件。", file=sys.stderr)
        return 1

    setup_offline_env()
    import oxocr

    if not images:
        for name in ("thai_lines.png", "thai_page.png"):
            path = DATA_DIR / name
            if path.is_file():
                images.append((f"合成泰文 {name}", load_image(path)))
        if not images:
            print(f"未找到测试图 {DATA_DIR}/*.png，请传入 IMAGE 参数。", file=sys.stderr)
            return 1

    for title, image in images:
        dr = oxocr.ocr(image, model="openthai2.0_ocr", model_dir=MODELS_DIR)
        report(title, dr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())