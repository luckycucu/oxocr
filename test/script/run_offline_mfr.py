"""离线测试：公式识别（pix2text-mfr）。整图公式 → LaTeX，默认不跑后处理。

覆盖 pix2text_mfr、pix2text_mfr_1.5 与 pix2text_mfr_quantized 三个变体，权重缺失则跳过。

运行：
    PY=C:/ProgramData/miniconda3/envs/pyside6py310/python.exe
    OXOCR_OFFLINE=1 $PY test/script/run_offline_mfr.py [IMAGE ...]

不指定 IMAGE 时使用内置合成公式图；可传多张图片依次识别。
"""
from __future__ import annotations

import sys

from _common import MODELS_DIR, make_formula_image, parse_images, report, setup_offline_env


def main() -> int:
    images = parse_images("离线测试：公式识别（pix2text-mfr）")
    setup_offline_env()
    import oxocr

    if not images:
        images = [("合成公式", make_formula_image())]

    ran = False
    for name, dirname in (
        ("pix2text_mfr", "pix2text-mfr"),
        ("pix2text_mfr_1.5", "pix2text-mfr-1.5"),
        ("pix2text_mfr_quantized", "pix2text-mfr-quantized"),
    ):
        if not (MODELS_DIR / dirname / "encoder_model.onnx").is_file():
            continue
        for title, image in images:
            ran = True
            fr = oxocr.ocr(image, model=name, model_dir=MODELS_DIR)
            report(f"公式 {name} | {title}", fr)
            print("--------------")
            print(fr.text)
            print("--------------")

    if not ran:
        print(f"未在 {MODELS_DIR} 找到 pix2text-mfr 离线权重，请先放置模型文件。", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())