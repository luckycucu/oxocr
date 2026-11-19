"""离线测试：公式识别（LaTeX-OCR）。整图公式 → LaTeX，默认跳过核心前/后处理。

权重缺失则跳过。运行：
    PY=C:/ProgramData/miniconda3/envs/pyside6py310/python.exe
    OXOCR_OFFLINE=1 $PY test/script/run_offline_latex_ocr.py [IMAGE ...]

不指定 IMAGE 时使用内置合成公式图；可传多张图片依次识别。
"""
from __future__ import annotations

import sys

from _common import MODELS_DIR, make_formula_image, parse_images, report, setup_offline_env


def main() -> int:
    images = parse_images("离线测试：公式识别（LaTeX-OCR）")
    setup_offline_env()
    import oxocr

    if not (MODELS_DIR / "rapid-latex-ocr" / "encoder.onnx").is_file():
        print(
            f"未在 {MODELS_DIR / 'rapid-latex-ocr'} 找到 LaTeX-OCR 离线权重，请先放置模型文件。",
            file=sys.stderr,
        )
        return 1

    if not images:
        images = [("合成公式", make_formula_image())]

    for title, image in images:
        fr = oxocr.ocr(image, model="rapid_latex_ocr", model_dir=MODELS_DIR)
        report(f"公式 rapid_latex_ocr | {title}", fr)
        print("--------------")
        print(fr.text)
        print("--------------")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
