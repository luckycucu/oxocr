"""离线测试：PP-OCR（ppocrv6_tiny）。不联网，用 test/models 下的权重。

运行：
    PY=C:/ProgramData/miniconda3/envs/pyside6py310/python.exe
    OXOCR_OFFLINE=1 $PY test/script/run_offline_ppocr.py [IMAGE ...]

不指定 IMAGE 时使用内置合成图；可传多张图片依次识别。
"""
from __future__ import annotations

import sys

from _common import MODELS_DIR, make_image, parse_images, setup_offline_env

MODEL = "ppocrv6_tiny"


def main() -> int:
    images = parse_images("离线测试：PP-OCR（ppocrv6_tiny）")

    if not (MODELS_DIR / "rapid-ocr" / "PP-OCRv6" / "det" / "PP-OCRv6_det_tiny.onnx").is_file():
        print(f"未在 {MODELS_DIR} 找到离线权重，请先放置模型文件。", file=sys.stderr)
        return 1

    setup_offline_env()
    import oxocr

    if not images:
        images = [("合成图", make_image())]

    print("可用模型：")
    for m in oxocr.list_models(installed_only=True):
        print("  ", m.name, m.name_chinese)

    for title, image in images:
        result = oxocr.ocr(image, model=MODEL, model_dir=MODELS_DIR, mode="para")
        print(f"\n===== {title} =====")
        print("[para]", repr(result.text))
        print("[line]", repr(result.as_text("line")))
        print("[raw] ", repr(result.as_text("raw")))
        print("\n条目（box 为原图坐标）：")
        for it in result.items:
            print(f"   {it.text!r:24s} score={it.score:.3f} rect={it.box.as_rect()}")
        print("\n耗时：", result.elapse.as_dict())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())