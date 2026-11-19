"""离线测试：读光 OCR（duguang）。三档：small/large=DB 检测，max=SegLink 检测。

权重缺失的档位自动跳过。

运行：
    PY=C:/ProgramData/miniconda3/envs/pyside6py310/python.exe
    OXOCR_OFFLINE=1 $PY test/script/run_offline_duguang.py [IMAGE ...]

不指定 IMAGE 时使用内置合成图；可传多张图片依次识别。
"""
from __future__ import annotations

import sys

from _common import MODELS_DIR, make_image, parse_images, report, setup_offline_env


def main() -> int:
    images = parse_images("离线测试：读光 OCR（duguang）")
    setup_offline_env()
    import oxocr

    if not images:
        images = [("合成图", make_image())]

    ran = False
    for tier in ("small", "large", "max"):
        name = f"duguang_ocr_{tier}"
        if not (MODELS_DIR / "duguang-ocr" / tier / "rec" / "model.onnx").is_file():
            continue
        for title, image in images:
            ran = True
            dr = oxocr.ocr(image, model=name, model_dir=MODELS_DIR, mode="para")
            report(f"读光 {name} | {title}", dr)

    if not ran:
        print(f"未在 {MODELS_DIR} 找到 duguang-ocr 离线权重，请先放置模型文件。", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())