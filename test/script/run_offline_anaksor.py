"""离线测试：anaksor-ocr 高棉语 OCR（KM+EN）。

权重缺失自动报错提示。默认合成 Latin 图必跑；真实高棉语照片
test/data/anaksor/sign.jpg 存在则一并演示。

运行：
    PY=C:/ProgramData/miniconda3/envs/pyside6py310/python.exe
    OXOCR_OFFLINE=1 $PY test/script/run_offline_anaksor.py [IMAGE ...]

指定 IMAGE 时只识别所传图片（可多张）；不指定则照旧。
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

SIGN_JPG = REPO_ROOT / "test" / "data" / "anaksor" / "sign.jpg"


def main() -> int:
    images = parse_images("离线测试：anaksor-ocr 高棉语 OCR（KM+EN）")

    if not (MODELS_DIR / "anaksor-ocr" / "khmer_rec_svtrv2_latest.onnx").is_file():
        print(f"未在 {MODELS_DIR}/anaksor-ocr 找到离线权重，请先放置模型文件。", file=sys.stderr)
        return 1

    setup_offline_env()
    import oxocr

    if not images:
        images = [("合成 Latin", make_image(("Hello oxocr offline", "Second line text", "Khmer OCR KM+EN")))]
        if SIGN_JPG.is_file():
            images.append(("真实高棉语 sign.jpg", load_image(SIGN_JPG)))
        else:
            print(f"\n（跳过真实照片：{SIGN_JPG} 不存在）")

    for title, image in images:
        dr = oxocr.ocr(image, model="anaksor_ocr", model_dir=MODELS_DIR, mode="para")
        report(title, dr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())