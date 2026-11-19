"""anaksor-ocr 家族 ModelSpec 清单（纯数据，不导入推理实现）。

单条 anaksor_ocr：mobile DBNet++ 行检测 + SVTRv2-B CTC 行识别（高棉语 + 英语/拉丁）。
字典是必需文件（非 gated role，恒必需）：rec ONNX 无 character metadata。
score_mode=slow、max_candidates=1000、无 dilate、max_ratio=20 是权重绑定的
训练/导出常量，不暴露为参数。
"""

from __future__ import annotations

from datetime import date

from oxocr.models.base import HUGGINGFACE_REPO, MODELSCOPE_REPO, Capabilities, ModelFile, ModelSpec
from oxocr.models.languages import names

__all__ = ["ANAKSOR_SPECS"]

_IMPL = "oxocr.models.anaksor_ocr.pipeline:AnaksorModel"
_CAPS = Capabilities(backend="onnxruntime")

_LANGUAGES = ("km", "en")
_ZH, _EN = names(_LANGUAGES)

ANAKSOR_SPECS: dict[str, ModelSpec] = {
    "anaksor_ocr": ModelSpec(
        name="anaksor_ocr",
        family="anaksor_ocr",
        impl=_IMPL,
        files=(
            ModelFile(
                "det",
                "anaksor-ocr/khmer_det_repvit_dbpp_latest.onnx",
                sha256="9568bd2e1c410d86bd75b32a2c57ad5b4168650341e11d872c06b2b017e34c9c",
                size=12967273,
            ),
            ModelFile(
                "rec",
                "anaksor-ocr/khmer_rec_svtrv2_latest.onnx",
                sha256="afe130ebd442b35cff553efc91213219d33afc5f1c1883a0ddad402a8d1c96ec",
                size=81802061,
            ),
            ModelFile(
                "dict",
                "anaksor-ocr/khmer_dict_v3.txt",
                sha256="c1b460ce84665af03d92fb43e64752ca11afe0a72ba163cbcea48ecd08d57483",
                size=645,
            ),
        ),
        name_chinese="anaksor 高棉语 OCR（KM+EN）",
        raw_author="Sokheng Din",
        raw_author_country="France",
        raw_urls=(
            "https://github.com/SokhengDin/anaksor-ocr",
            "https://huggingface.co/Sokheng/anaksor-ocr",
        ),
        raw_release_date=date(2026, 11, 4),
        raw_license="Apache-2.0",
        raw_model_ids={
            "det": "khmer_det_repvit_dbpp_latest",
            "rec": "khmer_rec_svtrv2_latest",
            "dict": "khmer_dict_v3",
        },
        modelscope_repo=MODELSCOPE_REPO,
        huggingface_repo=HUGGINGFACE_REPO,
        model_type="text",
        languages=_LANGUAGES,
        languages_chinese=_ZH,
        languages_english=_EN,
        size_bytes=94769979,
        star_rating=3,
        model_architecture={"det": "DBNet++ (RepViT + RSEFPN + ASF)", "rec": "SVTRv2-B + CTC"},
        commercial_use="yes",
        defaults={
            "det_limit_side_len": 1280,
            "det_min_side_len": 736,
            "det_thresh": 0.3,
            "det_box_thresh": 0.5,
            "det_unclip_ratio": 1.5,
            "rec_batch_num": 32,
            "crop_pad_y": 0.16,
            "crop_pad_x": 0.10,
            "try_vertical": True,
            "try_180": False,
        },
        capabilities=_CAPS,
    ),
}
