"""openthai-ocr 家族 ModelSpec 清单（纯数据，不导入推理实现）。

单条 openthai2.0_ocr：PP-OCRv6-tiny DB 行检测 + iApp 泰语 CTC 行识别（泰语 + 英语）。
字典是必需文件（非 gated role，恒必需）：rec ONNX 无 character metadata。
det 阈值取 det.yml PostProcess 实测覆盖值（0.2/0.4/1.4），rec 过滤取 config.json
作者推荐值（drop_below 0.8 等）；height=48（权重输入固定）、max_w=3200、
max_candidates=3000、min_size=3、ori sample/flip_below/flip_margin 是权重/策略
绑定常量，不暴露为参数。
"""

from __future__ import annotations

from datetime import date

from oxocr.models.base import HUGGINGFACE_REPO, MODELSCOPE_REPO, Capabilities, ModelFile, ModelSpec
from oxocr.models.languages import names

__all__ = ["OPENTHAI_SPECS"]

_IMPL = "oxocr.models.openthai_ocr.pipeline:OpenThaiModel"
_CAPS = Capabilities(backend="onnxruntime")

_LANGUAGES = ("th", "en")
_ZH, _EN = names(_LANGUAGES)

OPENTHAI_SPECS: dict[str, ModelSpec] = {
    "openthai2.0_ocr": ModelSpec(
        name="openthai2.0_ocr",
        family="openthai_ocr",
        impl=_IMPL,
        files=(
            ModelFile(
                "det",
                "openthai2.0-ocr/det.onnx",
                sha256="193bab7a04fca699a6c82e6abb5b81bdb28177f0abd4062552b04908dafb19f8",
                size=1780590,
            ),
            ModelFile(
                "rec",
                "openthai2.0-ocr/rec.onnx",
                sha256="82ec89533346faadcb30df96703dad2dd7521de65709e60d83fd9745b8968124",
                size=2392837,
            ),
            ModelFile(
                "dict",
                "openthai2.0-ocr/rec_dict.txt",
                sha256="259ef2636cec0eeae225ae9fd9a63fa9305285650c22d7ec34fe8a82b24c2f1a",
                size=1783,
            ),
        ),
        name_chinese="OpenThai 泰语 OCR（TH+EN）",
        raw_author="iApp Technology (OpenThai / AIEAT)",
        raw_author_country="Thailand",
        raw_urls=("https://iapp.co.th/en/blog/openthai2p0-ocr-how-to-run",),
        raw_release_date=date(2026, 11, 1),  # v1.0.0, November 2026（模型卡片只到月）
        raw_license="Apache-2.0",
        raw_model_ids={
            "det": "PP-OCRv6_tiny_det",
            "rec": "openthai2.0-ocr-text-rec",
            "dict": "rec_dict (528 chars)",
        },
        modelscope_repo=MODELSCOPE_REPO,
        huggingface_repo=HUGGINGFACE_REPO,
        model_type="text",
        languages=_LANGUAGES,
        languages_chinese=_ZH,
        languages_english=_EN,
        size_bytes=4175210,
        star_rating=3,
        model_architecture={"det": "PP-OCRv6-tiny DB", "rec": "PP-OCRv6-tiny + CTC (Thai head)"},
        commercial_use="yes",
        defaults={
            "det_limit_side_len": 1280,
            "det_thresh": 0.2,
            "det_box_thresh": 0.4,
            "det_unclip_ratio": 1.4,
            "split_gap": 1.2,
            "rec_batch_num": 16,
            "rec_color": "flatten",
            "drop_below": 0.8,
            "min_chars": 2,
            "junk_below": 0.9,
            "junk_invalid": 0.05,
            "orientation": True,
            "deskew_above": 2.5,
        },
        capabilities=_CAPS,
    ),
}