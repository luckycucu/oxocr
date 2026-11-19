"""读光 OCR 家族 ModelSpec 清单（纯数据，不导入推理实现）。

三档 small / large / max，det 分辨率与权重成对使用（变体表选择）。
词表三档相同（7643 行），是必需文件（非 gated role，恒必需）。
不随附 configuration.json：rec 预处理常量按实测固定。
"""
from __future__ import annotations

from collections.abc import Mapping
from datetime import date
from typing import Any

from oxocr.errors import UnknownParameterError
from oxocr.models.base import HUGGINGFACE_REPO, MODELSCOPE_REPO, Capabilities, ModelFile, ModelSpec
from oxocr.models.languages import names

__all__ = [
    "DUGUANG_SPECS",
    "DET_VARIANTS",
    "det_file_for",
]

_IMPL = "oxocr.models.duguang_ocr.pipeline:DuguangModel"
_CAPS = Capabilities(backend="onnxruntime")

_RAW_AUTHOR = "通义实验室（阿里巴巴） / Tongyi Lab (Alibaba)"
_RAW_COUNTRY = "China"
_RAW_LICENSE = "Apache-2.0"
_CUR_AUTHOR = "luckycucu"
_CUR_COUNTRY = "China"
_CUR_URLS = ("https://modelscope.cn/models/mscoder/duguang-ocr-onnx-v2", "https://github.com/luckycucu/duguang-ocr-onnx")
_CUR_DATE = date(2025, 5, 17)

_REC_CONVNEXT = "https://modelscope.cn/models/iic/cv_convnextTiny_ocr-recognition-general_damo"
# 每档：原始开源日期（ModelScope 页面创建）、原始地址（rec 为主溯源）、原始模型 ID、
# 架构、总大小（字节，test/models 实测 det+rec+vocab）、星级。
_TIER_META: dict[str, dict[str, Any]] = {
    "small": {
        "release": date(2023, 5, 29),
        "urls": (
            "https://modelscope.cn/models/iic/cv_LightweightEdge_ocr-recognitoin-general_damo",
            "https://modelscope.cn/models/iic/cv_proxylessnas_ocr-detection-db-line-level_damo",
        ),
        "model_ids": {
            "det": "iic/cv_proxylessnas_ocr-detection-db-line-level_damo",
            "rec": "iic/cv_LightweightEdge_ocr-recognitoin-general_damo",
        },
        "arch": {"det": "DBNet (ProxylessNAS)", "rec": "LightweightEdge"},
        "size_bytes": 13363576,
        "star_rating": 3,
    },
    "large": {
        "release": date(2023, 2, 6),
        "urls": (
            _REC_CONVNEXT,
            "https://modelscope.cn/models/iic/cv_resnet18_ocr-detection-db-line-level_damo",
        ),
        "model_ids": {
            "det": "iic/cv_resnet18_ocr-detection-db-line-level_damo",
            "rec": "iic/cv_convnextTiny_ocr-recognition-general_damo",
        },
        "arch": {"det": "DBNet (ResNet18)", "rec": "ConvNextViT"},
        "size_bytes": 125593879,
        "star_rating": 4,
    },
    "max": {
        "release": date(2022, 8, 19),
        "urls": (
            _REC_CONVNEXT,
            "https://modelscope.cn/models/iic/cv_resnet18_ocr-detection-line-level_damo",
        ),
        "model_ids": {
            "det": "iic/cv_resnet18_ocr-detection-line-level_damo",
            "rec": "iic/cv_convnextTiny_ocr-recognition-general_damo",
        },
        "arch": {"det": "SegLink (ResNet18)", "rec": "ConvNextViT"},
        "size_bytes": 158746749,
        "star_rating": 4,
    },
}

DET_VARIANTS: dict[str, dict[int, str]] = {
    "duguang_ocr_small": {1024: "duguang-ocr/small/det/model_1024x1024.onnx"},
    "duguang_ocr_large": {512: "duguang-ocr/large/det/model_512x512.onnx"},
    "duguang_ocr_max": {512: "duguang-ocr/max/det/model_512x512.onnx"},
}

_DB_PARAMS: dict[str, Any] = {
    "det_thresh": 0.2,
    "det_box_thresh": 0.3,
    "det_unclip_ratio": 1.5,
    "rec_batch_num": 6,
}
_SEG_PARAMS: dict[str, Any] = {
    "det_node_thresh": 0.4,
    "det_link_thresh": 0.6,
    "rec_batch_num": 6,
}


def det_file_for(spec: ModelSpec, params: Mapping[str, Any]) -> ModelFile:
    """按 params['det_image_size'] 选 det 权重；未列入表的分辨率抛 UnknownParameterError。"""
    size = int(params.get("det_image_size") or spec.defaults["det_image_size"])
    variants = DET_VARIANTS[spec.name]
    if size not in variants:
        raise UnknownParameterError(
            "det_image_size", allowed=[str(s) for s in sorted(variants)], context=f"模型 {spec.name}"
        )
    return ModelFile("det", variants[size])


def _spec(tier: str, display: str, seglink: bool) -> ModelSpec:
    name = f"duguang_ocr_{tier}"
    size = next(iter(DET_VARIANTS[name]))
    params = dict(_SEG_PARAMS if seglink else _DB_PARAMS)
    meta = _TIER_META[tier]
    languages = ("zh", "en")
    zh, en = names(languages)
    return ModelSpec(
        name=name,
        family="duguang_ocr",
        impl=_IMPL,
        files=(
            ModelFile("det", DET_VARIANTS[name][size]),
            ModelFile("rec", f"duguang-ocr/{tier}/rec/model.onnx"),
            ModelFile("vocab", f"duguang-ocr/{tier}/rec/vocab.txt"),
        ),
        name_chinese=display,
        raw_author=_RAW_AUTHOR,
        raw_author_country=_RAW_COUNTRY,
        raw_urls=meta["urls"],
        raw_release_date=meta["release"],
        raw_license=_RAW_LICENSE,
        raw_model_ids=meta["model_ids"],
        modelscope_repo=MODELSCOPE_REPO,
        huggingface_repo=HUGGINGFACE_REPO,
        current_author=_CUR_AUTHOR,
        current_author_country=_CUR_COUNTRY,
        current_urls=_CUR_URLS,
        current_release_date=_CUR_DATE,
        current_license=_RAW_LICENSE,
        model_type="text",
        languages=languages,
        languages_chinese=zh,
        languages_english=en,
        size_bytes=meta["size_bytes"],
        star_rating=meta["star_rating"],
        model_architecture=meta["arch"],
        commercial_use="yes",
        defaults={**params, "det_image_size": size},
        capabilities=_CAPS,
    )


DUGUANG_SPECS: dict[str, ModelSpec] = {
    "duguang_ocr_small": _spec("small", "读光 small（轻量最快）", seglink=False),
    "duguang_ocr_large": _spec("large", "读光 large（均衡）", seglink=False),
    "duguang_ocr_max": _spec("max", "读光 max（效果最好）", seglink=True),
}
