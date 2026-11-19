"""Pix2Text 家族 ModelSpec 清单（纯数据，不导入推理实现）。

公式识别（MFR）：TrOCR VisionEncoderDecoder，整图公式 → LaTeX，无检测模型。
权重目录 test/models/pix2text-mfr{,-1.5,-quantized}，文件路径相对 model_dir。
三个目录的词表（1868/1200）与起始 token（1/2）不同：start/eos 由各目录
config.json 在 load() 时读取，清单里不得硬编码。
"""
from __future__ import annotations

from datetime import date

from oxocr.models.base import HUGGINGFACE_REPO, MODELSCOPE_REPO, Capabilities, ModelFile, ModelSpec
from oxocr.models.languages import names

__all__ = ["MFR_SPECS"]

_IMPL = "oxocr.models.pix2text.pipeline:MfrModel"
# 无检测模型；LaTeX 不能过自然语言修饰（纠错/空白规范化），后处理默认关闭
_CAPS = Capabilities(
    detection=False, recognition=True, own_postprocess=True, backend="onnxruntime"
)
_DEFAULTS = {"max_new_tokens": 256, "repeat_limit": 12}

_RAW_AUTHOR = "breezedeus"
_RAW_COUNTRY = "China"
_RAW_LICENSE = "MIT"
_RAW_GITHUB = "https://github.com/breezedeus/Pix2Text"
_ARCH = {"encoder": "DeiT", "decoder": "TrOCR"}
_LANGUAGES = ("latex",)
_ZH, _EN = names(_LANGUAGES)

_FILES = (
    ("encoder", "encoder_model.onnx"),
    ("decoder", "decoder_model.onnx"),
    ("tokenizer", "tokenizer.json"),
    ("config", "config.json"),
)


def _spec(
    name: str,
    dirname: str,
    display: str,
    *,
    raw_model_id: str,
    raw_release_date: date | None,
    size_bytes: int,
    current_author: str | None = None,
    current_author_country: str | None = None,
    current_urls: tuple[str, ...] | None = None,
    current_release_date: date | None = None,
) -> ModelSpec:
    return ModelSpec(
        name=name,
        family="pix2text",
        impl=_IMPL,
        files=tuple(ModelFile(role, f"{dirname}/{filename}") for role, filename in _FILES),
        name_chinese=display,
        raw_author=_RAW_AUTHOR,
        raw_author_country=_RAW_COUNTRY,
        raw_urls=(_RAW_GITHUB, f"https://huggingface.co/{raw_model_id}"),
        raw_release_date=raw_release_date,
        raw_license=_RAW_LICENSE,
        raw_model_ids={"rec": raw_model_id},
        modelscope_repo=MODELSCOPE_REPO,
        huggingface_repo=HUGGINGFACE_REPO,
        current_author=current_author,
        current_author_country=current_author_country,
        current_urls=current_urls,
        current_release_date=current_release_date,
        current_license=_RAW_LICENSE,
        model_type="latex",
        languages=_LANGUAGES,
        languages_chinese=_ZH,
        languages_english=_EN,
        size_bytes=size_bytes,
        star_rating=3,
        model_architecture=_ARCH,
        commercial_use="yes",
        defaults=dict(_DEFAULTS),
        capabilities=_CAPS,
    )


MFR_SPECS: dict[str, ModelSpec] = {
    "pix2text_mfr": _spec(
        "pix2text_mfr",
        "pix2text-mfr",
        "Pix2Text 公式识别",
        raw_model_id="breezedeus/pix2text-mfr",
        raw_release_date=date(2024, 2, 12),  # huggingface release date
        size_bytes=117655644,
    ),
    "pix2text_mfr_1.5": _spec(
        "pix2text_mfr_1.5",
        "pix2text-mfr-1.5",
        "Pix2Text 公式识别 1.5",
        raw_model_id="breezedeus/pix2text-mfr-1.5",
        raw_release_date=date(2025, 7, 7),
        size_bytes=119651764,
    ),
    "pix2text_mfr_quantized": _spec(
        "pix2text_mfr_quantized",
        "pix2text-mfr-quantized",
        "Pix2Text 公式识别（int8 量化）",
        raw_model_id="breezedeus/pix2text-mfr",
        raw_release_date=date(2024, 2, 12),
        size_bytes=53241851,
        current_author="Brian314",
        current_author_country="South Korea",
        current_urls=("https://huggingface.co/Brian314/pix2text-mfr-quantized",),
        current_release_date=date(2024, 6, 18),
    ),
}
