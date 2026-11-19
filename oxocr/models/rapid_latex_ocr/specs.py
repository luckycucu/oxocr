"""LaTeX-OCR 家族 ModelSpec 清单
"""
from __future__ import annotations

from datetime import date

from oxocr.models.base import HUGGINGFACE_REPO, MODELSCOPE_REPO, Capabilities, ModelFile, ModelSpec
from oxocr.models.languages import names

__all__ = ["LATEX_OCR_SPECS"]

_IMPL = "oxocr.models.rapid_latex_ocr.pipeline:LatexOcrModel"
# 无检测模型；自带对比度拉伸/裁剪/反色前处理（核心前处理默认关闭，避免 pad_border 干扰）；
# LaTeX 不能过自然语言修饰（纠错/空白规范化），后处理默认关闭
_CAPS = Capabilities(
    detection=False, recognition=True, own_preprocess=True, own_postprocess=True,
    backend="onnxruntime",
)
_DEFAULTS = {"max_new_tokens": 512, "repeat_limit": 12}

_LANGUAGES = ("latex",)
_ZH, _EN = names(_LANGUAGES)

LATEX_OCR_SPECS: dict[str, ModelSpec] = {
    "rapid_latex_ocr": ModelSpec(
        name="rapid_latex_ocr",
        family="latex_ocr",
        impl=_IMPL,
        files=(
            ModelFile("resizer", "rapid-latex-ocr/image_resizer.onnx"),
            ModelFile("encoder", "rapid-latex-ocr/encoder.onnx"),
            ModelFile("decoder", "rapid-latex-ocr/decoder.onnx"),
            ModelFile("tokenizer", "rapid-latex-ocr/tokenizer.json"),
        ),
        name_chinese="LaTeX-OCR 公式识别",
        raw_author="lukas-blecher",
        raw_author_country="Germany",  # Github 主页显示的国家
        raw_urls=("https://github.com/lukas-blecher/LaTeX-OCR","https://huggingface.co/spaces/lukbl/LaTeX-OCR"),
        raw_release_date=date(2022, 5, 25),  # 以 HuggingFace 为准
        raw_license="MIT",
        raw_model_ids={"rec": "lukas-blecher/LaTeX-OCR (pix2tex)"},
        modelscope_repo=MODELSCOPE_REPO,
        huggingface_repo=HUGGINGFACE_REPO,
        current_author="RapidAI(SWHL)",
        current_author_country="China",
        current_urls=("https://github.com/RapidAI/RapidLaTeXOCR",),
        current_release_date=date(2023, 7, 15),
        current_license="MIT",
        model_type="latex",
        languages=_LANGUAGES,
        languages_chinese=_ZH,
        languages_english=_EN,
        size_bytes=178952787,
        star_rating=4,
        model_architecture={
            "resizer": "WidthBucket CNN",
            "encoder": "ViT (ResNet backbone)",
            "decoder": "Transformer (autoregressive)",
        },
        commercial_use="yes",
        defaults=dict(_DEFAULTS),
        capabilities=_CAPS,
    ),
}
