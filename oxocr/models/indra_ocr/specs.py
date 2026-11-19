"""Indra OCR 家族 ModelSpec 清单（纯数据，不导入推理实现）。

5 个专用模型：每模型固定一个识别器（不做运行时语种路由）。
文本：det/rec/meta 必需，lm/orient 可选（缺文件不阻断加载）。
math：enc/dec_init/dec_step/math_meta 必需（整图直识，无检测/方向模型）。
rec_height/rec_window/PAD_*/MIN_CONF/LM_BEAM 等是权重绑定的训练/导出常量，
不暴露为参数。权重来自 Hugging Face datalab-to/indra-ocr（tag v1.1.0），
许可 AI PUBS OPEN RAIL-M LICENSE (MODIFIED)，商用需审许可条款。
"""

from __future__ import annotations

from datetime import date

from oxocr.models.base import HUGGINGFACE_REPO, MODELSCOPE_REPO, Capabilities, ModelFile, ModelSpec
from oxocr.models.languages import names

__all__ = ["INDRA_SPECS"]

_IMPL = "oxocr.models.indra_ocr.pipeline:IndraModel"
_CAPS_TEXT = Capabilities(backend="onnxruntime")
_CAPS_MATH = Capabilities(
    detection=False, recognition=True, own_postprocess=True, backend="onnxruntime"
)
_DIR = "indra-ocr"

_RAW_AUTHOR = "Datalab"
_RAW_COUNTRY = "America"
_RAW_URLS = (
    "https://github.com/datalab-to/indra",
    "https://huggingface.co/datalab-to/indra-ocr",
)
_RAW_LICENSE = "AI PUBS OPEN RAIL-M (modified)"
_RAW_DATE = date(2026, 10, 19)  # 权重 tag v1.1.0
_COMMERCIAL = (
    "restricted(研究/个人/年营收或融资低于2000万美元的初创免费；"
    "不得与 Datalab 及其关联方 API 竞争；商用需审 AI PUBS OPEN RAIL-M (MODIFIED) 条款)"
)
_TEXT_ARCH = {"det": "LineNet", "rec": "CTC", "orient": "OrientCls"}
_MATH_ARCH = {"rec": "Encoder-Decoder (attention)"}

_TEXT_DEFAULTS = {
    "det_long_side": 1024,
    "det_thresh": 0.5,
    "adaptive": True,
    "rec_batch_num": 64,
    "use_lm": True,
    "deskew": True,
    "auto_orient": True,
}

# 上游未公布拉丁语种逐项清单（README/模型卡仅称"约 50 种"），此表为推导清单：
# 词表覆盖 Latin-1/Ext-A/Ext-B/Ext-Additional（含 ɓɗƙƴɛɔŋə 等非洲字母与越南语 ƠƯ），
# 46 种沿用 rapid_ocr _V6_LATIN 码表，ha/ff/ak/yo 为上游点名新增，en 为主要语言；
# latin 作为语系族标识保留并置首。
_LATIN_LANGS = (
    "en", "fr", "de", "it", "es", "pt", "nl", "pl", "ro", "cs", "sv", "no",
    "da", "fi", "hu", "tr", "vi", "id", "ms", "az", "af", "bs", "hr", "cy",
    "et", "ga", "is", "ku", "lt", "lv", "mt", "mi", "oc", "sk", "sl", "sq",
    "sw", "tl", "uz", "la", "rs_latin", "ca", "eu", "gl", "lb", "rm", "qu",
    "ha", "ff", "ak", "yo",
)

_DET = ModelFile(
    "det", f"{_DIR}/det.onnx",
    sha256="f2bd4cba23e9c328cf7ec824634e423e191a2c6aacbbf35c62bd79dd70f62f59",
    size=389948,
)
_META = ModelFile(
    "meta", f"{_DIR}/meta.json",
    sha256="4ce4c94b46f5d644033c02e718322014ff4081270f385038c571912b56696339",
    size=211869,
)
_ORIENT = ModelFile(
    "orient", f"{_DIR}/orient.onnx",
    sha256="20eaf670f55a7cf9facfe1bdd6c7b8e40be75bbf4d48a3df43f66aa3bbd874fd",
    size=371802,
)


def _text_spec(
    name: str,
    rec: str,
    rec_sha: str,
    rec_size: int,
    lm: tuple[str, str, int] | None,
    display: str,
    languages: tuple[str, ...],
    size_bytes: int,
) -> ModelSpec:
    files = [
        _DET,
        ModelFile("rec", f"{_DIR}/{rec}", sha256=rec_sha, size=rec_size),
        _META,
        _ORIENT,
    ]
    if lm is not None:  # rtl 无 lm 文件
        fname, sha, size = lm
        files.append(ModelFile("lm", f"{_DIR}/{fname}", sha256=sha, size=size))
    zh, en = names(languages)
    return ModelSpec(
        name=name,
        family="indra_ocr",
        impl=_IMPL,
        files=tuple(files),
        name_chinese=display,
        raw_author=_RAW_AUTHOR,
        raw_author_country=_RAW_COUNTRY,
        raw_urls=_RAW_URLS,
        raw_release_date=_RAW_DATE,
        raw_license=_RAW_LICENSE,
        raw_model_ids={
            "det": "datalab-to/indra-ocr/det.onnx",
            "rec": f"datalab-to/indra-ocr/{rec}",
        },
        modelscope_repo=MODELSCOPE_REPO,
        huggingface_repo=HUGGINGFACE_REPO,
        model_type="text",
        languages=languages,
        languages_chinese=zh,
        languages_english=en,
        size_bytes=size_bytes,
        star_rating=4,
        model_architecture=_TEXT_ARCH,
        commercial_use=_COMMERCIAL,
        defaults=dict(_TEXT_DEFAULTS),
        capabilities=_CAPS_TEXT,
    )


INDRA_SPECS: dict[str, ModelSpec] = {
    "indra_ocr_latin": _text_spec(
        "indra_ocr_latin", "rec.onnx",
        "475972a2710be4a3cbfeb2bc7f3624ffd563de07de42476fc7a290d4e1ece484", 5042161,
        ("lm.npz", "6c0fb618a363a0acdd05d4f5b0edd9e8146d2d990541b87f6130bbfe17cea58d", 15695957),
        "Indra 拉丁文 OCR", ("latin",) + _LATIN_LANGS + ("cyrillic", "el"), 21711737,
    ),
    "indra_ocr_cjk": _text_spec(
        "indra_ocr_cjk", "rec_cjk.onnx",
        "4a035516a17e62cdf33a4fe2d88077654cb8841cff024685cdef4e351694ef50", 10176018,
        ("lm_cjk.npz", "6d64f13f4f90f4e529bd63340c6df76238d007feaef2abb903d8cd28d92bc2f3", 18955160),
        "Indra 中日韩 OCR", ("zh", "zh_tw", "ja", "ko"), 30104797,
    ),
    "indra_ocr_rtl": _text_spec(
        "indra_ocr_rtl", "rec_rtl.onnx",
        "468796bd684486a8388c30c8b4ed2a26e3eeb1f8ff02831c24c2e12c79a2a834", 4812710,
        None,
        "Indra 右向左 OCR（阿拉伯/希伯来）", ("ar", "fa", "ur", "ps", "ku", "he"), 5786329,
    ),
    "indra_ocr_world": _text_spec(
        "indra_ocr_world", "rec_world.onnx",
        "c8d1e59f21b04966f179d2d2d8e22b230d22110b3f89855bbbf2624a58c40043", 5627990,
        ("lm_world.npz", "34df5e74a2fcbb9cf5e7f360a559375ffa2e04d2d2943a42308c177bcb444d5d", 11641215),
        "Indra 多语种 OCR（印度/东南亚）",
        ("hi", "th", "bn", "ta", "te", "my", "km", "lo", "si", "ka", "hy", "am"),
        18242824,
    ),
    "indra_ocr_math": ModelSpec(
        name="indra_ocr_math",
        family="indra_ocr",
        impl=_IMPL,
        files=(
            ModelFile(
                "enc", f"{_DIR}/math_enc.onnx",
                sha256="d4e4a6c40dad1ee988d32c6403b1590ca42864d4483bbe1e671c5dbb999a5a97",
                size=9889575,
            ),
            ModelFile(
                "dec_init", f"{_DIR}/math_dec_init.onnx",
                sha256="919417fde9962c3318aeb478c1fe3955dd2bb8cd34476c68debed2ba39e5dcd6",
                size=1356364,
            ),
            ModelFile(
                "dec_step", f"{_DIR}/math_dec_step.onnx",
                sha256="5ff6e6cbaf0dea70314f66222a1469736a8c5e7994fab06a5b4e53c90469239f",
                size=8429418,
            ),
            ModelFile(
                "math_meta", f"{_DIR}/math_meta.json",
                sha256="85d9715fb8a77791b55ef89f57fb41e72eb3170ec58a7c3bee3b394268bf2b10",
                size=7797,
            ),
        ),
        name_chinese="Indra 公式识别",
        raw_author=_RAW_AUTHOR,
        raw_author_country=_RAW_COUNTRY,
        raw_urls=_RAW_URLS,
        raw_release_date=_RAW_DATE,
        raw_license=_RAW_LICENSE,
        raw_model_ids={
            "enc": "datalab-to/indra-ocr/math_enc.onnx",
            "dec_init": "datalab-to/indra-ocr/math_dec_init.onnx",
            "dec_step": "datalab-to/indra-ocr/math_dec_step.onnx",
        },
        modelscope_repo=MODELSCOPE_REPO,
        huggingface_repo=HUGGINGFACE_REPO,
        model_type="latex",
        languages=("latex",),
        languages_chinese=("latex",),
        languages_english=("latex",),
        size_bytes=19683154,
        star_rating=4,
        model_architecture=_MATH_ARCH,
        commercial_use=_COMMERCIAL,
        defaults={"math_min_conf": 0.0},
        capabilities=_CAPS_MATH,
    ),
}
