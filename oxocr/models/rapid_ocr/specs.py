"""PP-OCR 家族 ModelSpec 清单（纯数据，不导入推理实现）。

权重目录与参数对齐 examples/RapidOCR-main/python/rapidocr：
det/rec 的前后处理各版本完全一致，差异只在权重文件；
cls 则按模型区分输入形状（v2.0 为 3×48×192，textline_ori 为固定 3×80×160）。

新增组合 = 往下面的表里加一行；文件路径相对 model_dir，权重统一置于 rapid-ocr/ 下。
权重经 hub 仓库下载（ModelScope 优先，HuggingFace 兜底），仓库内路径与 files.filename
一致；不校验 sha256（仓库文件可能更新，发版后哈希无法跟随）。

元数据（作者/许可/语言/架构等）按 docs/模型信息字段.md 填写：
原始作者为百度飞桨 PaddlePaddle，当前 ONNX 权重由 RapidAI/RapidOCR 转换发布；
语言列表与排序依据 PaddleOCR 官方文档（v6 支持 50 种语言，v5 各模型见官方语种表）。
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from typing import Any

from oxocr.errors import UnknownParameterError
from oxocr.models.base import HUGGINGFACE_REPO, MODELSCOPE_REPO, Capabilities, ModelFile, ModelSpec
from oxocr.models.languages import names

__all__ = [
    "SPECS",
    "PPOCR_DEFAULTS",
    "CLS_MODELS",
    "CLS_BUNDLE_SPEC",
    "ClsVariant",
    "cls_variant",
    "cls_file_for",
]


@dataclass(frozen=True)
class ClsVariant:
    """方向分类权重及其**固定**输入形状；两者必须成对使用。"""

    filename: str
    image_shape: tuple[int, int, int]


# 三个权重都是 2 类（0/180），差别只在 backbone 与输入形状。
# textline_ori 的 80×160 是 ONNX 图里的静态维度，喂 48×192 会直接报错。
CLS_MODELS: dict[str, ClsVariant] = {
    "v2_0_cls": ClsVariant(
        "rapid-ocr/PP-OCR-cls/ch_ppocr_mobile_v2.0_cls_mobile.onnx", (3, 48, 192)
    ),
    "textline_ori_mobile": ClsVariant(
        "rapid-ocr/PP-OCR-cls/ch_PP-LCNet_x0_25_textline_ori_cls_mobile.onnx", (3, 80, 160)
    ),
    "textline_ori_server": ClsVariant(
        "rapid-ocr/PP-OCR-cls/ch_PP-LCNet_x1_0_textline_ori_cls_server.onnx", (3, 80, 160)
    ),
}

DEFAULT_CLS = "v2_0_cls"

# ppocr_cls 下载 bundle：一次拉取全部方向分类权重（全家族共用）。
# 非推理模型（impl 为空），不进 catalog，仅供 download_model("ppocr_cls") 使用。
CLS_BUNDLE_SPEC = ModelSpec(
    name="ppocr_cls",
    family="ppocr",
    impl="",
    files=tuple(ModelFile(name, variant.filename) for name, variant in CLS_MODELS.items()),
    modelscope_repo=MODELSCOPE_REPO,
    huggingface_repo=HUGGINGFACE_REPO,
)

PPOCR_DEFAULTS: dict[str, Any] = {
    "det_limit_side_len": 736,
    "det_thresh": 0.3,
    "det_box_thresh": 0.5,
    "det_unclip_ratio": 1.6,
    "cls_thresh": 0.9,
    "cls_model": DEFAULT_CLS,
    "cls_batch_num": 6,
    "rec_image_shape": (3, 48, 320),
    "rec_batch_num": 6,
}

_IMPL = "oxocr.models.rapid_ocr.pipeline:PpOcrModel"
_CAPS = Capabilities(angle_classification=True, backend="onnxruntime")

_RAW_AUTHOR = "飞桨（百度） / PaddlePaddle (Baidu)"
_RAW_COUNTRY = "China"
_RAW_URLS = (
    "https://github.com/PaddlePaddle/PaddleOCR",
    "https://modelscope.cn/models/PaddlePaddle/PaddleOCR",
)
_RAW_LICENSE = "Apache-2.0"
_CUR_AUTHOR = "RapidAI(SWHL)"
_CUR_COUNTRY = "China"
_CUR_URLS = (
    "https://github.com/RapidAI/RapidOCR",
    "https://modelscope.cn/models/RapidAI/RapidOCR",
)

_CLS_ARCH = {
    "v2_0_cls": "AngleCls (MobileNetV3)",
    "textline_ori_mobile": "TextlineOri (PP-LCNet_x0_25)",
    "textline_ori_server": "TextlineOri (PP-LCNet_x1_0)",
}

_LANG_LABEL = {
    "ch": "中英",
    "ch_doc": "中英文档",
    "en": "英文",
    "japan": "日文",
    "korean": "韩文",
    "arabic": "阿拉伯文",
    "devanagari": "天城文",
    "ka": "卡纳达文",
    "ta": "泰米尔文",
    "te": "泰卢固文",
    "latin": "拉丁文",
    "chinese_cht": "繁体中文",
    "cyrillic": "西里尔文",
    "el": "希腊文",
    "eslav": "东斯拉夫文",
    "th": "泰文",
}

# ---- 官方语言列表（短代码沿用 PaddleOCR 写法，顺序按官方）----
# PP-OCRv6：中/繁/英/日 + 46 种拉丁语系，共 50 种；tiny 不含日文（49 种）。
_V6_LATIN = (
    "fr", "de", "it", "es", "pt", "nl", "pl", "ro", "cs", "sv", "no", "da",
    "fi", "hu", "tr", "vi", "id", "ms", "az", "af", "bs", "hr", "cy", "et",
    "ga", "is", "ku", "lt", "lv", "mt", "mi", "oc", "sk", "sl", "sq", "sw",
    "tl", "uz", "la", "rs_latin", "ca", "eu", "gl", "lb", "rm", "qu",
)
_V6_LANGS = ("zh", "zh_tw", "en", "ja") + _V6_LATIN
_V6_TINY_LANGS = ("zh", "zh_tw", "en") + _V6_LATIN
# PP-OCRv5 中文模型：简体/繁体/英文/日文。
_V5_CH = ("zh", "zh_tw", "en", "ja")
_V5_LATIN = (
    "fr", "de", "af", "it", "es", "bs", "pt", "cs", "cy", "da", "et", "ga",
    "hr", "uz", "hu", "rs_latin", "id", "oc", "is", "lt", "mi", "ms", "nl",
    "no", "pl", "sk", "sl", "sq", "sv", "sw", "tl", "tr", "la", "az", "ku",
    "lv", "mt", "pi", "ro", "vi", "fi", "eu", "gl", "lb", "rm", "ca", "qu",
)
_V5_CYRILLIC = (
    "ru", "be", "uk", "rs_cyrillic", "bg", "mn", "ab", "ady", "kbd", "av",
    "dar", "inh", "ce", "lki", "lez", "tab", "kk", "ky", "tg", "mk", "tt",
    "cv", "ba", "mhr", "mo", "udm", "kv", "os", "bua", "xal", "tyv", "sah",
    "kaa", "en",
)
_V5_ARABIC = ("ar", "fa", "ug", "ur", "ps", "ku", "sd", "bal", "en")
_V5_DEVANAGARI = (
    "hi", "mr", "ne", "bh", "mai", "ang", "bho", "mah", "sck", "new", "gom",
    "sa", "bgc", "en",
)
_V5_ESLAV = ("ru", "be", "uk", "en")

# (语种键, rec 文件名, languages)；mobile/server 由文件名后缀推导
_V3_REC = (
    ("latin", "latin_PP-OCRv3_rec_mobile.onnx", ("latin", "en")),
    ("chinese_cht", "chinese_cht_PP-OCRv3_rec_mobile.onnx", ("zh_tw",)),
    ("cyrillic", "cyrillic_PP-OCRv3_rec_mobile.onnx", ("cyrillic",)),
)

_V4_REC = (
    ("ch", "ch_PP-OCRv4_rec_mobile.onnx", ("zh", "en")),
    ("ch", "ch_PP-OCRv4_rec_server.onnx", ("zh", "en")),
    ("ch_doc", "ch_doc_PP-OCRv4_rec_server.onnx", ("zh", "en")),
    ("en", "en_PP-OCRv4_rec_mobile.onnx", ("en",)),
    ("japan", "japan_PP-OCRv4_rec_mobile.onnx", ("ja",)),
    ("korean", "korean_PP-OCRv4_rec_mobile.onnx", ("ko",)),
    ("arabic", "arabic_PP-OCRv4_rec_mobile.onnx", ("ar",)),
    ("devanagari", "devanagari_PP-OCRv4_rec_mobile.onnx", ("devanagari",)),
    ("ka", "ka_PP-OCRv4_rec_mobile.onnx", ("ka",)),
    ("ta", "ta_PP-OCRv4_rec_mobile.onnx", ("ta",)),
    ("te", "te_PP-OCRv4_rec_mobile.onnx", ("te",)),
)

_V5_REC = (
    ("ch", "ch_PP-OCRv5_rec_mobile.onnx", _V5_CH),
    ("ch", "ch_PP-OCRv5_rec_server.onnx", _V5_CH),
    ("en", "en_PP-OCRv5_rec_mobile.onnx", ("en",)),
    ("latin", "latin_PP-OCRv5_rec_mobile.onnx", _V5_LATIN),
    ("cyrillic", "cyrillic_PP-OCRv5_rec_mobile.onnx", _V5_CYRILLIC),
    ("eslav", "eslav_PP-OCRv5_rec_mobile.onnx", _V5_ESLAV),
    ("arabic", "arabic_PP-OCRv5_rec_mobile.onnx", _V5_ARABIC),
    ("devanagari", "devanagari_PP-OCRv5_rec_mobile.onnx", _V5_DEVANAGARI),
    ("el", "el_PP-OCRv5_rec_mobile.onnx", ("el", "en")),
    ("korean", "korean_PP-OCRv5_rec_mobile.onnx", ("ko", "en")),
    ("ta", "ta_PP-OCRv5_rec_mobile.onnx", ("ta", "en")),
    ("te", "te_PP-OCRv5_rec_mobile.onnx", ("te", "en")),
    ("th", "th_PP-OCRv5_rec_mobile.onnx", ("th", "en")),
)

# 原始开源日期（PaddleOCR 官方发布）：v3=2.5、v4=release/2.7、
# v5 中文=3.0.0、v5 英/泰/希腊=3.2.0、v5 其余小语种=3.3.0、v6=3.7.0。
_V3_DATE = date(2022, 5, 9)
_V4_DATE = date(2023, 8, 7)
_V6_DATE = date(2026, 6, 11)
_V5_DATE = {
    "ch": date(2025, 5, 20),
    "en": date(2025, 8, 21),
    "th": date(2025, 8, 21),
    "el": date(2025, 8, 21),
}
_V5_DATE_DEFAULT = date(2025, 10, 16)

# 模型总大小（字节，test/models 实测 det+rec+默认 cls 合计）与星级（编辑评级，0–5）。
_SIZE_BYTES = {
    "ppocrv6_small": 31749509,
    "ppocrv6_tiny": 6904963,
    "ppocrv6_medium": 139334970,
    "ppocrv5_ch_mobile": 22469390,
    "ppocrv5_ch_server": 173714298,
    "ppocrv5_en_mobile": 13710435,
    "ppocrv5_latin_mobile": 13742597,
    "ppocrv5_cyrillic_mobile": 13912176,
    "ppocrv5_eslav_mobile": 13749886,
    "ppocrv5_arabic_mobile": 13861912,
    "ppocrv5_devanagari_mobile": 13778445,
    "ppocrv5_el_mobile": 13670434,
    "ppocrv5_korean_mobile": 19326832,
    "ppocrv5_ta_mobile": 13748010,
    "ppocrv5_te_mobile": 13761186,
    "ppocrv5_th_mobile": 13753378,
    "ppocrv4_ch_mobile": 16189007,
    "ppocrv4_ch_server": 204468368,
    "ppocrv4_ch_doc_server": 208864944,
    "ppocrv4_en_mobile": 12984093,
    "ppocrv4_japan_mobile": 15084384,
    "ppocrv4_korean_mobile": 29398829,
    "ppocrv4_arabic_mobile": 13016255,
    "ppocrv4_devanagari_mobile": 13019262,
    "ppocrv4_ka_mobile": 13012430,
    "ppocrv4_ta_mobile": 27661661,
    "ppocrv4_te_mobile": 27672870,
    "ppocrv3_latin_mobile": 11985362,
    "ppocrv3_chinese_cht_mobile": 14159707,
    "ppocrv3_cyrillic_mobile": 11979584,
    "ppocrv3_en_mobile": 11985430,
}
_STAR_RATING = {
    "ppocrv6_small": 5,
    "ppocrv6_tiny": 4,
    "ppocrv6_medium": 5,
    "ppocrv5_ch_mobile": 4,
    "ppocrv5_ch_server": 5,
    "ppocrv4_ch_mobile": 4,
    "ppocrv4_ch_server": 4,
    "ppocrv4_ch_doc_server": 4,
}


def cls_variant(name: str, context: str) -> ClsVariant:
    if name not in CLS_MODELS:
        raise UnknownParameterError(
            "cls_model", allowed=sorted(CLS_MODELS), context=context
        )
    return CLS_MODELS[name]


def cls_file_for(spec: ModelSpec, params: Mapping[str, Any]) -> ModelFile:
    """按 params['cls_model'] 选出 cls 权重；未指定时用 spec 声明的默认。"""
    name = params.get("cls_model") or spec.defaults.get("cls_model", DEFAULT_CLS)
    return ModelFile("cls", cls_variant(str(name), f"模型 {spec.name}").filename)


def _size_of(rec_filename: str) -> str:
    return "server" if rec_filename.endswith("_server.onnx") else "mobile"


def _stem(filename: str) -> str:
    return filename.rsplit("/", 1)[-1].removesuffix(".onnx")


def _spec(
    name: str,
    display: str,
    det: str,
    rec: str,
    languages: tuple[str, ...],
    cls_model: str,
    release: date,
    det_arch: str,
    rec_arch: str,
) -> ModelSpec:
    cls_filename = CLS_MODELS[cls_model].filename
    zh, en = names(languages)
    return ModelSpec(
        name=name,
        family="ppocr",
        impl=_IMPL,
        files=(
            ModelFile("det", det),
            ModelFile("rec", rec),
            ModelFile("cls", cls_filename),
        ),
        name_chinese=display,
        raw_author=_RAW_AUTHOR,
        raw_author_country=_RAW_COUNTRY,
        raw_urls=_RAW_URLS,
        raw_release_date=release,
        raw_license=_RAW_LICENSE,
        raw_model_ids={
            "det": _stem(det),
            "rec": _stem(rec),
            "cls": _stem(cls_filename),
        },
        modelscope_repo=MODELSCOPE_REPO,
        huggingface_repo=HUGGINGFACE_REPO,
        current_author=_CUR_AUTHOR,
        current_author_country=_CUR_COUNTRY,
        current_urls=_CUR_URLS,
        current_release_date=None,
        current_license=_RAW_LICENSE,
        model_type="text",
        languages=languages,
        languages_chinese=zh,
        languages_english=en,
        size_bytes=_SIZE_BYTES[name],
        star_rating=_STAR_RATING.get(name, 4 if name.startswith("ppocrv5") else 3),
        model_architecture={"det": det_arch, "rec": rec_arch, "cls": _CLS_ARCH[cls_model]},
        commercial_use="yes",
        defaults={**PPOCR_DEFAULTS, "cls_model": cls_model},
        capabilities=_CAPS,
    )


def _v6_spec(size: str, display: str) -> ModelSpec:
    # 参考实现的 default_models.yaml 没有 v6 cls 段，沿用 v4 的 2 类权重。
    languages = _V6_TINY_LANGS if size == "tiny" else _V6_LANGS
    return _spec(
        f"ppocrv6_{size}",
        display,
        f"rapid-ocr/PP-OCRv6/det/PP-OCRv6_det_{size}.onnx",
        f"rapid-ocr/PP-OCRv6/rec/PP-OCRv6_rec_{size}.onnx",
        languages,
        DEFAULT_CLS,
        _V6_DATE,
        "DBNet (PPLCNetV4)",
        "SVTR_LCNet (PPLCNetV4)",
    )


def _det_arch(version: int, size: str) -> str:
    if version == 5:
        return "DBNet (PPHGNetV2_B4)" if size == "server" else "DBNet (PPLCNetV3)"
    return "DBNet (PPHGNet_small)" if size == "server" else "DBNet (PPLCNetV3)"


def _rec_arch(version: int, lang: str, size: str) -> str:
    if version == 5:
        if lang == "ch" and size == "server":
            return "SVTR_HGNet (PPHGNetV2_B4)"
        return "SVTR_LCNet (PPLCNetV3)"
    if lang in ("ch", "ch_doc") and size == "server":
        return "SVTR_HGNet (PPHGNet_small)"
    if lang in ("ch", "ch_doc", "en"):
        return "SVTR_LCNet (PPLCNetV3)"
    return "SVTR (MobileNetV1Enhance)"


def _v45_specs(version: int, rec_table, cls_model: str) -> dict[str, ModelSpec]:
    """v4/v5：rec 有 mobile/server 之分，det 跟随同档位；非中文 rec 也配 ch det。"""
    out: dict[str, ModelSpec] = {}
    for lang, filename, languages in rec_table:
        size = _size_of(filename)
        name = f"ppocrv{version}_{lang}_{size}"
        release = _V4_DATE if version == 4 else _V5_DATE.get(lang, _V5_DATE_DEFAULT)
        out[name] = _spec(
            name,
            f"PP-OCRv{version} {_LANG_LABEL[lang]} {size}",
            f"rapid-ocr/PP-OCRv{version}/det/ch_PP-OCRv{version}_det_{size}.onnx",
            f"rapid-ocr/PP-OCRv{version}/rec/{filename}",
            languages,
            cls_model,
            release,
            _det_arch(version, size),
            _rec_arch(version, lang, size),
        )
    return out


def _v3_specs() -> dict[str, ModelSpec]:
    """v3 只有 en/multi 两个 det，没有 ch det；默认配通用 multi det。"""
    out: dict[str, ModelSpec] = {}
    for lang, filename, languages in _V3_REC:
        name = f"ppocrv3_{lang}_mobile"
        out[name] = _spec(
            name,
            f"PP-OCRv3 {_LANG_LABEL[lang]} mobile",
            "rapid-ocr/PP-OCRv3/det/multi_PP-OCRv3_det_mobile.onnx",
            f"rapid-ocr/PP-OCRv3/rec/{filename}",
            languages,
            DEFAULT_CLS,
            _V3_DATE,
            "DBNet (MobileNetV3)",
            "SVTR (MobileNetV1Enhance)",
        )
    out["ppocrv3_en_mobile"] = _spec(
        "ppocrv3_en_mobile",
        "PP-OCRv3 英文 mobile（en 检测器）",
        "rapid-ocr/PP-OCRv3/det/en_PP-OCRv3_det_mobile.onnx",
        "rapid-ocr/PP-OCRv3/rec/latin_PP-OCRv3_rec_mobile.onnx",
        ("en",),
        DEFAULT_CLS,
        _V3_DATE,
        "DBNet (MobileNetV3)",
        "SVTR (MobileNetV1Enhance)",
    )
    return out


SPECS: dict[str, ModelSpec] = {
    "ppocrv6_small": _v6_spec("small", "PP-OCRv6 small（默认，均衡）"),
    "ppocrv6_tiny": _v6_spec("tiny", "PP-OCRv6 tiny（最快）"),
    "ppocrv6_medium": _v6_spec("medium", "PP-OCRv6 medium（更准）"),
    # 参考实现按 ocr_version 选 cls 形状：v5 → 80×160，v4 → 48×192。
    **_v45_specs(5, _V5_REC, "textline_ori_mobile"),
    **_v45_specs(4, _V4_REC, DEFAULT_CLS),
    **_v3_specs(),
}
