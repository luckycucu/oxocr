"""pytest 公共夹具：强制离线、定位 test/models、提供合成图像工具。"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
TEST_MODELS_DIR = REPO_ROOT / "test" / "models"

# 全程离线：缺文件即报错，绝不联网下载
os.environ.setdefault("OXOCR_OFFLINE", "1")


@pytest.fixture(scope="session")
def models_dir() -> Path:
    """离线权重目录（test/models）。缺失则跳过需要权重的用例。"""
    if not (TEST_MODELS_DIR / "rapid-ocr" / "PP-OCRv6" / "det" / "PP-OCRv6_det_small.onnx").is_file():
        pytest.skip("离线模型权重未放置在 test/models，跳过模型用例")
    return TEST_MODELS_DIR


@pytest.fixture
def model_dir_env(models_dir, monkeypatch):
    """把 OXOCR_MODEL_DIR 指向 test/models，供不显式传参的门面调用使用。"""
    monkeypatch.setenv("OXOCR_MODEL_DIR", str(models_dir))
    return models_dir


def _load_font(size: int):
    from PIL import ImageFont

    for name in ("arialbd.ttf", "arial.ttf", "segoeui.ttf", "calibrib.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def make_text_image(
    lines: list[str],
    *,
    width: int = 640,
    height: int = 200,
    dark: bool = False,
    font_size: int = 44,
    margin: int = 20,
    font: str | None = None,
) -> np.ndarray:
    """生成含若干行文本的 BGR 图，用于 OCR 集成用例。

    font: 指定字体文件名（如 msyh.ttc 渲染中文）；缺省走 _load_font 拉丁字体。
    """
    from PIL import Image, ImageDraw, ImageFont

    bg = (32, 32, 32) if dark else (250, 250, 250)
    fg = (238, 238, 238) if dark else (12, 12, 12)
    img = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(img)
    if font is not None:
        try:
            font_obj = ImageFont.truetype(font, font_size)
        except OSError:
            font_obj = _load_font(font_size)
    else:
        font_obj = _load_font(font_size)
    y = margin
    for line in lines:
        draw.text((margin, y), line, fill=fg, font=font_obj)
        y += font_size + 16
    return np.array(img)[:, :, ::-1].copy()


@pytest.fixture
def sample_image():
    return make_text_image(["Hello OCR 2026", "Second line text"])


@pytest.fixture
def small_image():
    # 触发 pad_border（宽/高 < 150）
    return make_text_image(["Hi"], width=120, height=60, font_size=28)


@pytest.fixture
def dark_image():
    return make_text_image(["Dark theme here"], dark=True)


MFR_DIR = TEST_MODELS_DIR / "pix2text-mfr-1.5"
MFR_QUANT_DIR = TEST_MODELS_DIR / "pix2text-mfr-quantized"
MFR_V1_DIR = TEST_MODELS_DIR / "pix2text-mfr"


@pytest.fixture(scope="session")
def mfr_models_dir() -> Path:
    """公式识别权重目录（pix2text-mfr-*）。缺失则跳过，与 PP-OCR 权重互不依赖。"""
    if (
        not (MFR_DIR / "encoder_model.onnx").is_file()
        or not (MFR_QUANT_DIR / "encoder_model.onnx").is_file()
        or not (MFR_V1_DIR / "encoder_model.onnx").is_file()
    ):
        pytest.skip("公式模型权重未放置在 test/models/pix2text-mfr-*，跳过")
    return TEST_MODELS_DIR


@pytest.fixture
def mfr_model_dir_env(mfr_models_dir, monkeypatch):
    """把 OXOCR_MODEL_DIR 指向 test/models，供公式模型的门面调用使用。"""
    monkeypatch.setenv("OXOCR_MODEL_DIR", str(mfr_models_dir))
    return mfr_models_dir


@pytest.fixture
def formula_image():
    return make_text_image(["E=mc^2"], width=480, height=160, font_size=56)


DUGUANG_DIR = TEST_MODELS_DIR / "duguang-ocr"


@pytest.fixture(scope="session")
def duguang_models_dir() -> Path:
    """读光权重目录（duguang-ocr/{small,large,max}）。缺失则跳过，与 PP-OCR/MFR 互不依赖。"""
    if not (DUGUANG_DIR / "small" / "det" / "model_1024x1024.onnx").is_file():
        pytest.skip("读光模型权重未放置在 test/models/duguang-ocr，跳过")
    return TEST_MODELS_DIR


@pytest.fixture
def duguang_model_dir_env(duguang_models_dir, monkeypatch):
    """把 OXOCR_MODEL_DIR 指向 test/models，供读光模型的门面调用使用。"""
    monkeypatch.setenv("OXOCR_MODEL_DIR", str(duguang_models_dir))
    return duguang_models_dir


ANAKSOR_DIR = TEST_MODELS_DIR / "anaksor-ocr"


@pytest.fixture(scope="session")
def anaksor_models_dir() -> Path:
    """anaksor 权重目录（anaksor-ocr/）。缺失则跳过，与其他家族互不依赖。"""
    if not (ANAKSOR_DIR / "khmer_rec_svtrv2_latest.onnx").is_file():
        pytest.skip("anaksor 模型权重未放置在 test/models/anaksor-ocr，跳过")
    return TEST_MODELS_DIR


@pytest.fixture
def anaksor_model_dir_env(anaksor_models_dir, monkeypatch):
    """把 OXOCR_MODEL_DIR 指向 test/models，供 anaksor 模型的门面调用使用。"""
    monkeypatch.setenv("OXOCR_MODEL_DIR", str(anaksor_models_dir))
    return anaksor_models_dir


OPENTHAI_DIR = TEST_MODELS_DIR / "openthai2.0-ocr"


@pytest.fixture(scope="session")
def openthai_models_dir() -> Path:
    """OpenThai 权重目录（openthai2.0-ocr/）。缺失则跳过，与其他家族互不依赖。"""
    if not (OPENTHAI_DIR / "rec.onnx").is_file():
        pytest.skip("OpenThai 模型权重未放置在 test/models/openthai2.0-ocr，跳过")
    return TEST_MODELS_DIR


@pytest.fixture
def openthai_model_dir_env(openthai_models_dir, monkeypatch):
    """把 OXOCR_MODEL_DIR 指向 test/models，供 OpenThai 模型的门面调用使用。"""
    monkeypatch.setenv("OXOCR_MODEL_DIR", str(openthai_models_dir))
    return openthai_models_dir


INDRA_DIR = TEST_MODELS_DIR / "indra-ocr"


@pytest.fixture(scope="session")
def indra_models_dir() -> Path:
    """Indra 权重目录（indra-ocr/）。缺失则跳过，与其他家族互不依赖。"""
    if not (INDRA_DIR / "rec.onnx").is_file():
        pytest.skip("Indra 模型权重未放置在 test/models/indra-ocr，跳过")
    return TEST_MODELS_DIR


@pytest.fixture
def indra_model_dir_env(indra_models_dir, monkeypatch):
    """把 OXOCR_MODEL_DIR 指向 test/models，供 Indra 模型的门面调用使用。"""
    monkeypatch.setenv("OXOCR_MODEL_DIR", str(indra_models_dir))
    return indra_models_dir


LATEX_OCR_DIR = TEST_MODELS_DIR / "rapid-latex-ocr"


@pytest.fixture(scope="session")
def latex_ocr_models_dir() -> Path:
    """LaTeX-OCR 权重目录（rapid-latex-ocr/）。缺失则跳过，与其他家族互不依赖。"""
    if not (LATEX_OCR_DIR / "encoder.onnx").is_file():
        pytest.skip("LaTeX-OCR 模型权重未放置在 test/models/rapid-latex-ocr，跳过")
    return TEST_MODELS_DIR


@pytest.fixture
def latex_ocr_model_dir_env(latex_ocr_models_dir, monkeypatch):
    """把 OXOCR_MODEL_DIR 指向 test/models，供 LaTeX-OCR 模型的门面调用使用。"""
    monkeypatch.setenv("OXOCR_MODEL_DIR", str(latex_ocr_models_dir))
    return latex_ocr_models_dir
