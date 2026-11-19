"""公式识别离线集成测试（pix2text-mfr，全程不联网）。

合成公式图（PIL 渲染）与训练分布有差距，因此不断言具体 LaTeX 内容，
只断言结构契约：非空输出、box 语义、退化防护、门控行为与执行后端一致性。
"""
from __future__ import annotations

import oxocr
import pytest
from oxocr.errors import ModelFileMissingError, UnknownParameterError

pytestmark = [pytest.mark.model, pytest.mark.integration]

MFR = "pix2text_mfr_1.5"
MFR_QUANT = "pix2text_mfr_quantized"
MFR_V1 = "pix2text_mfr"
ALL_MFR = [MFR, MFR_QUANT, MFR_V1]


@pytest.fixture(autouse=True)
def _clean_pool():
    oxocr.unload_all()
    yield
    oxocr.unload_all()


def test_list_models_reports_family(mfr_model_dir_env):
    infos = oxocr.list_models(family="pix2text")
    assert {m.name for m in infos} == set(ALL_MFR)
    assert all(m.installed for m in infos)
    assert all(m.languages == ("latex",) for m in infos)


@pytest.mark.parametrize("name", ALL_MFR)
def test_readtext_formula_returns_latex(mfr_model_dir_env, formula_image, name):
    r = oxocr.ocr(formula_image, model=name)
    assert r.items, f"{name} 未识别出条目"
    item = r.items[0]
    assert r.text.strip()
    assert item.meta["degenerate"] is False
    assert item.meta["num_tokens"] > 0
    assert 0.0 < item.score <= 1.0
    assert r.elapse.rec > 0
    # LaTeX 特征从宽：含反斜杠命令或数学符号即视为公式域输出
    assert "\\" in r.text or any(c.isalpha() for c in r.text)


@pytest.mark.parametrize("name", ALL_MFR)
def test_box_is_full_input_image(mfr_model_dir_env, formula_image, name):
    r = oxocr.ocr(formula_image, model=name)
    h, w = formula_image.shape[:2]
    assert r.items
    expected = [[0, 0], [w, 0], [w, h], [0, h]]
    for (x, y), (ex, ey) in zip(r.items[0].box.points, expected, strict=True):
        assert x == pytest.approx(ex, abs=1.0)
        assert y == pytest.approx(ey, abs=1.0)


def test_default_postprocess_off_and_explicit_override(mfr_model_dir_env, formula_image):
    r = oxocr.ocr(formula_image, model=MFR)
    assert r.config.postprocess.enabled is False
    # raw 直通：最终文本与条目原文一致（LaTeX 不被空白规范化/纠错改写）
    assert r.text.strip() == r.items[0].text.strip()

    r2 = oxocr.ocr(formula_image, model=MFR, postprocess=True)
    assert r2.config.postprocess.enabled is True


def test_max_new_tokens_param_reaches_model(mfr_model_dir_env, formula_image):
    # 1 步内必到不了 eos：要么退化（score=0 被过滤），要么空文本，总之无条目
    r = oxocr.ocr(
        formula_image,
        model=MFR,
        config={"model_options": {"params": {"max_new_tokens": 1}}},
    )
    assert r.items == []


def test_unknown_param_rejected(mfr_model_dir_env, formula_image):
    with pytest.raises(UnknownParameterError):
        oxocr.ocr(
            formula_image,
            model=MFR,
            config={"model_options": {"params": {"bogus": 1}}},
        )


def test_missing_file_raises_expected_path(mfr_model_dir_env, formula_image, tmp_path):
    with pytest.raises(ModelFileMissingError) as e:
        oxocr.ocr(formula_image, model=MFR, model_dir=tmp_path)
    assert "encoder_model.onnx" in str(e.value)


def test_subprocess_matches_inprocess(mfr_model_dir_env, formula_image):
    a = oxocr.readtext(formula_image, model=MFR_QUANT, backend="inprocess")
    b = oxocr.readtext(formula_image, model=MFR_QUANT, backend="subprocess")
    assert a == b


def test_resident_second_call_zero_load(mfr_model_dir_env, formula_image):
    eng = oxocr.OcrEngine(model=MFR_QUANT)
    r1 = eng.ocr(formula_image)
    r2 = eng.ocr(formula_image)
    assert r1.elapse.load > 0
    assert r2.elapse.load == pytest.approx(0.0, abs=1e-4)
    assert r1.text == r2.text
    eng.close()  # 常驻：close 只归还句柄，不卸载


def test_detect_only_returns_no_items(mfr_model_dir_env, formula_image):
    # 公式模型无检测：仅检测模式语义下返回空
    assert oxocr.OcrEngine(model=MFR).detect_only(formula_image) == []
