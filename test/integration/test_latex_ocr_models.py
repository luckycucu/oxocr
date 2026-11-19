"""LaTeX-OCR 公式识别离线集成测试（全程不联网，-m model）。

贪心解码确定性 + 高置信度（score≈1），对 4 张官方测试图（test/data/latex_ocr）
断言紧凑化后 LaTeX 精确 golden；其余为结构契约、前/后处理门控与执行后端一致性。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import oxocr
import pytest
from oxocr.errors import ModelFileMissingError, UnknownParameterError

pytestmark = [pytest.mark.model, pytest.mark.integration]

MODEL = "rapid_latex_ocr"
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA = REPO_ROOT / "test" / "data" / "latex_ocr"

# 探针实测的紧凑化后 LaTeX（贪心确定性，跨运行可复现）
GOLDEN = {
    "1.png": r"\exp\left[\int d^{4}x g\phi\bar{\psi}\psi\right]=\sum_{n=0}^{\infty}\frac{g^{n}}{n!}\left(\int d^{4}x\phi\bar{\psi}\psi\right)^{n}.",
    "2.png": r"x^{2}+y^{2}=1",
    "5.png": r"x={\frac{-b\pm{\sqrt{b^{2}-4a c\ }}}{2a}}",
    "6.png": r"{\frac{x^{2}}{a^{2}}}-{\frac{y^{2}}{b^{2}}}=1",
}


@pytest.fixture(autouse=True)
def _clean_pool():
    oxocr.unload_all()
    yield
    oxocr.unload_all()


def _load_bgr(name: str) -> np.ndarray:
    from PIL import Image

    path = DATA / name
    if not path.is_file():
        pytest.skip(f"官方测试图 {path} 缺失，跳过")
    return np.array(Image.open(path).convert("RGB"))[:, :, ::-1].copy()


def test_list_models_reports_family(latex_ocr_model_dir_env):
    infos = oxocr.list_models(family="latex_ocr")
    assert {m.name for m in infos} == {MODEL}
    assert all(m.installed for m in infos)
    assert all(m.languages == ("latex",) for m in infos)


@pytest.mark.parametrize("name", sorted(GOLDEN))
def test_golden_latex(latex_ocr_model_dir_env, name):
    r = oxocr.ocr(_load_bgr(name), model=MODEL)
    assert r.items, f"{name} 未识别出条目"
    item = r.items[0]
    assert item.meta["degenerate"] is False
    assert item.meta["num_tokens"] > 0
    assert item.score > 0.98
    assert r.text == GOLDEN[name]


def test_box_is_full_input_image(latex_ocr_model_dir_env):
    img = _load_bgr("2.png")
    r = oxocr.ocr(img, model=MODEL)
    h, w = img.shape[:2]
    assert r.items
    expected = [[0, 0], [w, 0], [w, h], [0, h]]
    for (x, y), (ex, ey) in zip(r.items[0].box.points, expected, strict=True):
        assert x == pytest.approx(ex, abs=1.0)
        assert y == pytest.approx(ey, abs=1.0)


def test_default_preprocess_off(latex_ocr_model_dir_env):
    # own_preprocess：默认跳过核心前处理（pad_border 补边会干扰模型内部预处理）
    r = oxocr.ocr(_load_bgr("2.png"), model=MODEL)
    assert r.config.preprocess.enabled is False


def test_explicit_preprocess_override(latex_ocr_model_dir_env):
    r = oxocr.ocr(_load_bgr("2.png"), model=MODEL, preprocess=True)
    assert r.config.preprocess.enabled is True


def test_default_postprocess_off_and_explicit_override(latex_ocr_model_dir_env):
    r = oxocr.ocr(_load_bgr("2.png"), model=MODEL)
    assert r.config.postprocess.enabled is False
    assert r.text.strip() == r.items[0].text.strip()

    r2 = oxocr.ocr(_load_bgr("2.png"), model=MODEL, postprocess=True)
    assert r2.config.postprocess.enabled is True


def test_max_new_tokens_param_reaches_model(latex_ocr_model_dir_env):
    # 1 步内必到不了 eos：退化（score=0 被过滤）或空文本，总之无条目
    r = oxocr.ocr(
        _load_bgr("2.png"),
        model=MODEL,
        config={"model_options": {"params": {"max_new_tokens": 1}}},
    )
    assert r.items == []


def test_unknown_param_rejected(latex_ocr_model_dir_env):
    with pytest.raises(UnknownParameterError):
        oxocr.ocr(
            _load_bgr("2.png"),
            model=MODEL,
            config={"model_options": {"params": {"bogus": 1}}},
        )


def test_missing_file_raises_expected_path(latex_ocr_model_dir_env, tmp_path):
    with pytest.raises(ModelFileMissingError) as e:
        oxocr.ocr(_load_bgr("2.png"), model=MODEL, model_dir=tmp_path)
    assert "encoder.onnx" in str(e.value)


def test_subprocess_matches_inprocess(latex_ocr_model_dir_env):
    img = _load_bgr("2.png")
    a = oxocr.readtext(img, model=MODEL, backend="inprocess")
    b = oxocr.readtext(img, model=MODEL, backend="subprocess")
    assert a == b


def test_resident_second_call_zero_load(latex_ocr_model_dir_env):
    img = _load_bgr("2.png")
    eng = oxocr.OcrEngine(model=MODEL)
    r1 = eng.ocr(img)
    r2 = eng.ocr(img)
    assert r1.elapse.load > 0
    assert r2.elapse.load == pytest.approx(0.0, abs=1e-4)
    assert r1.text == r2.text
    eng.close()


def test_detect_only_returns_no_items(latex_ocr_model_dir_env):
    assert oxocr.OcrEngine(model=MODEL).detect_only(_load_bgr("2.png")) == []


def test_terminates_on_non_formula(latex_ocr_model_dir_env, formula_image):
    # 非公式图也须在 max_new_tokens 步内终止不挂起（不断言退化，公式模型对文本图也产 LaTeX 样式输出）
    r = oxocr.ocr(formula_image, model=MODEL)
    assert r.elapse.rec > 0
