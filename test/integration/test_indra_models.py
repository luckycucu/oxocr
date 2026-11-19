"""Indra OCR 离线集成测试（全程不联网，-m model）。

合成文本图（PIL 渲染）与训练分布有差距，不断言逐字 golden，只断言结构契约
与语种特征；真实样图 test/data/indra/*.png 做从宽特征断言。
PIL 无 raqm：阿拉伯/希伯来合成图按码位从左到右绘制（视觉序 = 逻辑序的翻转），
故 rtl 用例先 flip 再渲染，模型翻回后应还原逻辑序。
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import oxocr
import pytest
from oxocr.errors import ModelFileMissingError, UnknownParameterError
from oxocr.models.indra_ocr.rtl import flip

pytestmark = [pytest.mark.model, pytest.mark.integration]

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA = REPO_ROOT / "test" / "data" / "indra"
TEXT_MODELS = ["indra_ocr_latin", "indra_ocr_cjk", "indra_ocr_rtl", "indra_ocr_world"]


@pytest.fixture(autouse=True)
def _clean_pool():
    oxocr.unload_all()
    yield
    oxocr.unload_all()


def _load_bgr(name: str) -> np.ndarray:
    from PIL import Image

    path = DATA / name
    if not path.is_file():
        pytest.skip(f"真实样图 {path} 缺失，跳过")
    return np.array(Image.open(path).convert("RGB"))[:, :, ::-1].copy()


def _has_cjk(text: str) -> bool:
    return any("一" <= c <= "鿿" for c in text)


def _has_hangul(text: str) -> bool:
    return any("가" <= c <= "힣" or "ᄀ" <= c <= "ᇿ" for c in text)


def _has_thai(text: str) -> bool:
    return any("฀" <= c <= "๿" for c in text)


def _has_arabic(text: str) -> bool:
    return any("؀" <= c <= "ۿ" for c in text)


def _has_hebrew(text: str) -> bool:
    return any("֐" <= c <= "׿" for c in text)


def test_list_models_reports_family(indra_model_dir_env):
    infos = oxocr.list_models(family="indra_ocr")
    assert {m.name for m in infos} == {*TEXT_MODELS, "indra_ocr_math"}
    assert all(m.installed for m in infos)
    assert all(m.downloadable for m in infos)
    by_name = {m.name: m for m in infos}
    assert by_name["indra_ocr_cjk"].languages == ("zh", "zh_tw", "ja", "ko")
    assert by_name["indra_ocr_math"].languages == ("latex",)


def test_latin_synthetic(indra_model_dir_env, sample_image):
    r = oxocr.ocr(sample_image, model="indra_ocr_latin")
    assert r.items
    assert any("Hello" in it.text for it in r.items)
    for it in r.items:
        assert 0.0 < it.score <= 1.0
    assert r.elapse.det > 0
    assert r.elapse.rec > 0


def test_cjk_synthetic(indra_model_dir_env):
    from conftest import make_text_image

    img = make_text_image(["你好世界 OCR", "中文识别测试"], font="msyh.ttc", font_size=40)
    r = oxocr.ocr(img, model="indra_ocr_cjk")
    assert r.items
    assert _has_cjk(r.text)
    assert any("你好" in it.text for it in r.items)


def test_cjk_real_essay(indra_model_dir_env):
    r = oxocr.ocr(_load_bgr("cjk.png"), model="indra_ocr_cjk")
    assert r.items
    assert _has_cjk(r.text)


def test_cjk_real_korean_jamo_composed(indra_model_dir_env):
    r = oxocr.ocr(_load_bgr("korean.png"), model="indra_ocr_cjk")
    assert r.items
    assert _has_hangul(r.text)
    # 至少部分音节已 NFC 合成（非纯 jamo）
    assert any("가" <= c <= "힣" for c in r.text)


def test_rtl_arabic_logical_order(indra_model_dir_env):
    from conftest import make_text_image

    logical = "مرحبا بالعالم"
    # PIL 无 raqm：渲染 flip(逻辑序) 得到正确视觉序，模型应翻回逻辑序
    img = make_text_image([flip(logical)], font="arial.ttf", font_size=36)
    r = oxocr.ocr(img, model="indra_ocr_rtl")
    assert r.items
    assert _has_arabic(r.text)
    assert any("مرحبا" in it.text for it in r.items)


def test_rtl_hebrew_logical_order(indra_model_dir_env):
    from conftest import make_text_image

    logical = "שלום עולם"
    img = make_text_image([flip(logical)], font="arial.ttf", font_size=36)
    r = oxocr.ocr(img, model="indra_ocr_rtl")
    assert r.items
    assert _has_hebrew(r.text)
    assert any("שלום" in it.text for it in r.items)


def test_world_real_thai(indra_model_dir_env):
    r = oxocr.ocr(_load_bgr("world.png"), model="indra_ocr_world")
    assert r.items
    assert _has_thai(r.text)


def test_math_real_equations(indra_model_dir_env):
    # 整图直识：单行/扁宽公式样图（math_line.png）应整图读回一条 LaTeX
    r = oxocr.ocr(_load_bgr("math_line.png"), model="indra_ocr_math")
    assert r.items
    assert all("$" in it.text for it in r.items)
    assert all(it.score >= 0.9 for it in r.items)
    assert any("\\mathbb" in it.text or "\\rho" in it.text for it in r.items)


def test_math_tall_display_low_conf_not_rejected(indra_model_dir_env):
    # math.png 是高显示公式：64px 编码器竖直分辨率不足、读法置信低，
    # 但整图直识不拒收，仍输出低置信条目（score=conf 供调用方过滤）
    r = oxocr.ocr(_load_bgr("math.png"), model="indra_ocr_math")
    assert r.items
    assert all("$" in it.text for it in r.items)
    assert all(it.score < 0.9 for it in r.items)


def test_math_whole_image_single_box(indra_model_dir_env):
    # 整图直识只产一条条目（无行检测切分）
    r = oxocr.ocr(_load_bgr("math_line.png"), model="indra_ocr_math")
    assert len(r.items) == 1


def test_math_synthetic_literal_lenient(indra_model_dir_env, formula_image):
    # 字面 E=mc^2（未渲染上下标）：整图直识 conf 低，只断言不抛异常
    r = oxocr.ocr(formula_image, model="indra_ocr_math")
    assert isinstance(r.items, list)


def test_stability_repeated_inference(indra_model_dir_env, sample_image):
    eng = oxocr.OcrEngine(model="indra_ocr_latin")
    texts = []
    for _ in range(10):
        r = eng.ocr(sample_image)
        texts.append(r.text)
    assert all(t == texts[0] for t in texts)
    assert "Hello" in texts[0]
    eng.close()


def test_boxes_within_original_image(indra_model_dir_env, sample_image):
    r = oxocr.ocr(sample_image, model="indra_ocr_latin")
    h, w = sample_image.shape[:2]
    assert r.items
    for it in r.items:
        for x, y in it.box.points:
            assert -1 <= x <= w + 1
            assert -1 <= y <= h + 1


def test_small_image_pad_coords_in_origin(indra_model_dir_env, small_image):
    r = oxocr.ocr(small_image, model="indra_ocr_latin")
    h, w = small_image.shape[:2]
    for it in r.items:
        for x, y in it.box.points:
            assert -1 <= x <= w + 1 and -1 <= y <= h + 1


@pytest.mark.parametrize("k", [1, 2, 3])
def test_auto_orient_quarter_turn(indra_model_dir_env, k):
    # 真实页面转 k 个四分转：orient 应转正并读回文本，box 逆映射回送入图
    turned = np.ascontiguousarray(np.rot90(_load_bgr("letter.png"), k))
    r = oxocr.ocr(turned, model="indra_ocr_latin")
    assert r.items
    assert any("LETTER TO SHAREHOLDERS" in it.text for it in r.items)
    h, w = turned.shape[:2]
    for it in r.items:
        for x, y in it.box.points:
            assert -1 <= x <= w + 1 and -1 <= y <= h + 1


def test_deskew_tilted(indra_model_dir_env, sample_image):
    from PIL import Image

    pil = Image.fromarray(sample_image[:, :, ::-1]).rotate(4.0, fillcolor=(250, 250, 250))
    tilted = np.array(pil)[:, :, ::-1].copy()
    r = oxocr.ocr(tilted, model="indra_ocr_latin")
    assert r.items
    assert any("Hello" in it.text for it in r.items)


@pytest.mark.parametrize("model", TEXT_MODELS)
def test_detect_only_score(indra_model_dir_env, sample_image, model):
    r = oxocr.ocr(sample_image, model=model, config={"model_options": {"use_rec": False}})
    assert r.items
    for it in r.items:
        assert it.text == ""
        assert 0.0 < it.score <= 1.0


def test_math_detect_only_empty(indra_model_dir_env, sample_image):
    r = oxocr.ocr(sample_image, model="indra_ocr_math", config={"model_options": {"use_rec": False}})
    assert r.items == []


@pytest.mark.parametrize("model", TEXT_MODELS)
def test_recognize_only_no_error(indra_model_dir_env, sample_image, model):
    r = oxocr.ocr(sample_image, model=model, config={"model_options": {"use_det": False}})
    assert isinstance(r.items, list)


def test_math_recognize_only_whole_image(indra_model_dir_env):
    # math 无检测：use_det 开关不影响整图直识
    img = _load_bgr("math_line.png")
    r = oxocr.ocr(img, model="indra_ocr_math", config={"model_options": {"use_det": False}})
    assert r.items
    for it in r.items:
        assert "$" in it.text


def test_recognize_crop_line(indra_model_dir_env):
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (400, 64), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 40)
    except OSError:
        font = ImageFont.load_default()
    draw.text((10, 8), "Hello line crop", fill=(0, 0, 0), font=font)
    crop = np.array(img)[:, :, ::-1].copy()
    eng = oxocr.OcrEngine(model="indra_ocr_latin")
    it = eng.recognize_crop(crop)
    assert it is not None
    assert it.text.strip()
    eng.close()


def test_unknown_param_rejected(indra_model_dir_env, sample_image):
    with pytest.raises(UnknownParameterError):
        oxocr.ocr(sample_image, model="indra_ocr_latin", config={"model_options": {"params": {"bogus": 1}}})


def test_math_param_on_text_rejected(indra_model_dir_env, sample_image):
    with pytest.raises(UnknownParameterError):
        oxocr.ocr(
            sample_image, model="indra_ocr_latin",
            config={"model_options": {"params": {"math_min_conf": 0.8}}},
        )


@pytest.mark.parametrize(
    "params",
    [
        {"det_thresh": 0.9},
        {"rec_batch_num": 2},
        {"use_lm": False},
        {"deskew": False},
        {"auto_orient": False},
        {"adaptive": False},
        {"det_long_side": 768},
    ],
)
def test_params_work(indra_model_dir_env, sample_image, params):
    r = oxocr.ocr(sample_image, model="indra_ocr_latin", config={"model_options": {"params": params}})
    assert isinstance(r.items, list)


def test_missing_files_raises(indra_model_dir_env, sample_image, tmp_path):
    with pytest.raises(ModelFileMissingError) as e:
        oxocr.ocr(sample_image, model="indra_ocr_latin", model_dir=tmp_path)
    assert "indra-ocr" in str(e.value)


def test_missing_lm_orient_still_loads(indra_model_dir_env, sample_image, tmp_path):
    src = Path(indra_model_dir_env) / "indra-ocr"
    dst = tmp_path / "indra-ocr"
    dst.mkdir()
    for name in ("det.onnx", "rec.onnx", "meta.json"):
        shutil.copy(src / name, dst)
    # 故意不复制 lm.npz / orient.onnx：可选，缺文件不阻断
    r = oxocr.ocr(sample_image, model="indra_ocr_latin", model_dir=tmp_path)
    assert any("Hello" in it.text for it in r.items)


def test_missing_meta_raises(indra_model_dir_env, sample_image, tmp_path):
    src = Path(indra_model_dir_env) / "indra-ocr"
    dst = tmp_path / "indra-ocr"
    dst.mkdir()
    for name in ("det.onnx", "rec.onnx"):
        shutil.copy(src / name, dst)
    with pytest.raises(ModelFileMissingError) as e:
        oxocr.ocr(sample_image, model="indra_ocr_latin", model_dir=tmp_path)
    assert "meta" in str(e.value)


def test_subprocess_matches_inprocess(indra_model_dir_env, sample_image):
    a = oxocr.readtext(sample_image, model="indra_ocr_latin", backend="inprocess")
    b = oxocr.readtext(sample_image, model="indra_ocr_latin", backend="subprocess")
    assert a == b


def test_resident_second_call_zero_load(indra_model_dir_env, sample_image):
    eng = oxocr.OcrEngine(model="indra_ocr_latin")
    r1 = eng.ocr(sample_image)
    r2 = eng.ocr(sample_image)
    assert r1.elapse.load > 0
    assert r2.elapse.load == pytest.approx(0.0, abs=1e-4)
    assert r1.text == r2.text
    eng.close()


def test_resident_kept_after_close(indra_model_dir_env, sample_image):
    eng = oxocr.OcrEngine(model="indra_ocr_latin")
    eng.ocr(sample_image)
    eng.close()
    assert any(m.name == "indra_ocr_latin" for m in oxocr.loaded_models())


@pytest.mark.parametrize("fill", [0, 255])
def test_empty_image_returns_empty(indra_model_dir_env, fill):
    empty = np.full((100, 100, 3), fill, dtype=np.uint8)
    r = oxocr.ocr(empty, model="indra_ocr_latin")
    assert r.items == []


def test_math_postprocess_default_off(indra_model_dir_env):
    r = oxocr.ocr(_load_bgr("math_line.png"), model="indra_ocr_math")
    assert r.config.postprocess.enabled is False
    assert r.items
    assert "$" in r.text  # LaTeX 原样输出，未被纠错/归一化改写


def test_math_postprocess_explicit_on(indra_model_dir_env):
    r = oxocr.ocr(_load_bgr("math_line.png"), model="indra_ocr_math", postprocess=True)
    assert r.config.postprocess.enabled is True


def test_text_postprocess_default_on(indra_model_dir_env, sample_image):
    r = oxocr.ocr(sample_image, model="indra_ocr_latin")
    assert r.config.postprocess.enabled is True


def test_use_cls_ignored(indra_model_dir_env, sample_image):
    r = oxocr.ocr(sample_image, model="indra_ocr_latin", use_cls=True)
    assert r.text.strip()
    assert all(it.angle == 0.0 for it in r.items)
