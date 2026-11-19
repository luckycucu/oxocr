"""读光 OCR 离线集成测试（duguang-ocr，全程不联网）。

合成文本图（PIL 渲染）与训练分布有差距，因此不断言逐字内容，
只断言结构契约：非空输出、box 语义、门控行为与执行后端一致性。
"""
from __future__ import annotations

import numpy as np
import oxocr
import pytest
from oxocr.errors import ModelFileMissingError, UnknownParameterError

pytestmark = [pytest.mark.model, pytest.mark.integration]

ALL_DUGUANG = ["duguang_ocr_small", "duguang_ocr_large", "duguang_ocr_max"]
DB_MODELS = ["duguang_ocr_small", "duguang_ocr_large"]


@pytest.fixture(autouse=True)
def _clean_pool():
    oxocr.unload_all()
    yield
    oxocr.unload_all()


def test_list_models_reports_family(duguang_model_dir_env):
    infos = oxocr.list_models(family="duguang_ocr")
    assert {m.name for m in infos} == set(ALL_DUGUANG)
    assert all(m.installed for m in infos)
    assert all(m.languages == ("zh", "en") for m in infos)


@pytest.mark.parametrize("name", ALL_DUGUANG)
def test_readtext_returns_text(duguang_model_dir_env, sample_image, name):
    r = oxocr.ocr(sample_image, model=name)
    assert r.items, f"{name} 未识别出条目"
    assert r.text.strip()
    assert "Hello" in r.text or "OCR" in r.text
    for it in r.items:
        assert 0.0 < it.score <= 1.0
    assert r.elapse.det > 0
    assert r.elapse.rec > 0


@pytest.mark.parametrize("name", ALL_DUGUANG)
def test_boxes_within_original_image(duguang_model_dir_env, sample_image, name):
    r = oxocr.ocr(sample_image, model=name)
    h, w = sample_image.shape[:2]
    assert r.items
    for it in r.items:
        for x, y in it.box.points:
            assert -1 <= x <= w + 1
            assert -1 <= y <= h + 1


@pytest.mark.parametrize("name", ALL_DUGUANG)
def test_small_image_pad_coords_in_origin(duguang_model_dir_env, small_image, name):
    r = oxocr.ocr(small_image, model=name)
    h, w = small_image.shape[:2]
    for it in r.items:
        for x, y in it.box.points:
            assert -1 <= x <= w + 1 and -1 <= y <= h + 1


@pytest.mark.parametrize("name", DB_MODELS)
def test_detect_only_db_score(duguang_model_dir_env, sample_image, name):
    r = oxocr.ocr(sample_image, model=name, config={"model_options": {"use_rec": False}})
    assert r.items
    for it in r.items:
        assert it.text == ""
        assert 0.0 < it.score <= 1.0
        assert "det_score" not in it.meta


def test_detect_only_seglink_score(duguang_model_dir_env, sample_image):
    r = oxocr.ocr(sample_image, model="duguang_ocr_max", config={"model_options": {"use_rec": False}})
    assert r.items
    for it in r.items:
        assert it.text == ""
        assert it.score == 1.0


@pytest.mark.parametrize("name", ALL_DUGUANG)
def test_recognize_only_no_error(duguang_model_dir_env, sample_image, name):
    r = oxocr.ocr(sample_image, model=name, config={"model_options": {"use_det": False}})
    assert isinstance(r.items, list)


def test_unknown_param_rejected(duguang_model_dir_env, sample_image):
    with pytest.raises(UnknownParameterError):
        oxocr.ocr(sample_image, model="duguang_ocr_small",
                  config={"model_options": {"params": {"bogus": 1}}})


def test_det_thresh_param_works(duguang_model_dir_env, sample_image):
    r = oxocr.ocr(sample_image, model="duguang_ocr_large",
                  config={"model_options": {"params": {"det_thresh": 0.9}}})
    assert isinstance(r.items, list)


def test_rec_batch_num_param_works(duguang_model_dir_env, sample_image):
    r = oxocr.ocr(sample_image, model="duguang_ocr_large",
                  config={"model_options": {"params": {"rec_batch_num": 2}}})
    assert r.text.strip()


def test_unknown_det_image_size_rejected(duguang_model_dir_env, sample_image):
    with pytest.raises(UnknownParameterError) as e:
        oxocr.ocr(sample_image, model="duguang_ocr_small",
                  config={"model_options": {"params": {"det_image_size": 512}}})
    assert "1024" in str(e.value)


def test_det_limit_side_len_not_accepted(duguang_model_dir_env, sample_image):
    with pytest.raises(UnknownParameterError):
        oxocr.ocr(sample_image, model="duguang_ocr_small", det_limit_side_len=640)


def test_missing_vocab_raises(duguang_model_dir_env, sample_image, tmp_path):
    with pytest.raises(ModelFileMissingError) as e:
        oxocr.ocr(sample_image, model="duguang_ocr_small", model_dir=tmp_path)
    assert "vocab" in str(e.value)


def test_missing_det_raises(duguang_model_dir_env, sample_image, tmp_path):
    with pytest.raises(ModelFileMissingError) as e:
        oxocr.ocr(sample_image, model="duguang_ocr_large", model_dir=tmp_path)
    assert "det" in str(e.value)


def test_subprocess_matches_inprocess(duguang_model_dir_env, sample_image):
    a = oxocr.readtext(sample_image, model="duguang_ocr_large", backend="inprocess")
    b = oxocr.readtext(sample_image, model="duguang_ocr_large", backend="subprocess")
    assert a == b


def test_resident_second_call_zero_load(duguang_model_dir_env, sample_image):
    eng = oxocr.OcrEngine(model="duguang_ocr_large")
    r1 = eng.ocr(sample_image)
    r2 = eng.ocr(sample_image)
    assert r1.elapse.load > 0
    assert r2.elapse.load == pytest.approx(0.0, abs=1e-4)
    assert r1.text == r2.text
    eng.close()


def test_resident_kept_after_close(duguang_model_dir_env, sample_image):
    eng = oxocr.OcrEngine(model="duguang_ocr_large")
    eng.ocr(sample_image)
    eng.close()
    assert any(m.name == "duguang_ocr_large" for m in oxocr.loaded_models())


@pytest.mark.parametrize("name", ALL_DUGUANG)
def test_empty_image_returns_empty(duguang_model_dir_env, name):
    empty = np.zeros((100, 100, 3), dtype=np.uint8)
    r = oxocr.ocr(empty, model=name)
    assert r.items == []


def test_postprocess_default_on(duguang_model_dir_env, sample_image):
    r = oxocr.ocr(sample_image, model="duguang_ocr_large")
    assert r.config.postprocess.enabled is True


def test_use_cls_ignored(duguang_model_dir_env, sample_image):
    r = oxocr.ocr(sample_image, model="duguang_ocr_large", use_cls=True)
    assert r.text.strip()
    assert all(it.angle == 0.0 for it in r.items)
