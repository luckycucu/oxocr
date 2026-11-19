"""anaksor-ocr 离线集成测试（全程不联网）。

合成文本图（PIL 渲染）与训练分布有差距，不断言逐字内容，只断言结构契约；
真实高棉语照片 test/data/anaksor/sign.jpg 做从宽特征断言（含高棉字符 + PHNOM PENH）。
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import oxocr
import pytest
from oxocr.errors import ModelFileMissingError, UnknownParameterError

pytestmark = [pytest.mark.model, pytest.mark.integration]

MODEL = "anaksor_ocr"
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SIGN_JPG = REPO_ROOT / "test" / "data" / "anaksor" / "sign.jpg"


@pytest.fixture(autouse=True)
def _clean_pool():
    oxocr.unload_all()
    yield
    oxocr.unload_all()


@pytest.fixture(scope="session")
def khmer_sign_image() -> np.ndarray:
    if not SIGN_JPG.is_file():
        pytest.skip("真实高棉语照片 test/data/anaksor/sign.jpg 缺失，跳过")
    from PIL import Image

    return np.array(Image.open(SIGN_JPG).convert("RGB"))[:, :, ::-1].copy()


def _has_khmer(text: str) -> bool:
    return any("ក" <= c <= "៿" for c in text)


def test_list_models_reports_family(anaksor_model_dir_env):
    infos = oxocr.list_models(family="anaksor_ocr")
    assert {m.name for m in infos} == {MODEL}
    assert all(m.installed for m in infos)
    assert all(m.languages == ("km", "en") for m in infos)
    assert all(m.downloadable for m in infos)


def test_readtext_synthetic_latin(anaksor_model_dir_env, sample_image):
    r = oxocr.ocr(sample_image, model=MODEL)
    assert r.items
    assert r.text.strip()
    assert any(c.isascii() and c.isalpha() for c in r.text)
    for it in r.items:
        assert 0.0 < it.score <= 1.0
        assert it.meta.get("rotation") in ("up", "cw", "ccw", "180")
    assert r.elapse.det > 0
    assert r.elapse.rec > 0


def test_real_khmer_photo(anaksor_model_dir_env, khmer_sign_image):
    r = oxocr.ocr(khmer_sign_image, model=MODEL)
    assert r.items
    assert _has_khmer(r.text)
    assert "PHNOM PENH" in r.text
    assert any(_has_khmer(it.text) for it in r.items)


def test_boxes_within_original_image(anaksor_model_dir_env, sample_image):
    r = oxocr.ocr(sample_image, model=MODEL)
    h, w = sample_image.shape[:2]
    assert r.items
    for it in r.items:
        for x, y in it.box.points:
            assert -1 <= x <= w + 1
            assert -1 <= y <= h + 1


def test_small_image_pad_coords_in_origin(anaksor_model_dir_env, small_image):
    r = oxocr.ocr(small_image, model=MODEL)
    h, w = small_image.shape[:2]
    for it in r.items:
        for x, y in it.box.points:
            assert -1 <= x <= w + 1 and -1 <= y <= h + 1


def test_detect_only_score(anaksor_model_dir_env, sample_image):
    r = oxocr.ocr(sample_image, model=MODEL, config={"model_options": {"use_rec": False}})
    assert r.items
    for it in r.items:
        assert it.text == ""
        assert 0.0 < it.score <= 1.0
        assert "det_score" not in it.meta


def test_recognize_only_no_error(anaksor_model_dir_env, sample_image):
    r = oxocr.ocr(sample_image, model=MODEL, config={"model_options": {"use_det": False}})
    assert isinstance(r.items, list)


def test_recognize_crop_line(anaksor_model_dir_env):
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (400, 64), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 40)
    except OSError:
        font = ImageFont.load_default()
    draw.text((10, 8), "Hello line crop", fill=(0, 0, 0), font=font)
    crop = np.array(img)[:, :, ::-1].copy()
    eng = oxocr.OcrEngine(model=MODEL)
    it = eng.recognize_crop(crop)
    assert it is not None
    assert it.text.strip()
    eng.close()


def test_unknown_param_rejected(anaksor_model_dir_env, sample_image):
    with pytest.raises(UnknownParameterError):
        oxocr.ocr(sample_image, model=MODEL, config={"model_options": {"params": {"bogus": 1}}})


def test_duguang_param_rejected(anaksor_model_dir_env, sample_image):
    with pytest.raises(UnknownParameterError):
        oxocr.ocr(
            sample_image, model=MODEL, config={"model_options": {"params": {"det_image_size": 512}}}
        )


@pytest.mark.parametrize(
    "params",
    [{"det_thresh": 0.9}, {"rec_batch_num": 2}, {"try_vertical": False}, {"try_180": True}],
)
def test_params_work(anaksor_model_dir_env, sample_image, params):
    r = oxocr.ocr(sample_image, model=MODEL, config={"model_options": {"params": params}})
    assert isinstance(r.items, list)


def test_missing_files_raises(anaksor_model_dir_env, sample_image, tmp_path):
    with pytest.raises(ModelFileMissingError) as e:
        oxocr.ocr(sample_image, model=MODEL, model_dir=tmp_path)
    assert "anaksor-ocr" in str(e.value)


def test_missing_dict_alone_raises(anaksor_model_dir_env, sample_image, tmp_path):
    src = Path(anaksor_model_dir_env) / "anaksor-ocr"
    dst = tmp_path / "anaksor-ocr"
    dst.mkdir()
    shutil.copy(src / "khmer_det_repvit_dbpp_latest.onnx", dst)
    shutil.copy(src / "khmer_rec_svtrv2_latest.onnx", dst)
    # 故意不复制 dict：恒必需，即便 use_rec=False 也报错
    with pytest.raises(ModelFileMissingError) as e:
        oxocr.ocr(sample_image, model=MODEL, model_dir=tmp_path)
    assert "dict" in str(e.value)


def test_subprocess_matches_inprocess(anaksor_model_dir_env, sample_image):
    a = oxocr.readtext(sample_image, model=MODEL, backend="inprocess")
    b = oxocr.readtext(sample_image, model=MODEL, backend="subprocess")
    assert a == b


def test_resident_second_call_zero_load(anaksor_model_dir_env, sample_image):
    eng = oxocr.OcrEngine(model=MODEL)
    r1 = eng.ocr(sample_image)
    r2 = eng.ocr(sample_image)
    assert r1.elapse.load > 0
    assert r2.elapse.load == pytest.approx(0.0, abs=1e-4)
    assert r1.text == r2.text
    eng.close()


def test_resident_kept_after_close(anaksor_model_dir_env, sample_image):
    eng = oxocr.OcrEngine(model=MODEL)
    eng.ocr(sample_image)
    eng.close()
    assert any(m.name == MODEL for m in oxocr.loaded_models())


@pytest.mark.parametrize("fill", [0, 255])
def test_empty_image_returns_empty(anaksor_model_dir_env, fill):
    empty = np.full((100, 100, 3), fill, dtype=np.uint8)
    r = oxocr.ocr(empty, model=MODEL)
    assert r.items == []


def test_postprocess_default_on(anaksor_model_dir_env, sample_image):
    r = oxocr.ocr(sample_image, model=MODEL)
    assert r.config.postprocess.enabled is True


def test_use_cls_ignored(anaksor_model_dir_env, sample_image):
    r = oxocr.ocr(sample_image, model=MODEL, use_cls=True)
    assert r.text.strip()
    assert all(it.angle == 0.0 for it in r.items)
