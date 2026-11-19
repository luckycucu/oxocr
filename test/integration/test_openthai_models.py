"""openthai-ocr 离线集成测试（全程不联网）。

固化合成泰文图 test/data/openthai/*.png（tahoma 渲染，冒烟已逐字验证）做从宽断言；
页级方向/deskew/过滤 meta 经 load_model + infer 直调断言（核心门面不传递
ModelOutput.meta，与 indra 先例一致）。
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import oxocr
import pytest
from oxocr.errors import ModelFileMissingError, UnknownParameterError
from oxocr.models.base import InferRequest
from oxocr.runtime import imageops

pytestmark = [pytest.mark.model, pytest.mark.integration]

MODEL = "openthai2.0_ocr"
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = REPO_ROOT / "test" / "data" / "openthai"


@pytest.fixture(autouse=True)
def _clean_pool():
    oxocr.unload_all()
    yield
    oxocr.unload_all()


def _load_png(name: str) -> np.ndarray:
    from PIL import Image

    path = DATA_DIR / name
    if not path.is_file():
        pytest.skip(f"测试图 {path} 缺失，跳过")
    return np.array(Image.open(path).convert("RGB"))[:, :, ::-1].copy()


@pytest.fixture(scope="session")
def thai_lines() -> np.ndarray:
    return _load_png("thai_lines.png")


@pytest.fixture(scope="session")
def thai_page() -> np.ndarray:
    return _load_png("thai_page.png")


def _has_thai(text: str) -> bool:
    return any("\u0e00" <= c <= "\u0e7f" for c in text)


def _loaded_model(**options):
    m = oxocr.load_model(MODEL, num_threads=2, **options)
    m.ensure_loaded()
    return m


def test_list_models_reports_family(openthai_model_dir_env):
    infos = oxocr.list_models(family="openthai_ocr")
    assert {m.name for m in infos} == {MODEL}
    assert all(m.installed for m in infos)
    assert all(m.languages == ("th", "en") for m in infos)
    assert all(m.downloadable for m in infos)


def test_thai_lines_read(openthai_model_dir_env, thai_lines):
    r = oxocr.ocr(thai_lines, model=MODEL)
    assert len(r.items) == 2
    assert _has_thai(r.text)
    assert "Hello Thai OCR" in r.text
    for it in r.items:
        assert 0.8 < it.score <= 1.0
        assert it.meta.get("row") in (0, 1)
    assert r.elapse.det > 0
    assert r.elapse.rec > 0


def test_thai_page_read(openthai_model_dir_env, thai_page):
    r = oxocr.ocr(thai_page, model=MODEL)
    assert len(r.items) == 6
    assert all(_has_thai(it.text) for it in r.items)
    assert [it.meta["row"] for it in r.items] == list(range(6))


def test_boxes_within_original_image(openthai_model_dir_env, thai_lines):
    r = oxocr.ocr(thai_lines, model=MODEL)
    h, w = thai_lines.shape[:2]
    assert r.items
    for it in r.items:
        for x, y in it.box.points:
            assert -1 <= x <= w + 1
            assert -1 <= y <= h + 1


def test_small_line_crop_fallback(openthai_model_dir_env, thai_lines):
    # 小图（h<=200, w>=1.5h）det 无框时整图当一行；坐标仍在原图范围
    crop = thai_lines[30:78, 28:535]
    r = oxocr.ocr(crop, model=MODEL)
    h, w = crop.shape[:2]
    assert r.items
    assert _has_thai(r.text)
    for it in r.items:
        for x, y in it.box.points:
            assert -1 <= x <= w + 1 and -1 <= y <= h + 1


@pytest.mark.parametrize(("k", "rotation"), [(1, 90), (-1, 270), (2, 180)])
def test_page_orientation(openthai_model_dir_env, thai_page, k, rotation):
    m = _loaded_model()
    try:
        upright = m.infer(thai_page, InferRequest())
        rotated = np.ascontiguousarray(np.rot90(thai_page, k))
        out = m.infer(rotated, InferRequest())
        assert out.meta["rotation"] == rotation
        assert [it.text for it in out.items] == [it.text for it in upright.items]
        h, w = rotated.shape[:2]
        for it in out.items:
            for x, y in it.box.points:
                assert -1 <= x <= w + 1 and -1 <= y <= h + 1
    finally:
        m.unload()


def test_orientation_disabled(openthai_model_dir_env, thai_page):
    m = _loaded_model()
    try:
        rotated = np.ascontiguousarray(np.rot90(thai_page, 2))
        out = m.infer(rotated, InferRequest(params={"orientation": False}))
        assert out.meta["rotation"] == 0
    finally:
        m.unload()


def test_deskew(openthai_model_dir_env, thai_page):
    m = _loaded_model()
    try:
        tilted = imageops.rotate(thai_page, 6, fill=(255, 255, 255), interpolation="bicubic", expand=True)
        out = m.infer(tilted, InferRequest())
        assert len(out.items) == 6
        assert abs(out.meta["text_angle"]) > 4  # 报原页倾角（残差+矫正量）
        upright = m.infer(thai_page, InferRequest())
        assert [it.text for it in out.items] == [it.text for it in upright.items]
    finally:
        m.unload()


def test_page_meta_and_dropped(openthai_model_dir_env, thai_page):
    m = _loaded_model()
    try:
        out = m.infer(thai_page, InferRequest())
        assert out.meta["rotation"] == 0
        assert isinstance(out.meta["text_angle"], float)
        assert isinstance(out.meta["dropped"], list)
    finally:
        m.unload()


def test_text_score_tightens_drop_below(openthai_model_dir_env, thai_page):
    m = _loaded_model()
    try:
        # 行置信度 ~0.96：text_score=0.99 收紧过滤 → 全弃
        out = m.infer(thai_page, InferRequest(text_score=0.99))
        assert out.items == []
        assert out.meta["dropped"]
        assert all(d["reason"] == "low_confidence" for d in out.meta["dropped"])
        # 放松 drop_below 后 text_score=0.5 恢复
        out = m.infer(thai_page, InferRequest(text_score=0.5, params={"drop_below": 0.0}))
        assert len(out.items) == 6
    finally:
        m.unload()


def test_detect_only_score(openthai_model_dir_env, thai_page):
    r = oxocr.ocr(thai_page, model=MODEL, config={"model_options": {"use_rec": False}})
    assert r.items
    for it in r.items:
        assert it.text == ""
        assert 0.0 < it.score <= 1.0


def test_recognize_only_no_error(openthai_model_dir_env, thai_lines):
    r = oxocr.ocr(thai_lines, model=MODEL, config={"model_options": {"use_det": False}})
    assert isinstance(r.items, list)


def test_recognize_crop_line(openthai_model_dir_env, thai_lines):
    crop = thai_lines[30:78, 28:535]
    eng = oxocr.OcrEngine(model=MODEL)
    it = eng.recognize_crop(crop)
    assert it is not None
    assert _has_thai(it.text)
    eng.close()


def test_unknown_param_rejected(openthai_model_dir_env, thai_lines):
    with pytest.raises(UnknownParameterError):
        oxocr.ocr(thai_lines, model=MODEL, config={"model_options": {"params": {"bogus": 1}}})


def test_duguang_param_rejected(openthai_model_dir_env, thai_lines):
    with pytest.raises(UnknownParameterError):
        oxocr.ocr(thai_lines, model=MODEL, config={"model_options": {"params": {"det_image_size": 512}}})


@pytest.mark.parametrize(
    "params",
    [
        {"det_thresh": 0.9},
        {"rec_batch_num": 2},
        {"rec_color": "bgr"},
        {"drop_below": 0.0},
        {"split_gap": 0},
        {"orientation": False},
    ],
)
def test_params_work(openthai_model_dir_env, thai_lines, params):
    r = oxocr.ocr(thai_lines, model=MODEL, config={"model_options": {"params": params}})
    assert isinstance(r.items, list)


def test_invalid_rec_color_rejected(openthai_model_dir_env, thai_lines):
    from oxocr.errors import InferenceError

    with pytest.raises(InferenceError):
        oxocr.ocr(thai_lines, model=MODEL, config={"model_options": {"params": {"rec_color": "bogus"}}})


def test_missing_files_raises(openthai_model_dir_env, thai_lines, tmp_path):
    with pytest.raises(ModelFileMissingError) as e:
        oxocr.ocr(thai_lines, model=MODEL, model_dir=tmp_path)
    assert "openthai2.0-ocr" in str(e.value)


def test_missing_dict_alone_raises(openthai_model_dir_env, thai_lines, tmp_path):
    src = Path(openthai_model_dir_env) / "openthai2.0-ocr"
    dst = tmp_path / "openthai2.0-ocr"
    dst.mkdir()
    shutil.copy(src / "det.onnx", dst)
    shutil.copy(src / "rec.onnx", dst)
    # 故意不复制 dict：恒必需，即便 use_rec=False 也报错
    with pytest.raises(ModelFileMissingError) as e:
        oxocr.ocr(thai_lines, model=MODEL, model_dir=tmp_path)
    assert "dict" in str(e.value)


def test_subprocess_matches_inprocess(openthai_model_dir_env, thai_lines):
    a = oxocr.readtext(thai_lines, model=MODEL, backend="inprocess")
    b = oxocr.readtext(thai_lines, model=MODEL, backend="subprocess")
    assert a == b


def test_resident_second_call_zero_load(openthai_model_dir_env, thai_lines):
    eng = oxocr.OcrEngine(model=MODEL)
    r1 = eng.ocr(thai_lines)
    r2 = eng.ocr(thai_lines)
    assert r1.elapse.load > 0
    assert r2.elapse.load == pytest.approx(0.0, abs=1e-4)
    assert r1.text == r2.text
    eng.close()


def test_resident_kept_after_close(openthai_model_dir_env, thai_lines):
    eng = oxocr.OcrEngine(model=MODEL)
    eng.ocr(thai_lines)
    eng.close()
    assert any(m.name == MODEL for m in oxocr.loaded_models())


@pytest.mark.parametrize("fill", [0, 255])
def test_empty_image_returns_empty(openthai_model_dir_env, fill):
    empty = np.full((100, 100, 3), fill, dtype=np.uint8)
    r = oxocr.ocr(empty, model=MODEL)
    assert r.items == []


def test_postprocess_default_on(openthai_model_dir_env, thai_lines):
    r = oxocr.ocr(thai_lines, model=MODEL)
    assert r.config.postprocess.enabled is True


def test_use_cls_ignored(openthai_model_dir_env, thai_lines):
    r = oxocr.ocr(thai_lines, model=MODEL, use_cls=True)
    assert r.text.strip()
    assert all(it.angle == 0.0 for it in r.items)