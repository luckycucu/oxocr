"""离线真实模型集成测试（test/models，全程不联网）。"""
from __future__ import annotations

import numpy as np
import oxocr
import pytest
from oxocr.errors import UnknownParameterError
from oxocr.models.rapid_ocr.specs import CLS_MODELS, SPECS

pytestmark = [pytest.mark.model, pytest.mark.integration]


@pytest.fixture(autouse=True)
def _clean_pool():
    oxocr.unload_all()
    yield
    oxocr.unload_all()


def test_readtext_nonempty(model_dir_env, sample_image):
    text = oxocr.readtext(sample_image)
    assert text.strip()
    assert "Hello" in text or "OCR" in text


def test_boxes_within_original_image(model_dir_env, sample_image):
    r = oxocr.ocr(sample_image)
    h, w = sample_image.shape[:2]
    for it in r.items:
        for x, y in it.box.points:
            assert -1 <= x <= w + 1
            assert -1 <= y <= h + 1


def test_as_text_modes_all_work(model_dir_env, sample_image):
    r = oxocr.ocr(sample_image, mode="para")
    for mode in ("raw", "line", "para", "layout"):
        assert isinstance(r.as_text(mode), str)


@pytest.mark.parametrize("name", ["ppocrv6_tiny", "ppocrv6_small", "ppocrv6_medium"])
def test_each_v6_model_detects(model_dir_env, sample_image, name):
    text = oxocr.readtext(sample_image, model=name)
    assert text.strip()


def test_subprocess_matches_inprocess(model_dir_env, sample_image):
    a = oxocr.readtext(sample_image, model="ppocrv6_tiny", backend="inprocess")
    b = oxocr.readtext(sample_image, model="ppocrv6_tiny", backend="subprocess")
    assert a == b


def test_dark_image_recognized(model_dir_env, dark_image):
    text = oxocr.readtext(dark_image, model="ppocrv6_tiny")
    assert "Dark" in text or "theme" in text


def test_small_image_pad_coords_in_origin(model_dir_env, small_image):
    r = oxocr.ocr(small_image, model="ppocrv6_tiny")
    h, w = small_image.shape[:2]
    assert r.items
    for it in r.items:
        for x, y in it.box.points:
            assert -1 <= x <= w + 1 and -1 <= y <= h + 1


def test_resident_second_call_zero_load(model_dir_env, sample_image):
    eng = oxocr.OcrEngine(model="ppocrv6_tiny")
    r1 = eng.ocr(sample_image)
    r2 = eng.ocr(sample_image)
    assert r1.elapse.load > 0
    assert r2.elapse.load == pytest.approx(0.0, abs=1e-4)
    eng.close()  # 常驻：close 只归还句柄，不卸载


def test_resident_kept_after_engine_close(model_dir_env, sample_image):
    eng = oxocr.OcrEngine(model="ppocrv6_tiny")
    eng.ocr(sample_image)
    eng.close()
    assert any(m.name == "ppocrv6_tiny" for m in oxocr.loaded_models())


def test_non_resident_not_in_pool(model_dir_env, sample_image):
    eng = oxocr.OcrEngine(model="ppocrv6_tiny", resident=False)
    eng.ocr(sample_image)
    eng.close()
    assert not any(m.name == "ppocrv6_tiny" for m in oxocr.loaded_models())


def test_config_dict_override(model_dir_env, sample_image):
    cfg = {"model": "ppocrv6_tiny", "infer": {"text_score": 0.95},
           "model_options": {"params": {"det_limit_side_len": 640}}}
    r = oxocr.ocr(sample_image, config=cfg)
    assert r.config.model_options.params["det_limit_side_len"] == 640


def test_output_json(model_dir_env, sample_image):
    r = oxocr.ocr(sample_image, output="json", model="ppocrv6_tiny")
    import json

    data = json.loads(r.as_json())
    assert data["model"] == "ppocrv6_tiny"
    assert "blocks" in data


def test_missing_model_files_offline(tmp_path, sample_image):
    from oxocr.errors import ModelFileMissingError

    with pytest.raises(ModelFileMissingError):
        oxocr.readtext(sample_image, model="ppocrv5_ch_server", model_dir=tmp_path)


def test_crop_and_recognize(model_dir_env, sample_image):
    r = oxocr.ocr(sample_image, model="ppocrv6_small")
    crop = r.crop(0)
    assert crop.ndim == 3 and crop.shape[2] == 3


@pytest.mark.parametrize(
    "name",
    [
        "ppocrv3_latin_mobile",
        "ppocrv3_en_mobile",
        "ppocrv4_ch_mobile",
        "ppocrv4_ch_server",
        "ppocrv4_en_mobile",
        "ppocrv5_ch_mobile",
        "ppocrv5_ch_server",
        "ppocrv5_en_mobile",
    ],
)
def test_legacy_versions_read_latin(model_dir_env, sample_image, name):
    assert "Hello" in oxocr.readtext(sample_image, model=name)


@pytest.mark.parametrize("name", sorted(SPECS))
def test_every_spec_runs_offline(model_dir_env, sample_image, name):
    """逐条清单跑通推理：捕获字典长度/输入形状/输出维度与权重不匹配。"""
    r = oxocr.ocr(sample_image, model=name)
    h, w = sample_image.shape[:2]
    for it in r.items:
        for x, y in it.box.points:
            assert -1 <= x <= w + 1 and -1 <= y <= h + 1


@pytest.mark.parametrize(
    "cls_name", ["v2_0_cls", "textline_ori_mobile", "textline_ori_server"]
)
def test_cls_variants_fix_upside_down(model_dir_env, sample_image, cls_name):
    upside = np.ascontiguousarray(np.rot90(sample_image, 2))
    r = oxocr.ocr(
        upside,
        model="ppocrv6_small",
        use_cls=True,
        config={"model_options": {"params": {"cls_model": cls_name}}},
    )
    assert "Hello" in r.text
    assert any(it.angle == 180.0 for it in r.items)


def test_v5_defaults_to_textline_cls(model_dir_env, sample_image):
    """v5 默认 cls 是 80×160 的 textline_ori，必须能直接跑通而非形状报错。"""
    upside = np.ascontiguousarray(np.rot90(sample_image, 2))
    r = oxocr.ocr(upside, model="ppocrv5_ch_mobile", use_cls=True)
    assert "Hello" in r.text
    assert any(it.angle == 180.0 for it in r.items)


def test_cls_override_reaches_loaded_model(model_dir_env):
    model = oxocr.load_model(
        "ppocrv6_small", use_cls=True, params={"cls_model": "textline_ori_server"}
    )
    try:
        assert model.classifier.cls_image_shape == [3, 80, 160]
        assert model.files["cls"].name == CLS_MODELS["textline_ori_server"].filename.split("/")[-1]
    finally:
        model.unload()


def test_unknown_cls_model_rejected(model_dir_env, sample_image):
    with pytest.raises(UnknownParameterError):
        oxocr.readtext(
            sample_image,
            model="ppocrv6_small",
            use_cls=True,
            config={"model_options": {"params": {"cls_model": "not-a-cls"}}},
        )


def test_subprocess_v5_with_textline_cls(model_dir_env, sample_image):
    """v5 默认的 80×160 cls 与 cls_model 参数都必须能穿过子进程序列化边界。"""
    upside = np.ascontiguousarray(np.rot90(sample_image, 2))
    cfg = {"model": "ppocrv5_ch_mobile", "model_options": {"use_cls": True}}
    a = oxocr.readtext(upside, config=cfg, backend="inprocess")
    b = oxocr.readtext(upside, config=cfg, backend="subprocess")
    assert a == b
    assert "Hello" in b
