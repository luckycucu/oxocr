"""Indra specs 单测：纯数据、白名单、可选 role、与离线权重目录一致。"""
from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import pytest
from oxocr.config import ModelOptions
from oxocr.errors import ModelFileMissingError
from oxocr.models.indra_ocr.specs import INDRA_SPECS
from oxocr.registry import registry

REPO = Path(__file__).resolve().parent.parent.parent
INDRA_DIR = REPO / "test" / "models" / "indra-ocr"

TEXT_NAMES = {"indra_ocr_latin", "indra_ocr_cjk", "indra_ocr_rtl", "indra_ocr_world"}
ALL_NAMES = TEXT_NAMES | {"indra_ocr_math"}
TEXT_PARAMS = {
    "det_long_side", "det_thresh", "adaptive", "rec_batch_num",
    "use_lm", "deskew", "auto_orient",
}
MATH_PARAMS = {"math_min_conf"}


def test_five_specs_family_impl():
    assert set(INDRA_SPECS) == ALL_NAMES
    for spec in INDRA_SPECS.values():
        assert spec.family == "indra_ocr"
        assert spec.impl == "oxocr.models.indra_ocr.pipeline:IndraModel"
        assert spec.license == "AI PUBS OPEN RAIL-M (modified)"
        assert all(f.url is None for f in spec.files)  # 自动下载暂缓


def test_capabilities():
    for name in TEXT_NAMES:
        caps = INDRA_SPECS[name].capabilities
        assert caps.detection and caps.recognition
        assert not caps.own_postprocess and not caps.angle_classification
    math = INDRA_SPECS["indra_ocr_math"].capabilities
    assert not math.detection and math.recognition and math.own_postprocess


def test_params_whitelist():
    for name in TEXT_NAMES:
        assert set(INDRA_SPECS[name].defaults) == TEXT_PARAMS
    assert set(INDRA_SPECS["indra_ocr_math"].defaults) == MATH_PARAMS
    assert INDRA_SPECS["indra_ocr_latin"].defaults["det_long_side"] == 1024
    assert INDRA_SPECS["indra_ocr_math"].defaults["math_min_conf"] == 0.0


def test_roles_and_optionality():
    for name in TEXT_NAMES:
        roles = {f.role for f in INDRA_SPECS[name].files}
        assert {"det", "rec", "meta", "orient"} <= roles
        assert ("lm" in roles) == (name != "indra_ocr_rtl")  # rtl 无 lm
    math_roles = {f.role for f in INDRA_SPECS["indra_ocr_math"].files}
    assert math_roles == {"enc", "dec_init", "dec_step", "math_meta"}


def test_resolve_files_optional_lm_orient(tmp_path):
    spec = INDRA_SPECS["indra_ocr_latin"]
    for f in spec.files:
        if f.role not in ("lm", "orient"):
            p = tmp_path / f.filename
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"x")
    files = registry.resolve_files(spec, tmp_path, ModelOptions())
    assert set(files) == {"det", "rec", "meta"}
    installed, _ = registry._installed_paths(spec, tmp_path)
    assert installed


def test_resolve_files_missing_meta_raises(tmp_path):
    spec = INDRA_SPECS["indra_ocr_math"]
    for f in spec.files:
        if f.role != "math_meta":
            p = tmp_path / f.filename
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"x")
    with pytest.raises(ModelFileMissingError) as e:
        registry.resolve_files(spec, tmp_path, ModelOptions())
    assert "math_meta" in e.value.missing


def test_specs_import_lightweight():
    """specs 纯数据：导入不得触发 onnxruntime 与推理实现。"""
    code = (
        "import sys; from oxocr.models.indra_ocr.specs import INDRA_SPECS;"
        "assert len(INDRA_SPECS) == 5;"
        "assert 'onnxruntime' not in sys.modules;"
        "assert 'oxocr.models.indra_ocr.pipeline' not in sys.modules"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


@pytest.mark.skipif(not INDRA_DIR.is_dir(), reason="离线权重未放置")
def test_files_match_offline_weights():
    for spec in INDRA_SPECS.values():
        for f in spec.files:
            p = REPO / "test" / "models" / f.filename
            assert p.is_file(), f"{spec.name}: 缺 {f.filename}"
            assert p.stat().st_size == f.size, f"{spec.name}: {f.filename} size 不符"
            assert hashlib.sha256(p.read_bytes()).hexdigest() == f.sha256, (
                f"{spec.name}: {f.filename} sha256 不符"
            )
