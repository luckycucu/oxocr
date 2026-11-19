"""模型下载：源优先级/回退、force、离线、幂等、progress 与 CLI（mock hub，无网络）。"""
from __future__ import annotations

from pathlib import Path

import oxocr
import pytest
from oxocr.cli import main
from oxocr.errors import ModelDownloadError, UnknownParameterError
from oxocr.models.base import ModelFile, ModelSpec
from oxocr.registry import download, registry


@pytest.fixture(autouse=True)
def online(monkeypatch):
    # conftest 全局 OXOCR_OFFLINE=1，下载用例默认解除
    monkeypatch.setenv("OXOCR_OFFLINE", "0")
    monkeypatch.delenv("OXOCR_DOWNLOAD_SOURCE", raising=False)


def _fake(calls, fail_on=()):
    def downloader(repo, filename, directory, force):
        calls.append({"repo": repo, "filename": filename, "force": force})
        if filename in fail_on:
            raise RuntimeError(f"boom: {filename}")
        path = Path(directory) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fake-weight")

    return downloader


def _fail(message):
    def downloader(*args):
        raise RuntimeError(message)

    return downloader


@pytest.fixture
def hub(monkeypatch):
    calls = {"modelscope": [], "huggingface": []}
    monkeypatch.setitem(download._DOWNLOADERS, "modelscope", _fake(calls["modelscope"]))
    monkeypatch.setitem(download._DOWNLOADERS, "huggingface", _fake(calls["huggingface"]))
    return calls


def test_auto_modelscope_first(hub, tmp_path):
    paths = oxocr.download_model("ppocrv6_tiny", model_dir=tmp_path)
    assert set(paths) == {"det", "rec", "cls"}
    assert all(p.is_file() for p in paths.values())
    spec = registry.resolve_spec("ppocrv6_tiny")
    assert [c["filename"] for c in hub["modelscope"]] == (
        ["config.json"] + [f.filename for f in spec.files]
    )
    assert hub["huggingface"] == []
    assert all(c["repo"] == "mscoder/oxocr" for c in hub["modelscope"])


def test_auto_fallback_to_huggingface(hub, monkeypatch, tmp_path):
    monkeypatch.setitem(download._DOWNLOADERS, "modelscope", _fail("ms down"))
    paths = oxocr.download_model("ppocrv6_tiny", model_dir=tmp_path)
    assert set(paths) == {"det", "rec", "cls"}
    assert [c["filename"] for c in hub["huggingface"]] == (
        ["config.json"] + [f.filename for f in registry.resolve_spec("ppocrv6_tiny").files]
    )
    assert all(c["repo"] == "Luoaho/oxocr" for c in hub["huggingface"])


def test_fallback_resumes_remaining_files(monkeypatch, tmp_path):
    ms_calls: list = []
    hf_calls: list = []
    spec = registry.resolve_spec("ppocrv6_tiny")
    second = spec.files[1].filename
    monkeypatch.setitem(download._DOWNLOADERS, "modelscope", _fake(ms_calls, fail_on=(second,)))
    monkeypatch.setitem(download._DOWNLOADERS, "huggingface", _fake(hf_calls))
    oxocr.download_model("ppocrv6_tiny", model_dir=tmp_path)
    assert len(ms_calls) == 3  # config.json + 第一个成功，第二个失败即停
    assert [c["filename"] for c in hf_calls] == (
        ["config.json"] + [f.filename for f in spec.files[1:]]
    )


def test_both_sources_fail(hub, monkeypatch, tmp_path):
    for src in ("modelscope", "huggingface"):
        monkeypatch.setitem(download._DOWNLOADERS, src, _fail(f"{src} down"))
    with pytest.raises(ModelDownloadError) as e:
        oxocr.download_model("ppocrv6_tiny", model_dir=tmp_path)
    assert "modelscope" in str(e.value) and "huggingface" in str(e.value)


def test_source_huggingface_only(hub, tmp_path):
    oxocr.download_model("ppocrv6_tiny", model_dir=tmp_path, source="huggingface")
    assert hub["modelscope"] == []
    assert len(hub["huggingface"]) == 4  # config.json + 3 个权重


def test_source_modelscope_no_fallback(hub, monkeypatch, tmp_path):
    monkeypatch.setitem(download._DOWNLOADERS, "modelscope", _fail("ms down"))
    with pytest.raises(ModelDownloadError):
        oxocr.download_model("ppocrv6_tiny", model_dir=tmp_path, source="modelscope")
    assert hub["huggingface"] == []


def test_invalid_source(tmp_path):
    with pytest.raises(UnknownParameterError):
        oxocr.download_model("ppocrv6_tiny", model_dir=tmp_path, source="github")


def test_env_default_source(hub, monkeypatch, tmp_path):
    monkeypatch.setenv("OXOCR_DOWNLOAD_SOURCE", "huggingface")
    oxocr.download_model("ppocrv6_tiny", model_dir=tmp_path)
    assert hub["modelscope"] == []
    assert len(hub["huggingface"]) == 4  # config.json + 3 个权重


def test_offline_missing_files(hub, monkeypatch, tmp_path):
    monkeypatch.setenv("OXOCR_OFFLINE", "1")
    with pytest.raises(ModelDownloadError) as e:
        oxocr.download_model("ppocrv6_tiny", model_dir=tmp_path)
    assert "OXOCR_OFFLINE" in str(e.value)
    assert hub["modelscope"] == [] and hub["huggingface"] == []


def test_offline_already_installed(hub, monkeypatch, tmp_path):
    spec = registry.resolve_spec("ppocrv6_tiny")
    for f in spec.files:
        path = tmp_path / f.filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")
    monkeypatch.setenv("OXOCR_OFFLINE", "1")
    paths = oxocr.download_model("ppocrv6_tiny", model_dir=tmp_path)
    assert set(paths) == {"det", "rec", "cls"}
    assert hub["modelscope"] == [] and hub["huggingface"] == []


def test_skip_existing_files(hub, tmp_path):
    spec = registry.resolve_spec("ppocrv6_tiny")
    det = tmp_path / spec.files[0].filename
    det.parent.mkdir(parents=True, exist_ok=True)
    det.write_bytes(b"x")
    oxocr.download_model("ppocrv6_tiny", model_dir=tmp_path)
    assert [c["filename"] for c in hub["modelscope"]] == (
        ["config.json"] + [f.filename for f in spec.files[1:]]
    )


def test_force_redownloads_all(hub, tmp_path):
    spec = registry.resolve_spec("ppocrv6_tiny")
    for f in spec.files:
        path = tmp_path / f.filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")
    oxocr.download_model("ppocrv6_tiny", model_dir=tmp_path, force=True)
    assert len(hub["modelscope"]) == 4  # config.json + 3 个权重
    assert all(c["force"] is True for c in hub["modelscope"])


# ---------------- config.json 下载统计 ----------------

def test_config_always_forced(hub, tmp_path):
    # 即使本地已有 config.json 且 force=False，也要强制请求一次以计入 hub 下载量
    (tmp_path / "config.json").write_bytes(b"{}")
    oxocr.download_model("ppocrv6_tiny", model_dir=tmp_path)
    config_calls = [c for c in hub["modelscope"] if c["filename"] == "config.json"]
    assert len(config_calls) == 1
    assert config_calls[0]["force"] is True
    assert config_calls[0]["repo"] == "mscoder/oxocr"
    assert hub["huggingface"] == []


def test_config_failure_does_not_break_download(hub, monkeypatch, tmp_path):
    monkeypatch.setitem(
        download._DOWNLOADERS, "modelscope", _fake(hub["modelscope"], fail_on=("config.json",))
    )
    paths = oxocr.download_model("ppocrv6_tiny", model_dir=tmp_path)
    assert all(p.is_file() for p in paths.values())
    assert hub["huggingface"] == []  # config 失败不触发回退


def test_no_config_when_nothing_to_download(hub, tmp_path):
    spec = registry.resolve_spec("ppocrv6_tiny")
    for f in spec.files:
        path = tmp_path / f.filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")
    oxocr.download_model("ppocrv6_tiny", model_dir=tmp_path)
    assert hub["modelscope"] == [] and hub["huggingface"] == []


def test_config_follows_source(hub, tmp_path):
    oxocr.download_model("ppocrv6_tiny", model_dir=tmp_path, source="huggingface")
    assert hub["modelscope"] == []
    config_calls = [c for c in hub["huggingface"] if c["filename"] == "config.json"]
    assert len(config_calls) == 1
    assert config_calls[0]["repo"] == "Luoaho/oxocr"


def test_config_from_modelscope_only_repo(hub, monkeypatch, tmp_path):
    spec = ModelSpec(
        name="unit_ms_only", family="unitfake",
        impl="oxocr.models.rapid_ocr.pipeline:PpOcrModel",
        files=(ModelFile("model", "x.onnx"),),
        modelscope_repo="mscoder/oxocr",
    )
    registry.register_spec(spec)
    try:
        oxocr.download_model("unit_ms_only", model_dir=tmp_path)
        assert [c["filename"] for c in hub["modelscope"]] == ["config.json", "x.onnx"]
        assert hub["huggingface"] == []
    finally:
        registry._RUNTIME_SPECS.pop("unit_ms_only", None)


def test_spec_without_repo(monkeypatch, tmp_path):
    spec = ModelSpec(
        name="unit_norepo", family="unitfake",
        impl="oxocr.models.rapid_ocr.pipeline:PpOcrModel",
        files=(ModelFile("model", "x.onnx"),),
    )
    registry.register_spec(spec)
    try:
        with pytest.raises(ModelDownloadError) as e:
            oxocr.download_model("unit_norepo", model_dir=tmp_path)
        assert "未配置下载仓库" in str(e.value)
    finally:
        registry._RUNTIME_SPECS.pop("unit_norepo", None)


def test_progress_callback(hub, tmp_path):
    events: list = []
    oxocr.download_model(
        "ppocrv6_tiny", model_dir=tmp_path,
        progress=lambda name, done, total: events.append((name, done, total)),
    )
    spec = registry.resolve_spec("ppocrv6_tiny")
    assert len(events) == 2 * len(spec.files)
    assert events[0][1] == 0 and events[1][1] == events[1][2]


def test_downloadable_flag(tmp_path):
    infos = oxocr.list_models(model_dir=tmp_path)
    assert all(m.downloadable for m in infos)


def test_all_builtin_specs_have_repos():
    from oxocr.registry.catalog import BUILTIN_SPECS

    for spec in BUILTIN_SPECS.values():
        assert spec.modelscope_repo == "mscoder/oxocr", spec.name
        assert spec.huggingface_repo == "Luoaho/oxocr", spec.name


def test_modelscope_retry_env(monkeypatch, tmp_path):
    import os

    import modelscope_hub

    class FakeApi:
        def download_file(self, *args, **kwargs):
            pass

    monkeypatch.setattr(modelscope_hub, "HubApi", FakeApi)
    monkeypatch.delenv("MODELSCOPE_DOWNLOAD_MAX_RETRIES", raising=False)
    download._download_modelscope("mscoder/oxocr", "x.onnx", tmp_path, False)
    assert os.environ["MODELSCOPE_DOWNLOAD_MAX_RETRIES"] == "3"
    # 尊重用户显式设置
    monkeypatch.setenv("MODELSCOPE_DOWNLOAD_MAX_RETRIES", "7")
    download._download_modelscope("mscoder/oxocr", "x.onnx", tmp_path, False)
    assert os.environ["MODELSCOPE_DOWNLOAD_MAX_RETRIES"] == "7"


# ---------------- ppocr_cls 下载 bundle ----------------

def test_pull_ppocr_cls_downloads_all_variants(hub, tmp_path):
    from oxocr.models.rapid_ocr.specs import CLS_MODELS

    paths = oxocr.download_model("ppocr_cls", model_dir=tmp_path)
    assert set(paths) == set(CLS_MODELS)
    assert all(p.is_file() for p in paths.values())
    assert {c["filename"] for c in hub["modelscope"]} == (
        {v.filename for v in CLS_MODELS.values()} | {"config.json"}
    )
    assert hub["huggingface"] == []


def test_pull_ppocr_cls_skips_existing(hub, tmp_path):
    from oxocr.models.rapid_ocr.specs import CLS_MODELS

    for variant in CLS_MODELS.values():
        path = tmp_path / variant.filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")
    paths = oxocr.download_model("ppocr_cls", model_dir=tmp_path)
    assert set(paths) == set(CLS_MODELS)
    assert hub["modelscope"] == [] and hub["huggingface"] == []


def test_ppocr_cls_not_a_catalog_model(tmp_path):
    from oxocr.errors import ModelNotFoundError

    assert all(m.name != "ppocr_cls" for m in oxocr.list_models(model_dir=tmp_path))
    with pytest.raises(ModelNotFoundError):
        oxocr.get_model_spec("ppocr_cls")


# ---------------- CLI ----------------

@pytest.fixture
def dl_spy(monkeypatch):
    calls: dict = {}

    def fake_download(name, **kwargs):
        calls["name"] = name
        calls["kwargs"] = kwargs
        return {"det": Path("d.onnx"), "rec": Path("r.onnx")}

    monkeypatch.setattr(oxocr, "download_model", fake_download)
    return calls


def test_cli_download(dl_spy, tmp_path, capsys):
    code = main(["models", "download", "ppocrv6_tiny",
                 "--source", "modelscope", "--force", "-D", str(tmp_path)])
    assert code == 0
    assert dl_spy["name"] == "ppocrv6_tiny"
    assert dl_spy["kwargs"] == {
        "model_dir": str(tmp_path), "force": True, "source": "modelscope",
    }
    out = capsys.readouterr().out
    assert "ppocrv6_tiny" in out and "d.onnx" in out


@pytest.mark.parametrize("argv", [
    ["models", "pull", "ppocrv6_tiny"],
    ["download", "ppocrv6_tiny"],
    ["pull", "ppocrv6_tiny"],
])
def test_cli_download_aliases(dl_spy, argv):
    assert main(argv) == 0
    assert dl_spy["name"] == "ppocrv6_tiny"
    assert dl_spy["kwargs"]["source"] is None


def test_cli_pull_ppocr_cls(dl_spy):
    assert main(["pull", "ppocr_cls"]) == 0
    assert dl_spy["name"] == "ppocr_cls"


def test_cli_download_error(monkeypatch, capsys):
    def fake_download(name, **kwargs):
        raise ModelDownloadError("网络不可用")

    monkeypatch.setattr(oxocr, "download_model", fake_download)
    assert main(["models", "download", "ppocrv6_tiny"]) == 1
    assert "网络不可用" in capsys.readouterr().err


def test_cli_download_help(capsys):
    with pytest.raises(SystemExit) as e:
        main(["models", "download", "-h"])
    assert e.value.code == 0
    assert "--source" in capsys.readouterr().out
