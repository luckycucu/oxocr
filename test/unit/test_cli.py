"""oxocr CLI：Docker 风格子命令、顶层简写、-h 双语帮助、旧写法拒绝。"""
from __future__ import annotations

import types
from datetime import date
from pathlib import Path

import numpy as np
import oxocr
import pytest
from oxocr.cli import main
from oxocr.types import Box, TextItem
from PIL import Image


@pytest.fixture
def ocr_spy(monkeypatch):
    calls: dict = {}

    def fake_ocr(image, **kwargs):
        calls["image"] = image
        calls["kwargs"] = kwargs
        return types.SimpleNamespace(text="识别结果")

    monkeypatch.setattr(oxocr, "ocr", fake_ocr)
    return calls


@pytest.fixture
def list_spy(monkeypatch):
    calls: dict = {}

    def fake_list(**kwargs):
        calls["kwargs"] = kwargs
        return [types.SimpleNamespace(installed=True, name="m1", family="f1",
                                      name_chinese="模型一")]

    def fake_spec(name):
        return types.SimpleNamespace(
            name="m1", family="f1", name_chinese="模型一",
            raw_release_date=date(2026, 6, 11),
            size_bytes=30 * 1048576,
            license="Apache-2.0",
            languages=("zh", "zh_tw", "en", "ja"),
            languages_chinese=("简体中文", "繁体中文", "英语", "日语"),
            languages_english=("Chinese (Simplified)", "Traditional Chinese",
                               "English", "Japanese"),
        )

    monkeypatch.setattr(oxocr, "list_models", fake_list)
    monkeypatch.setattr(oxocr, "get_model_spec", fake_spec)
    return calls


# ---------------- 双语帮助 ----------------

def test_help_default_english(capsys):
    with pytest.raises(SystemExit) as e:
        main(["-h"])
    assert e.value.code == 0
    out = capsys.readouterr().out
    assert "usage:" in out
    assert "recognize an image" in out
    assert "show this help message and exit" in out


@pytest.mark.parametrize("argv", [
    ["-h", "--lang", "zh"],
    ["--lang", "zh", "-h"],
    ["--help", "--lang=zh"],
])
def test_help_chinese_order_independent(argv, capsys):
    with pytest.raises(SystemExit) as e:
        main(argv)
    assert e.value.code == 0
    out = capsys.readouterr().out
    assert "用法:" in out
    assert "识别图像" in out
    assert "显示此帮助信息并退出" in out


def test_run_help_chinese(capsys):
    with pytest.raises(SystemExit) as e:
        main(["run", "-h", "--lang", "zh"])
    assert e.value.code == 0
    out = capsys.readouterr().out
    assert "用法: oxocr run" in out
    assert "图像路径" in out
    assert "排版模式" in out
    assert "--lang" in out


def test_models_ls_help_chinese(capsys):
    with pytest.raises(SystemExit) as e:
        main(["models", "ls", "-h", "--lang", "zh"])
    assert e.value.code == 0
    out = capsys.readouterr().out
    assert "列出可用模型" in out
    assert "仅列出已安装模型" in out


@pytest.mark.parametrize("argv", [
    ["-h", "--lang", "fr"],
    ["--lang", "fr", "-h"],
    ["--lang=fr", "-h"],
])
def test_help_invalid_lang(argv, capsys):
    with pytest.raises(SystemExit) as e:
        main(argv)
    assert e.value.code == 2
    assert "invalid choice" in capsys.readouterr().err


def test_lang_missing_value():
    with pytest.raises(SystemExit) as e:
        main(["--lang"])
    assert e.value.code == 2


# ---------------- version ----------------

def test_version_flag(capsys):
    with pytest.raises(SystemExit) as e:
        main(["-v"])
    assert e.value.code == 0
    assert capsys.readouterr().out.strip() == f"oxocr {oxocr.__version__}"


def test_version_command(capsys):
    assert main(["version"]) == 0
    assert capsys.readouterr().out.strip() == f"oxocr {oxocr.__version__}"


# ---------------- run ----------------

def test_run_all_options(ocr_spy, capsys):
    rc = main(["run", "a.png", "--model", "x", "--mode", "layout", "--output", "json",
               "--cls", "--text-score", "0.6", "--threads", "4",
               "--model-dir", "/m", "--debug"])
    assert rc == 0
    assert capsys.readouterr().out.strip() == "识别结果"
    assert ocr_spy["image"] == "a.png"
    kw = ocr_spy["kwargs"]
    assert kw["model"] == "x"
    assert kw["mode"] == "layout"
    assert kw["output"] == "json"
    assert kw["model_dir"] == "/m"
    assert kw["debug"] is True
    assert kw["use_cls"] is True
    assert kw["text_score"] == 0.6
    assert kw["num_threads"] == 4
    assert "backend" not in kw


def test_new_run_short_options(ocr_spy):
    assert main(["run", "a.png", "-m", "x", "-M", "raw", "-o", "json",
                 "-c", "-s", "0.7", "-t", "2", "-D", "/m"]) == 0
    kw = ocr_spy["kwargs"]
    assert kw["model"] == "x"
    assert kw["mode"] == "raw"
    assert kw["output"] == "json"
    assert kw["model_dir"] == "/m"
    assert kw["use_cls"] is True
    assert kw["text_score"] == 0.7
    assert kw["num_threads"] == 2


def test_format_alias(ocr_spy):
    assert main(["run", "a.png", "--format", "json"]) == 0
    assert ocr_spy["kwargs"]["output"] == "json"


def test_run_defaults(ocr_spy):
    assert main(["run", "a.png"]) == 0
    kw = ocr_spy["kwargs"]
    assert kw["model"] == "ppocrv6_small"
    assert kw["mode"] == "para"
    assert kw["output"] == "text"
    assert kw["model_dir"] is None
    assert kw["debug"] is False
    assert "use_cls" not in kw
    assert "text_score" not in kw
    assert "num_threads" not in kw


@pytest.mark.parametrize("argv", [
    ["run", "a.png", "--lang", "zh"],
    ["--lang", "zh", "run", "a.png"],
])
def test_lang_ignored_in_business(argv, ocr_spy):
    assert main(argv) == 0
    assert "lang" not in ocr_spy["kwargs"]


def test_run_error(monkeypatch, capsys):
    def boom(image, **kwargs):
        raise oxocr.OxocrError("boom")

    monkeypatch.setattr(oxocr, "ocr", boom)
    assert main(["run", "a.png"]) == 1
    assert "错误" in capsys.readouterr().err


def test_run_missing_image():
    with pytest.raises(SystemExit) as e:
        main(["run"])
    assert e.value.code == 2


@pytest.mark.parametrize("argv", [
    ["a.png"],
    ["a.png", "--model", "x"],
    ["-D", "/m", "a.png"],
])
def test_flat_form_rejected(argv, capsys):
    with pytest.raises(SystemExit) as e:
        main(argv)
    assert e.value.code == 2
    assert "invalid choice" in capsys.readouterr().err


@pytest.mark.parametrize("argv", [
    ["--list-models"],
    ["--list-models", "--installed-only"],
])
def test_legacy_list_models_rejected(argv, capsys):
    with pytest.raises(SystemExit) as e:
        main(argv)
    assert e.value.code == 2
    assert "unrecognized arguments" in capsys.readouterr().err


@pytest.mark.parametrize("argv", [
    ["run", "a.png", "--backend", "inprocess"],
    ["run", "a.png", "-b", "inprocess"],
])
def test_backend_removed(argv, capsys):
    with pytest.raises(SystemExit) as e:
        main(argv)
    assert e.value.code == 2
    assert "unrecognized arguments" in capsys.readouterr().err


# ---------------- run --vis 可视化 ----------------

VIS_FONT = (Path(__file__).resolve().parents[2]
            / "examples" / "duguang-ocr-onnx-v2.5" / "dgocr"
            / "AlibabaPuHuiTi-3-45-Light.ttf")


@pytest.fixture
def vis_env(monkeypatch, tmp_path):
    img = tmp_path / "in.png"
    Image.fromarray(np.full((120, 200, 3), 255, np.uint8)).save(img)
    items = [
        TextItem(text="hello", score=0.99, box=Box.from_rect(10, 10, 110, 25)),
        TextItem(text="world", score=0.95, box=Box.from_rect(20, 60, 160, 30)),
    ]
    monkeypatch.setattr(oxocr, "ocr",
                        lambda image, **kw: types.SimpleNamespace(text="识别结果", items=items))
    return tmp_path, img


@pytest.fixture
def vis_font() -> str:
    if not VIS_FONT.is_file():
        pytest.skip("仓库示例字体缺失，跳过文字绘制用例")
    return str(VIS_FONT)


def test_run_vis_default_box_txt(vis_env, vis_font, capsys):
    tmp, img = vis_env
    out = tmp / "vis.png"
    assert main(["run", str(img), "--vis", str(out), "--font", vis_font]) == 0
    captured = capsys.readouterr()
    assert captured.out.strip() == "识别结果"  # stdout 不受污染
    assert "Visualization saved" in captured.err
    with Image.open(out) as saved:
        assert saved.size == (400, 120)  # 双联图宽度翻倍


def test_run_vis_mode_box_needs_no_font(vis_env, capsys):
    tmp, img = vis_env
    out = tmp / "vis.png"
    assert main(["run", str(img), "--vis", str(out), "--vis-mode", "box"]) == 0
    assert "Visualization saved" in capsys.readouterr().err
    with Image.open(out) as saved:
        assert saved.size == (200, 120)


def test_run_vis_mode_ocr(vis_env, vis_font):
    tmp, img = vis_env
    out = tmp / "vis.jpg"
    assert main(["run", str(img), "--vis", str(out),
                 "--vis-mode", "ocr", "--font", vis_font]) == 0
    with Image.open(out) as saved:
        assert saved.width > 600  # 缩放后拼接文字面板


def test_run_vis_saved_message_chinese(vis_env, vis_font, capsys):
    tmp, img = vis_env
    out = tmp / "vis.png"
    assert main(["run", str(img), "--vis", str(out),
                 "--font", vis_font, "--lang", "zh"]) == 0
    assert "可视化已保存" in capsys.readouterr().err


def test_run_vis_invalid_mode(vis_env):
    tmp, img = vis_env
    with pytest.raises(SystemExit) as e:
        main(["run", str(img), "--vis", str(tmp / "v.png"), "--vis-mode", "fancy"])
    assert e.value.code == 2


def test_run_vis_save_failure(vis_env, capsys):
    tmp, img = vis_env
    bad = tmp / "no_such_dir" / "vis.png"
    assert main(["run", str(img), "--vis", str(bad), "--vis-mode", "box"]) == 1
    assert "错误" in capsys.readouterr().err


def test_run_vis_bad_font(vis_env, capsys):
    tmp, img = vis_env
    out = tmp / "vis.png"
    assert main(["run", str(img), "--vis", str(out), "--font", "no_such_font.ttf"]) == 1
    assert "错误" in capsys.readouterr().err


def test_run_vis_help_chinese(capsys):
    with pytest.raises(SystemExit) as e:
        main(["run", "-h", "--lang", "zh"])
    assert e.value.code == 0
    out = capsys.readouterr().out
    assert "可视化" in out
    assert "--vis-mode" in out
    assert "--font" in out


# ---------------- models：子命令与简写 ----------------

@pytest.mark.parametrize("argv", [
    ["models", "list"],
    ["models", "ls"],
    ["model", "list"],
    ["models"],
    ["ls"],
    ["list"],
    ["--lang", "zh", "ls"],
])
def test_models_list_forms(argv, list_spy, capsys):
    assert main(argv) == 0
    out = capsys.readouterr().out
    assert "[V] m1" in out
    assert list_spy["kwargs"]["installed_only"] is False


@pytest.mark.parametrize("argv", [
    ["ls", "-i"],
    ["list", "--installed-only"],
    ["models", "ls", "-i"],
])
def test_installed_only_forms(argv, list_spy):
    assert main(argv) == 0
    assert list_spy["kwargs"]["installed_only"] is True


def test_models_list_model_dir(list_spy):
    assert main(["ls", "-D", "/m"]) == 0
    assert list_spy["kwargs"]["model_dir"] == "/m"


def test_models_list_table_columns(list_spy, capsys):
    assert main(["ls"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].split() == ["NAME", "RELEASED", "SIZE(MB)", "LICENSE", "LANGUAGES"]
    assert lines[1].startswith("[V] m1")
    assert "2026-06-11" in lines[1]
    assert "30.0" in lines[1]
    assert "Apache-2.0" in lines[1]
    assert "zh, zh_tw, en, ..." in lines[1]


def test_models_list_table_chinese(list_spy, capsys):
    assert main(["ls", "--lang", "zh"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].split() == ["名称", "开源日期", "大小(MB)", "开源协议", "支持语言"]
    assert "简体中文, 繁体中文, 英语, ..." in lines[1]


def _list_one(monkeypatch, *, installed=False, release=None, size=0, license="",
              languages=(), languages_chinese=()):
    monkeypatch.setattr(oxocr, "list_models", lambda **kw: [
        types.SimpleNamespace(installed=installed, name="m1", family="f1",
                              name_chinese="模型一")])
    monkeypatch.setattr(oxocr, "get_model_spec", lambda name: types.SimpleNamespace(
        raw_release_date=release, size_bytes=size, license=license,
        languages=languages, languages_chinese=languages_chinese))


def test_models_list_table_missing_values(monkeypatch, capsys):
    _list_one(monkeypatch)
    assert main(["ls"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[1].startswith("[ ] m1")
    assert lines[1].split() == ["[", "]", "m1", "-", "-", "-", "-"]


def test_models_list_table_license_short(monkeypatch, capsys):
    _list_one(monkeypatch, license="AI PUBS OPEN RAIL-M (modified)")
    assert main(["ls"]) == 0
    row = capsys.readouterr().out.splitlines()[1]
    assert "RAIL-M (modified)" in row
    assert "AI PUBS" not in row


def test_models_list_table_few_languages(monkeypatch, capsys):
    _list_one(monkeypatch, release=date(2022, 5, 25), size=178952787,
              languages=("zh", "en"), languages_chinese=("简体中文", "英语"))
    assert main(["ls"]) == 0
    row = capsys.readouterr().out.splitlines()[1]
    assert row.endswith("zh, en")
    assert "..." not in row
    assert "170.7" in row


def test_models_list_table_empty(monkeypatch, capsys):
    monkeypatch.setattr(oxocr, "list_models", lambda **kw: [])
    assert main(["ls", "-i"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 1
    assert lines[0].split() == ["NAME", "RELEASED", "SIZE(MB)", "LICENSE", "LANGUAGES"]


# ---------------- models：排序 ----------------

def _fake_models(monkeypatch, models):
    infos = [types.SimpleNamespace(installed=m.get("installed", False),
                                   name=m["name"], family=m["family"],
                                   name_chinese=m.get("name_chinese", ""))
             for m in models]
    specs = {m["name"]: types.SimpleNamespace(
        name=m["name"], family=m["family"],
        name_chinese=m.get("name_chinese", ""),
        raw_release_date=m.get("date"), size_bytes=m.get("size", 0),
        license=m.get("license", ""),
        languages=m.get("languages", ()),
        languages_chinese=m.get("languages_chinese", ()),
        languages_english=m.get("languages_english", ())) for m in models}
    monkeypatch.setattr(oxocr, "list_models", lambda **kw: infos)
    monkeypatch.setattr(oxocr, "get_model_spec", lambda name: specs[name])


def _names(capsys) -> list[str]:
    return [line[4:].split()[0] for line in capsys.readouterr().out.splitlines()[1:]]


def test_sort_default_released_desc(monkeypatch, capsys):
    _fake_models(monkeypatch, [
        {"name": "old", "family": "f", "date": date(2022, 1, 1)},
        {"name": "new", "family": "f", "date": date(2026, 1, 1)},
        {"name": "mid", "family": "f", "date": date(2024, 1, 1)},
    ])
    assert main(["ls"]) == 0
    assert _names(capsys) == ["new", "mid", "old"]


def test_sort_size_desc_and_reverse(monkeypatch, capsys):
    _fake_models(monkeypatch, [
        {"name": "a", "family": "f", "size": 10 * 1048576},
        {"name": "b", "family": "f", "size": 30 * 1048576},
        {"name": "c", "family": "f", "size": 20 * 1048576},
    ])
    assert main(["ls", "--sort", "size"]) == 0
    assert _names(capsys) == ["b", "c", "a"]
    assert main(["ls", "--sort", "size", "-r"]) == 0
    assert _names(capsys) == ["a", "c", "b"]


def test_sort_name_asc_and_reverse(monkeypatch, capsys):
    _fake_models(monkeypatch, [
        {"name": "beta", "family": "f"},
        {"name": "Alpha", "family": "f"},
        {"name": "gamma", "family": "f"},
    ])
    assert main(["ls", "--sort", "name"]) == 0
    assert _names(capsys) == ["Alpha", "beta", "gamma"]
    assert main(["ls", "--sort", "name", "--reverse"]) == 0
    assert _names(capsys) == ["gamma", "beta", "Alpha"]


def test_sort_family_rejected(capsys):
    with pytest.raises(SystemExit) as e:
        main(["ls", "--sort", "family"])
    assert e.value.code == 2
    assert "invalid choice" in capsys.readouterr().err


def test_sort_missing_values_always_last(monkeypatch, capsys):
    _fake_models(monkeypatch, [
        {"name": "nodate", "family": "f", "size": 0},
        {"name": "dated", "family": "f", "date": date(2022, 1, 1),
         "size": 5 * 1048576},
    ])
    for argv in (["ls"], ["ls", "-r"],
                 ["ls", "--sort", "size"], ["ls", "--sort", "size", "-r"]):
        assert main(argv) == 0
        assert _names(capsys) == ["dated", "nodate"]


def test_sort_tiebreak_name_asc(monkeypatch, capsys):
    _fake_models(monkeypatch, [
        {"name": "b", "family": "f", "date": date(2026, 1, 1)},
        {"name": "a", "family": "f", "date": date(2026, 1, 1)},
    ])
    assert main(["ls"]) == 0
    assert _names(capsys) == ["a", "b"]
    assert main(["ls", "-r"]) == 0
    assert _names(capsys) == ["a", "b"]


# ---------------- models：search ----------------

SEARCH_MODELS = [
    {"name": "ppocrv6_small", "family": "ppocr", "name_chinese": "PP-OCRv6 小型",
     "date": date(2026, 6, 11), "size": 30 * 1048576,
     "languages": ("zh", "zh_tw", "en"),
     "languages_chinese": ("简体中文", "繁体中文", "英文"),
     "languages_english": ("Chinese (Simplified)", "Traditional Chinese",
                           "English")},
    {"name": "duguang_ocr_small", "family": "duguang_ocr", "name_chinese": "读光 small",
     "date": date(2023, 5, 29), "size": 12 * 1048576,
     "languages": ("zh", "en"),
     "languages_chinese": ("简体中文", "英文"),
     "languages_english": ("Chinese (Simplified)", "English")},
    {"name": "indra_ocr_rtl", "family": "indra_ocr",
     "date": date(2026, 10, 19), "size": 5 * 1048576,
     "languages": ("ar", "fa"),
     "languages_chinese": ("阿拉伯文", "波斯文"),
     "languages_english": ("Arabic", "Persian")},
    {"name": "ppocrv4_devanagari_mobile", "family": "ppocr",
     "date": date(2023, 8, 7), "size": 12 * 1048576,
     "languages": ("devanagari",),
     "languages_chinese": ("天城文",),
     "languages_english": ("Devanagari",)},
    {"name": "rapid_latex_ocr", "family": "latex_ocr",
     "date": date(2022, 5, 25), "size": 170 * 1048576,
     "languages": ("latex",),
     "languages_chinese": ("latex",),
     "languages_english": ("latex",)},
]


@pytest.mark.parametrize("argv", [
    ["search", "zh"],
    ["models", "search", "zh"],
])
def test_search_forms_and_lang_prefix(argv, monkeypatch, capsys):
    _fake_models(monkeypatch, SEARCH_MODELS)
    assert main(argv) == 0
    assert _names(capsys) == ["ppocrv6_small", "duguang_ocr_small"]


def test_search_lang_code_prefix_not_substring(monkeypatch, capsys):
    _fake_models(monkeypatch, SEARCH_MODELS)
    assert main(["search", "ar", "-f", "lang"]) == 0
    assert _names(capsys) == ["indra_ocr_rtl"]


def test_search_lang_chinese_and_english_names(monkeypatch, capsys):
    _fake_models(monkeypatch, SEARCH_MODELS)
    assert main(["search", "中文"]) == 0
    assert _names(capsys) == ["ppocrv6_small", "duguang_ocr_small"]
    assert main(["search", "chinese"]) == 0
    assert _names(capsys) == ["ppocrv6_small", "duguang_ocr_small"]
    assert main(["search", "阿拉伯", "--lang", "zh"]) == 0
    assert _names(capsys) == ["indra_ocr_rtl"]


def test_search_name_includes_chinese_name(monkeypatch, capsys):
    _fake_models(monkeypatch, SEARCH_MODELS)
    assert main(["search", "PPOCRV6", "-f", "name"]) == 0
    assert _names(capsys) == ["ppocrv6_small"]
    assert main(["search", "读光", "-f", "name"]) == 0
    assert _names(capsys) == ["duguang_ocr_small"]


def test_search_field_name_excludes_lang(monkeypatch, capsys):
    _fake_models(monkeypatch, SEARCH_MODELS)
    assert main(["search", "zh", "-f", "name"]) == 1
    assert capsys.readouterr().out == ""


def test_search_field_family(monkeypatch, capsys):
    _fake_models(monkeypatch, SEARCH_MODELS)
    assert main(["search", "indra", "-f", "family"]) == 0
    assert _names(capsys) == ["indra_ocr_rtl"]


def test_search_field_all(monkeypatch, capsys):
    _fake_models(monkeypatch, SEARCH_MODELS)
    assert main(["search", "latex", "-f", "all"]) == 0
    assert _names(capsys) == ["rapid_latex_ocr"]
    assert main(["search", "ppocr", "-f", "all"]) == 0
    assert _names(capsys) == ["ppocrv6_small", "ppocrv4_devanagari_mobile"]


def test_search_default_fields_name_and_lang(monkeypatch, capsys):
    models = SEARCH_MODELS + [{"name": "modelx", "family": "secretfam"}]
    _fake_models(monkeypatch, models)
    assert main(["search", "duguang"]) == 0
    assert _names(capsys) == ["duguang_ocr_small"]
    # 仅 family 命中 → 默认 name+lang 不返回
    assert main(["search", "secretfam"]) == 1
    assert main(["search", "secretfam", "-f", "family"]) == 0
    assert _names(capsys) == ["modelx"]


def test_search_lang_english_word_prefix(monkeypatch, capsys):
    _fake_models(monkeypatch, SEARCH_MODELS)
    # Chinese 含子串 hi，但按词前缀不应命中
    assert main(["search", "hi", "-f", "lang"]) == 1
    assert main(["search", "arab", "-f", "lang"]) == 0
    assert _names(capsys) == ["indra_ocr_rtl"]


def test_search_sorted_and_installed_only(monkeypatch, capsys):
    calls: dict = {}

    def fake_list(**kwargs):
        calls["kwargs"] = kwargs
        return [types.SimpleNamespace(installed=m.get("installed", False),
                                      name=m["name"], family=m["family"],
                                      name_chinese=m.get("name_chinese", ""))
                for m in SEARCH_MODELS]

    _fake_models(monkeypatch, SEARCH_MODELS)
    monkeypatch.setattr(oxocr, "list_models", fake_list)
    assert main(["search", "zh", "-i", "--sort", "size"]) == 0
    assert calls["kwargs"]["installed_only"] is True
    assert _names(capsys) == ["ppocrv6_small", "duguang_ocr_small"]


@pytest.mark.parametrize("argv, message", [
    (["search", "no_such_thing"], "no matching models"),
    (["search", "no_such_thing", "--lang", "zh"], "未找到匹配的模型"),
])
def test_search_no_match(argv, message, monkeypatch, capsys):
    _fake_models(monkeypatch, SEARCH_MODELS)
    assert main(argv) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert message in captured.err


def test_search_help_chinese(capsys):
    with pytest.raises(SystemExit) as e:
        main(["search", "-h", "--lang", "zh"])
    assert e.value.code == 0
    out = capsys.readouterr().out
    assert "搜索关键词" in out
    assert "检索字段" in out


@pytest.mark.parametrize("argv", [
    ["models", "info", "ppocrv6_small"],
    ["models", "inspect", "ppocrv6_small"],
    ["inspect", "ppocrv6_small"],
])
def test_models_info(argv, tmp_path, capsys):
    assert main(argv + ["-D", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "ppocrv6_small" in out
    assert "Family:" in out
    assert "Installed:        no" in out
    assert "Files:" in out


def test_models_info_installed(models_dir, capsys):
    assert main(["inspect", "ppocrv6_small", "-D", str(models_dir)]) == 0
    assert "Installed:        yes" in capsys.readouterr().out


def test_models_info_unknown(capsys):
    assert main(["inspect", "no_such_model"]) == 1
    assert "错误" in capsys.readouterr().err


# ---------------- 无子命令 fallback ----------------

def test_no_args_prints_help(capsys):
    assert main([]) == 2
    assert "usage:" in capsys.readouterr().out


def test_lang_only_prints_chinese_help(capsys):
    assert main(["--lang", "zh"]) == 2
    assert "用法:" in capsys.readouterr().out
