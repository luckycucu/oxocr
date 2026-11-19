"""命令行入口：Docker 风格子命令 oxocr <COMMAND> <ACTION> [OPTIONS]。

-h 默认英文，-h --lang zh 输出中文（与 -h 顺序无关）。
"""
from __future__ import annotations

import argparse
import re
import sys
import unicodedata
from functools import partial
from pathlib import Path

PROG = "oxocr"

MODELS_SHORTCUTS = {"ls", "list", "inspect", "search", "download", "pull"}


def T(en: str, zh: str, lang: str) -> str:
    return zh if lang == "zh" else en


class BilingualHelpFormatter(argparse.HelpFormatter):
    def __init__(self, prog, lang="en", **kw):
        self.lang = lang
        super().__init__(prog, **kw)

    def _format_usage(self, usage, actions, groups, prefix):
        if prefix is None:
            prefix = "用法: " if self.lang == "zh" else "usage: "
        return super()._format_usage(usage, actions, groups, prefix)

    def _fill_text(self, text, width, indent):
        # 保留 epilog 示例的换行（默认 HelpFormatter 会折成一段）
        return "".join(indent + line for line in text.splitlines(keepends=True))


def _detect_lang(argv: list[str]) -> str:
    # action="help" 遇到 -h 立即打印退出，等不到 --lang 解析，只能预扫描；
    # 非法值原样返回，由 main 统一报错（否则 -h 在前时会跳过校验）
    for i, a in enumerate(argv):
        if a == "--lang":
            return argv[i + 1] if i + 1 < len(argv) else "en"
        if a.startswith("--lang="):
            return a.split("=", 1)[1]
    return "en"


def _first_non_option(argv: list[str]) -> str | None:
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--lang":
            i += 2
            continue
        if a.startswith("--lang=") or a.startswith("-"):
            i += 1
            continue
        return a
    return None


def _normalize_argv(argv: list[str]) -> list[str]:
    """顶层简写 ls / list / inspect / search → models 子命令。"""
    first = _first_non_option(argv)
    if first in MODELS_SHORTCUTS:
        idx = argv.index(first)
        return argv[:idx] + ["models"] + argv[idx:]
    return argv


def _decorate(p: argparse.ArgumentParser, lang: str) -> argparse.ArgumentParser:
    p.add_argument("-h", "--help", action="help",
                   help=T("show this help message and exit",
                          "显示此帮助信息并退出", lang))
    p.add_argument("--lang", choices=["en", "zh"], default="en",
                   help=T("help language, used with -h (default: en)",
                          "帮助语言，与 -h 配合使用（默认 en）", lang))
    if lang == "zh":
        p._positionals.title = "位置参数"
        p._optionals.title = "选项"
    return p


def _new_parser(lang: str, **kwargs) -> argparse.ArgumentParser:
    kwargs.setdefault("formatter_class", partial(BilingualHelpFormatter, lang=lang))
    kwargs.setdefault("add_help", False)
    return _decorate(argparse.ArgumentParser(**kwargs), lang)


def _new_subparser(sub, name: str, lang: str, **kwargs) -> argparse.ArgumentParser:
    kwargs.setdefault("formatter_class", partial(BilingualHelpFormatter, lang=lang))
    kwargs.setdefault("add_help", False)
    return _decorate(sub.add_parser(name, **kwargs), lang)


def _add_global_options(sp: argparse.ArgumentParser, lang: str) -> None:
    sp.add_argument("-D", "--model-dir", default=None, metavar="DIR",
                    help=T("model root directory (overrides OXOCR_MODEL_DIR)",
                           "模型根目录（覆盖 OXOCR_MODEL_DIR）", lang))
    sp.add_argument("--debug", action="store_true",
                    help=T("enable debug logging", "输出调试日志", lang))


def _add_list_options(sp: argparse.ArgumentParser, lang: str) -> None:
    sp.add_argument("-i", "--installed-only", action="store_true",
                    help=T("only list installed models",
                           "仅列出已安装模型", lang))
    sp.add_argument("--sort", choices=["released", "name", "size"],
                    default="released", metavar="FIELD",
                    help=T("sort by field (default: %(default)s)",
                           "排序字段（默认 %(default)s）", lang))
    sp.add_argument("-r", "--reverse", action="store_true",
                    help=T("reverse sort order", "反转排序方向", lang))


def _version_text() -> str:
    from oxocr import __version__

    return f"{PROG} {__version__}"


class _VersionAction(argparse.Action):
    def __init__(self, option_strings, dest, **kwargs):
        super().__init__(option_strings, dest, nargs=0, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        print(_version_text())
        parser.exit()


def _build_parser(lang: str = "en") -> argparse.ArgumentParser:
    epilog_en = (
        "Examples:\n"
        "  oxocr run image.png\n"
        "  oxocr run image.png -m ppocrv6_small -o json\n"
        "  oxocr run image.png -M layout -o json -c\n"
        "  oxocr run image.png --vis out.png\n"
        "  oxocr models ls -i\n"
        "  oxocr ls -i\n"
        "  oxocr models inspect ppocrv6_small\n"
        "  oxocr models download ppocrv6_tiny\n"
        "  oxocr search zh -f lang\n"
        "  oxocr -h --lang zh          # Chinese help\n"
    )
    epilog_zh = (
        "示例:\n"
        "  oxocr run image.png\n"
        "  oxocr run image.png -m ppocrv6_small -o json\n"
        "  oxocr run image.png -M layout -o json -c\n"
        "  oxocr run image.png --vis out.png\n"
        "  oxocr models ls -i\n"
        "  oxocr ls -i\n"
        "  oxocr models inspect ppocrv6_small\n"
        "  oxocr models download ppocrv6_tiny\n"
        "  oxocr search zh -f lang\n"
        "  oxocr -h --lang zh          # 中文帮助\n"
    )

    p = _new_parser(
        lang,
        prog=PROG,
        description=T("oxocr command-line OCR tool",
                      "oxocr 命令行识别工具", lang),
        epilog=epilog_zh if lang == "zh" else epilog_en,
    )
    p.add_argument("-v", "--version", action=_VersionAction,
                   help=T("show program's version number and exit",
                          "显示版本号并退出", lang))

    sub = p.add_subparsers(dest="command", metavar="COMMAND")

    # ---- run ----
    run = _new_subparser(
        sub, "run", lang,
        help=T("recognize an image", "识别图像", lang),
        description=T("Recognize an image and print the result",
                      "识别图像并输出结果", lang),
    )
    run.add_argument("image", help=T("path to the image", "图像路径", lang))
    run.add_argument("-m", "--model", default="ppocrv6_small",
                     help=T("model name (default: %(default)s)",
                            "模型名（默认 %(default)s）", lang))
    run.add_argument("-M", "--mode", choices=["raw", "line", "para", "layout"],
                     default="para",
                     help=T("layout mode (default: %(default)s)",
                            "排版模式（默认 %(default)s）", lang))
    run.add_argument("-o", "--output", "--format", dest="output",
                     choices=["text", "json"], default="text",
                     help=T("output format (default: %(default)s)",
                            "输出格式（默认 %(default)s）", lang))
    run.add_argument("-c", "--cls", action="store_true",
                     help=T("enable orientation classification",
                            "开启方向分类", lang))
    run.add_argument("-s", "--text-score", type=float, default=None, metavar="FLOAT",
                     help=T("recognition confidence threshold",
                            "识别置信度阈值", lang))
    run.add_argument("-t", "--threads", type=int, default=None, metavar="N",
                     help=T("number of inference threads", "推理线程数", lang))
    run.add_argument("--vis", default=None, metavar="PATH",
                     help=T("save a visualization image to PATH (format by extension)",
                            "保存可视化图像到 PATH（格式按扩展名）", lang))
    run.add_argument("--vis-mode", choices=["box_txt", "box", "ocr"],
                     default="box_txt", metavar="MODE",
                     help=T("visualization mode: box_txt (side-by-side boxes and "
                            "text), box (boxes only), ocr (boxes + text panel) "
                            "(default: %(default)s)",
                            "可视化模式：box_txt 双联图（框+文字）、box 仅画框、"
                            "ocr 画框+文字面板（默认 %(default)s）", lang))
    run.add_argument("--font", default=None, metavar="PATH",
                     help=T("font file (ttf/ttc) for visualization text "
                            "(default: auto-detect system font)",
                            "可视化文字字体文件（ttf/ttc，默认自动探测系统字体）", lang))
    _add_global_options(run, lang)
    run.set_defaults(func=_cmd_run)

    # ---- models ----
    models = _new_subparser(
        sub, "models", lang,
        aliases=["model"],
        help=T("manage models", "模型管理", lang),
        description=T("Manage OCR models", "模型管理", lang),
    )
    msub = models.add_subparsers(dest="models_cmd", metavar="ACTION")

    mlist = _new_subparser(
        msub, "list", lang,
        aliases=["ls"],
        help=T("list available models", "列出可用模型", lang),
        description=T("List available models", "列出可用模型", lang),
    )
    _add_list_options(mlist, lang)
    _add_global_options(mlist, lang)
    mlist.set_defaults(func=_cmd_models_list)

    minfo = _new_subparser(
        msub, "info", lang,
        aliases=["inspect"],
        help=T("show model information", "显示模型信息", lang),
        description=T("Show model information", "显示模型信息", lang),
    )
    minfo.add_argument("name", help=T("model name", "模型名", lang))
    _add_global_options(minfo, lang)
    minfo.set_defaults(func=_cmd_models_info)

    msearch = _new_subparser(
        msub, "search", lang,
        help=T("search models by name or language",
               "按名称或语言搜索模型", lang),
        description=T("Search models by name or language",
                      "按名称或语言搜索模型", lang),
    )
    msearch.add_argument("keyword", help=T("search keyword", "搜索关键词", lang))
    msearch.add_argument("-f", "--field", choices=["name", "family", "lang", "all"],
                         default=None, metavar="FIELD",
                         help=T("fields to search: name, family, lang, all "
                                "(default: name+lang)",
                                "检索字段：name、family、lang、all"
                                "（默认 name+lang）", lang))
    _add_list_options(msearch, lang)
    _add_global_options(msearch, lang)
    msearch.set_defaults(func=_cmd_models_search)

    mdownload = _new_subparser(
        msub, "download", lang,
        aliases=["pull"],
        help=T("download model weights", "下载模型权重", lang),
        description=T("Download model weights (ModelScope first, HuggingFace fallback)",
                      "下载模型权重（ModelScope 优先，失败回退 HuggingFace）", lang),
    )
    mdownload.add_argument("name", help=T("model name", "模型名", lang))
    mdownload.add_argument("--source", choices=["auto", "modelscope", "huggingface"],
                           default=None, metavar="SRC",
                           help=T("download source (default: auto, "
                                  "or env OXOCR_DOWNLOAD_SOURCE)",
                                  "下载源（默认 auto，可用 OXOCR_DOWNLOAD_SOURCE 设置）", lang))
    mdownload.add_argument("--force", action="store_true",
                           help=T("re-download even if files exist",
                                  "文件已存在也重新下载", lang))
    _add_global_options(mdownload, lang)
    mdownload.set_defaults(func=_cmd_models_download)

    models.set_defaults(models_cmd="list", installed_only=False,
                        sort="released", reverse=False,
                        func=_cmd_models_list, model_dir=None, debug=False)

    # ---- version ----
    ver = _new_subparser(
        sub, "version", lang,
        help=T("show version", "显示版本", lang),
        description=T("Show version", "显示版本", lang),
    )
    ver.set_defaults(func=_cmd_version)

    return p


def _save_visualization(args, result) -> None:
    from PIL import Image

    from oxocr.visual import draw_boxes, draw_ocr, draw_items

    boxes = [it.box for it in result.items]
    txts = [it.text for it in result.items]
    scores = [it.score for it in result.items]
    if args.vis_mode == "box":
        vis = draw_boxes(args.image, boxes, scores)
    elif args.vis_mode == "ocr":
        vis = draw_ocr(args.image, boxes, txts, scores, font_path=args.font)
    else:
        vis = draw_items(args.image, result.items, font_path=args.font)
    Image.fromarray(vis[:, :, ::-1]).save(args.vis)


def _cmd_run(args) -> int:
    import oxocr

    overrides: dict = {}
    if args.cls:
        overrides["use_cls"] = True
    if args.text_score is not None:
        overrides["text_score"] = args.text_score
    if args.threads is not None:
        overrides["num_threads"] = args.threads

    try:
        result = oxocr.ocr(
            args.image,
            model=args.model,
            mode=args.mode,
            output=args.output,
            model_dir=args.model_dir,
            debug=args.debug,
            **overrides,
        )
    except oxocr.OxocrError as e:
        print(f"错误: {e}", file=sys.stderr)
        return 1

    # result.text 反映 --output 选定的渲染格式（text/json）
    print(result.text)

    if args.vis:
        try:
            _save_visualization(args, result)
        except (oxocr.OxocrError, OSError, ValueError) as e:
            print(f"错误: {e}", file=sys.stderr)
            return 1
        print(T(f"Visualization saved: {args.vis}",
                f"可视化已保存: {args.vis}", args.lang), file=sys.stderr)
    return 0


def _disp_width(text: str) -> int:
    # 中文等全角字符在终端占 2 列，按显示宽度对齐
    return sum(2 if unicodedata.east_asian_width(c) in "FW" else 1 for c in text)


# 长协议在表格中缩写显示，完整值见 models inspect
_LICENSE_SHORT = {"AI PUBS OPEN RAIL-M (modified)": "RAIL-M (modified)"}


def _pad(text: str, width: int, *, right: bool = False) -> str:
    fill = " " * max(0, width - _disp_width(text))
    return fill + text if right else text + fill


def _model_table_rows(args, zh: bool) -> list[dict]:
    import oxocr

    rows = []
    for info in oxocr.list_models(installed_only=args.installed_only,
                                  model_dir=args.model_dir):
        spec = oxocr.get_model_spec(info.name)
        # ASCII 标记：✓ 在 GBK 控制台/重定向下会抛 UnicodeEncodeError
        mark = "[V]" if info.installed else "[ ]"
        released = spec.raw_release_date.isoformat() if spec.raw_release_date else "-"
        size = f"{spec.size_bytes / 1048576:.1f}" if spec.size_bytes else "-"
        langs = spec.languages_chinese if zh else spec.languages
        shown = ", ".join(langs[:3]) + (", ..." if len(langs) > 3 else "")
        license = spec.license or "-"
        rows.append({"mark": mark, "name": info.name,
                     "released": released, "size": size,
                     "license": _LICENSE_SHORT.get(license, license),
                     "langs": shown or "-",
                     "date": spec.raw_release_date, "bytes": spec.size_bytes,
                     "spec": spec})
    return rows


def _sort_rows(rows: list[dict], field: str, reverse: bool) -> list[dict]:
    # released/size 自然方向为降序（最新/最大在前），name 为升序
    desc = field in ("released", "size")
    if reverse:
        desc = not desc
    if field == "name":
        return sorted(rows, key=lambda r: r["name"].lower(), reverse=desc)
    attr = "date" if field == "released" else "bytes"
    valued = [r for r in rows if r[attr]]
    missing = [r for r in rows if not r[attr]]
    # 两遍稳定排序：同值按名称升序兜底，不受 reverse 影响
    valued.sort(key=lambda r: r["name"].lower())
    valued.sort(key=lambda r: r[attr], reverse=desc)
    missing.sort(key=lambda r: r["name"].lower())
    return valued + missing


def _print_model_table(rows: list[dict], zh: bool) -> None:
    headers = (("名称", "开源日期", "大小(MB)", "开源协议", "支持语言") if zh else
               ("NAME", "RELEASED", "SIZE(MB)", "LICENSE", "LANGUAGES"))
    cols = ("name", "released", "size", "license", "langs")
    widths = [max([_disp_width(headers[i])] + [_disp_width(r[c]) for r in rows])
              for i, c in enumerate(cols)]
    gaps = ("  ", "  ", "   ", "  ", "")

    def line(cells: tuple[str, ...]) -> str:
        return "".join(_pad(c, widths[i], right=(i == 2)) + gaps[i]
                       for i, c in enumerate(cells)).rstrip()

    print("    " + line(headers))
    for r in rows:
        print(r["mark"] + " " + line(tuple(r[c] for c in cols)))


def _cmd_models_list(args) -> int:
    zh = args.lang == "zh"
    rows = _sort_rows(_model_table_rows(args, zh), args.sort, args.reverse)
    _print_model_table(rows, zh)
    return 0


def _match_spec(spec, keyword: str, field: str | None) -> bool:
    q = keyword.lower()
    if field is None:
        fields = ("name", "lang")
    elif field == "all":
        fields = ("name", "family", "lang")
    else:
        fields = (field,)
    for f in fields:
        if f == "name":
            if q in spec.name.lower() or q in spec.name_chinese.lower():
                return True
        elif f == "family":
            if q in spec.family.lower():
                return True
        # 语言代码前缀匹配（zh 命中 zh/zh_tw）；中文名子串；英文名按词前缀，
        # 避免 ar 误中 Devanagari、hi 误中 Chinese
        elif any(c.lower().startswith(q) for c in spec.languages) or any(
                q in n.lower() for n in spec.languages_chinese) or any(
                w.startswith(q) for n in spec.languages_english
                for w in re.split(r"[^a-z0-9]+", n.lower()) if w):
            return True
    return False


def _cmd_models_search(args) -> int:
    zh = args.lang == "zh"
    rows = [r for r in _model_table_rows(args, zh)
            if _match_spec(r["spec"], args.keyword, args.field)]
    if not rows:
        print(T("no matching models", "未找到匹配的模型", args.lang), file=sys.stderr)
        return 1
    _print_model_table(_sort_rows(rows, args.sort, args.reverse), zh)
    return 0


def _cmd_models_info(args) -> int:
    import oxocr
    from oxocr.registry.registry import resolve_model_dir

    try:
        spec = oxocr.get_model_spec(args.name)
    except oxocr.ModelNotFoundError as e:
        print(f"错误: {e}", file=sys.stderr)
        return 1

    model_dir = resolve_model_dir(args.model_dir)
    info = next(m for m in oxocr.list_models(model_dir=model_dir)
                if m.name == spec.name)
    cap = spec.capabilities
    flags = [k for k in ("detection", "recognition", "angle_classification",
                         "char_boxes", "own_preprocess", "own_postprocess")
             if getattr(cap, k)]
    arch = ", ".join(f"{k}={v}" for k, v in spec.model_architecture.items()) or "-"
    urls = "\n" + "\n".join(f"{'':<18}{u}" for u in spec.urls) if spec.urls else "-"
    rows = [
        ("Name", spec.name),
        ("Chinese name", spec.name_chinese or spec.name),
        ("Family", spec.family),
        ("Type", spec.model_type),
        ("Languages", ", ".join(spec.languages) or "-"),
        ("Languages (zh)", ", ".join(spec.languages_chinese) or "-"),
        ("Author", spec.author or "-"),
        ("Country", spec.author_country or "-"),
        ("Release date", spec.release_date.isoformat() if spec.release_date else "-"),
        ("License", spec.license or "-"),
        ("Commercial use", spec.commercial_use),
        ("Star rating", str(spec.star_rating)),
        ("Size", f"{spec.size_bytes / 1048576:.1f} MB" if spec.size_bytes else "-"),
        ("Architecture", arch),
        ("Installed", "yes" if info.installed else "no"),
        ("Downloadable", "yes" if info.downloadable else "no"),
        ("Model dir", str(model_dir)),
        ("Requires", ", ".join(spec.requires) or "-"),
        ("Capabilities", f"{', '.join(flags) or '-'} (backend: {cap.backend})"),
        ("URLs", urls),
    ]
    for label, value in rows:
        print(f"{label + ':':<18}{value}")
    print("Files:")
    for f in spec.files:
        mark = "V" if Path(info.paths[f.role]).is_file() else "X"
        size = f" ({f.size / 1048576:.1f} MB)" if f.size else ""
        print(f"  [{mark}] {f.role:6s} {f.filename}{size}")
    return 0


def _cmd_models_download(args) -> int:
    import oxocr

    try:
        paths = oxocr.download_model(
            args.name, model_dir=args.model_dir, force=args.force, source=args.source,
        )
    except oxocr.OxocrError as e:
        print(f"错误: {e}", file=sys.stderr)
        return 1
    print(T(f"Model {args.name} downloaded:", f"模型 {args.name} 下载完成:", args.lang))
    for role, path in paths.items():
        print(f"  [{role}] {path}")
    return 0


def _cmd_version(_args) -> int:
    print(_version_text())
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    lang = _detect_lang(argv)
    argv = _normalize_argv(argv)
    parser = _build_parser(lang=lang if lang in ("en", "zh") else "en")
    if lang not in ("en", "zh"):
        parser.error(f"argument --lang: invalid choice: {lang!r} (choose from 'en', 'zh')")
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 2
    return args.func(args) or 0


if __name__ == "__main__":
    raise SystemExit(main())
