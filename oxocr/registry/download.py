"""模型下载与缓存目录。

逐文件下载到 model_dir（不走 hub 缓存）：ModelScope 优先，失败回退 HuggingFace，
source 参数 / OXOCR_DOWNLOAD_SOURCE 可强制单源。OXOCR_OFFLINE=1 禁止任何网络行为。
ModelScope 失败重试 3 次；HuggingFace 重试次数由库内置（5 次，无公开配置项）。
现阶段不校验 sha256：仓库文件可能更新，包发版后硬编码哈希无法跟随。
HuggingFace 按 config.json 的请求计下载量：每个源实际下载权重前，都会向该源
仓库根目录额外请求一次 config.json（尽力而为，失败不影响下载）。
"""
from __future__ import annotations

import hashlib
import logging
import os
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from oxocr.errors import ChecksumMismatchError, MissingDependencyError, ModelDownloadError

if TYPE_CHECKING:
    from oxocr.models.base import ModelSpec

__all__ = ["cache_dir", "is_offline", "download_model", "verify_sha256", "SOURCES"]

logger = logging.getLogger("oxocr.download")

SOURCES = ("auto", "modelscope", "huggingface")


def cache_dir() -> Path:
    """下载缓存与用户纠错规则目录：OXOCR_CACHE_DIR > ~/.oxocr。"""
    env = os.environ.get("OXOCR_CACHE_DIR")
    return Path(env) if env else Path.home() / ".oxocr"


def is_offline() -> bool:
    return os.environ.get("OXOCR_OFFLINE", "") == "1"


def verify_sha256(path: str | Path, expected: str) -> None:
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            sha.update(chunk)
    actual = sha.hexdigest()
    if actual.lower() != expected.lower():
        raise ChecksumMismatchError(str(path), expected, actual)


def _resolve_source(source: str | None) -> str:
    from oxocr.errors import UnknownParameterError

    value = source if source is not None else os.environ.get("OXOCR_DOWNLOAD_SOURCE", "auto")
    value = str(value).strip().lower()
    if value not in SOURCES:
        raise UnknownParameterError("source", allowed=list(SOURCES), context="download_model")
    return value


def _download_modelscope(repo: str, filename: str, directory: Path, force: bool) -> None:
    # 库在导入时读取该环境变量（默认 5），须在首次导入前设置；尊重用户显式值
    os.environ.setdefault("MODELSCOPE_DOWNLOAD_MAX_RETRIES", "3")
    try:
        from modelscope_hub import HubApi
    except ImportError as e:
        raise MissingDependencyError("modelscope-hub") from e
    HubApi().download_file(repo, "model", filename, local_dir=directory, force=force)


def _download_huggingface(repo: str, filename: str, directory: Path, force: bool) -> None:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise MissingDependencyError("huggingface-hub") from e
    hf_hub_download(repo_id=repo, filename=filename, local_dir=directory, force_download=force)


_DOWNLOADERS = {"modelscope": _download_modelscope, "huggingface": _download_huggingface}


def _fetch_config(src: str, repo: str, directory: Path) -> None:
    # force=True：本地已有 config.json 时也要发出网络请求，否则 hub 不计下载量
    try:
        _DOWNLOADERS[src](repo, "config.json", directory, True)
    except Exception as e:
        logger.debug("从 %s 获取 config.json 失败（不影响模型下载）: %s", src, e)


def _resolve_download_spec(name: str) -> ModelSpec:
    """普通模型走 registry；ppocr_cls 为下载 bundle（全部方向分类权重，非推理模型）。"""
    if name == "ppocr_cls":
        from oxocr.models.rapid_ocr.specs import CLS_BUNDLE_SPEC

        return CLS_BUNDLE_SPEC
    from oxocr.registry.registry import resolve_spec

    return resolve_spec(name)


def download_model(
    name: str,
    *,
    model_dir: str | Path | None = None,
    force: bool = False,
    progress: Callable[[str, int, int], None] | None = None,
    source: str | None = None,
) -> dict[str, Path]:
    """确保模型文件本地齐备，返回 role → 路径。

    name 为模型名，或下载 bundle "ppocr_cls"（一次下载全部方向分类权重）。
    source: "auto"（默认，ModelScope 优先、失败回退 HuggingFace）| "modelscope" |
    "huggingface"；None 时读 OXOCR_DOWNLOAD_SOURCE。已存在的文件跳过，force 全部重下。
    """
    from oxocr.config import ModelOptions
    from oxocr.registry.registry import resolve_files, resolve_model_dir

    spec = _resolve_download_spec(name)
    directory = resolve_model_dir(model_dir)
    source = _resolve_source(source)

    targets = [f for f in spec.files if force or not (directory / f.filename).is_file()]
    if not targets:
        return resolve_files(spec, directory, ModelOptions())
    if is_offline():
        raise ModelDownloadError(
            f"OXOCR_OFFLINE=1 已禁止网络下载，模型 {name!r} 缺少文件，请手动放置到 {directory}"
        )

    repos = {"modelscope": spec.modelscope_repo, "huggingface": spec.huggingface_repo}
    order = ("modelscope", "huggingface") if source == "auto" else (source,)
    usable = [s for s in order if repos[s]]
    if not usable:
        raise ModelDownloadError(
            f"模型 {name!r} 未配置下载仓库，请手动下载文件并放置到 {directory}"
        )

    errors: list[str] = []
    for src in usable:
        pending = [f for f in targets if force or not (directory / f.filename).is_file()]
        if not pending:
            break
        _fetch_config(src, repos[src], directory)
        try:
            for f in pending:
                if progress:
                    progress(f.filename, 0, f.size or 0)
                _DOWNLOADERS[src](repos[src], f.filename, directory, force)
                if progress:
                    progress(f.filename, f.size or 0, f.size or 0)
        except Exception as e:
            errors.append(f"[{src}] {e}")
            logger.warning("从 %s 下载模型 %s 失败: %s", src, name, e)

    missing = [f for f in targets if not (directory / f.filename).is_file()]
    if missing:
        detail = "\n".join(f"  {err}" for err in errors)
        raise ModelDownloadError(
            f"模型 {name!r} 下载失败（源: {', '.join(usable)}）：\n{detail}\n"
            f"请检查网络后重试，或手动下载文件放置到 {directory}"
        )

    return resolve_files(spec, directory, ModelOptions())
