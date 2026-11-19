"""oxocr 异常类型。所有异常携带可操作的修复提示。"""
from __future__ import annotations

__all__ = [
    "OxocrError",
    "ModelNotFoundError",
    "ModelFileMissingError",
    "ModelDownloadError",
    "ChecksumMismatchError",
    "MissingDependencyError",
    "InvalidImageError",
    "UnknownParameterError",
    "InferenceError",
    "SubprocessError",
    "SubprocessTimeoutError",
    "StageError",
    "PluginNotFoundError",
]


class OxocrError(Exception):
    """oxocr 所有异常的基类。"""


class ModelNotFoundError(OxocrError):
    def __init__(self, name: str, available: list[str] | None = None):
        self.name = name
        self.available = available or []
        hint = f"可用模型: {', '.join(self.available)}" if self.available else "尚无注册模型"
        super().__init__(f"未找到模型 {name!r}。{hint}")


class ModelFileMissingError(OxocrError):
    def __init__(self, model: str, missing: dict[str, str]):
        self.model = model
        self.missing = missing
        lines = "\n".join(f"  [{role}] {path}" for role, path in missing.items())
        super().__init__(
            f"模型 {model!r} 缺少本地文件：\n{lines}\n"
            "请将模型文件放置到上述路径（或经 model_dir / OXOCR_MODEL_DIR 指定目录）。"
        )


class ModelDownloadError(OxocrError):
    def __init__(self, message: str, url: str | None = None, sha256: str | None = None):
        self.url = url
        self.sha256 = sha256
        detail = message
        if url:
            detail += f"（URL: {url}）"
        super().__init__(detail)


class ChecksumMismatchError(ModelDownloadError):
    def __init__(self, path: str, expected: str, actual: str):
        self.path = path
        super().__init__(
            f"文件校验失败: {path}\n  期望 sha256: {expected}\n  实际 sha256: {actual}"
        )


class MissingDependencyError(OxocrError):
    def __init__(self, package: str, extra: str | None = None):
        self.package = package
        hint = f"pip install oxocr[{extra}]" if extra else f"pip install {package}"
        super().__init__(f"缺少依赖 {package!r}，请执行: {hint}")


class InvalidImageError(OxocrError):
    """空图 / 无法解码的输入。"""


class UnknownParameterError(OxocrError):
    def __init__(self, name: str, allowed: list[str] | None = None, context: str = ""):
        self.name = name
        self.allowed = allowed or []
        where = f"（{context}）" if context else ""
        hint = f"可用参数: {', '.join(self.allowed)}" if self.allowed else ""
        super().__init__(f"未知参数 {name!r}{where}。{hint}".rstrip("。"))


class InferenceError(OxocrError):
    """模型推理失败。"""


class SubprocessError(OxocrError):
    def __init__(self, message: str, stderr_tail: str = "", returncode: int | None = None):
        self.stderr_tail = stderr_tail
        self.returncode = returncode
        detail = message
        if stderr_tail:
            detail += f"\n子进程 stderr（尾部）:\n{stderr_tail}"
        super().__init__(detail)


class SubprocessTimeoutError(SubprocessError):
    def __init__(self, timeout: float):
        self.timeout = timeout
        super().__init__(f"OCR 子进程超时（>{timeout}s），已终止")


class StageError(OxocrError):
    """前/后处理插件异常（on_error="raise" 时抛出）。"""

    def __init__(self, stage: str, kind: str, cause: BaseException):
        self.stage = stage
        self.kind = kind
        self.cause = cause
        super().__init__(f"{kind} 插件 {stage!r} 执行失败: {cause}")


class PluginNotFoundError(OxocrError):
    def __init__(self, name: str, group: str, available: list[str] | None = None):
        self.name = name
        self.group = group
        hint = f"可用插件: {', '.join(available)}" if available else ""
        super().__init__(f"未注册的 {group} 插件 {name!r}。{hint}".rstrip("。"))
