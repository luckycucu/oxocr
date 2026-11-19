"""执行器协议与两种实现。

- InProcessExecutor：默认。resident=True 从常驻池借会话，close() 只归还句柄不卸载；
  resident=False 持有私有会话，close() 时 unload + gc。
- SubprocessExecutor：不驻留特例。每次 run 拉起一次性子进程，退出即由 OS 回收内存。
"""
from __future__ import annotations

import gc
import json
import os
import struct
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

from oxocr.config import ModelOptions, SubprocessOptions
from oxocr.errors import SubprocessError, SubprocessTimeoutError
from oxocr.models.base import InferRequest, ModelOutput, ModelSpec
from oxocr.runtime.images import to_png_bytes
from oxocr.runtime.pool import ModelPool, ModelSession

RESULT_PREFIX = "OXOCR_RESULT:"

__all__ = ["Executor", "InProcessExecutor", "SubprocessExecutor"]


class Executor(Protocol):
    def run(self, image, request: InferRequest) -> ModelOutput: ...

    def close(self) -> None: ...

    @property
    def is_loaded(self) -> bool: ...


class InProcessExecutor:
    def __init__(
        self,
        spec: ModelSpec,
        files: Mapping[str, Path],
        options: ModelOptions,
        model_dir: str | Path,
        resident: bool = True,
    ):
        self.spec = spec
        self.resident = resident
        if resident:
            self._session = ModelPool.instance().get_or_create(
                spec, files, options, model_dir
            )
        else:
            self._session = ModelSession(spec, files, options, model_dir)

    @property
    def is_loaded(self) -> bool:
        return self._session.is_loaded

    def warm(self) -> None:
        """显式预热：触发 get-or-load（常驻池共享）。"""
        self._session.acquire_load()

    def run(self, image, request: InferRequest) -> ModelOutput:
        load = self._session.acquire_load()
        output = self._session.infer(image, request)
        output.elapse["load"] = load
        return output

    def close(self) -> None:
        if self.resident:
            return  # 会话归池所有，进程结束才释放
        self._session.unload()
        gc.collect()

    def __enter__(self) -> InProcessExecutor:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


class SubprocessExecutor:
    def __init__(
        self,
        spec: ModelSpec,
        files: Mapping[str, Path],
        options: ModelOptions,
        subprocess_options: SubprocessOptions | None = None,
    ):
        self.spec = spec
        self.files = {role: str(path) for role, path in files.items()}
        self.options = options
        self.subprocess_options = subprocess_options or SubprocessOptions()

    @property
    def is_loaded(self) -> bool:
        return False

    def warm(self) -> None:
        return None  # 子进程后端无常驻会话；仅校验由构造时完成

    def _build_command(self) -> tuple[list[str], dict[str, str] | None]:
        so = self.subprocess_options
        if getattr(sys, "frozen", False):
            cmd = [sys.executable, so.frozen_flag]
            env: dict[str, str] | None = dict(os.environ, PYTHONUTF8="1")
            return cmd, env
        python = so.python or sys.executable
        cmd = [python, "-X", "utf8", "-m", "oxocr.runtime.worker"]
        # 保证未安装场景（src 布局）子进程也能 import oxocr
        pkg_root = str(Path(__file__).resolve().parent.parent.parent)
        env = dict(os.environ)
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = pkg_root + (os.pathsep + existing if existing else "")
        env["PYTHONUTF8"] = "1"
        return cmd, env

    @staticmethod
    def _frame(header: dict[str, Any], png: bytes) -> bytes:
        raw = json.dumps(header, ensure_ascii=True).encode("utf-8")
        return struct.pack(">I", len(raw)) + raw + png

    @staticmethod
    def _parse_output(stdout_b: bytes) -> dict[str, Any]:
        result_line = None
        for line in stdout_b.decode("utf-8", errors="replace").splitlines():
            if line.startswith(RESULT_PREFIX):
                result_line = line[len(RESULT_PREFIX):]
        if result_line is None:
            raise SubprocessError("OCR 子进程未返回结果行")
        return json.loads(result_line)

    def run(self, image, request: InferRequest) -> ModelOutput:
        header = {
            "spec": self.spec.to_dict(),
            "files": self.files,
            "options": self.options.to_dict(),
            "request": request.to_dict(),
        }
        payload = self._frame(header, to_png_bytes(image))
        cmd, env = self._build_command()
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=creationflags,
                env=env,
            )
        except OSError as e:
            raise SubprocessError(f"无法启动 OCR 子进程: {e}") from e
        try:
            out_b, err_b = proc.communicate(payload, timeout=self.subprocess_options.timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()
            raise SubprocessTimeoutError(self.subprocess_options.timeout) from None
        if proc.returncode != 0:
            detail = err_b.decode("utf-8", errors="replace").strip()[-800:]
            raise SubprocessError(
                f"OCR 子进程异常退出 (code={proc.returncode})", detail, proc.returncode
            )
        data = self._parse_output(out_b)
        output = ModelOutput.from_dict(data)
        return output

    def close(self) -> None:
        return None

    def __enter__(self) -> SubprocessExecutor:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
