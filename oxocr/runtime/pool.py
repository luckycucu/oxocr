"""进程级常驻模型池 ModelPool。

默认语义：**模型只加载一次，驻留至进程结束；不驻留是需要显式选择的特例。**
- 无 LRU、无超时驱逐、无容量上限——进程活着，会话就在。
- 同 key 全局单实例；get-or-load 加锁，并发只建一个。
- 同一会话的 infer 由 ModelSession 串行化（模型内部可能非线程安全）。
- 需要真并行：不同 options key 多实例，或子进程后端。
"""
from __future__ import annotations

import atexit
import os
import threading
import time
from collections.abc import Mapping
from pathlib import Path

from oxocr.config import ModelOptions
from oxocr.models.base import InferRequest, ModelOutput, ModelSpec, OcrModel
from oxocr.registry.registry import load_impl
from oxocr.types import LoadedModelInfo

__all__ = ["ModelPool", "ModelSession", "pool_instance"]


def _key(name: str, model_dir: str | Path, options: ModelOptions) -> tuple:
    return (name, os.path.abspath(str(model_dir)), options.fingerprint())


class ModelSession:
    """常驻会话句柄：持有模型实例与串行化/生命周期锁。"""

    def __init__(
        self,
        spec: ModelSpec,
        files: Mapping[str, Path],
        options: ModelOptions,
        model_dir: str | Path,
    ):
        self.spec = spec
        self.files = dict(files)
        self.options = options
        self.model_dir = str(model_dir)
        self.key = _key(spec.name, model_dir, options)
        self.model: OcrModel | None = None
        self.loaded_at = 0.0
        self.infer_count = 0
        self._state_lock = threading.Lock()  # 保护 model 构造与 load/unload
        self._run_lock = threading.Lock()  # 串行化 infer
        self._cond = threading.Condition(self._state_lock)
        self._busy = 0

    def _build(self) -> OcrModel:
        if self.model is None:
            impl = load_impl(self.spec)
            self.model = impl(self.spec, self.files, self.options)
        self.model.validate_options(self.options)
        return self.model

    def acquire_load(self) -> float:
        """get-or-load，返回本次调用实际产生的加载耗时（已加载则 ≈ 0）。"""
        with self._cond:
            model = self._build()
            if model.is_loaded:
                return 0.0
            t0 = time.perf_counter()
            model.load()
            self.loaded_at = time.time()
            return time.perf_counter() - t0

    def infer(self, image, request: InferRequest) -> ModelOutput:
        self.acquire_load()
        with self._run_lock:
            with self._cond:
                self._busy += 1
            try:
                assert self.model is not None
                output = self.model.infer(image, request)
                self.infer_count += 1
                return output
            finally:
                with self._cond:
                    self._busy -= 1
                    self._cond.notify_all()

    @property
    def is_loaded(self) -> bool:
        return self.model is not None and self.model.is_loaded

    def unload(self) -> bool:
        """等待进行中的推理完成后释放会话。"""
        with self._cond:
            while self._busy > 0:
                self._cond.wait()
            if self.model is not None and self.model.is_loaded:
                self.model.unload()
                return True
            return False

    def info(self) -> LoadedModelInfo:
        return LoadedModelInfo(
            name=self.spec.name,
            options_key=self.key[2],
            loaded_at=self.loaded_at,
            infer_count=self.infer_count,
        )


class ModelPool:
    _instance: ModelPool | None = None
    _instance_lock = threading.Lock()

    def __init__(self) -> None:
        self._sessions: dict[tuple, ModelSession] = {}
        self._lock = threading.Lock()

    @classmethod
    def instance(cls) -> ModelPool:
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = ModelPool()
                atexit.register(cls._instance._atexit_release)
            return cls._instance

    def _atexit_release(self) -> None:
        try:
            self.unload_all()
        except Exception:
            pass

    def get_or_create(
        self,
        spec: ModelSpec,
        files: Mapping[str, Path],
        options: ModelOptions,
        model_dir: str | Path,
    ) -> ModelSession:
        """进程级 get-or-load：首次命中惰性建会话，之后复用同一实例。"""
        key = _key(spec.name, model_dir, options)
        with self._lock:
            session = self._sessions.get(key)
            if session is None:
                session = ModelSession(spec, files, options, model_dir)
                self._sessions[key] = session
            return session

    def acquire(
        self,
        spec: ModelSpec,
        files: Mapping[str, Path],
        options: ModelOptions,
        model_dir: str | Path,
    ) -> OcrModel:
        session = self.get_or_create(spec, files, options, model_dir)
        session.acquire_load()
        return session.model  # type: ignore[return-value]

    def release(self, session: ModelSession) -> None:
        """归还句柄，不 unload（会话归池所有）。"""

    def unload(
        self,
        name: str,
        *,
        model_dir: str | Path | None = None,
        options: ModelOptions | None = None,
    ) -> int:
        with self._lock:
            targets = [
                (key, session)
                for key, session in self._sessions.items()
                if key[0] == name
                and (model_dir is None or key[1] == os.path.abspath(str(model_dir)))
                and (options is None or key[2] == options.fingerprint())
            ]
            for key, _ in targets:
                del self._sessions[key]
        released = 0
        for _, session in targets:
            if session.unload():
                released += 1
        return released

    def unload_all(self) -> int:
        with self._lock:
            sessions = list(self._sessions.values())
            self._sessions.clear()
        return sum(1 for s in sessions if s.unload())

    def snapshot(self) -> list[LoadedModelInfo]:
        with self._lock:
            return [s.info() for s in self._sessions.values() if s.is_loaded]


def pool_instance() -> ModelPool:
    return ModelPool.instance()
