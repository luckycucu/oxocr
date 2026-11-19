"""ONNX Runtime 会话薄封装。onnxruntime 在构造时才导入（保持包导入轻量）。"""
from __future__ import annotations

import math
import os
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from oxocr.errors import MissingDependencyError

__all__ = ["OrtSession", "resolve_num_threads"]


def resolve_num_threads(num_threads: int) -> int:
    if num_threads and num_threads > 0:
        return int(num_threads)
    return max(1, math.ceil((os.cpu_count() or 4) / 2))


class OrtSession:
    def __init__(
        self,
        model_path: str | Path,
        num_threads: int = 0,
        providers: Sequence[str] = ("CPUExecutionProvider",),
    ):
        try:
            import onnxruntime as ort
        except ImportError as e:
            raise MissingDependencyError("onnxruntime") from e
        threads = resolve_num_threads(num_threads)
        sess_opt = ort.SessionOptions()
        sess_opt.log_severity_level = 4
        sess_opt.enable_cpu_mem_arena = False
        sess_opt.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_opt.intra_op_num_threads = threads
        sess_opt.inter_op_num_threads = threads
        self.session = ort.InferenceSession(
            str(model_path), sess_options=sess_opt, providers=list(providers)
        )

    def __call__(self, input_data: np.ndarray) -> np.ndarray:
        input_name = self.session.get_inputs()[0].name
        return self.session.run(None, {input_name: input_data})[0]

    def metadata(self) -> dict[str, str]:
        return self.session.get_modelmeta().custom_metadata_map

    def have_key(self, key: str = "character") -> bool:
        return key in self.metadata()

    def get_character_list(self, key: str = "character") -> list[str]:
        return self.metadata()[key].splitlines()
