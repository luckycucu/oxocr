"""MfrModel：TrOCR VisionEncoderDecoder 公式识别（整图公式 → LaTeX）。

无检测模型：box 恒为送入图整框，核心前处理的几何改动（如 pad_border 平移）
由现有仿射逆映射统一还原。decoder 是 use_cache=false 的 optimum 导出
（无 KV-cache），贪心解码每步喂全序列（O(n²)），退化防护（连续重复 / 超长）
沿用 OCR-buddy 的验证阈值。start/eos 从各模型目录 config.json 读取——
pix2text-mfr-1.5 起始 token 是 1，1.0 与量化版是 2，不可硬编码。
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from oxocr.errors import InferenceError
from oxocr.models.base import Capabilities, InferRequest, ModelOutput, OcrModel
from oxocr.models.pix2text.tokenizer import MfrTokenizer
from oxocr.runtime.session import OrtSession
from oxocr.types import Box, TextItem

__all__ = ["MfrModel"]

# decoder max_position_embeddings=1024（含起始位），生成 token 数上限 1023
_MAX_POSITION_TOKENS = 1023


def _read_generation_config(path: str | Path) -> tuple[int, int]:
    """从 config.json 读 (decoder_start_token_id, eos_token_id)。"""
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    dec = cfg.get("decoder") or {}
    start = cfg.get("decoder_start_token_id", dec.get("decoder_start_token_id"))
    eos = cfg.get("eos_token_id", dec.get("eos_token_id"))
    if start is None or eos is None:
        raise InferenceError(
            f"{path} 缺少 decoder_start_token_id / eos_token_id，无法确定解码起止 token"
        )
    return int(start), int(eos)


def _preprocess(image: np.ndarray) -> np.ndarray:
    """BGR uint8 → [1,3,384,384] float32。

    对齐 preprocessor_config.json：384×384 非保比缩放（resample=3，bicubic）、
    v/255、(v-0.5)/0.5、RGB。encoder 输出序列长 578 是静态维度，输入尺寸不可变。
    """
    from PIL import Image

    rgb = np.ascontiguousarray(image[:, :, ::-1])
    pil = Image.fromarray(rgb).resize((384, 384), Image.Resampling.BICUBIC)
    arr = np.asarray(pil, dtype=np.float32) / 255.0
    arr = (arr - 0.5) / 0.5
    return arr.transpose(2, 0, 1)[np.newaxis, ...]


class MfrModel(OcrModel):
    family = "pix2text"
    capabilities = Capabilities(
        detection=False, recognition=True, own_postprocess=True, backend="onnxruntime"
    )

    def __init__(self, spec, files, options):
        super().__init__(spec, files, options)
        self._encoder: OrtSession | None = None
        self._decoder: OrtSession | None = None
        self._tokenizer: MfrTokenizer | None = None
        self._start_id = 0
        self._eos_id = 0
        self._max_new_tokens = 256
        self._repeat_limit = 12

    def load(self) -> None:
        p = self.params()
        threads = self.options.num_threads
        providers = self.options.providers
        self._encoder = OrtSession(
            self.files["encoder"], num_threads=threads, providers=providers
        )
        self._decoder = OrtSession(
            self.files["decoder"], num_threads=threads, providers=providers
        )
        self._tokenizer = MfrTokenizer(self.files["tokenizer"])
        self._start_id, self._eos_id = _read_generation_config(self.files["config"])
        self._max_new_tokens = max(1, min(int(p["max_new_tokens"]), _MAX_POSITION_TOKENS))
        self._repeat_limit = max(2, int(p["repeat_limit"]))
        self._loaded = True

    def unload(self) -> None:
        self._encoder = None
        self._decoder = None
        self._tokenizer = None
        self._loaded = False

    def _greedy_decode(self, ehs: np.ndarray) -> tuple[list[int], list[float], bool]:
        """返回 (含起始位的 token 序列, 各文本 token 的 log 概率, 是否退化)。

        decoder 双输入，走原生 session.run（OrtSession.__call__ 只支持单输入）。
        词表宽度随权重不同（1868/1200），argmax 域以 logits 实际宽度为准。
        """
        ids = [self._start_id]
        logps: list[float] = []
        repeats = 1
        for _ in range(self._max_new_tokens):
            logits = self._decoder.session.run(
                None,
                {
                    "input_ids": np.array([ids], dtype=np.int64),
                    "encoder_hidden_states": ehs,
                },
            )[0]
            last = logits[0, -1]
            next_id = int(last.argmax())
            if next_id == self._eos_id:
                return ids, logps, False
            m = float(last.max())
            lse = m + float(np.log(np.exp(last - m).sum()))
            logps.append(float(last[next_id]) - lse)
            repeats = repeats + 1 if next_id == ids[-1] else 1
            if repeats >= self._repeat_limit:
                return ids, logps, True
            ids.append(next_id)
        return ids, logps, True  # 达到长度上限仍未 eos → 退化

    def infer(self, image: np.ndarray, request: InferRequest) -> ModelOutput:
        if not self._loaded:
            raise InferenceError(f"模型 {self.spec.name} 尚未 load()")
        if not request.use_rec:
            # 公式模型无检测：仅检测模式无条目可给
            return ModelOutput(items=[], elapse={})
        t0 = time.perf_counter()
        ehs = self._encoder.session.run(None, {"pixel_values": _preprocess(image)})[0]
        t_enc = time.perf_counter() - t0
        t1 = time.perf_counter()
        ids, logps, degenerate = self._greedy_decode(ehs)
        t_dec = time.perf_counter() - t1

        latex = self._tokenizer.decode(ids[1:])
        score = float(np.exp(np.mean(logps))) if logps else 0.0
        if degenerate:
            score = 0.0
        meta = {
            "degenerate": degenerate,
            "num_tokens": len(ids) - 1,
            "enc": t_enc,
            "dec": t_dec,
        }
        elapse = {"rec": t_enc + t_dec}
        if not latex.strip() or score < request.text_score:
            return ModelOutput(items=[], elapse=elapse, meta=meta)
        h, w = image.shape[:2]
        return ModelOutput(
            items=[
                TextItem(
                    text=latex,
                    score=score,
                    box=Box.from_rect(0.0, 0.0, float(w), float(h)),
                    meta={"degenerate": degenerate, "num_tokens": len(ids) - 1},
                )
            ],
            elapse=elapse,
            meta=meta,
        )
