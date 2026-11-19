"""LatexOcrModel：pix2tex 系 ViT encoder + 自回归 decoder 公式识别（整图公式 → LaTeX）。

无检测模型：box 恒为送入图整框，核心前处理的几何改动由现有仿射逆映射统一还原。
比 pix2text MFR 多一个 image_resizer（迭代预测宽度桶）；decoder 三输入（含 bool mask）、
无 KV-cache，贪心解码每步喂全序列（O(n²)）。参考的 temperature=1e-5 采样实质是贪心，
此处用确定性 argmax（域取词表宽 1175，logits 宽 8000），可复现且与参考输出一致。
bos/eos 从 tokenizer.json 的 added_tokens 按名读取，不硬编码。
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import numpy as np
from PIL import Image

from oxocr.errors import InferenceError
from oxocr.models.base import Capabilities, InferRequest, ModelOutput, OcrModel
from oxocr.models.rapid_latex_ocr.preproc import resize_loop
from oxocr.models.pix2text.tokenizer import MfrTokenizer
from oxocr.runtime.session import OrtSession
from oxocr.types import Box, TextItem

__all__ = ["LatexOcrModel"]

_MAX_SEQ_LEN = 512  # decoder 位置编码上限，超出时滑窗取末 512（对齐参考）


def _read_bos_eos(path: str | Path) -> tuple[int, int]:
    """从 tokenizer.json 的 added_tokens 按内容名读 ([BOS], [EOS]) 的 id。"""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    by_content = {t["content"]: t["id"] for t in data.get("added_tokens", ())}
    bos, eos = by_content.get("[BOS]"), by_content.get("[EOS]")
    if bos is None or eos is None:
        raise InferenceError(
            f"{path} 缺少 [BOS]/[EOS] added_tokens，无法确定解码起止 token"
        )
    return int(bos), int(eos)


def _compact_latex(s: str) -> str:
    """去除 LaTeX 多余空白（参考 main.post_process 原样平移，模型内部输出整理）。"""
    text_reg = r"(\\(operatorname|mathrm|text|mathbf)\s?\*? {.*?})"
    letter = "[a-zA-Z]"
    noletter = r"[\W_^\d]"
    names = [x[0].replace(" ", "") for x in re.findall(text_reg, s)]
    s = re.sub(text_reg, lambda match: str(names.pop(0)), s)
    news = s
    while True:
        s = news
        news = re.sub(rf"(?!\\ )({noletter})\s+?({noletter})", r"\1\2", s)
        news = re.sub(rf"(?!\\ )({noletter})\s+?({letter})", r"\1\2", news)
        news = re.sub(rf"({letter})\s+?({noletter})", r"\1\2", news)
        if news == s:
            break
    return s


class LatexOcrModel(OcrModel):
    family = "latex_ocr"
    capabilities = Capabilities(
        detection=False, recognition=True, own_preprocess=True, own_postprocess=True,
        backend="onnxruntime",
    )

    def __init__(self, spec, files, options):
        super().__init__(spec, files, options)
        self._resizer: OrtSession | None = None
        self._encoder: OrtSession | None = None
        self._decoder: OrtSession | None = None
        self._tokenizer: MfrTokenizer | None = None
        self._bos = 0
        self._eos = 0
        self._vocab = 0
        self._max_new_tokens = 512
        self._repeat_limit = 12

    def load(self) -> None:
        p = self.params()
        threads = self.options.num_threads
        providers = self.options.providers
        self._resizer = OrtSession(self.files["resizer"], num_threads=threads, providers=providers)
        self._encoder = OrtSession(self.files["encoder"], num_threads=threads, providers=providers)
        self._decoder = OrtSession(self.files["decoder"], num_threads=threads, providers=providers)
        self._tokenizer = MfrTokenizer(self.files["tokenizer"])
        self._bos, self._eos = _read_bos_eos(self.files["tokenizer"])
        self._vocab = self._tokenizer.vocab_size
        self._max_new_tokens = max(1, int(p["max_new_tokens"]))
        self._repeat_limit = max(2, int(p["repeat_limit"]))
        self._loaded = True

    def unload(self) -> None:
        self._resizer = None
        self._encoder = None
        self._decoder = None
        self._tokenizer = None
        self._loaded = False

    def _greedy_decode(self, ctx: np.ndarray) -> tuple[list[int], list[float], bool]:
        """返回 (含起始位的 token 序列, 各文本 token 的 log 概率, 是否退化)。

        decoder 三输入（x/mask/context），走原生 session.run。argmax 域取词表宽
        （logits 宽 8000，但 id≥vocab 无词表映射）；mask 与 x 超 512 时滑窗取末段。
        """
        ids = [self._bos]
        mask = np.full((1, 1), True, dtype=bool)
        logps: list[float] = []
        repeats = 1
        for _ in range(self._max_new_tokens):
            x = np.array([ids[-_MAX_SEQ_LEN:]], dtype=np.int64)
            mask = mask[:, -_MAX_SEQ_LEN:]
            logits = self._decoder.session.run(
                None, {"x": x, "mask": mask, "context": ctx}
            )[0]
            last = logits[0, -1]
            next_id = int(last[: self._vocab].argmax())
            if next_id == self._eos:
                return ids, logps, False
            m = float(last.max())
            lse = m + float(np.log(np.exp(last - m).sum()))
            logps.append(float(last[next_id]) - lse)
            repeats = repeats + 1 if next_id == ids[-1] else 1
            if repeats >= self._repeat_limit:
                return ids, logps, True
            ids.append(next_id)
            mask = np.pad(mask, [(0, 0), (0, 1)], constant_values=True)
        return ids, logps, True  # 达到长度上限仍未 eos → 退化

    def infer(self, image: np.ndarray, request: InferRequest) -> ModelOutput:
        if not self._loaded:
            raise InferenceError(f"模型 {self.spec.name} 尚未 load()")
        if not request.use_rec:
            return ModelOutput(items=[], elapse={})
        t0 = time.perf_counter()
        rgb = np.ascontiguousarray(image[:, :, ::-1])
        x = resize_loop(Image.fromarray(rgb), self._resizer.session)
        ctx = self._encoder.session.run(None, {"input": x})[0]
        t_enc = time.perf_counter() - t0
        t1 = time.perf_counter()
        ids, logps, degenerate = self._greedy_decode(ctx)
        t_dec = time.perf_counter() - t1

        latex = _compact_latex(self._tokenizer.decode(ids[1:]))
        score = float(np.exp(np.mean(logps))) if logps else 0.0
        if degenerate:
            score = 0.0
        meta = {
            "degenerate": degenerate,
            "num_tokens": len(ids) - 1,
            "enc": t_enc,
            "dec": t_dec,
            "input_hw": [int(x.shape[-2]), int(x.shape[-1])],
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
