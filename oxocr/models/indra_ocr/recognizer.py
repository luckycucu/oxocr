"""单识别器 CTC 贪心 + LM 门控束搜索（indra pipeline.recognize 的单组版）。

每模型固定一个识别器，不做运行时语种路由。裁剪窗口由 rec ONNX 静态高度 ÷
rec_height 推导并校验：base 读高窗（行 + 上下页面，window=3 → 96px），
组识别器读行带（window=1 → 32px）。score = CTC 保留帧概率算术平均；
LM 改写文本但 score 不变（同示例）。低于 MIN_CONF 的行返回空文本。
"""

from __future__ import annotations

import numpy as np

from oxocr.errors import InferenceError
from oxocr.models.indra_ocr.lm import NgramLM, beam_search
from oxocr.models.indra_ocr.preproc import crop_line, is_dark_line, pad_batch, width_batches
from oxocr.models.indra_ocr.vocab import Vocab
from oxocr.runtime.session import OrtSession

__all__ = ["LineRecognizer", "MIN_CONF"]

MIN_CONF = 0.5  # 低于此置信度的行无文本（空表单字段/勾选框/横线上的噪声读法）
LM_GATE = 0.9
LM_ALPHA = 0.5
LM_BETA = 1.5
LM_BEAM = 8
LM_CAND_FLOOR = 3e-3
OUT_GREEDY = ["ids", "conf", "lengths"]
OUT_LM = ["blank", "topv", "topi"]


class LineRecognizer:
    def __init__(
        self,
        rec_path,
        meta: dict,
        group: str,
        *,
        lm_path=None,
        batch_num: int = 64,
        num_threads: int = 0,
        providers=("CPUExecutionProvider",),
    ):
        self.session = OrtSession(rec_path, num_threads=num_threads, providers=providers)
        self.group = group
        self.height = int(meta["rec_height"])
        self.window = self._derive_window(meta, group)
        vocab_key = "vocab" if group == "base" else f"vocab_{group}"
        self.vocab = Vocab.from_json(meta[vocab_key])
        self.batch_num = int(batch_num)
        self.lm: NgramLM | None = None
        self.alpha, self.beta, self.gate = LM_ALPHA, LM_BETA, LM_GATE
        out_names = {o.name for o in self.session.session.get_outputs()}
        if lm_path is not None and set(OUT_LM) <= out_names:
            self.lm = NgramLM(lm_path)
            p = meta.get("lm_params", {}).get(group, {})
            self.alpha = float(p.get("alpha", LM_ALPHA))
            self.beta = float(p.get("beta", LM_BETA))
            self.gate = float(p.get("gate", LM_GATE))

    def _derive_window(self, meta: dict, group: str) -> int:
        shape = self.session.session.get_inputs()[0].shape
        h_dim = shape[2]
        expected = int(meta.get("rec_window", 1)) if group == "base" else 1
        if isinstance(h_dim, int):
            window = h_dim // self.height
            if window < 1 or window * self.height != h_dim:
                raise InferenceError(
                    f"rec 输入高度 {h_dim} 不是 rec_height {self.height} 的整数倍"
                )
        else:
            window = expected
        if window != expected:
            raise InferenceError(
                f"{self.group} 识别器裁剪窗口应为 {expected}（{expected * self.height}px），"
                f"权重给出 {window}；识别器与权重不可混用"
            )
        return window

    def __call__(
        self, gray: np.ndarray, boxes, *, use_lm: bool = True
    ) -> list[tuple[str, float]]:
        """每行 (text, conf)；conf < MIN_CONF 的行 text=""。boxes: [N, 4+]。"""
        if len(boxes) == 0:
            return []
        # 暗行判定每行只做一次，高窗与行带两份裁剪共享
        dark = (
            [is_dark_line(gray, b) for b in boxes]
            if self.window > 1
            else [None] * len(boxes)
        )
        crops = [
            crop_line(gray, b, self.height, window=self.window, dark=d)
            for b, d in zip(boxes, dark, strict=True)
        ]
        # 批宽度按行带编排（高窗与行带宽度相近，窄优先分组同样适用）
        band = (
            [crop_line(gray, b, self.height, dark=d) for b, d in zip(boxes, dark, strict=True)]
            if self.window > 1
            else crops
        )
        out: list[tuple[str, float]] = [("", 0.0)] * len(crops)
        lm_on = self.lm is not None and use_lm
        for part in width_batches([c.shape[1] for c in band], self.batch_num):
            arr, widths = pad_batch([crops[j] for j in part])
            res = self.session.session.run(
                OUT_GREEDY + (OUT_LM if lm_on else []),
                {"x": arr, "widths": widths},
            )
            best, pmax, lengths = res[:3]
            for k, j in enumerate(part):
                n = int(lengths[k])
                seq, pm = best[k, :n], pmax[k, :n]
                keep = (seq != 0) & np.concatenate([[True], seq[1:] != seq[:-1]])
                ids = seq[keep].tolist()
                conf = float(pm[keep].mean()) if keep.any() else 0.0
                if self.lm is not None and use_lm and MIN_CONF <= conf < self.gate:
                    ids, _steps = beam_search(
                        res[3][k, :n],
                        res[4][k, :n],
                        res[5][k, :n],
                        self.lm,
                        self.alpha,
                        self.beta,
                        LM_BEAM,
                        cand_floor=LM_CAND_FLOOR,
                    )
                out[j] = (
                    (self.vocab.decode(ids), conf) if conf >= MIN_CONF else ("", conf)
                )
        return out
