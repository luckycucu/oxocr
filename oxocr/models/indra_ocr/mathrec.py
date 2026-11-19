"""整图公式识别（移植自 indra_ocr/math.py 的解码部分）：注意力 KV-cache 解码。

CTC 识别器每列读一个符号，写不出分式/带限求和；MathReader 用注意力解码器
读 64px 高的整图缩放，每步一个 token，decoder 键值缓存增量拼接。
conf = 平均 token 概率，随读法返回供调用方过滤；read 的 min_conf 仅用于早停。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from oxocr.models.indra_ocr.preproc import pad_batch, width_batches
from oxocr.runtime.session import OrtSession

__all__ = ["MathReader", "detokenize"]

EARLY_STOP_MARGIN = 0.1  # 8 token 后均值低于 min_conf - 此值即早停（省步优化，min_conf=0 时不触发）
BOS, EOS, N_SPECIAL = 1, 2, 4


def detokenize(tokens: list[str]) -> str:
    """token → 文本；数学段（"$" token 之间）以空格连接。"""
    out: list[str] = []
    math: list[str] = []
    in_math = False
    for t in tokens:
        if t == "$":
            if in_math:
                out.append("$" + " ".join(math) + "$")
                math = []
            in_math = not in_math
        elif in_math:
            math.append(t)
        else:
            out.append(t)
    if in_math and math:
        out.append("$" + " ".join(math) + "$")
    return " ".join("".join(out).split())


class MathReader:
    def __init__(self, files, *, num_threads: int = 0, providers=("CPUExecutionProvider",)):
        kw = {"num_threads": num_threads, "providers": providers}
        self.enc = OrtSession(files["enc"], **kw)
        self.init = OrtSession(files["dec_init"], **kw)
        self.step = OrtSession(files["dec_step"], **kw)
        meta = json.loads(Path(files["math_meta"]).read_text(encoding="utf-8"))
        cfg = meta["config"]
        self.height = int(cfg["height"])
        self.max_tokens = int(cfg["max_tokens"])
        self.layers, self.heads = int(cfg["dec_layers"]), int(cfg["dec_heads"])
        self.head_dim = int(cfg["channels"]) // self.heads
        self.itos = ["", "", "", ""] + meta["tokens"]
        # 导出会丢弃未用输入（init 图不读 mask）
        self.init_inputs = {i.name for i in self.init.session.get_inputs()}
        self.step_inputs = {i.name for i in self.step.session.get_inputs()}

    def read(
        self, crops: list[np.ndarray], batch: int = 64, min_conf: float = 0.0
    ) -> list[tuple[str, float]]:
        """每份 64px 高整图缩放 → (含内联 LaTeX 的文本, 平均 token 概率)。"""
        out = [("", 0.0)] * len(crops)
        # 少而宽的批：每批一个至多 max_tokens 步的解码循环，
        # 一步对 8 行与 40 行的开销相近
        widths = [c.shape[1] for c in crops]
        for idx in width_batches(widths, batch, max_ratio=4.0, max_area=16 * 64 * 640):
            arr, widths_b = pad_batch([crops[j] for j in idx])
            mem, mask = self.enc.session.run(None, {"x": arr, "widths": widths_b})
            feed = {"memory": mem, "mask": mask}
            ck, cv = self.init.session.run(
                None, {k: v for k, v in feed.items() if k in self.init_inputs}
            )
            B = len(idx)
            sk = np.zeros((self.layers, B, self.heads, 0, self.head_dim), np.float32)
            sv = sk
            tok = np.full(B, BOS, np.int64)
            seqs: list[list[int]] = [[] for _ in range(B)]
            probs: list[list[float]] = [[] for _ in range(B)]
            done = np.zeros(B, bool)
            for t in range(self.max_tokens - 1):
                feed = {
                    "token": tok,
                    "pos": np.array(t, np.int64),
                    "self_k": sk,
                    "self_v": sv,
                    "cross_k": ck,
                    "cross_v": cv,
                    "mask": mask,
                }
                logits, sk, sv = self.step.session.run(
                    None, {k: v for k, v in feed.items() if k in self.step_inputs}
                )
                tok = logits.argmax(-1).astype(np.int64)
                e = np.exp(logits - logits.max(-1, keepdims=True))
                top = 1.0 / e.sum(-1)  # argmax token 的 softmax 概率
                for b in range(B):
                    if not done[b]:
                        probs[b].append(float(top[b]))
                        if tok[b] == EOS:
                            done[b] = True
                        elif (
                            len(probs[b]) >= 8
                            and np.mean(probs[b]) < min_conf - EARLY_STOP_MARGIN
                        ):
                            done[b] = True  # 拒收：保留低均值概率
                        elif tok[b] >= N_SPECIAL:
                            seqs[b].append(int(tok[b]))
                if done.all():
                    break
            for k, j in enumerate(idx):
                text = detokenize([self.itos[i] for i in seqs[k]])
                out[j] = (text, float(np.mean(probs[k])) if probs[k] else 0.0)
        return out
