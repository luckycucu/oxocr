"""anaksor SVTRv2-B 行识别（ONNX 后端，仅 CTC 头）。

宽高比桶分组批（同形状无 padding），批大小随 ratio² 缩减（全局注意力内存 ~ batch×width²）。
输出不做归一化：chars 保持原始码点（字簇切分依赖 marks/coeng 原始形态），
normalize() 由管线在每行最终文本上调用一次。
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from PIL import Image

from oxocr.errors import InferenceError
from oxocr.runtime.session import OrtSession

__all__ = ["SvtrRecognizer", "ctc_greedy", "ctc_positions", "read_dict"]

_MAX_RATIO = 20  # 权重绑定的训练/导出常量，不暴露为参数
_BASE_H = 48
_BASE_SHAPE = ((64, 64), (96, 48), (144, 48), (192, 48))


def read_dict(path: str | Path) -> list[str]:
    """字典每行一个字符（UTF-8），use_space_char 追加空格。"""
    with open(path, "rb") as f:
        chars = [line.decode("utf-8").strip("\n").strip("\r\n") for line in f]
    chars.append(" ")
    return chars


def ctc_greedy(probs: np.ndarray, classes: list[str]) -> list[tuple[str, float]]:
    """(B, T, C) 概率的 best-path CTC 解码：重复折叠、blank 去除。
    score = 保留帧概率的算术平均（空结果 0）。"""
    ids, best = probs.argmax(2), probs.max(2)
    out = []
    for row, p in zip(ids, best, strict=True):
        keep = row != 0
        keep[1:] &= row[1:] != row[:-1]
        text = "".join(classes[i] for i in row[keep])
        out.append((text, float(np.mean(p[keep])) if keep.any() else 0.0))
    return out


def ctc_positions(probs: np.ndarray, classes: list[str]) -> list[list[tuple[str, float, float]]]:
    """每行 [(char, x_frac, prob)]：x_frac = 帧段中心 / 总帧数，prob = 帧段最高概率。"""
    ids, best = probs.argmax(2), probs.max(2)
    out = []
    for row, p in zip(ids, best, strict=True):
        n, seq, t = len(row), [], 0
        while t < n:
            e = t
            while e + 1 < n and row[e + 1] == row[t]:
                e += 1
            if row[t] != 0:
                seq.append((classes[row[t]], (t + e + 1) / 2 / n, float(p[t : e + 1].max())))
            t = e + 1
        out.append(seq)
    return out


def _batch_for(ratio: float, batch_size: int) -> int:
    return batch_size if ratio <= 4 else max(1, int(batch_size * 16 / ratio**2))


class SvtrRecognizer:
    def __init__(
        self,
        model_path: str | Path,
        dict_path: str | Path,
        batch_num: int = 32,
        num_threads: int = 0,
        providers: Sequence[str] = ("CPUExecutionProvider",),
    ):
        self.session = OrtSession(model_path, num_threads=num_threads, providers=providers)
        self.batch_num = int(batch_num)
        self.classes = ["blank", *read_dict(dict_path)]
        meta = self.session.metadata()
        if meta.get("anaksor_kind") not in (None, "rec"):
            raise InferenceError(
                f"{Path(model_path).name} 是 {meta['anaksor_kind']} 模型，不是 rec 识别权重"
            )
        if meta.get("anaksor_head") not in (None, "ctc"):
            raise InferenceError(
                f"{Path(model_path).name} 的 head 为 {meta['anaksor_head']}，期望 ctc"
            )
        if meta.get("anaksor_classes") and int(meta["anaksor_classes"]) != len(self.classes):
            raise InferenceError(
                f"{Path(model_path).name} 有 {meta['anaksor_classes']} 类，"
                f"字典 {Path(dict_path).name} 给出 {len(self.classes)} 类；请使用匹配的字典"
            )

    @property
    def max_ratio(self) -> int:
        return _MAX_RATIO

    @staticmethod
    def shape_for(w: int, h: int) -> tuple[int, int]:
        ratio = min(max(1, round(w / max(h, 1))), _MAX_RATIO)
        if ratio <= len(_BASE_SHAPE):
            return _BASE_SHAPE[ratio - 1]
        return _BASE_H * ratio, _BASE_H

    def _preprocess(self, crop: np.ndarray) -> np.ndarray:
        w, h = self.shape_for(crop.shape[1], crop.shape[0])
        rgb = np.ascontiguousarray(crop[:, :, ::-1])
        resized = Image.fromarray(rgb).resize((w, h), Image.Resampling.BICUBIC)
        x = np.asarray(resized, dtype=np.float32) / 255.0
        return ((x - 0.5) / 0.5).transpose(2, 0, 1)

    def __call__(self, crops: list[np.ndarray]) -> list[dict]:
        """[{"text", "score", "chars"}]，text 未归一化，chars 供长行拼接。"""
        results: list[dict] = [{"text": "", "score": 0.0, "chars": []} for _ in crops]
        if not crops:
            return results
        groups: dict[tuple[int, int], list[int]] = defaultdict(list)
        for i, crop in enumerate(crops):
            groups[self.shape_for(crop.shape[1], crop.shape[0])].append(i)
        for (w, h), idxs in groups.items():
            n = _batch_for(w / h, self.batch_num)
            for b in range(0, len(idxs), n):
                chunk = idxs[b : b + n]
                batch = np.stack([self._preprocess(crops[i]) for i in chunk]).astype(np.float32)
                probs = self.session(batch)
                texts = ctc_greedy(probs, self.classes)
                chars = ctc_positions(probs, self.classes)
                for i, (text, score), ch in zip(chunk, texts, chars, strict=True):
                    results[i] = {"text": text, "score": score, "chars": ch}
        return results
