"""读光行识别（三档共用，ONNX 后端）。

3 chunk 前处理 + CTC 解码 + 重叠合并 + 几何平均 score。
组块语义按输出行数运行时推断：出=入 → 重叠合并；出=入/3 → 直接解码。
"""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from oxocr.errors import InferenceError
from oxocr.runtime import imageops
from oxocr.runtime.session import OrtSession

__all__ = ["TextRecognizer"]

_IMG_H = 32
_IMG_W = 804
_CHUNK_W = 300
_CHUNK_STEP = 252
_N_CHUNKS = 3


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def merge_strings_with_overlap(strings: list[str]) -> str:
    if not strings:
        return ""
    result = strings[0]
    for s in strings[1:]:
        max_overlap = min(len(result), len(s))
        best = 0
        for overlap in range(max_overlap, -1, -1):
            if result.endswith(s[:overlap]):
                best = overlap
                break
        result += s[best:]
    return result


class TextRecognizer:
    def __init__(
        self,
        model_path: str | Path,
        vocab_path: str | Path,
        batch_num: int = 6,
        num_threads: int = 0,
        providers: Sequence[str] = ("CPUExecutionProvider",),
    ):
        self.session = OrtSession(model_path, num_threads=num_threads, providers=providers)
        self.batch_num = int(batch_num)
        self.label_mapping = self._load_vocab(vocab_path)

    @staticmethod
    def _load_vocab(vocab_path: str | Path) -> dict[int, str]:
        lines = Path(vocab_path).read_text(encoding="utf-8").splitlines()
        return {i + 2: line for i, line in enumerate(lines)}

    def _preprocess(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        ratio = w / float(h)
        target_w = _IMG_W if ratio > _IMG_W / _IMG_H else int(_IMG_H * ratio)
        target_w = max(1, target_w)
        resized = imageops.resize(img, (target_w, _IMG_H))
        padded = np.zeros((_IMG_H, _IMG_W, 3), dtype=np.float32)
        padded[:, :target_w, :] = resized.astype(np.float32)
        chunks = []
        for i in range(_N_CHUNKS):
            left = _CHUNK_STEP * i
            chunks.append(padded[:, left:left + _CHUNK_W, :])
        data = np.stack(chunks, axis=0) / 255.0
        return data.transpose((0, 3, 1, 2)).astype(np.float32)

    def _ctc_decode(
        self, preds_idx: np.ndarray, preds_prob: np.ndarray
    ) -> tuple[str, list[float]]:
        ids = preds_idx
        probs = preds_prob
        selection = np.ones(len(ids), dtype=bool)
        selection[1:] = ids[1:] != ids[:-1]
        selection &= ids != 0
        selected_ids = ids[selection]
        selected_probs = probs[selection]
        chars = []
        kept_probs = []
        for idx, prob in zip(selected_ids, selected_probs, strict=True):
            idx = int(idx)
            if idx == 1:
                continue
            char = self.label_mapping.get(idx)
            if char is None:
                continue
            chars.append(char)
            kept_probs.append(float(prob))
        return "".join(chars), kept_probs

    def __call__(self, img_list: list[np.ndarray]) -> list[tuple[str, float]]:
        if not img_list:
            return []
        img_num = len(img_list)
        results: list[tuple[str, float]] = [("", 0.0)] * img_num
        for beg in range(0, img_num, self.batch_num):
            end = min(img_num, beg + self.batch_num)
            batch = np.concatenate(
                [self._preprocess(img_list[i]) for i in range(beg, end)], axis=0
            )
            preds = self.session(batch)
            out_probs = _softmax(preds)
            preds_idx = out_probs.argmax(axis=-1)
            preds_prob = out_probs.max(axis=-1)

            input_rows = batch.shape[0]
            output_rows = preds.shape[0]
            n_crops = end - beg

            if output_rows == input_rows:
                for i in range(n_crops):
                    chunk_texts = []
                    chunk_probs = []
                    for j in range(_N_CHUNKS):
                        row = i * _N_CHUNKS + j
                        text, probs = self._ctc_decode(preds_idx[row], preds_prob[row])
                        chunk_texts.append(text)
                        chunk_probs.extend(probs)
                    merged = merge_strings_with_overlap(chunk_texts)
                    score = float(np.exp(np.mean(np.log(chunk_probs)))) if chunk_probs else 0.0
                    results[beg + i] = (merged, score)
            elif output_rows * 3 == input_rows:
                for i in range(n_crops):
                    text, probs = self._ctc_decode(preds_idx[i], preds_prob[i])
                    score = float(np.exp(np.mean(np.log(probs)))) if probs else 0.0
                    results[beg + i] = (text, score)
            else:
                raise InferenceError(
                    f"读光 rec 输出行数 {output_rows} 与输入行数 {input_rows} 不匹配："
                    f"期望出=入（重叠合并）或出=入/3（直接解码）"
                )
        return results
