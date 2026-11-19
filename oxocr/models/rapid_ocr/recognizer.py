"""CTC 文本识别（ONNX 后端）。

字典双通道：优先 ONNX metadata `character`，缺失时回退 keys 文件；两者都无才报错。
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from oxocr.errors import InferenceError
from oxocr.runtime import imageops
from oxocr.runtime.session import OrtSession

__all__ = ["TextRecognizer"]


class TextRecognizer:
    def __init__(
        self,
        model_path: str | Path,
        keys: str | Path | None = None,
        rec_image_shape: Sequence[int] = (3, 48, 320),
        rec_batch_num: int = 6,
        num_threads: int = 0,
        providers: Sequence[str] = ("CPUExecutionProvider",),
    ):
        self.model_path = str(model_path)
        self.session = OrtSession(model_path, num_threads=num_threads, providers=providers)
        self.rec_image_shape = list(rec_image_shape)
        self.rec_batch_num = int(rec_batch_num)
        self.character = self._load_character(keys)

    def _load_character(self, keys: str | Path | None) -> list[str]:
        if self.session.have_key("character"):
            char_list = self.session.get_character_list("character")
        elif keys is not None and Path(keys).is_file():
            char_list = Path(keys).read_text(encoding="utf-8").splitlines()
        else:
            raise InferenceError(
                f"识别模型 {self.model_path} 缺少字典：ONNX metadata 无 'character'，"
                "且未提供可用的 keys 文件。请更换含字典的模型或经 ModelFile(role='keys') 提供字典文件。"
            )
        char_list = list(char_list)
        char_list.insert(0, "blank")
        char_list.append(" ")
        return char_list

    def _resize_norm_img(self, img: np.ndarray, max_wh_ratio: float) -> np.ndarray:
        _, img_h, img_w_max = self.rec_image_shape
        img_w = int(img_h * max_wh_ratio)
        h, w = img.shape[:2]
        ratio = w / float(h)
        resized_w = img_w if math.ceil(img_h * ratio) > img_w else int(math.ceil(img_h * ratio))
        resized = imageops.resize(img, (resized_w, img_h))
        resized = resized.astype("float32").transpose((2, 0, 1)) / 255
        resized = (resized - 0.5) / 0.5
        padded = np.zeros((3, img_h, img_w), dtype=np.float32)
        padded[:, :, :resized_w] = resized
        return padded

    def _ctc_decode(
        self, preds_idx: np.ndarray, preds_prob: np.ndarray
    ) -> list[tuple[str, float]]:
        results = []
        for batch_idx in range(len(preds_idx)):
            ids = preds_idx[batch_idx]
            probs = preds_prob[batch_idx]
            selection = np.ones(len(ids), dtype=bool)
            selection[1:] = ids[1:] != ids[:-1]
            selection &= ids != 0
            selected_ids = ids[selection]
            selected_probs = probs[selection]
            if len(selected_ids) == 0:
                results.append(("", 0.0))
            else:
                text = "".join(self.character[i] for i in selected_ids)
                results.append((text, round(float(np.mean(selected_probs)), 5)))
        return results

    def __call__(self, img_list: list[np.ndarray]) -> list[tuple[str, float]]:
        if not img_list:
            return []
        width_list = [img.shape[1] / float(img.shape[0]) for img in img_list]
        indices = np.argsort(np.array(width_list))
        img_num = len(img_list)
        rec_res: list[tuple[str, float]] = [("", 0.0)] * img_num
        for beg in range(0, img_num, self.rec_batch_num):
            end = min(img_num, beg + self.rec_batch_num)
            _, img_h, img_w = self.rec_image_shape[:3]
            max_wh_ratio = img_w / img_h
            for i in range(beg, end):
                h, w = img_list[indices[i]].shape[:2]
                max_wh_ratio = max(max_wh_ratio, w / float(h))
            batch = np.concatenate(
                [
                    self._resize_norm_img(img_list[indices[i]], max_wh_ratio)[np.newaxis, :]
                    for i in range(beg, end)
                ]
            ).astype(np.float32)
            preds = self.session(batch)
            preds_idx = preds.argmax(axis=2)
            preds_prob = preds.max(axis=2)
            for j, one_res in enumerate(self._ctc_decode(preds_idx, preds_prob)):
                rec_res[indices[beg + j]] = one_res
        return rec_res
