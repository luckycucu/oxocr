"""图像加载与颜色空间约定：内部统一 BGR uint8 (H, W, 3)。"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from oxocr.errors import InvalidImageError

__all__ = ["load_image", "to_png_bytes"]


def _pil_to_bgr(img: Image.Image) -> np.ndarray:
    if img.mode == "RGBA" or (img.mode == "P" and "transparency" in img.info):
        img = img.convert("RGBA")
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        img = bg
    elif img.mode != "RGB":
        img = img.convert("RGB")
    return np.array(img)[:, :, ::-1].copy()  # RGB → BGR


def load_image(image: Any) -> np.ndarray:
    """path/str/bytes/文件对象/PIL/ndarray → BGR uint8 (H, W, 3)。

    ndarray 约定为 BGR；灰度 (H, W) 自动扩为三通道，(H, W, 4) 按白底合成。
    三通道 ndarray 无法区分 RGB/BGR，一律按约定的 BGR 处理。
    """
    if isinstance(image, np.ndarray):
        arr = image
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        elif arr.ndim == 3 and arr.shape[2] == 4:
            pil = Image.fromarray(arr.astype(np.uint8), "RGBA")
            return _pil_to_bgr(pil)
        elif arr.ndim == 3 and arr.shape[2] == 1:
            arr = np.repeat(arr, 3, axis=2)
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise InvalidImageError(f"ndarray 形状非法: {arr.shape}，期望 (H, W, 3) BGR")
        if arr.size == 0 or arr.shape[0] == 0 or arr.shape[1] == 0:
            raise InvalidImageError("空图像无法识别")
        return arr.astype(np.uint8, copy=False)

    if isinstance(image, Image.Image):
        return _pil_to_bgr(image)

    if isinstance(image, (bytes, bytearray, memoryview)):
        data = bytes(image)
        if not data:
            raise InvalidImageError("空图像字节流")
        try:
            with Image.open(io.BytesIO(data)) as img:
                return _pil_to_bgr(img)
        except InvalidImageError:
            raise
        except Exception as e:
            raise InvalidImageError(f"无法解码图像字节流: {e}") from e

    if hasattr(image, "read"):
        return load_image(image.read())

    if isinstance(image, (str, Path)):
        path = Path(image)
        if not path.is_file():
            raise InvalidImageError(f"图像文件不存在: {path}")
        try:
            with Image.open(path) as img:
                return _pil_to_bgr(img)
        except Exception as e:
            raise InvalidImageError(f"无法读取图像 {path}: {e}") from e

    raise InvalidImageError(f"不支持的图像输入类型: {type(image).__name__}")


def to_png_bytes(img: np.ndarray) -> bytes:
    """BGR ndarray → 无损 PNG 字节（子进程协议输入）。"""
    pil = Image.fromarray(img[:, :, ::-1])  # BGR → RGB
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return buf.getvalue()
