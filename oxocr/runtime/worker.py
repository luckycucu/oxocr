"""OCR 一次性工作子进程入口。

协议（修正参考实现的 argv 传参，避免 Windows 长度/引号风险）：
    stdin  ← [4 字节 big-endian 头长度][UTF-8 JSON 头][图像字节 PNG 无损]
             头 = {"spec": ..., "files": {role: path}, "options": ..., "request": ...}
    stdout → 最后一行 `OXOCR_RESULT:<单行 JSON>`（前缀防第三方库日志污染）
    stderr ← 日志与错误；失败以非零码退出

子进程不读任何配置文件，全部参数由父进程经头传入；不触发网络。
冻结（PyInstaller）时由宿主 main() 分发到本模块 main()。
"""
from __future__ import annotations

import json
import struct
import sys

EXIT_OK = 0
EXIT_EMPTY_INPUT = 2
EXIT_DEPS = 3
EXIT_IMAGE = 4
EXIT_INFER = 5


def _read_frame(stream) -> tuple[dict, bytes]:
    length_bytes = stream.read(4)
    if len(length_bytes) < 4:
        raise ValueError("stdin 缺少长度前缀")
    (length,) = struct.unpack(">I", length_bytes)
    header = json.loads(stream.read(length).decode("utf-8"))
    image = stream.read()
    return header, image


def main() -> int:
    from oxocr.runtime.executor import RESULT_PREFIX

    stream = getattr(sys.stdin, "buffer", sys.stdin)
    try:
        header, image_bytes = _read_frame(stream)
    except Exception as e:
        print(f"读取输入帧失败: {e}", file=sys.stderr)
        return EXIT_EMPTY_INPUT
    if not image_bytes:
        print("empty image payload", file=sys.stderr)
        return EXIT_EMPTY_INPUT

    try:
        from oxocr.config import ModelOptions
        from oxocr.models.base import InferRequest, ModelSpec
        from oxocr.registry.registry import load_impl
        from oxocr.runtime.images import load_image
    except Exception as e:
        print(f"依赖加载失败: {e}", file=sys.stderr)
        return EXIT_DEPS

    try:
        spec = ModelSpec.from_dict(header["spec"])
        files = {role: path for role, path in header["files"].items()}
        options = ModelOptions.from_dict(header.get("options") or {})
        request = InferRequest.from_dict(header.get("request") or {})
        image = load_image(image_bytes)
    except Exception as e:
        print(f"invalid request/image: {e}", file=sys.stderr)
        return EXIT_IMAGE

    try:
        impl = load_impl(spec)
        model = impl(spec, files, options)
        model.load()
        try:
            output = model.infer(image, request)
        finally:
            model.unload()
        payload = output.to_dict()
    except Exception as e:
        print(f"{type(e).__name__}: {e}", file=sys.stderr)
        return EXIT_INFER

    sys.stdout.write("\n" + RESULT_PREFIX + json.dumps(payload, ensure_ascii=True) + "\n")
    sys.stdout.flush()
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
