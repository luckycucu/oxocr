"""右向左文本的视觉序 ↔ 逻辑序翻转（解码子集，移植自 indra_ocr/rtl.py）。

CTC 识别器按从左到右的视觉序输出字符，``flip`` 把视觉序翻回逻辑序
（存储/打字序），同一调用也可反向使用（自逆）：

- RTL 基行（RTL 字母多于 LTR 字母）：单元序反转。单元是一段保持自身顺序的
  LTR 文本（拉丁词、数字及其间的空格标点），或一个其它字符簇。
- LTR 基行：整行保序，每段 RTL 文本就地反转。

字符簇 = 基字 + 组合 mark（阿拉伯 harakat、希伯来 niqqud），mark 随基字。
镜像字符（括号）在右向左层级换成镜像形。这是 UAX #9 的小子集：无嵌入/隔离，
中性字符按 run 归并。文本层工具（visual_label / brackets_as_shapes 等）不接。
"""

from __future__ import annotations

import unicodedata

__all__ = ["is_rtl", "has_rtl", "rtl_base", "clusters", "flip"]

_MIRROR = dict(zip("()[]{}<>«»‹›", ")(][}{><»«›‹", strict=True))


def is_rtl(c: str) -> bool:
    return unicodedata.bidirectional(c) in ("R", "AL")


def _is_ltr(c: str) -> bool:
    # 数字（欧洲/阿拉伯-印度）在 RTL 文本中仍从左到右读
    return unicodedata.bidirectional(c) in ("L", "EN", "AN")


def has_rtl(text: str) -> bool:
    return any(is_rtl(c) for c in text)


def rtl_base(text: str) -> bool:
    """行是否右向左读：RTL 字母多于 LTR 字母。计数与顺序无关，逻辑/视觉序一致。"""
    r = sum(is_rtl(c) for c in text)
    lt = sum(unicodedata.bidirectional(c) == "L" for c in text)
    return r > lt


def clusters(text: str) -> list[str]:
    out: list[str] = []
    for c in text:
        if out and unicodedata.category(c) in ("Mn", "Me"):
            out[-1] += c
        else:
            out.append(c)
    return out


def _kind(cl: str) -> str:
    c = cl[0]
    return "R" if is_rtl(c) else "L" if _is_ltr(c) else "N"


def _runs(kinds: list[str], strong: str, stop: set[str]) -> list[tuple[int, int]]:
    """[start, end) 区间：从 strong 簇到 stop 类簇之前的最后一个 strong 簇。"""
    spans: list[tuple[int, int]] = []
    start = -1
    last = -1
    for i, k in enumerate(kinds + ["#"]):
        if k == strong:
            if start < 0:
                start = i
            last = i
        elif k in stop or k == "#":
            if start >= 0:
                spans.append((start, last + 1))
            start = last = -1
    return spans


def _mirror(cl: str) -> str:
    return _MIRROR.get(cl[0], cl[0]) + cl[1:]


def flip(text: str, mirror: bool = True, base_rtl: bool | None = None) -> str:
    """逻辑序 → 视觉序，视觉序 → 逻辑序（自逆）。

    base_rtl: 已知段落方向时传入；None 按 ``rtl_base`` 判定。
    """
    if not has_rtl(text):
        return text
    cls = clusters(text)
    kinds = [_kind(c) for c in cls]
    if rtl_base(text) if base_rtl is None else base_rtl:
        # 单元：LTR run 整体保留，其它簇各自一个；反转单元序
        units: list[str] = []
        spans = _runs(kinds, "L", {"R"})
        i = 0
        for a, b in spans + [(len(cls), len(cls))]:
            for j in range(i, a):
                units.append(_mirror(cls[j]) if mirror else cls[j])
            if a < b:
                units.append("".join(cls[a:b]))
            i = b
        return "".join(reversed(units))
    # LTR 基行：每段 RTL run 就地反转
    out = list(cls)
    for a, b in _runs(kinds, "R", {"L"}):
        seg = cls[a:b][::-1]
        out[a:b] = [_mirror(c) for c in seg] if mirror else seg
    return "".join(out)
