"""Indra OCR 家族：LineNet 行检测 + CTC 行识别（n-gram LM）+ 注意力公式解码。

5 个专用模型（indra_ocr_latin/cjk/rtl/world/math），每模型固定一个识别器，
不做运行时语种路由。移植自 datalab-to/indra 1.1.2（examples/indra-1.1.2）。
"""
