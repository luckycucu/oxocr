"""LaTeX-OCR 公式识别家族：pix2tex 系 ViT encoder + 自回归 decoder，整图公式 → LaTeX，无检测模型。

原始模型 lukas-blecher/LaTeX-OCR（pix2tex）；ONNX 权重由 RapidAI/RapidLatexOCR 转换
（移植参考 examples/RapidLaTeXOCR-main）。比 pix2text MFR 多一个
image_resizer.onnx（预测宽度桶、迭代缩放），预处理自带对比度拉伸/阈值裁剪/暗底反色/32 对齐。
"""
