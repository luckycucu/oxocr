<div align="center">
  <img src="docs/images/logo.svg" alt="Logo" width="400" height="400" />  
  <p>
    <a href="#许可证"><img src="https://img.shields.io/badge/License-MIT-blue.svg?logo=opensourceinitiative&logoColor=white" alt="License" /></a>
    <img src="https://img.shields.io/badge/Python-3.10+-FFD43B.svg?logo=python&logoColor=3776AB" alt="Python" />
    <img src="https://img.shields.io/badge/Runtime-ONNX-005CED.svg?logo=onnx&logoColor=white" alt="ONNX Runtime" />
    <img src="https://img.shields.io/badge/Models-45-orange.svg" alt="Models" />
  </p>
  <p>
    <b>简体中文</b> ·
    <a href="README.md">English</a>
  </p>
</div>

**基于 ONNX Runtime 的轻量级 Python OCR 库** —— 支持 paddleocr 系列、duguang_ocr 系列、indra_ocr 系列、pix2text_mfr（公式识别）等 45 个 OCR 模型。

- **轻量依赖**：主要依赖 numpy、Pillow、onnxruntime，**无需依赖 OpenCV**。
- **排版输出**：支持 raw / line / para / layout 共 4 种排版模式，可直接输出文本结果，也支持输出原始识别文本框数据。
- **可视化**：检测框与识别文字画回图像（`oxocr.visual` / `run --vis`），自动探测系统字体，不内置字体。
- **CLI**：采用 Docker 风格命令设计，简洁直观，易于上手。

## 快速开始

### 安装

```bash
pip install oxocr
```

要求 Python ≥ 3.10。从源码安装：`pip install -e .`（开发/测试依赖见 [docs/testing.md](docs/testing.md)）。

### 模型下载

模型权重不随包分发，首次使用前下载（ModelScope 优先，HuggingFace备份）：

```bash
oxocr download ppocrv6_small     # 默认存入 ~/.oxocr/models
```

也可在 Python 中下载：`oxocr.download_model("ppocrv6_small")`。

模型权重统一托管在以下仓库，可手动浏览或下载：

- ModelScope（国内优先）：[mscoder/oxocr](https://modelscope.cn/models/mscoder/oxocr)
- HuggingFace（备份源）：[Luoaho/oxocr](https://huggingface.co/Luoaho/oxocr)

可用 `--source modelscope|huggingface` 指定下载源；网络受限时可从上述仓库手动下载文件，放入模型目录（默认 `~/.oxocr/models`，可用 `OXOCR_MODEL_DIR` 自定义）即可离线使用。

### 识别

```python
import oxocr

# 一行拿文本（默认 ppocrv6_small + para 模式）
text = oxocr.readtext("shot.png")

# 完整结果，一次推理多种排版
result = oxocr.ocr("shot.png", mode="para")
print(result.text)
print(result.as_text("layout"))          # 换版式，不重新推理
for item in result.items:                 # box 恒为原图像素坐标
    print(item.text, item.score, item.box.as_rect())

# 复用引擎（批量场景，共享进程级常驻会话）
engine = oxocr.OcrEngine(model="ppocrv6_small", postprocess={"mode": "para"})
results = engine.batch(["a.png", "b.png"])
```

指定本地离线模型目录：

```python
text = oxocr.readtext("shot.png", model_dir="test/models")
```

### 可视化

把检测框与识别文字可视化：

```python
from oxocr.visual import draw_items, draw_ocr, draw_boxes
from PIL import Image

result = oxocr.ocr("shot.png")

vis = draw_items("shot.png", result.items)       # 双联图：左框右字（BGR ndarray）
Image.fromarray(vis[:, :, ::-1]).save("vis.png")

draw_ocr("shot.png", [it.box for it in result.items],
         [it.text for it in result.items])        # 画框 + 文字面板（含置信度）
draw_boxes("shot.png", [it.box for it in result.items])  # 仅画红色检测框
```

命令行一步到位：

```bash
oxocr run shot.png --vis vis.png                  # 双联可视化图
oxocr run shot.png --vis vis.png --vis-mode box   # 只画框
```

字体默认自动探测系统字体（优先中英文覆盖）；泰语、天城文等文种建议用 `font_path=` / `--font` 指定对应字体（ttf/ttc）。全部模式见 [docs/cli.md](docs/cli.md)。

## 模型清单

按开源日期排序

| 模型                           | 开源日期   | 大小 (MB) | 开源协议          | 支持语言                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| ------------------------------ | ---------- | --------- | ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `anaksor_ocr`                | 2026-11-04 | 90.4      | Apache-2.0        | 高棉文、英文                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| `openthai2.0_ocr`            | 2026-11-01 | 4.0       | Apache-2.0        | 泰文、英文                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| `indra_ocr_cjk`              | 2026-10-19 | 28.7      | RAIL-M (modified) | 简体中文、繁体中文、日文、韩文                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| `indra_ocr_latin`            | 2026-10-19 | 20.7      | RAIL-M (modified) | 拉丁文、英文、法文、德文、意大利文、西班牙文、葡萄牙文、荷兰文、波兰文、罗马尼亚文、捷克文、瑞典文、挪威文、丹麦文、芬兰文、匈牙利文、土耳其文、越南文、印度尼西亚文、马来文、阿塞拜疆文、南非荷兰文、波斯尼亚文、克罗地亚文、威尔士文、爱沙尼亚文、爱尔兰文、冰岛文、库尔德文、立陶宛文、拉脱维亚文、马耳他文、毛利文、欧西坦文、斯洛伐克文、斯洛文尼亚文、阿尔巴尼亚文、斯瓦希里文、塔加洛文、乌兹别克文、拉丁文、塞尔维亚文（拉丁）、加泰罗尼亚文、巴斯克文、加利西亚文、卢森堡文、罗曼什文、克丘亚文、豪萨文、富拉文、阿坎文、约鲁巴文、西里尔文、希腊文 |
| `indra_ocr_math`             | 2026-10-19 | 18.8      | RAIL-M (modified) | latex                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| `indra_ocr_rtl`              | 2026-10-19 | 5.5       | RAIL-M (modified) | 阿拉伯文、波斯文、乌尔都文、普什图文、库尔德文、希伯来文                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| `indra_ocr_world`            | 2026-10-19 | 17.4      | RAIL-M (modified) | 印地文、泰文、孟加拉文、泰米尔文、泰卢固文、缅甸文、高棉文、老挝文、僧伽罗文、卡纳达文、亚美尼亚文、阿姆哈拉文                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| `ppocrv6_medium`             | 2026-06-11 | 132.9     | Apache-2.0        | 简体中文、繁体中文、英文、日文、法文、德文、意大利文、西班牙文、葡萄牙文、荷兰文、波兰文、罗马尼亚文、捷克文、瑞典文、挪威文、丹麦文、芬兰文、匈牙利文、土耳其文、越南文、印度尼西亚文、马来文、阿塞拜疆文、南非荷兰文、波斯尼亚文、克罗地亚文、威尔士文、爱沙尼亚文、爱尔兰文、冰岛文、库尔德文、立陶宛文、拉脱维亚文、马耳他文、毛利文、欧西坦文、斯洛伐克文、斯洛文尼亚文、阿尔巴尼亚文、斯瓦希里文、塔加洛文、乌兹别克文、拉丁文、塞尔维亚文（拉丁）、加泰罗尼亚文、巴斯克文、加利西亚文、卢森堡文、罗曼什文、克丘亚文                                   |
| `ppocrv6_small`              | 2026-06-11 | 30.3      | Apache-2.0        | 简体中文、繁体中文、英文、日文、法文、德文、意大利文、西班牙文、葡萄牙文、荷兰文、波兰文、罗马尼亚文、捷克文、瑞典文、挪威文、丹麦文、芬兰文、匈牙利文、土耳其文、越南文、印度尼西亚文、马来文、阿塞拜疆文、南非荷兰文、波斯尼亚文、克罗地亚文、威尔士文、爱沙尼亚文、爱尔兰文、冰岛文、库尔德文、立陶宛文、拉脱维亚文、马耳他文、毛利文、欧西坦文、斯洛伐克文、斯洛文尼亚文、阿尔巴尼亚文、斯瓦希里文、塔加洛文、乌兹别克文、拉丁文、塞尔维亚文（拉丁）、加泰罗尼亚文、巴斯克文、加利西亚文、卢森堡文、罗曼什文、克丘亚文                                   |
| `ppocrv6_tiny`               | 2026-06-11 | 6.6       | Apache-2.0        | 简体中文、繁体中文、英文、法文、德文、意大利文、西班牙文、葡萄牙文、荷兰文、波兰文、罗马尼亚文、捷克文、瑞典文、挪威文、丹麦文、芬兰文、匈牙利文、土耳其文、越南文、印度尼西亚文、马来文、阿塞拜疆文、南非荷兰文、波斯尼亚文、克罗地亚文、威尔士文、爱沙尼亚文、爱尔兰文、冰岛文、库尔德文、立陶宛文、拉脱维亚文、马耳他文、毛利文、欧西坦文、斯洛伐克文、斯洛文尼亚文、阿尔巴尼亚文、斯瓦希里文、塔加洛文、乌兹别克文、拉丁文、塞尔维亚文（拉丁）、加泰罗尼亚文、巴斯克文、加利西亚文、卢森堡文、罗曼什文、克丘亚文                                         |
| `ppocrv5_arabic_mobile`      | 2025-10-16 | 13.2      | Apache-2.0        | 阿拉伯文、波斯文、维吾尔文、乌尔都文、普什图文、库尔德文、信德文、俾路支文、英文                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| `ppocrv5_cyrillic_mobile`    | 2025-10-16 | 13.3      | Apache-2.0        | 俄罗斯文、白俄罗斯文、乌克兰文、塞尔维亚文（西里尔）、保加利亚文、蒙古文、阿布哈兹文、阿迪赫文、卡巴尔达文、阿瓦尔文、达尔格瓦文、印古什文、车臣文、拉克文、列兹金文、塔巴萨兰文、哈萨克文、吉尔吉斯文、塔吉克文、马其顿文、鞑靼文、楚瓦什文、巴什基尔文、马里文、莫尔多瓦文、乌德穆尔特文、科米文、奥塞梯文、布里亚特文、卡尔梅克文、图瓦文、萨哈文、卡拉卡尔帕克文、英文                                                                                                                                                                                   |
| `ppocrv5_devanagari_mobile`  | 2025-10-16 | 13.1      | Apache-2.0        | 印地文、马拉地文、尼泊尔文、比哈尔文、迈蒂利文、古英文、博杰普尔文、马加希文、萨达里文、尼瓦尔文、孔卡尼文、梵文、哈里亚纳文、英文                                                                                                                                                                                                                                                                                                                                                                                                                           |
| `ppocrv5_eslav_mobile`       | 2025-10-16 | 13.1      | Apache-2.0        | 俄罗斯文、白俄罗斯文、乌克兰文、英文                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| `ppocrv5_korean_mobile`      | 2025-10-16 | 18.4      | Apache-2.0        | 韩文、英文                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| `ppocrv5_latin_mobile`       | 2025-10-16 | 13.1      | Apache-2.0        | 法文、德文、南非荷兰文、意大利文、西班牙文、波斯尼亚文、葡萄牙文、捷克文、威尔士文、丹麦文、爱沙尼亚文、爱尔兰文、克罗地亚文、乌兹别克文、匈牙利文、塞尔维亚文（拉丁）、印度尼西亚文、欧西坦文、冰岛文、立陶宛文、毛利文、马来文、荷兰文、挪威文、波兰文、斯洛伐克文、斯洛文尼亚文、阿尔巴尼亚文、瑞典文、斯瓦希里文、塔加洛文、土耳其文、拉丁文、阿塞拜疆文、库尔德文、拉脱维亚文、马耳他文、巴利文、罗马尼亚文、越南文、芬兰文、巴斯克文、加利西亚文、卢森堡文、罗曼什文、加泰罗尼亚文、克丘亚文                                                           |
| `ppocrv5_ta_mobile`          | 2025-10-16 | 13.1      | Apache-2.0        | 泰米尔文、英文                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| `ppocrv5_te_mobile`          | 2025-10-16 | 13.1      | Apache-2.0        | 泰卢固文、英文                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| `ppocrv5_el_mobile`          | 2025-08-21 | 13.0      | Apache-2.0        | 希腊文、英文                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| `ppocrv5_en_mobile`          | 2025-08-21 | 13.1      | Apache-2.0        | 英文                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| `ppocrv5_th_mobile`          | 2025-08-21 | 13.1      | Apache-2.0        | 泰文、英文                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| `pix2text_mfr_1.5`           | 2025-07-07 | 114.1     | MIT               | latex                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| `ppocrv5_ch_mobile`          | 2025-05-20 | 21.4      | Apache-2.0        | 简体中文、繁体中文、英文、日文                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| `ppocrv5_ch_server`          | 2025-05-20 | 165.7     | Apache-2.0        | 简体中文、繁体中文、英文、日文                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| `pix2text_mfr`               | 2024-02-12 | 112.2     | MIT               | latex                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| `pix2text_mfr_quantized`     | 2024-02-12 | 50.8      | MIT               | latex                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| `ppocrv4_arabic_mobile`      | 2023-08-07 | 12.4      | Apache-2.0        | 阿拉伯文                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| `ppocrv4_ch_doc_server`      | 2023-08-07 | 199.2     | Apache-2.0        | 简体中文、英文                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| `ppocrv4_ch_mobile`          | 2023-08-07 | 15.4      | Apache-2.0        | 简体中文、英文                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| `ppocrv4_ch_server`          | 2023-08-07 | 195.0     | Apache-2.0        | 简体中文、英文                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| `ppocrv4_devanagari_mobile`  | 2023-08-07 | 12.4      | Apache-2.0        | 天城文                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| `ppocrv4_en_mobile`          | 2023-08-07 | 12.4      | Apache-2.0        | 英文                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| `ppocrv4_japan_mobile`       | 2023-08-07 | 14.4      | Apache-2.0        | 日文                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| `ppocrv4_ka_mobile`          | 2023-08-07 | 12.4      | Apache-2.0        | 卡纳达文                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| `ppocrv4_korean_mobile`      | 2023-08-07 | 28.0      | Apache-2.0        | 韩文                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| `ppocrv4_ta_mobile`          | 2023-08-07 | 26.4      | Apache-2.0        | 泰米尔文                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| `ppocrv4_te_mobile`          | 2023-08-07 | 26.4      | Apache-2.0        | 泰卢固文                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| `duguang_ocr_small`          | 2023-05-29 | 12.7      | Apache-2.0        | 简体中文、英文                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| `duguang_ocr_large`          | 2023-02-06 | 119.8     | Apache-2.0        | 简体中文、英文                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| `duguang_ocr_max`            | 2022-08-19 | 151.4     | Apache-2.0        | 简体中文、英文                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| `rapid_latex_ocr`            | 2022-05-25 | 170.7     | MIT               | latex                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| `ppocrv3_chinese_cht_mobile` | 2022-05-09 | 13.5      | Apache-2.0        | 繁体中文                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| `ppocrv3_cyrillic_mobile`    | 2022-05-09 | 11.4      | Apache-2.0        | 西里尔文                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| `ppocrv3_en_mobile`          | 2022-05-09 | 11.4      | Apache-2.0        | 英文                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| `ppocrv3_latin_mobile`       | 2022-05-09 | 11.4      | Apache-2.0        | 拉丁文、英文                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |

- `RAIL-M (modified)` 是 AI PUBS OPEN RAIL-M (modified) 的缩写。
- 语言列为 `latex` 的模型是整图公式识别器（公式图 → LaTeX，无检测模型）。
- 模型详细信息请使用`oxocr inspect <name>` 命令查看。
- `ppocr`来源于RapidOCR

```python
oxocr.readtext("shot.png", model="ppocrv5_ch_server")
oxocr.readtext("shot.png", model="ppocrv4_korean_mobile")

# 方向分类：三个权重可选，输入形状随权重自动切换
oxocr.readtext("shot.png", use_cls=True,
               config={"model_options": {"params": {"cls_model": "textline_ori_server"}}})

# 公式识别：整张公式图 → LaTeX；默认跳过后处理，LaTeX 原样输出
oxocr.readtext("formula.png", model="pix2text_mfr")  # 1.0 原版
oxocr.readtext("formula.png", model="pix2text_mfr_1.5")
oxocr.readtext("formula.png", model="pix2text_mfr_quantized")  # int8 encoder，更快
oxocr.readtext("formula.png", model="rapid_latex_ocr")  # LaTeX-OCR（pix2tex），自带前处理，CPU 单公式 <1s

# 读光 OCR：中英文三档；max 用 SegLink 检测，small/large 用 DB 检测
oxocr.readtext("shot.png", model="duguang_ocr_large")
oxocr.readtext("shot.png", model="duguang_ocr_max")

# 高棉语 OCR：高棉语 + 英语/拉丁；长行自动切窗拼接，竖排/180° 可选
oxocr.readtext("shot.png", model="anaksor_ocr")

# 泰语 OCR：泰语 + 英语；整页自动转正/纠偏、分栏阅读序，CPU 约 0.4s/页
oxocr.readtext("shot.png", model="openthai2.0_ocr")

# Indra 多语种：按主导语种选专用模型（每个识别器也读拉丁文）
oxocr.readtext("shot.png", model="indra_ocr_cjk")     # 中日韩（韩文 jamo 自动合成音节）
oxocr.readtext("shot.png", model="indra_ocr_rtl")     # 阿拉伯/希伯来（输出逻辑序）
oxocr.readtext("formula.png", model="indra_ocr_math")  # 整图公式 → LaTeX（无检测，同 MFR）
```

## 命令行

Docker 风格命令（`run` / `models` / `version`），详见 [docs/cli.md](docs/cli.md)：

```bash
oxocr run shot.png -m ppocrv6_tiny -M layout  # 识别图片，输出多栏排版的文本结果
oxocr run shot.png --vis vis.png          # 识别并保存可视化图像
oxocr models ls                 		# 查看模型列表
oxocr models inspect ppocrv6_small		# 查看模型详细信息
oxocr -h --lang zh                 		# 中文帮助（默认英文）
```

## 许可证

代码以 MIT 许可证开源；模型权重不在该许可证范围内，遵循各自许可条款。
