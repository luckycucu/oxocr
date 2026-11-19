# 新增一个 OCR 模型

oxocr 的模型契约只有一句话：**像素进，带 box 的文本条目出**。写一个 `infer()` 就接完一个新模型，
前处理、4 种排版、纠错、子进程隔离全部自动继承。

配套接口定义见 [api.md](api.md) §4，架构决策见 [design.md](design.md) §4.1。

---

## 情形 A：PP-OCR 家族新版本（零代码）

v3~v6 共用 DB 检测 + CTC 识别 + 0/180 方向分类，前后处理逐版本一致
（对齐 `examples/RapidOCR-main/python/rapidocr`：det `736/min`、rec 高度 48、
字典取自 ONNX metadata 的 `character`）。差异只在权重文件与 cls 输入形状，
因此**新增组合 = 往 `oxocr/models/rapid_ocr/specs.py` 的表里加一行**：

```python
_V5_REC = (
    # (语种键, rec 文件名, languages)；mobile/server 由文件名后缀推导
    ("korean", "korean_PP-OCRv5_rec_mobile.onnx", ("korean",)),
)
```

清单据此生成 `ppocrv5_korean_mobile`：rec 用表里的文件，det 跟随同档位的
`ch_PP-OCRv5_det_{mobile,server}`，cls 用该版本默认。需要手写整条记录时：

```python
ModelSpec(
    name="ppocrv5_ch_server",
    family="ppocr",
    impl="oxocr.models.rapid_ocr.pipeline:PpOcrModel",
    files=(
        ModelFile("det", "rapid-ocr/PP-OCRv5/det/ch_PP-OCRv5_det_server.onnx"),
        ModelFile("rec", "rapid-ocr/PP-OCRv5/rec/ch_PP-OCRv5_rec_server.onnx"),
        ModelFile("cls", CLS_MODELS["textline_ori_mobile"].filename),
    ),
    defaults={**PPOCR_DEFAULTS, "cls_model": "textline_ori_mobile"},
    languages=("ch", "en"),
)
```

`defaults` 同时是该模型接受的 `params` **白名单**：`model_options.params` 里出现未声明的键会抛
`UnknownParameterError`。`ModelSpec` 只是纯数据，导入不触碰 onnxruntime。

### 方向分类模型（cls）

三个权重都是 2 类（0/180），但输入形状不同，**权重与形状必须成对使用**：

| `cls_model` | 权重 | 输入形状 |
| --- | --- | --- |
| `v2_0_cls` | `ch_ppocr_mobile_v2.0_cls_mobile` | `(3, 48, 192)` |
| `textline_ori_mobile` | `ch_PP-LCNet_x0_25_textline_ori_cls_mobile` | `(3, 80, 160)` 静态 |
| `textline_ori_server` | `ch_PP-LCNet_x1_0_textline_ori_cls_server` | `(3, 80, 160)` 静态 |

默认按版本选择（v5 → `textline_ori_mobile`，v3/v4/v6 → `v2_0_cls`），可用 params 覆盖：

```python
oxocr.readtext(
    "shot.png", model="ppocrv6_small", use_cls=True,
    config={"model_options": {"params": {"cls_model": "textline_ori_server"}}},
)
```

形状由所选权重自动推导，因此**没有** `cls_image_shape` 参数：textline_ori 的 80×160 是
ONNX 图里的静态维度，喂 48×192 会直接报 `INVALID_ARGUMENT`。
`cls_model` 参与常驻池指纹，换方向模型会得到独立实例并重新加载。

---

## 情形 B：全新 ONNX 模型家族（一个类 + 一条清单）

继承 `OcrModel`，实现三个方法：

```python
from oxocr.models.base import OcrModel, ModelSpec, Capabilities, InferRequest, ModelOutput
from oxocr.types import Box, TextItem
from oxocr.runtime.session import OrtSession


class MyOcrModel(OcrModel):
    family = "myocr"
    capabilities = Capabilities(detection=True, recognition=True, backend="onnxruntime")

    def load(self):
        # 建立会话。构造时不得调用，由引擎惰性触发
        self._sess = OrtSession(self.files["model"], self.options.num_threads)

    def unload(self):
        self._sess = None

    def infer(self, image, request: InferRequest) -> ModelOutput:
        # image: BGR uint8 (H, W, 3)，已完成前处理
        # 返回的 box 用「送入图坐标」——逆映射由核心负责，模型不必关心补边
        items = [TextItem(text="你好", score=0.98, box=Box.from_rect(10, 10, 60, 20))]
        return ModelOutput(items=items, elapse={"rec": 0.12})
```

契约要点（`OcrModel` 已内置）：

- `is_loaded` / `ensure_loaded()`：引擎/池负责惰性加载，`infer` 前一定已 `load()`。
- `validate_options()`：默认按 `spec.defaults` 校验 `params` 白名单，可覆写。
- `self.params()`：`spec.defaults ← options.params` 的合并结果。
- **不得**读配置文件、做排版合并、或 `print` 到 stdout（子进程协议占用 stdout）。

现成实例：`oxocr/models/pix2text/`（公式识别：无检测模型、box 恒为整图、
`own_postprocess=True` 使后处理默认关闭、encoder/decoder/tokenizer/config 四文件一组），
方案与取舍见 [mfr.md](mfr.md)；`oxocr/models/rapid_latex_ocr/`（LaTeX-OCR 公式识别：
无检测、box 恒为整图、`own_preprocess=True` 使核心前处理默认关闭——自带对比度拉伸/裁剪/反色，
resizer/encoder/decoder/tokenizer 四文件一组），方案与实测事实见
[latex-ocr.md](latex-ocr.md)；`oxocr/models/duguang_ocr/`（读光：DB / SegLink 检测 + 行识别、
det 分辨率变体表、词表必需文件、组块语义运行时推断），方案与实测事实见
[duguang-ocr-design.md](duguang-ocr-design.md)；`oxocr/models/anaksor_ocr/`（高棉语：DBNet++
检测 + SVTRv2 CTC 识别、字典必需文件、宽高比分桶批处理、长行切窗拼接、模型内高棉文规范化），
方案与实测事实见 [anaksor-ocr-design.md](anaksor-ocr-design.md)；`oxocr/models/openthai_ocr/`
（泰语：PP-OCRv6-tiny DB 检测 + 泰语 CTC 识别、字典必需文件、flatten 背景归一、
整页方向投票/deskew 与 tf 逆映射、XY-cut 分栏阅读序、模型内泰文规范化），
方案与实测事实见 [openthai2.0-ocr-design.md](openthai2.0-ocr-design.md)；`oxocr/models/indra_ocr/`
（多语种/公式：一个家族拆 5 个专用模型、LineNet 行检测、可选 lm/orient role、
模型内 deskew/四分转与 box 逆映射、math 页面公式提取器），方案与实测事实见
[indra-ocr-design.md](indra-ocr-design.md)。

然后在 `registry/catalog.py` 聚合，或运行时注册：

```python
import oxocr
oxocr.register_spec(ModelSpec(name="myocr_base", family="myocr",
                              impl="my_pkg.ocr:MyOcrModel", files=(...), defaults={...}))
oxocr.ocr("img.png", model="myocr_base")
```

---

## 情形 C：第三方包经 entry point 接入（不改 oxocr 源码）

```toml
[project.entry-points."oxocr.models"]
myocr = "my_pkg.ocr:MyOcrModel"   # 类上实现 get_specs() 返回 ModelSpec 列表
```

```python
class MyOcrModel(OcrModel):
    @classmethod
    def get_specs(cls):
        return [ModelSpec(name="myocr_base", family="myocr",
                          impl="my_pkg.ocr:MyOcrModel", files=(...), defaults={...})]
```

`oxocr.list_models()` 会惰性发现它（解析顺序：内置 catalog → 运行时注册 → entry points）。

---

## 非 ONNX 后端

同一条契约，把 `capabilities.backend` 设为 `"torch"` / `"remote"`，用 `requires` 声明额外依赖
（extras 名），缺依赖时 `load_impl` 抛 `MissingDependencyError` 并附 `pip install` 提示。
`OrtSession` 只是默认后端，模型内部想用什么都可以，只要 `infer` 遵守坐标约定。

---

## 离线模型与测试

`test/models/` 放置的权重按 `ModelSpec.files.filename` 的相对路径组织即可被离线解析：

```bash
OXOCR_OFFLINE=1 OXOCR_MODEL_DIR=test/models python -m pytest -m model
```

缺文件时抛 `ModelFileMissingError` 并列出**期望路径**，不会联网（自动下载按方案暂缓）。