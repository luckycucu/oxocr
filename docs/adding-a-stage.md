# 新增一个前 / 后处理插件

oxocr 将处理链拆成可插拔的阶段。**新增一个处理步骤只需实现一个 `apply()`（前处理）
或 `on_lines()` / `on_blocks()`（后处理）。** 接口定义见 [api.md](api.md) §5–6。

---

## 前处理插件

```python
from oxocr.preprocess.base import Preprocessor, PreContext, Affine, register_preprocessor


@register_preprocessor("deskew")
class Deskew:
    name = "deskew"
    default_options = {"enabled": True, "max_angle": 15.0}

    def apply(self, ctx: PreContext) -> PreContext | None:
        angle = estimate_skew(ctx.image)          # 你自己的算法
        ctx.image = rotate(ctx.image, angle)
        # 任何几何改动必须同步更新 transform，否则结果坐标会错
        ctx.transform = ctx.transform.then(
            Affine.rotation(angle, center=(ctx.image.shape[1] / 2, ctx.image.shape[0] / 2))
        )
        return ctx        # 返回 None 表示未改动图像
```

**铁律**：`ctx.transform` 记录「原图 → 当前送入图」的仿射变换；推理后核心统一执行
`transform.inverse` 逆映射，保证 `OcrResult.items[].box` 恒为原图像素坐标。

- `self.options`：管线执行前把 `default_options ← config.options["deskew"]` 合并后注入。
- `ctx.meta`：跨步骤共享（内置 `dark_to_light` 在此写 `theme`，供 `pad_border` 取补边颜色）。

## 后处理：四个正交轴

```
items ─①grouping→ lines ─②modifiers.on_lines→ ─③Assembler(mode)+Joiner→ blocks
                 ─②modifiers.on_blocks→ ─④Renderer(output)→ str|dict
```

### 修饰插件（Postprocessor）

两个 hook 都可选，管线只调用实现了的那个：

```python
from oxocr.postprocess.base import Postprocessor, PostContext, register_postprocessor


@register_postprocessor("sensitive_filter")
class SensitiveFilter:
    name = "sensitive_filter"
    default_options = {"enabled": True, "words": []}

    def on_blocks(self, blocks, ctx: PostContext):     # 合并后整段处理
        bad = set(self.options["words"])
        for b in blocks:
            if any(w in b.text for w in bad):
                b.text = "[已过滤]"
        return blocks

    # def on_lines(self, lines, ctx): ...   # 需在行内合并「之前」修正时用这个（如 ghost_filter）
```

> 为什么分两个 hook：幽灵 token 过滤必须在行内合并**前**（`on_lines`），整词纠错必须在
> 合并**后**（`on_blocks`）——「矣」与「键参数」原是两个 box，只有拼成词才能整词替换。

### 组装器（排版模式，第 5 种）

```python
from oxocr.postprocess.base import Assembler, register_assembler


@register_assembler("table")
class TableAssembler:
    name = "table"
    default_join = "smart_space"        # 该模式的默认 JoinPolicy

    def assemble(self, lines, ctx: PostContext, joiner):
        ...                             # 返回 list[TextBlock]
```

### 连接策略 / 渲染器

```python
from oxocr.postprocess.base import register_joiner, register_renderer

@register_joiner("comma")
class CommaJoiner:
    name = "comma"
    def join(self, items, ctx): return ", ".join(it.text for it in items)

@register_renderer("csv")
class CsvRenderer:
    name = "csv"
    def render(self, blocks, ctx): return "\n".join(b.text for b in blocks)
```

---

## 开关的四个层级（前后处理同理）

| 层级     | 写法                                                              | 场景                   |
| -------- | ----------------------------------------------------------------- | ---------------------- |
| 整阶段   | `oxocr.ocr(img, preprocess=False, postprocess=False)`           | 只要原始条目           |
| 步骤列表 | `preprocess={"steps": ["pad_border", "deskew"]}`                | 增删某步 / 改顺序      |
| 单步开关 | `preprocess={"options": {"dark_to_light": {"enabled": False}}}` | 保留配置便于回切       |
| 构造注入 | `OcrEngine(preprocess=[Deskew(), PadBorder()])`                 | 直接传实例，绕过注册表 |

后处理另有 `mode`（选组装器）与 `output`（选渲染器）两个正交开关：
`oxocr.ocr(img, mode="table", output="csv")`。

## 第三方 entry point

```toml
[project.entry-points]
"oxocr.preprocessors"  deskew  = "my_pkg.pre:Deskew"
"oxocr.postprocessors" my_fix  = "my_pkg.post:MyFix"
"oxocr.assemblers"     table   = "my_pkg.asm:TableAssembler"
"oxocr.joiners"        comma   = "my_pkg.join:CommaJoiner"
"oxocr.renderers"      csv     = "my_pkg.render:CsvRenderer"
```

## 契约由测试保证

`test/contract/test_extension_contract.py` 用 dummy 模型 + dummy 前处理断言：几何改动经
`transform` 逆映射回到原图坐标、`as_text(mode)` 不重新推理、四级开关生效。新增插件时，
参照它补一条契约用例即可。
