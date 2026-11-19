# oxocr CLI

Docker 风格命令行：`oxocr <COMMAND> <ACTION> [OPTIONS]`。
旧式扁平调用（`oxocr img.png --model ...`）与 `--list-models` **已移除**，需改用子命令写法。

## 命令树

```
oxocr [--lang {en,zh}] <COMMAND> ...

COMMAND:
  run      IMAGE [OPTIONS] [--vis PATH]       识别图像并输出结果（可选保存可视化图）
  models   {list|ls|info|inspect|search|download|pull} ...
                                              模型管理（别名 model；省略 ACTION 默认 list）
  version                                     显示版本（等价 -v/--version）

models ACTION:
  list | ls    [-i] [--sort FIELD] [-r] [-D DIR]   列出可用模型（表格）
  info | inspect NAME [-D DIR]                     显示模型详情
  search KEYWORD [-f FIELD] [-i] [--sort FIELD] [-r] [-D DIR]
                                                   按名称/语言搜索模型（表格）
  download | pull NAME [--source SRC] [--force] [-D DIR]
                                                   下载模型权重

顶层简写:
  oxocr list   [-i]        == oxocr models list
  oxocr ls     [-i]        == oxocr models ls
  oxocr inspect NAME       == oxocr models info
  oxocr search KEYWORD     == oxocr models search
  oxocr download NAME      == oxocr models download
  oxocr pull NAME          == oxocr models pull
```

## run 选项

| 选项                       | 说明                                   | 默认              |
| -------------------------- | -------------------------------------- | ----------------- |
| `IMAGE`                  | 图像路径（必填位置参数）               | —                |
| `-m, --model`            | 模型名                                 | `ppocrv6_small` |
| `-M, --mode`             | 排版模式 `raw/line/para/layout`      | `para`          |
| `-o, --output, --format` | 输出格式 `text/json`                 | `text`          |
| `-c, --cls`              | 开启方向分类                           | 关闭              |
| `-s, --text-score FLOAT` | 识别置信度阈值                         | 模型默认          |
| `-t, --threads N`        | 推理线程数                             | 模型默认          |
| `--vis PATH`             | 保存可视化图像到 PATH（格式按扩展名）  | 不保存            |
| `--vis-mode MODE`        | 可视化模式 `box_txt/box/ocr`         | `box_txt`       |
| `--font PATH`            | 可视化文字字体文件（ttf/ttc）          | 自动探测系统字体  |
| `-D, --model-dir DIR`    | 模型根目录（覆盖 `OXOCR_MODEL_DIR`） | 环境/缓存目录     |
| `--debug`                | 输出调试日志                           | 关闭              |

## run 可视化（--vis）

`--vis PATH` 在识别之余把检测框与识别文字画到图上并保存到 PATH
（PNG/JPG 等，格式按扩展名），纯 Pillow + numpy 实现，无 OpenCV 依赖。
stdout 仍只输出识别结果（便于管道），保存提示输出到 stderr。

| `--vis-mode` | 效果                                                                                    |
| -------------- | --------------------------------------------------------------------------------------- |
| `box_txt`    | 双联图（默认）：左半原图叠加半透明彩色框，右半白底按框位置透视绘制文字，宽度为原图 2 倍 |
| `box`        | 仅在原图副本上画红色检测框，尺寸不变                                                    |
| `ocr`        | 画框 + 右侧文字面板（编号/文本/置信度），图缩放到最长边 600                             |

```bash
oxocr run shot.png --vis out.png                    # 双联可视化图
oxocr run shot.png --vis out.png --vis-mode box     # 只画框
oxocr run shot.png --vis out.jpg --vis-mode ocr     # 画框 + 文字面板
oxocr run shot.png --vis out.png --font C:/Windows/Fonts/msyh.ttc
```

字体说明：

- 不内置字体。默认按候选链探测系统字体（Windows 微软雅黑/宋体、macOS 苹方、
  Linux Noto CJK/文泉驿等，优先中英文覆盖），`--font` 可指定任意 ttf/ttc
- 单个字体缺字形会显示为方框：识别泰语/天城文等文种时建议 `--font` 指定对应字体
  （如泰语 `leelawui.ttf`、Noto Sans Thai）
- libraqm 可用时自动启用 RAQM 排版（阿拉伯文连写、天城文组合等复杂文种）
- 找不到可用字体且未指定 `--font` 时报错退出（`box` 模式不画文字，无需字体）

## models list / ls

| 选项                     | 说明                                   | 默认          |
| ------------------------ | -------------------------------------- | ------------- |
| `-i, --installed-only` | 仅列出已安装模型                       | 关闭          |
| `--sort FIELD`         | 排序字段 `released/name/size`        | `released`  |
| `-r, --reverse`        | 反转排序方向                           | 关闭          |
| `-D, --model-dir DIR`  | 模型根目录（覆盖 `OXOCR_MODEL_DIR`） | 环境/缓存目录 |
| `--debug`              | 输出调试日志                           | 关闭          |

输出为 docker ps 风格表格：首行标题，无边框线，列宽按内容自适应。
默认按**发布日期降序**（最新在前），排序规则见下文。

```
$ oxocr ls
    NAME                        RELEASED    SIZE(MB)   LICENSE            LANGUAGES
[ ] anaksor_ocr                 2026-11-04      90.4   Apache-2.0         km, en
[ ] indra_ocr_cjk               2026-10-19      28.7   RAIL-M (modified)  zh, zh_tw, ja, ...
[ ] indra_ocr_latin             2026-10-19      20.7   RAIL-M (modified)  latin, en, fr, ...
[ ] ppocrv6_small               2026-06-11      30.3   Apache-2.0         zh, zh_tw, en, ...
...
```

各列含义：

- 行首标记：`[V]` 已安装 / `[ ]` 未安装（ASCII 标记，兼容 GBK 控制台；标题行该列留空）
- `NAME`：模型名（家族不单独显示，可用 `models inspect` 查看）
- `RELEASED`：**raw 开源日期**（ISO 格式），缺失显示 `-`
- `SIZE(MB)`：权重大小，MB、1 位小数、右对齐；未知显示 `-`
- `LICENSE`：开源协议；长协议缩写显示（`RAIL-M (modified)` = AI PUBS OPEN RAIL-M (modified)），
  完整协议与商用限制见 `models inspect`；未知显示 `-`
- `LANGUAGES`：支持语言，只显示前 3 个，超出加 `...`；`--lang zh` 时显示中文语言名

`--lang zh` 时标题行与语言列切换为中文（中文按终端全角宽度对齐）：

```
$ oxocr ls --lang zh
    名称                        开源日期    大小(MB)   开源协议           支持语言
[ ] anaksor_ocr                 2026-11-04      90.4   Apache-2.0         高棉文, 英文
[ ] indra_ocr_cjk               2026-10-19      28.7   RAIL-M (modified)  简体中文, 繁体中文, 日文, ...
...
```

`-i` 无已安装模型时只输出标题行，退出码 0。完整模型信息用 `models inspect` 查看。

## models info / inspect

`info` / `inspect NAME`：输出模型名、家族、语种、安装状态、能力、文件清单等；
安装状态按 `-D/--model-dir` 指定的目录计算。`-D/--model-dir`、`--debug` 同上。

## models search

`oxocr search KEYWORD`（等价 `oxocr models search KEYWORD`），输出与 `ls` 相同的表格，
默认按发布日期降序。无匹配时 stderr 输出 `no matching models`
（`--lang zh` 时 `未找到匹配的模型`），退出码 1（grep 惯例，便于脚本判断）。

| 选项                     | 说明                                   | 默认          |
| ------------------------ | -------------------------------------- | ------------- |
| `KEYWORD`              | 搜索关键词（必填位置参数，单个）       | —            |
| `-f, --field FIELD`    | 检索字段 `name/family/lang/all`      | name+lang     |
| `-i, --installed-only` | 仅搜索已安装模型                       | 关闭          |
| `--sort FIELD`         | 排序字段 `released/name/size`        | `released`  |
| `-r, --reverse`        | 反转排序方向                           | 关闭          |
| `-D, --model-dir DIR`  | 模型根目录（覆盖 `OXOCR_MODEL_DIR`） | 环境/缓存目录 |
| `--debug`              | 输出调试日志                           | 关闭          |

字段匹配规则（均不区分大小写）：

| `-f` 取值 | 检索范围                           | 匹配方式                                                 |
| ----------- | ---------------------------------- | -------------------------------------------------------- |
| `name`    | 模型名 + 中文名                    | 子串                                                     |
| `family`  | 家族                               | 子串                                                     |
| `lang`    | 语言代码 + 中文语言名 + 英文语言名 | 代码**前缀**；中文名子串；英文名**按词前缀** |
| `all`     | 以上全部                           | 同上                                                     |
| 缺省        | `name` + `lang`                | 同上                                                     |

语言匹配示例：`zh` 命中 `zh`/`zh_tw`；`ar` 只命中 `ar`（不会误中 devanagari）；
`中文` 命中 简体中文/繁体中文；`chinese` 命中 Chinese (Simplified) 等；
`arab` 命中 Arabic（英文名按词前缀，避免 `hi` 误中 Chinese）。

```
oxocr search zh                 # 支持中文（zh/zh_tw）的模型
oxocr search 中文                # 按中文语言名匹配，结果同上
oxocr search chinese            # 按英文语言名匹配
oxocr search ppocrv6 -f name    # 只查名称 → ppocrv6_tiny/small/medium
oxocr search 读光 -f name        # 中文名命中 → duguang_ocr_*
oxocr search latex -f lang      # 公式识别模型
oxocr search indra -f family    # indra_ocr 全家族
oxocr search zh -i --sort size  # 已安装 + 支持中文，按大小降序
```

## models download / pull

`oxocr models download NAME`（别名 `pull`，顶层简写 `oxocr download NAME`）：
从 hub 仓库逐文件下载模型权重到模型根目录，**ModelScope 优先，失败自动回退 HuggingFace**。
已存在的文件跳过（幂等），下载完成打印各角色文件路径；失败 stderr 输出 `错误: ...`，退出码 1。

| 选项                    | 说明                                   | 默认                                     |
| ----------------------- | -------------------------------------- | ---------------------------------------- |
| `NAME`                | 模型名（必填位置参数）                 | —                                       |
| `--source SRC`        | 下载源 `auto/modelscope/huggingface` | `auto`（或 `OXOCR_DOWNLOAD_SOURCE`） |
| `--force`             | 文件已存在也重新下载                   | 关闭                                     |
| `-D, --model-dir DIR` | 模型根目录（覆盖 `OXOCR_MODEL_DIR`） | 环境/缓存目录                            |
| `--debug`             | 输出调试日志                           | 关闭                                     |

```
oxocr models download ppocrv6_tiny                 # auto：ModelScope → HuggingFace
oxocr download ppocrv6_tiny --source modelscope    # 强制单源
oxocr pull ppocrv6_small -D ./models --force       # 指定目录并强制重下
oxocr pull ppocr_cls                               # 下载全部方向分类权重（3 个 cls）
```

说明：

- `ppocr_cls` 是**下载 bundle**（非推理模型，不出现在 `models ls`）：一次下载全部 3 个
  方向分类权重（`v2_0_cls` / `textline_ori_mobile` / `textline_ori_server`，ppocr 全家族共用）
- 所有内置模型共用下载仓库：ModelScope `mscoder/oxocr`、HuggingFace `Luoaho/oxocr`，
  仓库内文件路径与模型清单相对路径一致
- 失败重试：ModelScope 3 次（`MODELSCOPE_DOWNLOAD_MAX_RETRIES`，尊重用户显式值）；
  HuggingFace 由库内置（5 次，无公开配置项）
- **不校验 sha256**：仓库文件可能更新，包发版后硬编码哈希无法跟随
- `OXOCR_OFFLINE=1` 时禁止联网，缺文件直接报错
- HuggingFace 镜像可用 `HF_ENDPOINT` 环境变量（huggingface_hub 原生支持）
- 推理（`run`）**不会自动下载**，缺权重时报错并提示期望路径

## 排序规则（--sort / -r）

`ls` 与 `search` 共用。各字段默认方向：

| 字段         | 默认方向 | 说明                 |
| ------------ | -------- | -------------------- |
| `released` | 降序     | 最新发布在前（默认） |
| `size`     | 降序     | 最大在前             |
| `name`     | 升序     | A→Z，不区分大小写   |

- `-r/--reverse` 反转当前方向
- 缺失值（日期 `-`、大小未知）恒排末尾，不受方向影响
- 同值按名称升序兜底，不受 `-r` 影响

## 帮助语言（-h 的二级开关）

```
oxocr -h                       英文帮助（默认）
oxocr -h --lang zh             中文帮助（与 -h 顺序无关，也支持 --lang=zh）
oxocr run -h --lang zh         run 子命令中文帮助
oxocr models ls -h --lang zh   models list 中文帮助
oxocr search -h --lang zh      models search 中文帮助
```

注意：`--lang` 影响帮助输出，以及 `ls`/`search` 表格的标题行与语言列
（`zh` 时显示中文标题和中文语言名）；对 `run` 的识别结果无影响
（如 `oxocr run a.png --lang zh` 不改变输出）。
非法值（如 `--lang fr`）无论与 `-h` 的先后顺序，都报 `invalid choice`，退出码 2。

## 退出码

| 码 | 含义                                                                                                                                     |
| -- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| 0  | 成功                                                                                                                                     |
| 1  | 业务错误（模型不存在、缺权重、推理失败等，stderr 输出 `错误: ...`；search 无匹配时输出 `no matching models` / `未找到匹配的模型`） |
| 2  | 用法错误（argparse 报错；无子命令时打印帮助）                                                                                            |

## 注意

- `-D/--debug` 注册在各子命令上，须放在子命令之后：`oxocr run a.png -D DIR`，
  不能写成 `oxocr -D DIR run a.png`。
- 顶层简写 `ls/list/inspect/search/download/pull` 由 argv 预处理映射到 `models` 子命令，
  因此顶层 `invalid choice` 提示中不列出它们，但可正常使用。
