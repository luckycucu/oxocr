# 运行测试

## 1. 安装开发依赖

```bash
pip install -e ".[dev]"
```

`dev` 额外安装：`pytest`、`pytest-cov`、`ruff`、`mypy`、`opencv-python-headless`
（OpenCV 仅作 imageops 对照测试用，运行期不依赖）。

## 2. 测试分层

| 目录 | 内容 | 需要权重 |
| --- | --- | --- |
| `test/unit/` | 纯逻辑单测：几何/仿射、配置、注册表、CLI、词表、前后处理步骤等 | 否 |
| `test/contract/` | 扩展契约：dummy 模型 + dummy 前处理，验证「写一个 `infer()` 即接入」与常驻池语义 | 否 |
| `test/integration/` | 真实模型端到端：PP-OCR / MFR / LaTeX-OCR / 读光 / anaksor / Indra 各家族 | 是 |

pytest 标记（定义见 `pyproject.toml`）：

- `model`：需要本地 ONNX 权重（`test/models`，离线）。
- `integration`：集成测试（常驻池 / 子进程 / 真实模型）。

集成用例同时带 `model` 与 `integration` 两个标记，因此 `-m "not model"` 会一并排除。

## 3. 离线权重

`test/models/` **不入库**（`.gitignore` 排除），需自行放置。`test/conftest.py` 默认设置
`OXOCR_OFFLINE=1`，测试全程离线、绝不联网下载；某家族权重缺失时，相关用例自动 `skip`，
不影响其余用例。

按需下载要测的模型到 `test/models`：

```bash
oxocr download ppocrv6_tiny -D test/models
oxocr download ppocrv6_small -D test/models
oxocr download pix2text_mfr -D test/models
oxocr download rapid_latex_ocr -D test/models
oxocr download duguang_ocr_small -D test/models
oxocr download anaksor_ocr -D test/models
oxocr download indra_ocr_cjk -D test/models
```

或一次性下载全部模型：

```python
import oxocr

for m in oxocr.list_models():
    oxocr.download_model(m.name, model_dir="test/models")
```

各家族权重在 `test/models` 下的相对路径见 `oxocr.list_models()` 与 [model-info.md](model-info.md)。

## 4. 运行 pytest

在仓库根目录执行（`testpaths` 已指向 `test/`，conftest 已强制离线）：

```bash
pytest                              # 全部用例（缺权重的家族自动 skip）
pytest -m "not model"               # 仅无需权重的 unit + contract
pytest -m integration               # 仅集成用例
pytest test/unit/test_box.py        # 单个文件
pytest test/unit/test_box.py::test_xxx   # 单个用例
pytest -k anaksor                   # 按关键字筛选
```

覆盖率（`pytest-cov`）：

```bash
pytest --cov=oxocr --cov-report=term-missing
pytest --cov=oxocr --cov-report=html      # 生成 htmlcov/ 报告
```

## 5. 离线冒烟脚本

`test/script/run_offline_*.py` 不依赖 pytest，直接加载 `test/models` 权重识别合成图或指定图片，
用于快速肉眼验证各家族效果：

```bash
python test/script/run_offline_ppocr.py [IMAGE ...]
python test/script/run_offline_mfr.py [IMAGE ...]
python test/script/run_offline_latex_ocr.py [IMAGE ...]
python test/script/run_offline_duguang.py [IMAGE ...]
python test/script/run_offline_anaksor.py [IMAGE ...]
python test/script/run_offline_indra.py [IMAGE ...]
```

不传 `IMAGE` 时使用内置合成图；脚本自动设置 `OXOCR_OFFLINE=1` 与 `OXOCR_MODEL_DIR=test/models`。

## 6. 静态检查

```bash
ruff check .                # lint（规则见 pyproject.toml [tool.ruff]）
ruff format --check .       # 格式检查
mypy oxocr                  # 类型检查
```