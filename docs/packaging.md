# 打包与发布

oxocr 是纯 Python 包，用 [hatchling](https://hatch.pypa.io/) 构建，
产出平台无关的 `py3-none-any` wheel。模型权重不入库（`test/models/` 已被 `.gitignore` 排除），
也不随包分发：wheel 只含 `oxocr/` 包代码与内置数据（`oxocr/config_data/*.json`），
用户安装后自行下载权重（见 README「快速开始」）。

## 1. 准备工具

```bash
pip install build twine
```

- `build`：PEP 517 前端，调用 hatchling 产出 sdist 与 wheel。
- `twine`：校验产物并上传到 PyPI。

构建默认启用隔离环境，会自动联网获取 `hatchling>=1.21`；离线机器可先 `pip install hatchling`
再加 `--no-isolation`。

## 2. 同步版本号

发版前两处必须一致：

- `pyproject.toml` → `[project] version`
- `oxocr/__init__.py` → `__version__`

```bash
grep -n "version" pyproject.toml
grep -n "__version__" oxocr/__init__.py
```

PyPI 不允许覆盖已发布的版本号，任何改动后重新上传都要升版本。

## 3. 构建

在仓库根目录执行：

```bash
python -m build            # 同时产出 sdist(.tar.gz) 与 wheel(.whl) 到 dist/
python -m build --wheel    # 只构建 wheel
python -m build --sdist    # 只构建 sdist
```

产物示例（版本以实际为准）：

```
dist/
├── oxocr-0.1.0-py3-none-any.whl
└── oxocr-0.1.0.tar.gz
```

> 也可用 `pip wheel . --no-deps -w dist` 只构建 wheel，效果等同 `build --wheel`。
> 注意解释器须满足 `requires-python >= 3.10`，否则 pip 会拒绝构建。

## 4. 检查产物

```bash
twine check dist/*                              # 校验元数据与 README 渲染
python -m zipfile -l dist/oxocr-0.1.0-py3-none-any.whl   # 列出 wheel 内容
```

确认：

- 只包含 `oxocr/` 包与 `oxocr-0.1.0.dist-info/`，**不含** `test/`、`test/models/` 权重、`examples/`。
- 内置数据 `oxocr/config_data/ocr_corrections.json` 在包内（运行期纠错规则依赖它）。
- `twine check` 全部 PASSED。

## 5. 本地安装验证

建议在干净虚拟环境里装一遍，确认依赖与入口点正常：

```bash
python -m venv /tmp/oxocr-verify
# Windows: /tmp/oxocr-verify/Scripts/activate；Linux/macOS: source /tmp/oxocr-verify/bin/activate
pip install dist/oxocr-0.1.0-py3-none-any.whl

oxocr version
oxocr ls
python -c "import oxocr; print(oxocr.__version__)"
```

`oxocr ls` 能列出模型清单即说明包代码与内置数据完整；权重需另行 `oxocr download <name>`。

## 6. 发布到 PyPI

先传 TestPyPI 演练，确认无误再传正式 PyPI：

```bash
twine upload --repository testpypi dist/*       # 演练
pip install -i https://test.pypi.org/simple/ oxocr   # 从 TestPyPI 安装验证

twine upload dist/*                             # 正式发布
```

凭据用 PyPI API token（`~/.pypirc` 或环境变量）：

```bash
export TWINE_USERNAME=__token__
export TWINE_PASSWORD=pypi-xxxxxxxxxxxxxxxx     # 以 pypi- 开头的 token
```

发布后 `pip install oxocr` 即可安装。同一版本号不可重复上传，迭代请升版本后重新 `build`。
