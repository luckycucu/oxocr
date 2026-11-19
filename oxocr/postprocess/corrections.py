"""OCR 整词纠错（移植参考实现，去掉 xpin 硬编码）。

- 纠错发生在合并**之后**（on_blocks）：「矣」与「键参数」原为不同 box。
- 单次扫描：所有规则合成一条 alternation 正则，re.sub 不回扫替换结果。
- 按 mtime 失效：用户改完 json 下次识别即生效。
- 用户规则目录改为 ~/.oxocr（OXOCR_CACHE_DIR 可覆盖）。
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, NamedTuple

from oxocr.types import TextBlock

if TYPE_CHECKING:
    from oxocr.postprocess.base import PostContext

logger = logging.getLogger("oxocr.corrections")

__all__ = ["Rule", "RuleEngine", "Corrections", "correct_text"]

_DATA_FILE = "ocr_corrections.json"
# os.path.abspath 而非 resolve：frozen 下 __file__ 可能指向包内虚拟路径
_BUILTIN_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config_data", _DATA_FILE
)


def _default_user_path() -> str:
    from oxocr.registry.download import cache_dir

    return str(cache_dir() / _DATA_FILE)


class Rule(NamedTuple):
    wrong: str
    right: str
    regex: bool


def _parse_rules(data: Any, source: str) -> list[Rule]:
    if not isinstance(data, dict):
        logger.warning("%s 顶层结构不是对象，已忽略", source)
        return []
    raw_rules = data.get("rules")
    if not isinstance(raw_rules, list):
        logger.warning("%s 缺少 rules 数组，已忽略", source)
        return []
    rules: list[Rule] = []
    for index, item in enumerate(raw_rules):
        if not isinstance(item, dict):
            logger.warning("%s 第 %d 条规则不是对象，已跳过", source, index + 1)
            continue
        wrong, right = item.get("wrong"), item.get("right")
        if not isinstance(wrong, str) or not isinstance(right, str) or not wrong:
            logger.warning("%s 第 %d 条规则的 wrong/right 非法，已跳过", source, index + 1)
            continue
        rules.append(Rule(wrong, right, bool(item.get("regex", False))))
    return rules


def _read_rules_file(path: str) -> list[Rule]:
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        return []
    except (OSError, ValueError) as e:
        logger.warning("读取纠错配置失败，已忽略：%s（%s）", path, e)
        return []
    return _parse_rules(data, path)


def _merge_rules(builtin: list[Rule], user: list[Rule]) -> list[Rule]:
    merged = list(builtin)
    for rule in user:
        merged = [r for r in merged if r.wrong != rule.wrong]
        merged.append(rule)
    merged.sort(key=lambda r: len(r.wrong), reverse=True)
    return merged


def _build(rules: list[Rule]):
    branches: list[str] = []
    table: dict[str, str] = {}
    for index, rule in enumerate(rules):
        pattern = rule.wrong if rule.regex else re.escape(rule.wrong)
        try:
            re.compile(pattern)
        except re.error as e:
            logger.warning("规则正则非法，已跳过：%r（%s）", rule.wrong, e)
            continue
        name = f"r{index}"
        branches.append(f"(?P<{name}>{pattern})")
        table[name] = rule.right
    if not branches:
        return None, None
    try:
        compiled = re.compile("|".join(branches))
    except re.error as e:
        logger.warning("纠错规则无法编译，本轮跳过纠错（%s）", e)
        return None, None

    def _sub(match: re.Match) -> str:
        for name, value in match.groupdict().items():
            if value is not None and name in table:
                return table[name]
        return match.group(0)

    return compiled, _sub


def _mtime(path: str) -> float | None:
    try:
        return os.path.getmtime(path)
    except OSError:
        return None


class RuleEngine:
    """规则表 + 编译正则，按 mtime 失效缓存。路径可注入。"""

    def __init__(self, builtin_path: str | None = _BUILTIN_PATH, user_path: str | None = None):
        self.builtin_path = builtin_path
        self.user_path = user_path if user_path is not None else _default_user_path()
        self._lock = threading.Lock()
        self._stamp: tuple | None = None
        self._rules: list[Rule] = []
        self._pattern = None
        self._sub = None

    def _ensure_loaded(self) -> None:
        stamp = (
            _mtime(self.builtin_path) if self.builtin_path else None,
            _mtime(self.user_path) if self.user_path else None,
        )
        if stamp == self._stamp:
            return
        with self._lock:
            if stamp == self._stamp:
                return
            builtin = _read_rules_file(self.builtin_path) if self.builtin_path else []
            user = _read_rules_file(self.user_path) if self.user_path else []
            self._rules = _merge_rules(builtin, user)
            self._pattern, self._sub = _build(self._rules)
            self._stamp = stamp

    def rules(self) -> list[Rule]:
        self._ensure_loaded()
        return list(self._rules)

    def correct(self, text: str) -> str:
        self._ensure_loaded()
        if self._pattern is None or not text:
            return text
        return self._pattern.sub(self._sub, text)


_DEFAULT_ENGINE = RuleEngine()


def correct_text(text: str) -> str:
    return _DEFAULT_ENGINE.correct(text)


class Corrections:
    name: ClassVar[str] = "corrections"
    default_options: ClassVar[Mapping[str, Any]] = {
        "enabled": True,
        "user_rules": None,
        "builtin": True,
    }

    def __init__(self) -> None:
        self._engines: dict[tuple, RuleEngine] = {}

    def _engine(self, opts: Mapping[str, Any]) -> RuleEngine:
        builtin_path = _BUILTIN_PATH if opts.get("builtin", True) else None
        user_rules = opts.get("user_rules")
        user_path = str(Path(user_rules).expanduser()) if user_rules else _default_user_path()
        key = (builtin_path, user_path)
        if key not in self._engines:
            self._engines[key] = RuleEngine(builtin_path, user_path)
        return self._engines[key]

    def on_blocks(self, blocks: list[TextBlock], ctx: PostContext) -> list[TextBlock]:
        opts = getattr(self, "options", self.default_options)
        engine = self._engine(opts)
        for block in blocks:
            if block.text:
                block.text = engine.correct(block.text)
        return blocks
