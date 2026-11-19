from __future__ import annotations

import json
import os
import time

from oxocr.postprocess.corrections import Rule, RuleEngine, _merge_rules


def test_user_rule_overrides_builtin():
    builtin = [Rule("矣键参数", "关键参数", False)]
    user = [Rule("矣键参数", "关键设参", False)]
    merged = _merge_rules(builtin, user)
    assert merged[0].right == "关键设参"


def test_longest_rule_wins():
    rules = _merge_rules([Rule("参数", "参树", False)], [Rule("关键参数", "关键参树", False)])
    # 长度降序：整词规则先匹配
    assert rules[0].wrong == "关键参数"


def test_single_scan_no_chain(tmp_path):
    # A→B、B→C 时，re.sub 不回扫，A 只会被替换为 B，不会连锁成 C
    cfg = tmp_path / "c.json"
    cfg.write_text(
        json.dumps({"rules": [
            {"wrong": "甲", "right": "乙"},
            {"wrong": "乙", "right": "丙"},
        ]}),
        encoding="utf-8",
    )
    engine = RuleEngine(builtin_path=None, user_path=str(cfg))
    assert engine.correct("甲") == "乙"


def test_regex_rule(tmp_path):
    cfg = tmp_path / "c.json"
    cfg.write_text(
        json.dumps({"rules": [{"wrong": r"O(\d)R", "right": r"OCR-\1", "regex": True}]}),
        encoding="utf-8",
    )
    # right 按字面文本处理，反向引用不展开 → 命中时替换为字面串
    engine = RuleEngine(builtin_path=None, user_path=str(cfg))
    assert engine.correct("O2R") == "OCR-\\1"


def test_mtime_invalidation(tmp_path):
    cfg = tmp_path / "c.json"
    cfg.write_text(json.dumps({"rules": [{"wrong": "A", "right": "B"}]}), encoding="utf-8")
    engine = RuleEngine(builtin_path=None, user_path=str(cfg))
    assert engine.correct("A") == "B"
    time.sleep(0.01)
    cfg.write_text(json.dumps({"rules": [{"wrong": "A", "right": "Z"}]}), encoding="utf-8")
    os.utime(cfg, (time.time() + 1, time.time() + 1))
    assert engine.correct("A") == "Z"


def test_corrupt_config_returns_original(tmp_path):
    cfg = tmp_path / "c.json"
    cfg.write_text("{ not json", encoding="utf-8")
    engine = RuleEngine(builtin_path=None, user_path=str(cfg))
    assert engine.correct("hello") == "hello"


def test_builtin_file_loads():
    engine = RuleEngine(user_path=os.devnull)  # devnull 读不到规则
    assert engine.correct("矣键参数") == "关键参数"
