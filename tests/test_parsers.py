from pathlib import Path

import pandas as pd

from agent_mvp.parsers import parse_rule_library, profile_dataframe, read_dataset


def test_read_dataset_and_profile(tmp_path: Path) -> None:
    path = tmp_path / "data.csv"
    path.write_text("a,b\n1,x\n2,\n3,y\n", encoding="utf-8")

    df = read_dataset(path)
    profile = profile_dataframe(df)

    assert profile["rows"] == 3
    assert profile["columns"] == 2
    assert profile["column_profiles"][1]["missing_count"] == 1


def test_parse_text_rule_library(tmp_path: Path) -> None:
    path = tmp_path / "rules.md"
    path.write_text("R001: temperature > 80 异常\nR002: status == WARN\n", encoding="utf-8")

    rules = parse_rule_library(path, known_fields=["temperature", "status"])

    assert len(rules) == 2
    assert rules[0].rule_id == "R001"
    assert rules[0].condition == "temperature > 80"
    assert rules[1].target_fields == ["status"]


def test_parse_json_rule_library(tmp_path: Path) -> None:
    path = tmp_path / "rules.json"
    path.write_text(
        '{"rules":[{"id":"X1","name":"hot","condition":"temperature > 80","fields":["temperature"]}]}',
        encoding="utf-8",
    )

    rules = parse_rule_library(path, known_fields=["temperature"])

    assert rules[0].rule_id == "X1"
    assert rules[0].target_fields == ["temperature"]


def test_parse_yaml_structured_rule_library(tmp_path: Path) -> None:
    path = tmp_path / "rules.yaml"
    path.write_text(
        """
rules:
  - rule_id: R100
    name: 温度偏高
    field: temperature
    operator: ">"
    value: 80
    severity: high
    recommendation: 检查温控系统
    tags: [temperature, threshold]
""",
        encoding="utf-8",
    )

    rules = parse_rule_library(path, known_fields=["temperature"])

    assert rules[0].rule_id == "R100"
    assert rules[0].condition == "temperature > 80"
    assert rules[0].severity == "high"
    assert rules[0].recommendation == "检查温控系统"
    assert rules[0].tags == ["temperature", "threshold"]
