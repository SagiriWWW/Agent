"""Schema metadata helpers and local semantic suggestions.

这一层是“Schema/语义 Agent”的 MVP 版本：它先用稳定配置和本地启发式给
字段生成中文名、分组、角色和置信度。后续接入模型时，可以替换
suggest_schema_metadata 的候选生成逻辑，但报告和工具仍读取同一种结构。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

from .runtime import RunContext
from .utils import write_json


DEFAULT_SCHEMA_PATH = Path(__file__).resolve().parent.parent / "configs" / "telemetry_schema.yaml"
DEFAULT_SEMANTICS_PATH = Path(__file__).resolve().parent.parent / "configs" / "schema_semantics.yaml"
DEFAULT_MAPPING_OVERRIDES_PATH = Path(__file__).resolve().parent.parent / "configs" / "schema_mapping_overrides.yaml"


MEASUREMENT_LABELS = {
    "battery_remaining": "电池剩余电量",
    "current_battery": "电池电流",
    "voltage_battery": "电池电压",
    "current_consumed": "累计耗电",
    "Vcc": "供电电压",
    "direction": "方向",
    "speed": "速度",
    "altitude": "高度",
    "relative_alt": "相对高度",
    "pitch": "俯仰角",
    "pitchspeed": "俯仰角速度",
    "roll": "横滚角",
    "rollspeed": "横滚角速度",
    "yaw": "偏航角",
    "yawspeed": "偏航角速度",
    "hdg": "航向角",
    "vibration_x": "X 轴振动",
    "vibration_y": "Y 轴振动",
    "vibration_z": "Z 轴振动",
    "xacc": "X 轴加速度",
    "yacc": "Y 轴加速度",
    "zacc": "Z 轴加速度",
    "xgyro": "X 轴角速度",
    "ygyro": "Y 轴角速度",
    "zgyro": "Z 轴角速度",
    "xmag": "X 轴磁场",
    "ymag": "Y 轴磁场",
    "zmag": "Z 轴磁场",
    "press_abs": "绝对气压",
    "press_diff": "气压差",
    "temperature": "温度",
    "alt": "高度",
    "vx": "X 向速度",
    "vy": "Y 向速度",
    "vz": "Z 向垂直速度",
}


SOURCE_PREFIX_LABELS = {
    "1": "主",
    "147": "备用",
}


GROUP_BY_TOKEN = {
    "battery": "power",
    "current": "power",
    "voltage": "power",
    "vcc": "power",
    "speed": "navigation",
    "direction": "navigation",
    "alt": "navigation",
    "altitude": "navigation",
    "hdg": "navigation",
    "vx": "navigation",
    "vy": "navigation",
    "vz": "navigation",
    "pitch": "attitude",
    "roll": "attitude",
    "yaw": "attitude",
    "vibration": "vibration",
    "acc": "imu",
    "gyro": "imu",
    "mag": "imu",
    "press": "barometer",
    "temperature": "barometer",
}

STATISTIC_SUFFIXES = {
    "avg": "均值",
    "mean": "均值",
    "last": "末值",
    "raw": "原始值",
    "min": "最小值",
    "max": "最大值",
    "std": "标准差",
    "sum": "累计值",
    "count": "计数",
}

GENERIC_TOKENS = {
    "avg",
    "last",
    "mean",
    "median",
    "min",
    "max",
    "std",
    "sum",
    "count",
    "value",
    "values",
    "field",
    "fields",
    "data",
    "sensor",
    "metric",
    "pct",
    "percent",
    "percentage",
}


def load_schema_metadata(path: str | Path | None = None) -> dict[str, Any]:
    """Load the stable schema configuration if available."""
    schema_path = Path(path) if path else DEFAULT_SCHEMA_PATH
    if not schema_path.exists():
        return {}
    data = yaml.safe_load(schema_path.read_text(encoding="utf-8")) or {}
    return data if isinstance(data, dict) else {}


def load_schema_semantics(path: str | Path | None = None) -> dict[str, Any]:
    """Load stable semantic roles used for cross-dataset field mapping."""
    schema_path = Path(path) if path else DEFAULT_SEMANTICS_PATH
    if not schema_path.exists():
        return {}
    data = yaml.safe_load(schema_path.read_text(encoding="utf-8")) or {}
    return data if isinstance(data, dict) else {}


def load_schema_mapping_overrides(path: str | Path | None = None) -> dict[str, Any]:
    """Load human-confirmed schema mapping overrides.

    这个配置用于把已经人工确认过的客户字段语义或规则字段映射沉淀下来。
    它和 schema_semantics.yaml 的区别是：semantics 描述“可复用语义角色”，
    overrides 描述“某个具体字段/候选映射已经被确认过”。
    """
    override_path = Path(path) if path else DEFAULT_MAPPING_OVERRIDES_PATH
    if not override_path.exists():
        return {
            "path": str(override_path),
            "confirmed_mappings": {},
            "confirmed_rule_mappings": {},
            "raw": {},
        }
    data = yaml.safe_load(override_path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        data = {}
    return {
        "path": str(override_path),
        "confirmed_mappings": _normalize_confirmed_mapping_records(data.get("confirmed_mappings") or {}),
        "confirmed_rule_mappings": _normalize_confirmed_rule_mapping_records(data.get("confirmed_rule_mappings") or {}),
        "raw": data,
    }


def schema_mapping_override_for_field(field: str, overrides: dict[str, Any] | None) -> dict[str, Any] | None:
    """Return the confirmed semantic override for one dataset field."""
    if not isinstance(overrides, dict):
        return None
    mapping = (overrides.get("confirmed_mappings") or {}).get(_normalize_alias(field))
    return dict(mapping) if isinstance(mapping, dict) else None


def schema_mapping_override_for_rule_field(field: str, overrides: dict[str, Any] | None) -> dict[str, Any] | None:
    """Return the confirmed rule-field-to-dataset-field mapping if configured."""
    if not isinstance(overrides, dict):
        return None
    mapping = (overrides.get("confirmed_rule_mappings") or {}).get(_normalize_alias(field))
    return dict(mapping) if isinstance(mapping, dict) else None


def latest_schema_metadata(context: RunContext | None = None) -> dict[str, Any]:
    """Prefer run-level schema suggestions, then fall back to stable config."""
    if context is not None:
        latest = _latest_tool_result(context.tool_results, "suggest_schema_metadata")
        if isinstance(latest, dict) and isinstance(latest.get("schema"), dict):
            return latest["schema"]
    return load_schema_metadata()


def field_display_name(field: str, schema: dict[str, Any] | None = None) -> str:
    """Return the configured Chinese display name for one field."""
    schema = schema or load_schema_metadata()
    info = (schema.get("fields") or {}).get(field) if isinstance(schema, dict) else None
    if isinstance(info, dict) and info.get("display_name"):
        return str(info["display_name"])
    return infer_field_metadata(field)["display_name"]


def field_label(field: str, schema: dict[str, Any] | None = None, *, include_raw: bool = True) -> str:
    """Return a human-readable label, optionally keeping the raw field name."""
    display = field_display_name(field, schema)
    if not include_raw or display == field:
        return display
    return f"{display}({field})"


def label_expression(expression: Any, schema: dict[str, Any] | None = None, *, include_raw: bool = False) -> str:
    """Replace raw field names inside a rule/item expression with display names."""
    text = str(expression or "")
    if not text:
        return text
    schema = schema or load_schema_metadata()
    fields = (schema.get("fields") or {}).keys() if isinstance(schema, dict) else []
    for field in sorted(fields, key=len, reverse=True):
        display = field_label(str(field), schema, include_raw=include_raw)
        text = re.sub(rf"(?<![\w]){re.escape(str(field))}(?![\w])", display, text)
    return text


def suggest_schema_metadata(
    context: RunContext,
    *,
    max_fields: int = 120,
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    """Generate a schema suggestion artifact from dataset fields and rule bindings."""
    stable_schema = load_schema_metadata(config_path)
    stable_fields = stable_schema.get("fields") or {}
    profile = next(iter(context.dataset_profiles.values()), {})
    field_names = list(profile.get("column_names") or [])
    if not field_names:
        for dataset_id in context.dataset_file_ids:
            df = context.dataframes.get(dataset_id)
            if df is not None:
                field_names = [str(col) for col in df.columns]
                break
    field_names = field_names[: max(1, int(max_fields or 120))]

    rule_bound_fields = sorted({field for rule in context.rules for field in rule.target_fields})
    fields: dict[str, dict[str, Any]] = {}
    for field in field_names:
        if isinstance(stable_fields.get(field), dict):
            info = dict(stable_fields[field])
            info.setdefault("confidence", 0.95)
            info.setdefault("reason", "来自已固化的 telemetry_schema.yaml 配置。")
        else:
            info = infer_field_metadata(field)
        if field in rule_bound_fields:
            info["used_by_rule"] = True
            info["confidence"] = max(float(info.get("confidence", 0.0)), 0.86)
        fields[field] = info

    ignored_fields = [field for field, info in fields.items() if info.get("ignore")]
    analysis_columns = [
        field
        for field, info in fields.items()
        if not info.get("ignore") and info.get("role") == "feature"
    ]
    schema = {
        "schema_id": stable_schema.get("schema_id", "suggested_schema"),
        "dataset_kind": stable_schema.get("dataset_kind", "unknown"),
        "language": "zh-CN",
        "status": "suggested_by_schema_tool",
        "review_required": True,
        "field_groups": stable_schema.get("field_groups") or {},
        "ignored_fields": ignored_fields,
        "analysis_columns": analysis_columns,
        "fields": fields,
    }

    artifacts_dir = context.run_dir / "artifacts"
    yaml_path = artifacts_dir / "schema_suggestions.yaml"
    json_path = artifacts_dir / "schema_suggestions.json"
    yaml_path.write_text(yaml.safe_dump(schema, allow_unicode=True, sort_keys=False), encoding="utf-8")
    write_json(json_path, schema)
    return {
        "schema": schema,
        "schema_yaml_path": str(yaml_path),
        "schema_json_path": str(json_path),
        "field_count": len(fields),
        "rule_bound_fields": rule_bound_fields,
        "summary": f"已生成 {len(fields)} 个字段的中文名与分组建议，需人工复核后固化。",
    }


def infer_field_metadata(field: str) -> dict[str, Any]:
    """Infer display metadata from one raw field name."""
    lower = field.lower()
    if lower in {"label", "diagnosis"}:
        return {
            "display_name": "标签" if lower == "label" else "诊断说明",
            "group": "label",
            "role": "target",
            "ignore": True,
            "confidence": 0.95,
            "reason": "字段名表示标注或诊断目标，不参与无监督关联挖掘。",
        }
    if lower.startswith("00000_") or "time" in lower or "span" in lower:
        return {
            "display_name": _time_display_name(field),
            "group": "time",
            "role": "metadata",
            "ignore": True,
            "confidence": 0.9,
            "reason": "字段名表示时间或窗口跨度，默认作为元数据处理。",
        }

    prefix, measurement, suffix = _split_field(field)
    measurement_label = MEASUREMENT_LABELS.get(measurement, measurement.replace("_", " "))
    source_prefix = SOURCE_PREFIX_LABELS.get(prefix, "")
    display = f"{source_prefix}{measurement_label}".strip()
    if suffix == "avg":
        display = f"{display}均值"
    elif suffix == "last":
        display = f"{display}末值"

    return {
        "display_name": display or field,
        "group": _infer_group(measurement),
        "role": "feature",
        "ignore": False,
        "confidence": 0.72 if measurement in MEASUREMENT_LABELS else 0.55,
        "reason": "根据字段编号、英文测量名和统计后缀启发式生成，建议人工复核。",
    }


def infer_field_semantics(
    field: str,
    *,
    semantics: dict[str, Any] | None = None,
    schema_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Infer semantic role, unit and statistic hints for one raw field.

    这一步只输出可复核候选，不要求客户字段与当前样例字段同名。后续工具可用
    semantic_role 连接规则字段和客户字段，但仍需看置信度和人工确认标记。
    """
    semantics = semantics if semantics is not None else load_schema_semantics()
    schema_fields = schema_fields or {}
    stable_info = schema_fields.get(field) if isinstance(schema_fields, dict) else None
    metadata = infer_field_metadata(field)
    prefix, measurement, suffix = _split_field(field)
    tokens = sorted(_semantic_tokens(field))
    statistic = _statistic_from_field(field, suffix)
    source_hint = _source_hint(field, prefix)
    roles = semantics.get("semantic_roles") or {}
    best_role: dict[str, Any] | None = None
    best_score = 0.0
    best_method = ""
    normalized_field = _normalize_alias(field)

    for role_id, role in roles.items():
        if not isinstance(role, dict):
            continue
        aliases = [str(item) for item in role.get("aliases") or []]
        normalized_aliases = {_normalize_alias(alias) for alias in aliases}
        role_tokens = set(str(item).lower() for item in role.get("tokens") or [])
        source_codes = {str(item).lower() for item in role.get("source_codes") or []}
        if normalized_field in normalized_aliases:
            score = 0.98
            method = "alias"
        else:
            common = set(tokens) & role_tokens
            if not common:
                continue
            precision = len(common) / max(len(tokens), 1)
            recall = len(common) / max(len(role_tokens), 1)
            score = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
            if prefix and source_codes and prefix.lower() in source_codes:
                score += 0.08
            method = "token_overlap"
        if score > best_score:
            best_role = {"semantic_role": role_id, **role}
            best_score = min(score, 0.99)
            best_method = method

    if best_role and best_score >= 0.55:
        role_id = str(best_role.get("semantic_role"))
        display_name = str(best_role.get("display_name") or metadata.get("display_name") or field)
        statistic_label = STATISTIC_SUFFIXES.get(statistic, "")
        detected_unit = _unit_hint(field, measurement)
        if statistic_label and statistic not in {"raw"} and statistic_label not in display_name:
            display_name = f"{display_name}{statistic_label}"
        confidence = round(best_score, 3)
        if isinstance(stable_info, dict) and stable_info.get("display_name"):
            display_name = str(stable_info["display_name"])
            confidence = max(confidence, 0.9)
        return {
            "field": field,
            "semantic_role": role_id,
            "display_name": display_name,
            "group": best_role.get("group") or metadata.get("group") or "other",
            "role": metadata.get("role", "feature"),
            "ignore": bool(metadata.get("ignore", False)),
            "unit": detected_unit if detected_unit != "unknown" else best_role.get("unit", "unknown"),
            "statistic": statistic,
            "source_hint": source_hint,
            "confidence": confidence,
            "match_method": best_method,
            "tokens": tokens,
            "aliases": best_role.get("aliases") or [],
            "reason": "根据稳定语义角色配置和字段名线索生成，需在新客户数据中复核单位与统计口径。",
        }

    return {
        "field": field,
        "semantic_role": "",
        "display_name": str((stable_info or {}).get("display_name") or metadata.get("display_name") or field) if isinstance(stable_info, dict) else str(metadata.get("display_name") or field),
        "group": str((stable_info or {}).get("group") or metadata.get("group") or "other") if isinstance(stable_info, dict) else str(metadata.get("group") or "other"),
        "role": str((stable_info or {}).get("role") or metadata.get("role") or "feature") if isinstance(stable_info, dict) else str(metadata.get("role") or "feature"),
        "ignore": bool((stable_info or {}).get("ignore", metadata.get("ignore", False))) if isinstance(stable_info, dict) else bool(metadata.get("ignore", False)),
        "unit": _unit_hint(field, measurement),
        "statistic": statistic,
        "source_hint": source_hint,
        "confidence": min(float(metadata.get("confidence", 0.0) or 0.0), 0.58),
        "match_method": "heuristic_only",
        "tokens": tokens,
        "aliases": [],
        "reason": "未匹配到稳定语义角色，仅保留字段名启发式结果，建议人工确认。",
    }


def _split_field(field: str) -> tuple[str, str, str]:
    match = re.match(r"^(?P<prefix>\d+)_?(?P<body>.+?)(?:_(?P<suffix>avg|last))?$", field)
    if match:
        return match.group("prefix") or "", match.group("body") or field, match.group("suffix") or ""
    match = re.match(r"^(?P<body>.+?)(?:_(?P<suffix>avg|last))?$", field)
    if match:
        return "", match.group("body") or field, match.group("suffix") or ""
    return "", field, ""


def _statistic_from_field(field: str, suffix: str) -> str:
    lower = field.lower()
    if suffix:
        return suffix.lower()
    for candidate in STATISTIC_SUFFIXES:
        if lower.endswith(f"_{candidate}") or lower.endswith(f"-{candidate}"):
            return candidate
    return "raw"


def _source_hint(field: str, prefix: str) -> str:
    lower = field.lower()
    if prefix in SOURCE_PREFIX_LABELS:
        return SOURCE_PREFIX_LABELS[prefix]
    if "main" in lower or "primary" in lower:
        return "主"
    if "backup" in lower or "secondary" in lower or "reserve" in lower:
        return "备用"
    return ""


def _unit_hint(field: str, measurement: str) -> str:
    lower = f"{field} {measurement}".lower()
    if re.search(r"(^|[_-])(mv|millivolt|millivolts)($|[_-])", lower):
        return "millivolt"
    if "battery_remaining" in lower or "remaining" in lower or "pct" in lower or "percent" in lower:
        return "percent"
    if "voltage" in lower or "vcc" in lower:
        return "volt"
    if "current" in lower:
        return "ampere"
    if "temperature" in lower or lower.endswith("temp"):
        return "celsius"
    if "speed" in lower or lower.endswith("vx") or lower.endswith("vy") or lower.endswith("vz"):
        return "m/s"
    if "vibration" in lower:
        return "raw_vibration"
    if "alt" in lower or "altitude" in lower:
        return "meter"
    if "press" in lower:
        return "pressure"
    return "unknown"


def _semantic_tokens(field: str) -> set[str]:
    normalized = str(field).lower()
    normalized = re.sub(r"^\d+[_-]+", "", normalized)
    normalized = re.sub(r"(?:[_-](?:avg|last|mean|median|min|max|std|sum|count))+$", "", normalized)
    raw_tokens = re.split(r"[^0-9a-z\u4e00-\u9fff]+", normalized)
    return {token for token in raw_tokens if token and token not in GENERIC_TOKENS and not token.isdigit()}


def _normalize_alias(value: str) -> str:
    normalized = str(value).lower()
    normalized = re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "_", normalized)
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized


def _normalize_confirmed_mapping_records(raw_records: Any) -> dict[str, dict[str, Any]]:
    """Normalize confirmed dataset-field semantic overrides by field alias."""
    records: dict[str, dict[str, Any]] = {}
    if isinstance(raw_records, dict):
        iterable = []
        for field, value in raw_records.items():
            if isinstance(value, dict):
                iterable.append({"field": field, **value})
            else:
                iterable.append({"field": field, "semantic_role": value})
    elif isinstance(raw_records, list):
        iterable = [item for item in raw_records if isinstance(item, dict)]
    else:
        iterable = []

    for item in iterable:
        field = str(item.get("field") or item.get("dataset_field") or "").strip()
        if not field:
            continue
        record = dict(item)
        record["field"] = field
        records[_normalize_alias(field)] = record
    return records


def _normalize_confirmed_rule_mapping_records(raw_records: Any) -> dict[str, dict[str, Any]]:
    """Normalize confirmed rule-field mapping overrides by rule field alias."""
    records: dict[str, dict[str, Any]] = {}
    if isinstance(raw_records, dict):
        iterable = []
        for rule_field, value in raw_records.items():
            if isinstance(value, dict):
                iterable.append({"rule_field": rule_field, **value})
            else:
                iterable.append({"rule_field": rule_field, "dataset_field": value})
    elif isinstance(raw_records, list):
        iterable = [item for item in raw_records if isinstance(item, dict)]
    else:
        iterable = []

    for item in iterable:
        rule_field = str(item.get("rule_field") or "").strip()
        dataset_field = str(item.get("dataset_field") or item.get("field") or "").strip()
        if not rule_field or not dataset_field:
            continue
        record = dict(item)
        record["rule_field"] = rule_field
        record["dataset_field"] = dataset_field
        records[_normalize_alias(rule_field)] = record
    return records


def _infer_group(measurement: str) -> str:
    lower = measurement.lower()
    for token, group in GROUP_BY_TOKEN.items():
        if token in lower:
            return group
    return "other"


def _time_display_name(field: str) -> str:
    lower = field.lower()
    if "boot" in lower and "span" in lower:
        return "启动后时间跨度"
    if "boot" in lower:
        return "启动后时间"
    if "span" in lower:
        return "时间戳跨度"
    return "时间戳"


def _latest_tool_result(tool_results: list[dict[str, Any]], tool_name: str) -> Any:
    for item in reversed(tool_results):
        if item.get("tool") == tool_name:
            return item.get("result")
    return None
