"""File parsing and lightweight rule normalization.

该模块只做 MVP 级别的解析：把 CSV/XLSX 变成 DataFrame，把规则库文本或
结构化文件变成 RuleCard。复杂自然语言规则仍然会保留 description，
只有能识别成简单比较表达式的规则才会自动执行。
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from aircraft_agent.common.models import RuleCard
from aircraft_agent.common.utils import to_jsonable, truncate_text


def read_dataset(path: Path) -> pd.DataFrame:
    """Read a supported dataset file into pandas.

    数据分析工具统一消费 DataFrame，因此所有数据集入口都先在这里收敛。
    目前 MVP 只支持 CSV/XLSX；后续如果要接遥测流、数据库或 parquet，
    也应优先在这里扩展成 DataFrame 或兼容的中间结构。
    """
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return _read_csv(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported dataset format: {suffix}. Use CSV or XLSX.")


def _read_csv(path: Path) -> pd.DataFrame:
    """Try common encodings so Chinese CSV files work without extra config."""
    encodings = ["utf-8-sig", "utf-8", "gb18030", "latin1"]
    last_error: Exception | None = None
    for encoding in encodings:
        try:
            # utf-8-sig 可处理 Excel 导出的带 BOM CSV；gb18030 覆盖常见中文环境。
            return pd.read_csv(path, encoding=encoding)
        except Exception as exc:
            last_error = exc
    raise ValueError(f"Could not read CSV file {path.name}: {last_error}")


def profile_dataframe(df: pd.DataFrame, sample_rows: int = 5) -> dict[str, Any]:
    """Build a compact, JSON-safe profile of a DataFrame.

    画像结果会进入报告和模型上下文，所以要“足够有用但不能太大”：
    - 给出行列数、字段名、缺失率、唯一值数量；
    - 数值列给出 min/max/mean/median/std/q1/q3；
    - 类别列给出高频值；
    - 只保留少量样例行，避免把原始数据大批量交给模型。
    """
    profiles: list[dict[str, Any]] = []
    for column in df.columns:
        series = df[column]
        non_null = series.dropna()
        # 每一列都生成一个 profile，后续报告可以按字段展示数据质量和分布。
        profile: dict[str, Any] = {
            "name": str(column),
            "dtype": str(series.dtype),
            "missing_count": int(series.isna().sum()),
            "missing_rate": round(float(series.isna().mean()), 6) if len(series) else 0.0,
            "unique_count": int(non_null.nunique(dropna=True)),
            "sample_values": [to_jsonable(v) for v in non_null.head(5).tolist()],
        }
        if pd.api.types.is_numeric_dtype(series):
            # 数值列走统计描述，用于离群值提示、阈值判断和专家快速浏览。
            desc = series.describe()
            profile["numeric"] = {
                "min": to_jsonable(desc.get("min")),
                "max": to_jsonable(desc.get("max")),
                "mean": to_jsonable(desc.get("mean")),
                "median": to_jsonable(series.median()),
                "std": to_jsonable(desc.get("std")),
                "q1": to_jsonable(series.quantile(0.25)),
                "q3": to_jsonable(series.quantile(0.75)),
            }
        else:
            # 只对看起来像日期的字段尝试日期解析，避免 pandas 对普通类别值
            # 发出大量格式推断 warning。
            looks_date_like = _looks_date_like(non_null.head(20).astype(str).tolist())
            if looks_date_like:
                parsed_dates = pd.to_datetime(series, errors="coerce")
            else:
                parsed_dates = pd.Series([pd.NaT] * len(series), index=series.index)
            if looks_date_like and parsed_dates.notna().mean() >= 0.8 and len(non_null) > 0:
                profile["datetime"] = {
                    "min": to_jsonable(parsed_dates.min()),
                    "max": to_jsonable(parsed_dates.max()),
                }
            value_counts = non_null.astype(str).value_counts().head(10)
            # 类别/文本字段只保留前 10 个高频值，避免长尾值把报告撑大。
            profile["top_values"] = [{"value": k, "count": int(v)} for k, v in value_counts.items()]
        profiles.append(profile)

    # 样例行仅用于报告/模型理解字段含义，不作为完整原始数据传递。
    sample = df.head(sample_rows).where(pd.notnull(df.head(sample_rows)), None).to_dict(orient="records")
    return {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "column_names": [str(c) for c in df.columns],
        "column_profiles": profiles,
        "sample_rows": to_jsonable(sample),
    }


def _looks_date_like(values: list[str]) -> bool:
    """Heuristic date detector used before calling pandas.to_datetime.

    pandas 对任意字符串做日期推断可能产生大量 warning，也容易把普通编码误判。
    这里先做一个轻量正则判断，只有“看起来像日期”的列才进入日期解析。
    """
    if not values:
        return False
    dateish = 0
    pattern = re.compile(r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}|^\d{1,2}[-/]\d{1,2}[-/]\d{2,4}")
    for value in values:
        if pattern.search(value.strip()):
            dateish += 1
    return dateish / len(values) >= 0.6


def extract_rule_text(path: Path) -> str:
    """Extract plain text from supported rule-library file formats.

    规则库入口比数据集更宽：md/txt/json/yaml 是一等格式，docx/pdf 走文本抽取。
    这里的目标不是完美理解复杂文档，而是尽可能把规则文本抽出来，交给后续
    parse_rule_library 归一化成 RuleCard。
    """
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md", ".markdown", ".csv"}:
        return _read_text(path)
    if suffix == ".json":
        return path.read_text(encoding="utf-8")
    if suffix in {".yaml", ".yml"}:
        return path.read_text(encoding="utf-8")
    if suffix == ".docx":
        try:
            from docx import Document
        except Exception as exc:
            raise ValueError("python-docx is required for DOCX rule files.") from exc
        doc = Document(str(path))
        # MVP 只抽取段落文本；表格规则后续可以在这里继续扩展。
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except Exception as exc:
            raise ValueError("pypdf is required for PDF rule files.") from exc
        reader = PdfReader(str(path))
        # PDF 文本抽取受版式影响较大，因此报告里仍会保留 source_ref，方便人工核查。
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    raise ValueError(f"Unsupported rule-library format: {suffix}.")


def _read_text(path: Path) -> str:
    """Read text with common encodings, falling back to lossy decoding."""
    for encoding in ["utf-8-sig", "utf-8", "gb18030", "latin1"]:
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text(errors="ignore")


def parse_rule_library(path: Path, known_fields: list[str] | None = None) -> list[RuleCard]:
    """Parse a rule file into RuleCard records.

    known_fields 来自数据集字段名。它让解析器能识别“这条规则提到了哪个字段”，
    并优先用真实字段名抽取 condition，避免自然语言里相似词导致误绑定。
    """
    suffix = path.suffix.lower()
    if suffix == ".json":
        raw = json.loads(path.read_text(encoding="utf-8"))
        return _rules_from_structured(raw, path.name, known_fields)
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except Exception as exc:
            raise ValueError("PyYAML is required for YAML rule files.") from exc
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        return _rules_from_structured(raw, path.name, known_fields)

    text = extract_rule_text(path)
    return _rules_from_text(text, path.name, known_fields)


def _rules_from_structured(raw: Any, source_name: str, known_fields: list[str] | None) -> list[RuleCard]:
    """Normalize JSON/YAML rules into RuleCard objects.

    支持三种常见结构：
    - {"rules": [...]}；
    - 直接是规则列表；
    - 单个规则对象。
    这样用户可以从简单 demo 逐步过渡到结构化规则库，而不需要改工具链。
    """
    if isinstance(raw, dict) and isinstance(raw.get("rules"), list):
        items = raw["rules"]
    elif isinstance(raw, list):
        items = raw
    elif isinstance(raw, dict):
        items = [raw]
    else:
        items = [{"description": str(raw)}]

    rules: list[RuleCard] = []
    for idx, item in enumerate(items, 1):
        if isinstance(item, dict):
            # 结构化规则优先信任用户给出的 condition/fields/source_ref；
            # 如果缺失，再用 description 和 known_fields 做自动推断。
            condition = _structured_condition(item)
            target_fields = _structured_fields(item)
            description = str(item.get("description") or item.get("desc") or item.get("rationale") or condition or item)
            rule = RuleCard(
                rule_id=str(item.get("rule_id") or item.get("id") or f"R{idx:03d}"),
                name=str(item.get("name") or item.get("title") or f"Rule {idx}"),
                description=description,
                target_fields=target_fields,
                condition=condition,
                threshold=item.get("threshold", {}) if isinstance(item.get("threshold", {}), dict) else {},
                severity=str(item.get("severity") or "medium"),
                recommendation=str(item.get("recommendation") or item.get("action") or ""),
                rationale=str(item.get("rationale") or item.get("reason") or ""),
                tags=[str(v) for v in item.get("tags", []) or []],
                review_required=bool(item.get("review_required", False)),
                enabled=bool(item.get("enabled", True)),
                source_ref=str(item.get("source_ref") or source_name),
                confidence=float(item.get("confidence") or 0.8),
            )
        else:
            rule = _rule_from_line(str(item), idx, source_name, known_fields)
        if not rule.target_fields:
            # 即便用户没有写 target_fields，也尝试从规则文本中匹配数据集字段名。
            rule.target_fields = infer_target_fields(rule.description + " " + rule.condition, known_fields)
        rules.append(rule)
    return rules


def _structured_fields(item: dict[str, Any]) -> list[str]:
    """Extract target fields from flexible structured rule shapes."""
    fields = item.get("target_fields", item.get("fields"))
    if fields is None and item.get("field") is not None:
        fields = [item.get("field")]
    if isinstance(fields, str):
        fields = [fields]
    return [str(v) for v in fields or []]


def _structured_condition(item: dict[str, Any]) -> str:
    """Build an executable condition from either condition or field/operator/value."""
    if item.get("condition"):
        return str(item["condition"])
    field = item.get("field")
    operator = item.get("operator") or item.get("op")
    value = item.get("value")
    if field is None and isinstance(item.get("threshold"), dict):
        threshold = item["threshold"]
        field = threshold.get("field")
        operator = threshold.get("operator") or threshold.get("op")
        value = threshold.get("value")
    if field is not None and operator and value is not None:
        return f"{field} {operator} {_format_condition_value(value)}"
    return ""


def _format_condition_value(value: Any) -> str:
    """Format a structured rule value for the simple condition evaluator."""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def _rules_from_text(text: str, source_name: str, known_fields: list[str] | None) -> list[RuleCard]:
    """Treat each non-empty text line as one candidate rule.

    文本规则库在 MVP 里采用“每行一条规则”的简单约定，例如：
    R001: temperature > 80 异常
    复杂段落不会被强行拆成多条规则，而是保留为 description 供人工复核。
    """
    lines = []
    for line in text.splitlines():
        stripped = line.strip().strip("-*# \t")
        if stripped:
            lines.append(stripped)
    if not lines and text.strip():
        lines = [text.strip()]
    return [_rule_from_line(line, idx, source_name, known_fields) for idx, line in enumerate(lines, 1)]


def _rule_from_line(line: str, idx: int, source_name: str, known_fields: list[str] | None) -> RuleCard:
    """Convert one text line into a RuleCard and infer simple conditions.

    规则行可以带显式编号，如 `R001: ...`；没有编号时自动生成 R001/R002。
    condition 只抽取 MVP 支持的简单比较表达式，不能执行的内容仍留在
    description 中，不会被静默丢弃。
    """
    rule_id = f"R{idx:03d}"
    name = truncate_text(line, 80)
    match = re.match(r"^(?P<id>[A-Za-z]*\d{1,5})\s*[:：\-]\s*(?P<body>.+)$", line)
    if match:
        # 支持英文冒号、中文冒号、短横线三种常见编号分隔符。
        rule_id = match.group("id")
        line = match.group("body").strip()
        name = truncate_text(line, 80)
    condition = infer_condition(line, known_fields)
    return RuleCard(
        rule_id=rule_id,
        name=name,
        description=line,
        target_fields=infer_target_fields(line, known_fields),
        condition=condition,
        threshold={},
        source_ref=f"{source_name}:line:{idx}",
        confidence=0.5,
    )


def infer_target_fields(text: str, known_fields: list[str] | None) -> list[str]:
    """Find dataset column names mentioned in a rule description.

    这是一个保守匹配：只有规则文本中出现了真实字段名才认为命中。这样简单、
    可解释，但也意味着字段别名/中文别称需要后续用字段字典增强。
    """
    if not known_fields:
        return []
    lowered = text.lower()
    found = [field for field in known_fields if field.lower() in lowered]
    return found


def infer_condition(text: str, known_fields: list[str] | None) -> str:
    """Extract a simple executable condition from natural-language-ish text.

    当前支持的表达式形态包括：
    - temperature > 80
    - status == WARN
    - 字段 大于 80
    - 字段 contains XXX

    更复杂的括号、OR、多层逻辑暂不在解析阶段支持；复杂规则会保留原文，
    在 apply_rules 中被标为 unparseable 或待人工复核。
    """
    normalized = (
        text.replace("大于等于", ">=")
        .replace("不小于", ">=")
        .replace("小于等于", "<=")
        .replace("不大于", "<=")
        .replace("大于", ">")
        .replace("小于", "<")
        .replace("等于", "==")
        .replace("不等于", "!=")
    )
    operators = r"(>=|<=|==|!=|>|<|contains|not_contains)"
    if known_fields:
        # 先按真实字段名匹配，且长字段名优先，避免 temp 抢先匹配 temperature。
        for field in sorted(known_fields, key=len, reverse=True):
            pattern = re.compile(rf"({re.escape(field)})\s*{operators}\s*([^,;，；。]+)", re.IGNORECASE)
            match = pattern.search(normalized)
            if match:
                return f"{field} {match.group(2)} {_clean_condition_value(match.group(3).strip())}"
    # 没有字段表时退化为通用正则，保证单独解析规则文件时也能得到 condition。
    generic = re.search(rf"([\w\u4e00-\u9fff ._\-/]+?)\s*{operators}\s*([^,;，；。]+)", normalized, re.IGNORECASE)
    if generic:
        return f"{generic.group(1).strip()} {generic.group(2)} {_clean_condition_value(generic.group(3).strip())}"
    return ""


def _clean_condition_value(value: str) -> str:
    """Remove common natural-language labels after numeric rule values.

    例如 `temperature > 80 异常` 会清理成 `temperature > 80`。如果右侧不是
    数字，则保留原文本，例如 `status == WARN`。
    """
    stripped = value.strip()
    numeric = re.match(r"^([-+]?\d+(?:\.\d+)?)(?:\s*[\w\u4e00-\u9fff]+)?$", stripped)
    if numeric:
        return numeric.group(1)
    return stripped
