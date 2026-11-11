"""Tool discovery metadata and model-facing parameter schemas."""
from typing import Any

from aircraft_agent.common.tool_contracts import CORE_TOOL_CONTRACTS

TOOL_NAME_ALIASES = {
    "apply_rule_library": "apply_rules",
}

VALID_RULE_SEVERITIES = {"critical", "high", "medium", "low", "info"}


# 工具箱对外展示清单。它和 OPENAI_TOOL_SCHEMAS 的区别是：
# - TOOLBOX_CATALOG 给人、报告和 UI 看，强调分类、职责、可信边界；
# - OPENAI_TOOL_SCHEMAS 给模型看，强调函数名、参数 schema、调用约束。
TOOLBOX_CATALOG: list[dict[str, Any]] = [
    {
        "name": "profile_dataset",
        "category": "data_intake",
        "label": "数据画像",
        "role": "识别数据字段、类型、缺失率、分布和样例行。",
        "trust_boundary": "本地读取数据集，只向 Agent 返回结构化摘要。",
    },
    {
        "name": "parse_rule_library",
        "category": "rule_library",
        "label": "规则库解析",
        "role": "把规则文件解析成可执行或可追溯的 RuleCard。",
        "trust_boundary": "保留规则来源，不让模型改写原始规则。",
    },
    {
        "name": "validate_rule_library",
        "category": "rule_library",
        "label": "规则库校验",
        "role": "检查结构化规则的字段绑定、条件可执行性、严重程度和建议动作。",
        "trust_boundary": "只做静态校验和小范围可执行性检查，不修改原始规则库。",
    },
    {
        "name": "suggest_schema_metadata",
        "category": "schema",
        "label": "字段语义建议",
        "role": "根据字段名、数据画像和规则绑定生成字段中文名、分组和忽略字段建议。",
        "trust_boundary": "输出候选配置草案，不直接改变原始数据或分析结果。",
    },
    {
        "name": "review_schema_metadata",
        "category": "schema",
        "label": "字段语义复核清单",
        "role": "对 Schema 草案生成低置信度字段、忽略字段和规则绑定字段复核清单。",
        "trust_boundary": "只生成复核产物，不自动覆盖稳定配置。",
    },
    {
        "name": "review_schema_mapping",
        "category": "schema",
        "label": "Schema 语义映射",
        "role": "识别新数据字段的语义角色、单位和统计口径，并为规则字段生成候选映射。",
        "trust_boundary": "只输出候选映射和置信度，不自动改写字段名、规则条件或阈值。",
    },
    {
        "name": "review_dataset_adaptation",
        "category": "schema",
        "label": "数据适配审查",
        "role": "在执行规则前检查新数据字段与规则字段是否精确匹配、是否只有候选映射、是否不适用。",
        "trust_boundary": "只给出规则适用性和字段映射候选，不自动改写字段、规则或阈值。",
    },
    {
        "name": "review_schema_mapping_candidates",
        "category": "schema",
        "label": "映射候选复核辅助",
        "role": "把规则字段候选映射分为可考虑确认、需专家复核、统计口径风险和不适用，辅助人工处理。",
        "trust_boundary": "只生成复核建议清单，不写入人工决策、不自动推广全局配置。",
    },
    {
        "name": "review_generalization_readiness",
        "category": "validation",
        "label": "泛化验收门禁",
        "role": "综合 Schema 映射、数据适配和规则执行结果，判断是否存在把当前数据过拟合为自动结论的风险。",
        "trust_boundary": "只做验收门禁和风险分级，不修改规则库、字段映射或阈值。",
    },
    {
        "name": "build_mapping_execution_plan",
        "category": "schema",
        "label": "映射执行计划",
        "role": "把已确认字段映射转成可审查的规则字段重写预案，并标出单位、统计口径和回归验证风险。",
        "trust_boundary": "只生成执行计划，不自动改写规则、不改阈值、不直接执行映射后的规则。",
    },
    {
        "name": "validate_mapping_regression",
        "category": "validation",
        "label": "映射回归验证",
        "role": "验证已确认字段映射生成的规则重写预案是否能透明执行，并标出仍需复核的单位、统计口径和执行风险。",
        "trust_boundary": "只验证映射预案和生成证据，不自动写入全局配置、不修改规则库、不确认阈值业务正确。",
    },
    {
        "name": "check_missing_values",
        "category": "quality",
        "label": "缺失值检查",
        "role": "定位缺失字段并给出基础处理建议。",
        "trust_boundary": "统计计算由 pandas 本地完成。",
    },
    {
        "name": "detect_outliers",
        "category": "quality",
        "label": "离群值提示",
        "role": "用 IQR 或 z-score 对数值字段做透明离群值提示。",
        "trust_boundary": "阈值、边界和样例索引写入证据链。",
    },
    {
        "name": "apply_rules",
        "category": "rule_library",
        "label": "规则命中",
        "role": "把规则条件应用到数据集，返回命中行数和样例索引。",
        "trust_boundary": "只执行可解析规则，不可解析规则显式返回。",
    },
    {
        "name": "check_association_backend",
        "category": "mining",
        "label": "关联挖掘后端检查",
        "role": "检查当前关联规则挖掘将使用 Datamining FP-growth、mlxtend 还是本地 fallback。",
        "trust_boundary": "只检查后端状态，不执行挖掘。",
    },
    {
        "name": "mine_association_rules",
        "category": "mining",
        "label": "关联规则挖掘",
        "role": "将行数据转成事务集合，调用 FP-growth 挖掘高支持度/置信度关联，并按最小命中数过滤弱规则。",
        "trust_boundary": "本地挖掘算法执行，Agent 只解释结果。",
    },
    {
        "name": "compute_correlations",
        "category": "mining",
        "label": "相关性分析",
        "role": "计算数值字段 Pearson 相关性。",
        "trust_boundary": "返回字段对、相关系数和排序结果。",
    },
    {
        "name": "build_visualizations",
        "category": "output",
        "label": "可视化生成",
        "role": "把规则命中、数据质量、关联挖掘和相关性结果生成静态可视化看板。",
        "trust_boundary": "只读取已记录工具输出，生成本地 HTML/JSON 可视化产物。",
    },
    {
        "name": "summarize_findings",
        "category": "agent_support",
        "label": "结果摘要",
        "role": "压缩已完成工具输出，供报告和模型总结使用。",
        "trust_boundary": "摘要来自已记录工具结果。",
    },
    {
        "name": "validate_analysis",
        "category": "validation",
        "label": "结果校验",
        "role": "检查必需工具是否执行、规则是否命中、证据链是否完整。",
        "trust_boundary": "校验结果单独入库并写入证据链。",
    },
    {
        "name": "write_report",
        "category": "output",
        "label": "报告生成",
        "role": "生成 Markdown/HTML 报告。",
        "trust_boundary": "报告只引用 RunContext 中的文件、工具和证据状态。",
    },
]


# OpenAI function tool schema。模型会根据这些 description 和 parameters
# 选择要调用的工具；真实执行仍由 ToolExecutor 在本地完成。
OPENAI_TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "name": "profile_dataset",
        "description": "Profile a dataset locally: fields, types, missing rates, distributions, and safe sample rows.",
        "parameters": {
            "type": "object",
            "properties": {"dataset_id": {"type": "string", "description": "Optional dataset file id."}},
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "parse_rule_library",
        "description": "Parse the uploaded rule library into normalized RuleCard objects.",
        "parameters": {
            "type": "object",
            "properties": {"rule_file_id": {"type": "string", "description": "Optional rule file id."}},
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "validate_rule_library",
        "description": "Validate structured rules for field binding, executable conditions, severity metadata, and recommendations.",
        "parameters": {
            "type": "object",
            "properties": {
                "dataset_id": {"type": "string", "description": "Optional dataset file id."},
                "rule_file_id": {"type": "string", "description": "Optional rule file id."},
            },
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "suggest_schema_metadata",
        "description": "Suggest Chinese display names, groups, roles, and ignore flags for dataset fields. Output is a reviewable schema draft.",
        "parameters": {
            "type": "object",
            "properties": {
                "max_fields": {"type": "integer", "default": 120},
                "config_path": {"type": "string", "description": "Optional stable schema config path."},
            },
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "review_schema_metadata",
        "description": "Create a review checklist for schema suggestions, including low-confidence fields, ignored fields, and rule-bound fields.",
        "parameters": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "review_schema_mapping",
        "description": "Infer semantic roles, unit hints, statistic hints, and rule-field mapping candidates for unseen customer schemas. Does not rewrite rules automatically.",
        "parameters": {
            "type": "object",
            "properties": {
                "dataset_id": {"type": "string", "description": "Optional dataset file id."},
                "rule_file_id": {"type": "string", "description": "Optional rule library file id."},
                "semantics_path": {"type": "string", "description": "Optional semantic role config path."},
                "overrides_path": {"type": "string", "description": "Optional confirmed schema mapping override config path."},
            },
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "review_dataset_adaptation",
        "description": "Review whether the uploaded dataset schema can safely use the current rule library. Checks exact field bindings, suffix mismatches, semantic mapping candidates, and non-applicable rules before applying rules.",
        "parameters": {
            "type": "object",
            "properties": {
                "dataset_id": {"type": "string", "description": "Optional dataset file id."},
                "rule_file_id": {"type": "string", "description": "Optional rule library file id."},
            },
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "review_schema_mapping_candidates",
        "description": "Create a conservative human-review checklist for rule-field mapping candidates. It suggests which mappings may be confirmed, which require expert review, and which should stay not applicable. It never writes decisions or global overrides.",
        "parameters": {
            "type": "object",
            "properties": {
                "dataset_id": {"type": "string", "description": "Optional dataset file id."},
                "rule_file_id": {"type": "string", "description": "Optional rule library file id."},
                "min_confirm_score": {"type": "number", "default": 0.86},
            },
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "review_generalization_readiness",
        "description": "Gate whether the current run is generalization-ready without overfitting to the current dataset. Uses schema mapping, dataset adaptation, and rule execution evidence.",
        "parameters": {
            "type": "object",
            "properties": {
                "min_semantic_coverage": {"type": "number", "default": 0.6},
                "max_unresolved_rule_ratio": {"type": "number", "default": 0.35},
            },
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "build_mapping_execution_plan",
        "description": "Build a reviewable execution plan for confirmed schema mappings. Does not rewrite rules or thresholds automatically.",
        "parameters": {
            "type": "object",
            "properties": {
                "allow_statistic_mismatch": {"type": "boolean", "default": False},
                "allow_unknown_unit": {"type": "boolean", "default": False},
            },
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "validate_mapping_regression",
        "description": "Validate confirmed mapping execution plans by checking rewritten rule conditions, unit/statistic compatibility, and regression evidence. Does not promote mappings or modify rules.",
        "parameters": {
            "type": "object",
            "properties": {
                "dataset_id": {"type": "string", "description": "Optional dataset file id."},
                "require_strict_unit": {"type": "boolean", "default": True},
                "require_strict_statistic": {"type": "boolean", "default": True},
            },
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "check_missing_values",
        "description": "Summarize missing values by column and recommend basic handling.",
        "parameters": {
            "type": "object",
            "properties": {"dataset_id": {"type": "string"}},
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "detect_outliers",
        "description": "Report basic numeric outlier hints using IQR or z-score.",
        "parameters": {
            "type": "object",
            "properties": {
                "dataset_id": {"type": "string"},
                "method": {"type": "string", "enum": ["iqr", "zscore"], "default": "iqr"},
                "columns": {"type": "array", "items": {"type": "string"}},
            },
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "apply_rules",
        "description": "Apply normalized rules to the dataset and report matched rows and unparseable rules.",
        "parameters": {
            "type": "object",
            "properties": {"dataset_id": {"type": "string"}, "rule_file_id": {"type": "string"}},
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "apply_rule_library",
        "description": "Alias of apply_rules for rule-library checking.",
        "parameters": {
            "type": "object",
            "properties": {"dataset_id": {"type": "string"}, "rule_file_id": {"type": "string"}},
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "check_association_backend",
        "description": "Check which FP-growth backend will be used for association-rule mining.",
        "parameters": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "mine_association_rules",
        "description": "Mine association rules with FP-growth from categorical or binned numeric fields.",
        "parameters": {
            "type": "object",
            "properties": {
                "dataset_id": {"type": "string"},
                "columns": {"type": "array", "items": {"type": "string"}},
                "min_support": {"type": "number", "default": 0.2},
                "min_confidence": {"type": "number", "default": 0.7},
                "min_lift": {"type": "number", "default": 1.2},
                "min_count": {"type": "integer", "default": 2},
                "max_len": {"type": "integer", "default": 3},
                "max_antecedent": {"type": "integer", "default": 2},
                "max_rules": {"type": "integer", "default": 15},
                "max_rows": {"type": "integer", "default": 5000},
            },
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "compute_correlations",
        "description": "Compute numeric Pearson correlations and return strongest relationships.",
        "parameters": {
            "type": "object",
            "properties": {
                "dataset_id": {"type": "string"},
                "columns": {"type": "array", "items": {"type": "string"}},
                "top_n": {"type": "integer", "default": 20},
            },
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "discover_frequent_patterns",
        "description": "Return frequent single-field and pair-field patterns in categorical or binned data.",
        "parameters": {
            "type": "object",
            "properties": {
                "dataset_id": {"type": "string"},
                "columns": {"type": "array", "items": {"type": "string"}},
                "min_support": {"type": "number", "default": 0.1},
            },
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "build_visualizations",
        "description": "Build local static visualization artifacts for the final report from prior tool outputs.",
        "parameters": {
            "type": "object",
            "properties": {
                "max_items": {"type": "integer", "default": 8, "description": "Maximum rows per chart."},
            },
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "summarize_findings",
        "description": "Summarize all local tool outputs generated so far.",
        "parameters": {
            "type": "object",
            "properties": {"focus": {"type": "string", "description": "Optional focus for the summary."}},
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "validate_analysis",
        "description": "Validate whether the planned tool chain produced enough auditable outputs for the final report.",
        "parameters": {
            "type": "object",
            "properties": {
                "required_tools": {"type": "array", "items": {"type": "string"}},
                "strict": {"type": "boolean", "default": False},
            },
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "write_report",
        "description": "Write Markdown and HTML reports for the current run.",
        "parameters": {
            "type": "object",
            "properties": {"final_summary": {"type": "string"}},
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "record_evidence",
        "description": "Record a model-authored note in the evidence chain.",
        "parameters": {
            "type": "object",
            "properties": {
                "summary": {"type": "string"},
                "details": {"type": "object", "additionalProperties": True},
            },
            "required": ["summary"],
            "additionalProperties": False,
        },
    },
]


# Keep existing descriptions/defaults; advertise the optional v1 discriminator.
for _schema in OPENAI_TOOL_SCHEMAS:
    if _schema["name"] in CORE_TOOL_CONTRACTS:
        _schema["parameters"]["properties"]["schema_version"] = {
            "type": "integer", "enum": [1], "default": 1,
            "description": "Tool contract version; omission selects v1.",
        }
