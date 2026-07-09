"""Report rendering utilities.

报告内容完全来自 RunContext：输入文件、数据画像、规则卡片、工具调用摘要、
Agent 总结和证据链数量。HTML 渲染是一个极简 Markdown 子集，避免 MVP
阶段引入额外模板复杂度。
"""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any

from .runtime import RunContext
from .schema_metadata import field_label, label_expression, latest_schema_metadata
from .utils import stable_json, truncate_text


def build_markdown_report(context: RunContext, final_summary: str = "") -> str:
    """Render the current run state as a Markdown report.

    报告只从 RunContext 读取信息，不重新执行工具，也不重新读取原始数据。
    这样可以保证报告内容与 manifest/evidence_chain 中的状态一致。
    """
    schema = latest_schema_metadata(context)
    lines: list[str] = []
    title = "标准演示报告" if getattr(context, "run_kind", "analysis") == "demo" else "客户数据分析报告"
    lines.append(f"# {title}")
    lines.append("")
    lines.append(f"- 运行 ID: `{context.run_id}`")
    lines.append(f"- 用户目标: {context.user_goal}")
    lines.append(f"- 运行状态: {_cn_status(context.status)}")
    lines.append(f"- Agent 模式: 编排中枢负责计划、工具路由、结果校验；工具箱负责本地可信执行。")
    lines.append("")
    _append_acceptance_summary(lines, context)
    _append_visualization_summary(lines, context)
    _append_schema_summary(lines, context)
    _append_schema_mapping_summary(lines, context)
    _append_dataset_adaptation_summary(lines, context, schema)
    _append_mapping_execution_plan_summary(lines, context)
    _append_mapping_regression_summary(lines, context)
    _append_generalization_readiness_summary(lines, context)

    if context.toolbox_manifest:
        # 工具箱清单帮助读者理解：Agent 并不是直接“凭空分析”，而是在调度
        # 一组本地可信工具。
        lines.append("## 工具箱清单")
        categories: dict[str, list[dict[str, Any]]] = {}
        for tool in context.toolbox_manifest:
            categories.setdefault(str(tool.get("category", "other")), []).append(tool)
        for category, tools in categories.items():
            labels = ", ".join(f"{tool.get('label')}(`{tool.get('name')}`)" for tool in tools)
            lines.append(f"- {category}: {labels}")
        lines.append("")

    if context.agent_plan:
        # Agent 执行计划是 demo 的重点：展示先做什么、调用什么工具、预期什么。
        lines.append("## Agent 执行计划")
        for step in context.agent_plan:
            lines.append(
                f"- 步骤 {step.get('step')}: {step.get('thought')} -> `{step.get('action')}`；预期: {step.get('expected')}"
            )
        lines.append("")

    lines.append("## 输入文件")
    for file in context.files.values():
        # 只展示文件摘要和 sha256 前缀；完整哈希保存在 manifest/evidence。
        extra = ""
        if file.rows is not None and file.columns is not None:
            extra = f", rows={file.rows}, columns={file.columns}"
        lines.append(f"- `{file.original_name}` ({file.kind}, sha256={file.sha256[:16]}...{extra})")
    lines.append("")

    if context.dataset_profiles:
        # 数据概况来自 profile_dataset 的输出，避免报告直接遍历原始数据。
        lines.append("## 数据概况")
        for dataset_id, profile in context.dataset_profiles.items():
            file = context.files.get(dataset_id)
            label = file.original_name if file else dataset_id
            lines.append(f"### {label}")
            lines.append(f"- 行数: {profile.get('rows')}")
            lines.append(f"- 列数: {profile.get('columns')}")
            cols = ", ".join(field_label(str(col), schema) for col in profile.get("column_names", [])[:30])
            lines.append(f"- 字段: {cols}")
            missing = [
                f"{field_label(str(c['name']), schema)}({c['missing_rate']:.1%})"
                for c in profile.get("column_profiles", [])
                if c.get("missing_rate", 0) > 0
            ][:10]
            if missing:
                lines.append(f"- 存在缺失的字段: {', '.join(missing)}")
            lines.append("")

    _append_data_quality_summary(lines, context, schema)
    _append_rule_validation_summary(lines, context)

    if context.rules:
        # 规则库概况展示 RuleCard，而不是原始规则文件全文。
        lines.append("## 规则库概况")
        lines.append(f"- 解析规则数: {len(context.rules)}")
        for rule in context.rules[:20]:
            fields = ", ".join(field_label(str(field), schema) for field in rule.target_fields) if rule.target_fields else "未识别"
            condition = label_expression(rule.condition or "未抽取", schema, include_raw=True)
            lines.append(
                f"- `{rule.rule_id}` {rule.name}; 严重程度: {_severity_label(rule.severity)}；"
                f"字段: {fields}; 条件: `{condition}`"
            )
            if rule.recommendation:
                lines.append(f"  建议动作: {rule.recommendation}")
        lines.append("")

    _append_rule_execution_summary(lines, context, schema)

    if context.agent_trace:
        # Agent 轨迹是给人看的 Thought/Action/Observation 简化版。
        lines.append("## Agent 轨迹")
        for item in context.agent_trace:
            lines.append(
                f"- 步骤 {item.get('step')} `{item.get('action')}` [{_cn_status(str(item.get('status')))}]: {item.get('observation')}"
            )
        lines.append("")

    if context.validation_results:
        # 只展示最后一次校验结果；历史校验记录仍在 manifest 中。
        lines.append("## 结果校验")
        latest = context.validation_results[-1]
        lines.append(f"- 状态: {_cn_status(str(latest.get('status')))}")
        lines.append(f"- 证据链节点数: {latest.get('evidence_nodes')}")
        missing = latest.get("missing_tools") or []
        if missing:
            lines.append(f"- 缺失工具: {', '.join(missing)}")
        warnings = latest.get("warnings") or []
        if warnings:
            for warning in warnings:
                lines.append(f"- 告警: {warning}")
        else:
            lines.append("- 告警: 无")
        lines.append("")

    association_result = _latest_tool_result(context.tool_results, "mine_association_rules")
    if isinstance(association_result, dict):
        lines.append("## 关联规则挖掘结果")
        lines.append(f"- 后端: {association_result.get('engine')}")
        lines.append(f"- 事务数: {association_result.get('transactions', 0)}")
        lines.append(f"- 频繁项集数: {association_result.get('frequent_itemsets', 0)}")
        params = association_result.get("parameters") or {}
        if params:
            lines.append(
                "- 阈值: "
                f"min_support={params.get('min_support')}, "
                f"effective_min_support={params.get('effective_min_support')}, "
                f"min_confidence={params.get('min_confidence')}, "
                f"min_lift={params.get('min_lift')}, "
                f"min_count={params.get('min_count')}, "
                f"max_rules={params.get('max_rules')}"
            )
        for warning in association_result.get("warnings") or []:
            lines.append(f"- 样本提示: {warning}")
        rules = association_result.get("rules") or []
        if rules:
            lines.append("- 前 10 条规则: 最多展示前 10 条。")
            for idx, rule in enumerate(rules[:10], 1):
                lines.append(
                    f"- 规则 {idx}: `{label_expression(rule.get('antecedent'), schema)}` -> `{label_expression(rule.get('consequent'), schema)}`; "
                    f"命中行数={rule.get('matched_rows')}, "
                    f"支持度={rule.get('support')}, "
                    f"置信度={rule.get('confidence')}, "
                    f"提升度={rule.get('lift')}"
                )
        else:
            lines.append("- 前 10 条规则: 无满足当前阈值的关联规则。")
        lines.append("")

    _append_correlation_summary(lines, context, schema)

    lines.append("## 工具调用摘要")
    for idx, item in enumerate(context.tool_results, 1):
        # 每个工具应尽量返回 summary；没有时再把结构化结果截断展示。
        result = item.get("result", {})
        summary = result.get("summary") if isinstance(result, dict) else None
        if not summary:
            summary = truncate_text(stable_json(result), 240)
        lines.append(f"{idx}. `{item.get('tool')}`: {summary}")
    lines.append("")

    if final_summary or context.final_text:
        lines.append("## Agent 总结")
        lines.append(final_summary or context.final_text)
        lines.append("")

    _append_limitations(lines, context)
    _append_acceptance_checklist(lines, context)

    lines.append("## 证据链")
    lines.append(f"- 节点数: {len(context.evidence.nodes)}")
    lines.append("- 完整证据链见 `evidence_chain.json`。")
    return "\n".join(lines) + "\n"


def _latest_tool_result(tool_results: list[dict[str, Any]], tool_name: str) -> Any:
    for item in reversed(tool_results):
        if item.get("tool") == tool_name:
            return item.get("result")
    return None


def _append_acceptance_summary(lines: list[str], context: RunContext) -> None:
    """Add the first-page MVP conclusion for quick human review."""
    latest_validation = context.validation_results[-1] if context.validation_results else {}
    association = _latest_tool_result(context.tool_results, "mine_association_rules")
    backend = association.get("engine") if isinstance(association, dict) else None
    rule_execution = _latest_tool_result(context.tool_results, "apply_rules")
    profile = next(iter(context.dataset_profiles.values()), {})

    evaluated_rules = 0
    total_rule_results = 0
    if isinstance(rule_execution, dict):
        results = rule_execution.get("rule_results") or []
        total_rule_results = len(results)
        evaluated_rules = sum(1 for item in results if item.get("status") == "evaluated")

    validation_raw_status = str(latest_validation.get("status", "not_run"))
    validation_status = _cn_status(validation_raw_status)
    ready = bool(
        context.dataset_profiles
        and context.rules
        and evaluated_rules
        and isinstance(association, dict)
        and validation_raw_status == "passed"
    )
    if validation_raw_status == "needs_review":
        conclusion = "本次结果需复核，暂不建议作为业务诊断结论"
    elif ready:
        conclusion = "当前 MVP 闭环可验收"
    else:
        conclusion = "当前 MVP 闭环仍有待补齐项"

    lines.append("## MVP 验收摘要")
    lines.append(f"- 验收结论: {conclusion}。")
    lines.append(f"- 数据规模: {profile.get('rows', '未读取')} 行，{profile.get('columns', '未读取')} 列。")
    lines.append(f"- 规则执行: 已解析 {len(context.rules)} 条规则，成功执行 {evaluated_rules}/{total_rule_results or len(context.rules)} 条。")
    lines.append(f"- 关联规则后端: {backend or '未运行'}。")
    evidence_nodes = latest_validation.get("evidence_nodes", len(context.evidence.nodes))
    lines.append(f"- 结果校验: {validation_status}，证据链节点数 {evidence_nodes}。")
    for warning in (latest_validation.get("warnings") or [])[:3]:
        lines.append(f"- 关键提示: {warning}")
    lines.append("- 本轮范围: 以规则命中、FP-growth 关联规则、本地证据链和中文报告为核心；暂不接入独立异常检测项目。")
    lines.append("")


def _append_data_quality_summary(lines: list[str], context: RunContext, schema: dict[str, Any]) -> None:
    """Summarize missing values and basic outlier checks as data-quality evidence."""
    missing_result = _latest_tool_result(context.tool_results, "check_missing_values")
    outlier_result = _latest_tool_result(context.tool_results, "detect_outliers")
    if not isinstance(missing_result, dict) and not isinstance(outlier_result, dict):
        return

    lines.append("## 数据质量摘要")
    if isinstance(missing_result, dict):
        missing_columns = sorted(
            missing_result.get("columns_with_missing") or [],
            key=lambda item: (item.get("missing_rate", 0), item.get("missing_count", 0)),
            reverse=True,
        )
        if missing_columns:
            lines.append(f"- 缺失值: 发现 {len(missing_columns)} 个字段存在缺失，报告展示缺失率最高的前 10 个字段。")
            for item in missing_columns[:10]:
                lines.append(
                    f"- {field_label(str(item.get('column')), schema)}: 缺失 {item.get('missing_count')} 行，"
                    f"缺失率 {_format_rate(item.get('missing_rate'))}；建议: {item.get('recommendation')}"
                )
        else:
            lines.append("- 缺失值: 未发现存在缺失的字段。")

    if isinstance(outlier_result, dict):
        positive_outliers = [
            item for item in outlier_result.get("outliers") or [] if int(item.get("outlier_count") or 0) > 0
        ]
        positive_outliers.sort(key=lambda item: (item.get("outlier_count", 0), item.get("outlier_rate", 0)), reverse=True)
        method = outlier_result.get("method", "iqr")
        if positive_outliers:
            lines.append(f"- 离群值提示: 使用 {method} 方法发现 {len(positive_outliers)} 个字段存在离群点，报告展示前 8 个字段。")
            for item in positive_outliers[:8]:
                samples = ", ".join(str(i) for i in (item.get("sample_indices") or [])[:5]) or "无"
                lines.append(
                    f"- {field_label(str(item.get('column')), schema)}: 离群 {item.get('outlier_count')} 行，"
                    f"占比 {_format_rate(item.get('outlier_rate'))}；样例索引: {samples}"
                )
        else:
            checked = len(outlier_result.get("outliers") or [])
            lines.append(f"- 离群值提示: 使用 {method} 方法检查 {checked} 个数值字段，未发现明显离群点。")
    lines.append("")


def _append_rule_validation_summary(lines: list[str], context: RunContext) -> None:
    """Summarize structured rule-library validation results."""
    validation = _latest_tool_result(context.tool_results, "validate_rule_library")
    if not isinstance(validation, dict):
        return
    lines.append("## 规则库校验摘要")
    lines.append(
        f"- 校验结果: {validation.get('ready_count', 0)} 条可执行，"
        f"{validation.get('needs_review_count', 0)} 条需复核，{validation.get('disabled_count', 0)} 条已停用。"
    )
    md_path = validation.get("markdown_path")
    if md_path:
        lines.append(f"- 规则校验清单: `{Path(str(md_path)).name}`")
    needs_review = [item for item in validation.get("rules") or [] if item.get("status") == "needs_review"]
    if needs_review:
        lines.append("- 需复核规则: 报告展示前 5 条。")
        for item in needs_review[:5]:
            issues = "；".join(item.get("issues") or []) or "标记为人工复核"
            lines.append(f"- `{item.get('rule_id')}` {item.get('name')}: {issues}")
    lines.append("")


def _append_dataset_adaptation_summary(lines: list[str], context: RunContext, schema: dict[str, Any]) -> None:
    """Summarize cross-dataset rule applicability before rule execution."""
    adaptation = _latest_tool_result(context.tool_results, "review_dataset_adaptation")
    if not isinstance(adaptation, dict):
        return
    lines.append("## 数据适配摘要")
    lines.append(
        f"- 适配结果: {adaptation.get('ready_count', 0)} 条可直接执行，"
        f"{adaptation.get('confirmed_mapping_count', 0)} 条已确认字段映射，"
        f"{adaptation.get('needs_mapping_count', 0)} 条需字段映射确认，"
        f"{adaptation.get('not_applicable_count', 0)} 条不适用。"
    )
    lines.append(
        f"- 数据口径提示: {adaptation.get('dataset_kind_hint', 'unknown')}；"
        f"规则字段覆盖: {adaptation.get('mapped_field_count', 0)}/{adaptation.get('required_field_count', 0)}。"
    )
    md_path = adaptation.get("markdown_path")
    if md_path:
        lines.append(f"- 数据适配审查清单: `{Path(str(md_path)).name}`")
    review_rules = [
        item
        for item in adaptation.get("rules") or []
        if item.get("status") in {"mapping_confirmed", "needs_mapping_confirmation", "not_applicable", "needs_review"}
    ]
    if review_rules:
        lines.append("- 需关注规则: 报告展示前 5 条。")
        for item in review_rules[:5]:
            fields = ", ".join(field_label(str(field), schema) for field in item.get("required_fields") or []) or "未识别字段"
            lines.append(
                f"- `{item.get('rule_id')}` {item.get('name')}: {_cn_status(str(item.get('status')))}；"
                f"字段: {fields}；建议: {item.get('recommendation')}"
            )
    else:
        lines.append("- 需关注规则: 当前没有数据适配层面的阻断项。")
    lines.append("- 原则: 新客户数据与当前样例差异较大时，应先确认字段语义、单位和统计口径，再解释规则命中。")
    lines.append("")


def _append_generalization_readiness_summary(lines: list[str], context: RunContext) -> None:
    """Summarize the anti-overfitting generalization gate."""
    gate = _latest_tool_result(context.tool_results, "review_generalization_readiness")
    if not isinstance(gate, dict):
        return
    metrics = gate.get("metrics") or {}
    lines.append("## 泛化验收门禁")
    lines.append(f"- 门禁状态: {_cn_status(str(gate.get('status')))}。")
    lines.append(f"- 是否允许自动诊断解释: {'是' if gate.get('auto_diagnosis_allowed') else '否'}。")
    lines.append(f"- 结论: {gate.get('verdict')}")
    lines.append(
        f"- 核心指标: 字段语义覆盖率 {float(metrics.get('semantic_coverage') or 0):.1%}；"
        f"未解决规则 {metrics.get('unresolved_rule_count', 0)} 条；"
        f"不可解析规则 {metrics.get('unparseable_rule_count', 0)} 条。"
    )
    md_path = gate.get("markdown_path")
    if md_path:
        lines.append(f"- 泛化验收清单: `{Path(str(md_path)).name}`")
    checks = [item for item in gate.get("checks") or [] if item.get("status") != "passed"]
    if checks:
        lines.append("- 需关注门禁项: 报告展示前 5 条。")
        for item in checks[:5]:
            lines.append(f"- {_cn_status(str(item.get('status')))}｜{item.get('name')}: {item.get('message')}")
    lines.append("- 原则: 不得为了让当前数据集通过而改阈值、改规则或把候选映射直接当作已确认映射。")
    lines.append("")


def _append_mapping_execution_plan_summary(lines: list[str], context: RunContext) -> None:
    """Summarize confirmed mapping execution plan without implying execution."""
    plan = _latest_tool_result(context.tool_results, "build_mapping_execution_plan")
    if not isinstance(plan, dict):
        return
    lines.append("## 映射执行计划")
    lines.append(f"- 计划状态: {_cn_status(str(plan.get('status')))}。")
    lines.append(f"- 是否允许立即自动执行: {'是' if plan.get('auto_execution_allowed') else '否'}。")
    lines.append(
        f"- 规则计划: 无需重写 {plan.get('executable_without_mapping_count', 0)} 条，"
        f"待回归验证 {plan.get('ready_for_regression_count', 0)} 条，"
        f"需确认 {plan.get('needs_confirmation_count', 0)} 条，"
        f"需复核 {plan.get('needs_review_count', 0)} 条，"
        f"阻断 {plan.get('blocked_count', 0)} 条。"
    )
    md_path = plan.get("markdown_path")
    if md_path:
        lines.append(f"- 映射执行计划清单: `{Path(str(md_path)).name}`")
    attention = [item for item in plan.get("rules") or [] if item.get("status") != "executable_without_mapping"]
    if attention:
        lines.append("- 需关注计划: 报告展示前 5 条。")
        for item in attention[:5]:
            lines.append(f"- `{item.get('rule_id')}` {item.get('name')}: {_cn_status(str(item.get('status')))}；建议: {item.get('recommendation')}")
    lines.append("- 原则: 映射执行计划只生成字段重写预案，不自动修改规则条件或阈值。")
    lines.append("")


def _append_mapping_regression_summary(lines: list[str], context: RunContext) -> None:
    """Summarize mapping regression validation without promoting mappings."""
    validation = _latest_tool_result(context.tool_results, "validate_mapping_regression")
    if not isinstance(validation, dict):
        return
    lines.append("## 映射回归验证")
    lines.append(f"- 验证状态: {_cn_status(str(validation.get('status')))}。")
    lines.append(f"- 映射执行是否已验证: {'是' if validation.get('mapped_execution_validated') else '否'}。")
    lines.append(
        f"- 规则验证: 映射重写通过 {validation.get('regression_passed_count', 0)} 条，"
        f"无需映射 {validation.get('no_mapping_count', 0)} 条，"
        f"需确认 {validation.get('needs_confirmation_count', 0)} 条，"
        f"需复核 {validation.get('needs_review_count', 0)} 条，"
        f"阻断/失败 {int(validation.get('blocked_count') or 0) + int(validation.get('failed_count') or 0)} 条。"
    )
    md_path = validation.get("markdown_path")
    if md_path:
        lines.append(f"- 映射回归验证清单: `{Path(str(md_path)).name}`")
    attention = [item for item in validation.get("rules") or [] if item.get("status") not in {"passed_no_mapping", "regression_passed"}]
    if attention:
        lines.append("- 需关注验证项: 报告展示前 5 条。")
        for item in attention[:5]:
            lines.append(f"- `{item.get('rule_id')}` {item.get('name')}: {_cn_status(str(item.get('status')))}；建议: {item.get('recommendation')}")
    lines.append("- 原则: 映射回归只验证重写预案可执行，不自动沉淀全局配置，也不证明阈值业务正确。")
    lines.append("")


def _append_rule_execution_summary(lines: list[str], context: RunContext, schema: dict[str, Any]) -> None:
    """Show which business rules actually matched the dataset."""
    rule_execution = _latest_tool_result(context.tool_results, "apply_rules")
    if not isinstance(rule_execution, dict):
        return

    results = rule_execution.get("rule_results") or []
    evaluated = [item for item in results if item.get("status") == "evaluated"]
    unparseable = [item for item in results if item.get("status") == "unparseable"]
    matched = [item for item in evaluated if int(item.get("matched_rows") or 0) > 0]
    matched.sort(key=lambda item: (item.get("matched_rows", 0), item.get("match_rate", 0)), reverse=True)

    lines.append("## 规则命中摘要")
    lines.append(f"- 执行概况: 成功执行 {len(evaluated)} 条规则，无法解析 {len(unparseable)} 条规则。")
    if matched:
        lines.append("- 命中规则: 按命中行数展示前 10 条。")
        for item in matched[:10]:
            samples = ", ".join(str(i) for i in (item.get("sample_indices") or [])[:5]) or "无"
            lines.append(
                f"- `{item.get('rule_id')}` {item.get('name')}: 命中 {item.get('matched_rows')} 行，"
                f"命中率 {_format_rate(item.get('match_rate'))}；严重程度: {_severity_label(item.get('severity'))}；"
                f"条件: `{label_expression(item.get('condition'), schema, include_raw=True)}`；样例索引: {samples}"
            )
            if item.get("recommendation"):
                lines.append(f"  建议动作: {item.get('recommendation')}")
    elif evaluated:
        lines.append("- 命中规则: 当前数据没有触发已解析规则。")
    else:
        lines.append("- 命中规则: 当前没有可执行规则，不能判断是否触发专家规则。")
    if unparseable:
        for item in unparseable[:5]:
            lines.append(f"- 待人工复核: `{item.get('rule_id')}` {item.get('name')}: {item.get('message')}")
    lines.append("")


def _append_correlation_summary(lines: list[str], context: RunContext, schema: dict[str, Any]) -> None:
    """Summarize strongest numeric correlations as exploratory clues."""
    correlation_result = _latest_tool_result(context.tool_results, "compute_correlations")
    if not isinstance(correlation_result, dict):
        return

    correlations = correlation_result.get("correlations") or []
    lines.append("## 相关性摘要")
    if correlations:
        lines.append("- Pearson 相关性只表示线性同步变化，不代表因果；以下展示绝对相关性最高的前 10 对字段。")
        for item in correlations[:10]:
            lines.append(
                f"- {field_label(str(item.get('field_a')), schema)} 与 {field_label(str(item.get('field_b')), schema)}: "
                f"r={_format_float(item.get('correlation'))}，|r|={_format_float(item.get('abs_correlation'))}"
            )
    else:
        lines.append("- 当前数据可用于相关性分析的有效数值字段不足，或未形成可报告的相关字段对。")
    lines.append("")


def _append_limitations(lines: list[str], context: RunContext) -> None:
    """State the boundaries of the current MVP so the report is not overclaimed."""
    lines.append("## 分析局限")
    dataset_names = [context.files[file_id].original_name for file_id in context.dataset_file_ids if file_id in context.files]
    dataset_label = "、".join(dataset_names) if dataset_names else "本次上传数据集"
    lines.append(f"- 当前报告只分析本次输入数据（{dataset_label}），尚未做多数据集、多统计口径的自动融合。")
    lines.append("- `label` 字段在当前样例中全部为 0，`diagnosis` 为空，因此报告不输出监督分类或故障诊断结论。")
    lines.append("- 关联规则挖掘表示字段状态的统计共现，不等同于因果关系，也不能直接替代专家规则。")
    lines.append("- 本轮不接入独立异常检测项目；现有离群值输出只作为数据质量提示。")
    lines.append("- 报告结论基于当前输入文件、当前规则库和本地工具结果，输入变化后应重新运行并重新验收。")
    lines.append("")


def _append_acceptance_checklist(lines: list[str], context: RunContext) -> None:
    """Add a concrete checklist for the MVP handoff."""
    association = _latest_tool_result(context.tool_results, "mine_association_rules")
    rule_execution = _latest_tool_result(context.tool_results, "apply_rules")
    latest_validation = context.validation_results[-1] if context.validation_results else {}
    backend = association.get("engine") if isinstance(association, dict) else "未运行"
    rule_results = rule_execution.get("rule_results") if isinstance(rule_execution, dict) else []

    evaluated_rule_count = sum(1 for item in (rule_results or []) if item.get("status") == "evaluated")

    checks = [
        ("数据读取与画像", bool(context.dataset_profiles), "已生成行列数、字段列表和字段画像。"),
        ("规则库解析", bool(context.rules), f"已解析 {len(context.rules)} 条规则卡片。"),
        ("规则命中执行", evaluated_rule_count > 0, f"成功执行 {evaluated_rule_count}/{len(rule_results or [])} 条规则。"),
        ("FP-growth 关联挖掘", isinstance(association, dict), f"当前后端: {backend}。"),
        ("报告可视化", isinstance(_latest_tool_result(context.tool_results, "build_visualizations"), dict), "已生成图表片段和独立可视化看板。"),
        ("映射复核辅助", isinstance(_latest_tool_result(context.tool_results, "review_schema_mapping_candidates"), dict), "已将候选映射分组为可考虑确认、需专家复核和不适用。"),
        ("映射执行计划", isinstance(_latest_tool_result(context.tool_results, "build_mapping_execution_plan"), dict), "已为已确认映射生成可审查执行预案。"),
        ("映射回归验证", isinstance(_latest_tool_result(context.tool_results, "validate_mapping_regression"), dict), "已验证映射重写预案的可执行性并保留风险项。"),
        ("泛化验收门禁", isinstance(_latest_tool_result(context.tool_results, "review_generalization_readiness"), dict), "已检查是否存在围绕单一数据集过拟合的风险。"),
        ("结果校验", latest_validation.get("status") == "passed", f"校验状态: {_cn_status(str(latest_validation.get('status', 'not_run')))}。"),
        ("中文报告产物", True, "本文件由 report.md/report.html 生成流程输出。"),
        ("证据链记录", len(context.evidence.nodes) > 0, f"已记录 {len(context.evidence.nodes)} 个证据节点。"),
    ]

    lines.append("## MVP 验收检查项")
    for name, passed, detail in checks:
        status = "通过" if passed else "待补齐"
        lines.append(f"- {name}: {status}；{detail}")
    lines.append("- 外部异常检测项目: 本阶段不纳入验收范围。")
    lines.append("")


def _append_visualization_summary(lines: list[str], context: RunContext) -> None:
    visualization = _latest_tool_result(context.tool_results, "build_visualizations")
    if not isinstance(visualization, dict):
        return
    dashboard_path = visualization.get("dashboard_html_path")
    chart_count = visualization.get("chart_count", 0)
    lines.append("## 可视化看板")
    lines.append(f"- 图表数量: {chart_count}。")
    lines.append("- HTML 报告已嵌入可视化区块，独立看板可通过 Web 路由查看。")
    lines.append(f"- Web 路径: `/runs/{context.run_id}/visualizations`")
    if dashboard_path:
        lines.append(f"- 本地产物: `{Path(str(dashboard_path)).name}`")
    lines.append("")


def _append_schema_summary(lines: list[str], context: RunContext) -> None:
    schema_result = _latest_tool_result(context.tool_results, "suggest_schema_metadata")
    if not isinstance(schema_result, dict):
        return
    schema_review = _latest_tool_result(context.tool_results, "review_schema_metadata")
    schema = schema_result.get("schema") or {}
    fields = schema.get("fields") or {}
    ignored = schema.get("ignored_fields") or []
    rule_bound = schema_result.get("rule_bound_fields") or []
    yaml_path = schema_result.get("schema_yaml_path")
    lines.append("## 字段语义配置")
    lines.append(f"- 字段建议数: {len(fields)}。")
    lines.append(f"- 忽略字段数: {len(ignored)}。")
    lines.append(f"- 规则绑定字段数: {len(rule_bound)}。")
    lines.append("- 说明: 当前字段中文名由稳定配置和本地 Schema 建议工具生成，属于可复核草案；数据计算仍使用原始字段名。")
    if yaml_path:
        lines.append(f"- Schema 草案: `{Path(str(yaml_path)).name}`")
    if isinstance(schema_review, dict):
        lines.append(
            f"- Schema 复核: {schema_review.get('low_confidence_count', 0)} 个低置信度字段，"
            f"{schema_review.get('ignored_count', 0)} 个忽略字段，{len(schema_review.get('review_items') or [])} 个复核项。"
        )
        review_path = schema_review.get("markdown_path")
        if review_path:
            lines.append(f"- Schema 复核清单: `{Path(str(review_path)).name}`")
    lines.append("")


def _append_schema_mapping_summary(lines: list[str], context: RunContext) -> None:
    """Summarize semantic field-role mapping for customer-data generalization."""
    mapping = _latest_tool_result(context.tool_results, "review_schema_mapping")
    if not isinstance(mapping, dict):
        return
    lines.append("## Schema 语义映射")
    lines.append(
        f"- 字段语义覆盖: {mapping.get('semantic_field_count', 0)}/{mapping.get('field_count', 0)} 个字段匹配稳定语义角色。"
    )
    lines.append(
        f"- 规则字段映射: 精确匹配 {mapping.get('exact_mapping_count', 0)} 个，"
        f"已确认映射 {mapping.get('confirmed_mapping_count', 0)} 个，"
        f"候选映射 {mapping.get('candidate_mapping_count', 0)} 个，"
        f"缺少候选 {mapping.get('missing_mapping_count', 0)} 个，"
        f"歧义映射 {mapping.get('ambiguous_mapping_count', 0)} 个。"
    )
    md_path = mapping.get("markdown_path")
    if md_path:
        lines.append(f"- Schema 语义映射清单: `{Path(str(md_path)).name}`")
    attention = [
        item
        for item in mapping.get("rule_fields") or []
        if item.get("status") != "exact" or item.get("ambiguous")
    ]
    if attention:
        lines.append("- 需确认映射: 报告展示前 5 个。")
        for item in attention[:5]:
            selected = item.get("selected_field") or "无候选"
            lines.append(
                f"- `{item.get('rule_field')}` -> `{selected}`；"
                f"语义角色: {item.get('semantic_role') or '未匹配'}；候选数: {item.get('candidate_count')}。"
            )
    else:
        lines.append("- 需确认映射: 当前规则字段均为精确匹配。")
    lines.append("- 原则: 语义映射用于泛化适配审查，不自动改写规则条件或阈值。")
    lines.append("")


def _format_rate(value: Any) -> str:
    try:
        return f"{float(value):.1%}"
    except (TypeError, ValueError):
        return "N/A"


def _format_float(value: Any) -> str:
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "N/A"


def _cn_status(status: str) -> str:
    mapping = {
        "created": "已创建",
        "running": "运行中",
        "complete": "已完成",
        "completed": "已完成",
        "failed": "失败",
        "passed": "通过",
        "needs_review": "需复核",
        "blocked": "阻断",
        "mapping_confirmed": "映射已确认",
        "needs_mapping_confirmation": "需映射确认",
        "executable_without_mapping": "无需映射可执行",
        "ready_for_regression": "待回归验证",
        "not_applicable": "不适用",
        "disabled": "已停用",
        "not_run": "未运行",
        "offline": "离线模式",
    }
    return mapping.get(status, status)


def _severity_label(severity: Any) -> str:
    mapping = {
        "critical": "严重",
        "high": "高",
        "medium": "中",
        "low": "低",
        "info": "提示",
    }
    key = str(severity or "medium").lower()
    return mapping.get(key, str(severity or "中"))


def _render_inline(text: Any) -> str:
    """Render the tiny inline subset used by generated reports."""
    parts = str(text).split("`")
    if len(parts) == 1:
        return html.escape(str(text))
    rendered = []
    for index, part in enumerate(parts):
        escaped = html.escape(part)
        if index % 2:
            rendered.append(f"<code>{escaped}</code>")
        else:
            rendered.append(escaped)
    return "".join(rendered)


def _ordered_item_text(line: str) -> str | None:
    marker, separator, rest = line.partition(". ")
    if separator and marker.isdigit() and rest:
        return rest
    return None


def _intro_key_values(intro_lines: list[str]) -> list[tuple[str, str]]:
    pairs = []
    for item in intro_lines:
        separator = "：" if "：" in item else ":"
        if separator in item:
            key, value = item.split(separator, 1)
            pairs.append((key.strip(), value.strip()))
        elif item.strip():
            pairs.append(("", item.strip()))
    return pairs


def _lookup_intro_value(pairs: list[tuple[str, str]], *keys: str) -> str:
    for key, value in pairs:
        if key in keys:
            return value
    return ""


def _report_status_tone(status: Any) -> str:
    value = str(status or "").lower()
    if value in {"complete", "completed", "passed", "已完成", "通过"}:
        return "ok"
    if value in {"needs_review", "warning", "warn", "需复核", "待复核"}:
        return "warn"
    if value in {"failed", "blocked", "失败", "阻断"}:
        return "danger"
    return "info"


def _section_class(title: str) -> str:
    classes = []
    if title in {"MVP 验收摘要", "可视化看板", "规则命中摘要", "数据适配摘要"}:
        classes.append("section-priority")
    if any(keyword in title for keyword in ["局限", "门禁", "待", "复核"]):
        classes.append("section-attention")
    if any(keyword in title for keyword in ["证据", "校验", "检查项"]):
        classes.append("section-evidence")
    return " ".join(classes)


def _report_metric_cards(context: RunContext | None) -> list[tuple[str, str, str]]:
    if context is None:
        return []
    profile = next(iter(context.dataset_profiles.values()), {})
    rule_execution = _latest_tool_result(context.tool_results, "apply_rules")
    association = _latest_tool_result(context.tool_results, "mine_association_rules")
    validation = context.validation_results[-1] if context.validation_results else {}

    rule_results = rule_execution.get("rule_results") if isinstance(rule_execution, dict) else []
    evaluated = [item for item in (rule_results or []) if item.get("status") == "evaluated"]
    matched = [item for item in evaluated if int(item.get("matched_rows") or 0) > 0]
    total_rules = len(rule_results or []) or len(context.rules)
    association_count = len(association.get("rules") or []) if isinstance(association, dict) else 0
    association_engine = association.get("engine", "未运行") if isinstance(association, dict) else "未运行"
    evidence_nodes = validation.get("evidence_nodes", len(context.evidence.nodes))

    return [
        ("数据规模", f"{profile.get('rows', '未读取')} 行", f"{profile.get('columns', '未读取')} 列"),
        ("规则执行", f"{len(evaluated)}/{total_rules}", f"命中 {len(matched)} 条"),
        ("关联规则", str(association_count), str(association_engine)),
        ("校验状态", _cn_status(str(validation.get("status", "not_run"))), f"证据 {evidence_nodes} 节点"),
    ]


def _generic_intro_cards(intro_pairs: list[tuple[str, str]]) -> list[tuple[str, str, str]]:
    skip_keys = {"用户目标", "验收目标", "Agent 模式"}
    cards = []
    for key, value in intro_pairs:
        if not key or key in skip_keys:
            continue
        cards.append((key, value, ""))
        if len(cards) >= 4:
            break
    return cards


def _render_report_metric(title: str, value: str, note: str = "") -> str:
    note_html = f"<span>{_render_inline(note)}</span>" if note else ""
    return (
        '<article class="report-metric">'
        f"<span>{html.escape(title)}</span>"
        f"<strong>{_render_inline(value)}</strong>"
        f"{note_html}"
        "</article>"
    )


def _display_report_title(markdown_title: str, context: RunContext | None) -> str:
    if markdown_title == "Agent 工具箱 Demo 报告" and context is not None:
        return "标准演示报告" if getattr(context, "run_kind", "analysis") == "demo" else "客户数据分析报告"
    return markdown_title


def _report_hero(title: str, intro_lines: list[str], context: RunContext | None) -> str:
    intro_pairs = _intro_key_values(intro_lines)
    goal = _lookup_intro_value(intro_pairs, "用户目标", "验收目标")
    mode = _lookup_intro_value(intro_pairs, "Agent 模式")
    run_id = context.run_id if context else _lookup_intro_value(intro_pairs, "运行 ID", "批次 ID")
    run_status = context.status if context else _lookup_intro_value(intro_pairs, "运行状态", "批次状态")
    metric_cards = _report_metric_cards(context) or _generic_intro_cards(intro_pairs)
    metric_html = "".join(_render_report_metric(*card) for card in metric_cards)
    status_pill = (
        f'<span class="report-pill {_report_status_tone(run_status)}">运行：{_render_inline(_cn_status(str(run_status)))}</span>'
        if run_status
        else ""
    )
    run_id_pill = f'<span class="report-pill">ID：{_render_inline(run_id)}</span>' if run_id else ""
    mode_pill = f'<span class="report-pill">Agent 编排</span>' if mode else ""
    goal_html = f"<p>{_render_inline(goal)}</p>" if goal else "<p>本报告汇总本次分析任务的输入、执行、校验、可视化和证据链结果。</p>"
    return f"""
  <section class="report-cover" aria-label="报告摘要">
    <div>
      <span class="report-kicker">Analysis Report</span>
      <h1>{html.escape(title or "Agent MVP Report")}</h1>
      {goal_html}
      <div class="report-pills">{status_pill}{run_id_pill}{mode_pill}</div>
    </div>
    <div class="report-metrics">{metric_html}</div>
  </section>
"""


def _report_toc(items: list[tuple[str, str]]) -> str:
    if not items:
        return ""
    links = "".join(f'<a href="#{html.escape(section_id)}">{html.escape(title)}</a>' for section_id, title in items)
    return f'<aside class="report-toc" aria-label="报告目录"><strong>报告目录</strong>{links}</aside>'


def markdown_to_html(markdown: str, context: RunContext | None = None) -> str:
    """Convert generated Markdown into a run-kind-specific standalone HTML report."""
    if _is_demo_report(context):
        return _markdown_to_demo_html(markdown, context=context)
    return _markdown_to_analysis_html(markdown, context=context)


def _is_demo_report(context: RunContext | None) -> bool:
    return bool(context and getattr(context, "run_kind", "analysis") == "demo")


def _markdown_to_analysis_html(markdown: str, context: RunContext | None = None) -> str:
    """Convert the small Markdown subset produced above into standalone HTML.

    这里不是通用 Markdown 渲染器，只支持 build_markdown_report 生成的有限
    子集：标题、列表、段落。完整运行使用分析报告工作台结构展示。
    """
    visualization_html = _visualization_fragment(context)
    body_lines = []
    list_type: str | None = None
    section_open = False
    section_index = 0
    toc_items: list[tuple[str, str]] = []
    intro_lines: list[str] = []
    report_title = "Agent MVP Report"
    before_first_section = True

    def close_list() -> None:
        nonlocal list_type
        if list_type:
            body_lines.append(f"</{list_type}>")
            list_type = None

    def open_list(next_type: str) -> None:
        nonlocal list_type
        if list_type == next_type:
            return
        close_list()
        body_lines.append(f"<{next_type}>")
        list_type = next_type

    for line in markdown.splitlines():
        # 先 html.escape，再根据行前缀包成标签，避免报告内容注入 HTML。
        if line.startswith("# "):
            close_list()
            if section_open:
                body_lines.append("</section>")
                section_open = False
            report_title = line[2:]
        elif line.startswith("## "):
            before_first_section = False
            close_list()
            if section_open:
                body_lines.append("</section>")
            section_open = True
            section_index += 1
            title = line[3:]
            section_id = f"section-{section_index}"
            toc_items.append((section_id, title))
            section_class = _section_class(title)
            body_lines.append(
                f'<section id="{section_id}" class="report-section {section_class}">'
                '<div class="section-heading">'
                f'<span class="section-index">{section_index}</span>'
                f"<h2>{html.escape(title)}</h2>"
                "</div>"
            )
            if title == "可视化看板" and visualization_html:
                body_lines.append(visualization_html)
        elif line.startswith("### "):
            close_list()
            body_lines.append(f"<h3>{html.escape(line[4:])}</h3>")
        elif before_first_section:
            if line.startswith("- "):
                intro_lines.append(line[2:])
            elif line.strip():
                intro_lines.append(line.strip())
        elif line.startswith("- "):
            open_list("ul")
            body_lines.append(f"<li>{_render_inline(line[2:])}</li>")
        elif (ordered_item := _ordered_item_text(line)) is not None:
            open_list("ol")
            body_lines.append(f"<li>{_render_inline(ordered_item)}</li>")
        elif line.strip() == "":
            close_list()
        else:
            close_list()
            body_lines.append(f"<p>{_render_inline(line)}</p>")
    close_list()
    if section_open:
        body_lines.append("</section>")
    report_title = _display_report_title(report_title, context)
    hero = _report_hero(report_title, intro_lines, context)
    toc = _report_toc(toc_items)
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(report_title or "分析报告")}</title>
  <style id="runtime-artifact-polish">
    :root {{ color-scheme: light; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif; }}
    * {{ box-sizing: border-box; }}
    html {{ scroll-behavior: smooth; }}
    body {{ margin: 0; background: #f5f7fa; color: #17212b; }}
    main {{ max-width: 1320px; margin: 0 auto; padding: 28px 28px 60px; }}
    h1, h2, h3, p {{ margin: 0; letter-spacing: 0; }}
    h1 {{ font-size: 32px; line-height: 1.18; }}
    h2 {{ font-size: 19px; line-height: 1.25; }}
    h3 {{ font-size: 15px; margin: 18px 0 9px; }}
    p, li {{ line-height: 1.68; font-size: 14px; }}
    p {{ color: #38485a; }}
    a {{ color: #2563a8; text-decoration: none; font-weight: 700; }}
    a:hover {{ text-decoration: underline; }}
    code {{ background: #eef3f7; border: 1px solid #d6dde6; padding: 2px 5px; border-radius: 4px; overflow-wrap: anywhere; }}
    nav {{ margin-bottom: 16px; display: flex; gap: 9px; flex-wrap: wrap; align-items: center; }}
    nav a, nav span {{ border: 1px solid #d6dde6; background: rgba(255,255,255,.95); color: #243447; padding: 8px 10px; border-radius: 6px; text-decoration: none; font-weight: 700; }}
    nav a:hover {{ background: #fff; border-color: #aebdcc; text-decoration: none; }}
    .report-cover {{ display: grid; grid-template-columns: minmax(0, 1fr) 520px; gap: 22px; align-items: stretch; border: 1px solid #d5e0ea; background: #ffffff; border-radius: 8px; padding: 22px; margin-bottom: 18px; box-shadow: 0 10px 28px rgba(23, 33, 43, 0.06); }}
    .report-cover p {{ max-width: 780px; margin-top: 9px; }}
    .report-kicker {{ display: inline-block; color: #0b5f59; font-size: 12px; font-weight: 850; text-transform: uppercase; margin-bottom: 6px; }}
    .report-pills {{ display: flex; flex-wrap: wrap; gap: 8px; margin-top: 14px; }}
    .report-pill {{ display: inline-flex; align-items: center; min-height: 30px; border: 1px solid #d6dde6; background: #f8fbfc; color: #344457; border-radius: 6px; padding: 5px 9px; font-weight: 750; font-size: 13px; }}
    .report-pill.ok {{ border-color: #b9dec7; background: #e7f5ed; color: #17673a; }}
    .report-pill.warn {{ border-color: #e8cc85; background: #fff4dc; color: #7d4c00; }}
    .report-pill.danger {{ border-color: #f1b4ad; background: #fdecea; color: #a93a2f; }}
    .report-pill.info {{ border-color: #bdd1e9; background: #e7f0fb; color: #1d4f8f; }}
    .report-metrics {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; align-content: stretch; }}
    .report-metric {{ border: 1px solid #d6dde6; background: #f8fbfc; border-radius: 8px; padding: 13px; min-width: 0; }}
    .report-metric span {{ display: block; color: #617080; font-size: 13px; line-height: 1.5; }}
    .report-metric strong {{ display: block; color: #17212b; font-size: 22px; line-height: 1.15; margin: 4px 0; overflow-wrap: anywhere; }}
    .report-layout {{ display: grid; grid-template-columns: 238px minmax(0, 1fr); gap: 18px; align-items: start; }}
    .report-toc {{ position: sticky; top: 18px; border: 1px solid #d5e0ea; background: #ffffff; border-radius: 8px; padding: 14px; display: grid; gap: 6px; max-height: calc(100vh - 36px); overflow: auto; }}
    .report-toc strong {{ margin-bottom: 4px; }}
    .report-toc a {{ display: block; color: #344457; border-radius: 6px; padding: 7px 8px; font-size: 13px; font-weight: 700; }}
    .report-toc a:hover {{ background: #eef6f4; color: #0b5f59; text-decoration: none; }}
    .report-content {{ min-width: 0; display: grid; gap: 14px; }}
    .report-section {{ border: 1px solid #d6dde6; background: #fff; border-radius: 8px; padding: 18px; box-shadow: 0 1px 0 rgba(23, 33, 43, 0.02); scroll-margin-top: 18px; }}
    .report-section.section-priority {{ border-top: 4px solid #0f766e; }}
    .report-section.section-attention {{ border-top-color: #a15c00; }}
    .report-section.section-evidence {{ border-top-color: #2563a8; }}
    .section-heading {{ display: grid; grid-template-columns: 32px minmax(0, 1fr); gap: 10px; align-items: center; padding-bottom: 12px; border-bottom: 1px solid #e5ebf1; margin-bottom: 13px; }}
    .section-index {{ display: inline-grid; place-items: center; width: 28px; height: 28px; border-radius: 6px; background: #dff2ef; color: #0b5f59; font-size: 13px; font-weight: 850; }}
    .report-section ul, .report-section ol {{ display: grid; gap: 8px; margin: 0; padding: 0; }}
    .report-section ul {{ list-style: none; }}
    .report-section ol {{ list-style-position: inside; }}
    .report-section li {{ color: #344457; margin: 0; }}
    .report-section ul li {{ position: relative; padding-left: 18px; }}
    .report-section ul li::before {{ content: ""; position: absolute; left: 2px; top: .7em; width: 7px; height: 7px; border-radius: 50%; background: #0f766e; }}
    .report-section p + p, .report-section p + ul, .report-section p + ol {{ margin-top: 8px; }}
    @media (max-width: 1080px) {{ .report-cover {{ grid-template-columns: 1fr; }} .report-layout {{ grid-template-columns: 1fr; }} .report-toc {{ position: static; max-height: none; grid-template-columns: repeat(2, minmax(0, 1fr)); }} .report-toc strong {{ grid-column: 1 / -1; }} }}
    @media (max-width: 720px) {{ main {{ padding: 22px 14px 44px; }} h1 {{ font-size: 26px; }} .report-cover, .report-section {{ padding: 15px; }} .report-metrics, .report-toc {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body><main>{hero}<div class="report-layout">{toc}<article class="report-content">{''.join(body_lines)}</article></div></main></body></html>"""


def _render_demo_metric(title: str, value: str, note: str = "") -> str:
    note_html = f"<span>{_render_inline(note)}</span>" if note else ""
    return (
        '<article class="demo-metric">'
        f"<span>{html.escape(title)}</span>"
        f"<strong>{_render_inline(value)}</strong>"
        f"{note_html}"
        "</article>"
    )


def _demo_report_hero(title: str, intro_lines: list[str], context: RunContext | None) -> str:
    intro_pairs = _intro_key_values(intro_lines)
    goal = _lookup_intro_value(intro_pairs, "用户目标", "验收目标")
    run_id = context.run_id if context else _lookup_intro_value(intro_pairs, "运行 ID", "批次 ID")
    metric_cards = _report_metric_cards(context) or _generic_intro_cards(intro_pairs)
    metric_html = "".join(_render_demo_metric(*card) for card in metric_cards)
    return f"""
  <section class="demo-report-hero" aria-label="Demo 报告摘要">
    <div class="demo-hero-copy">
      <span class="demo-kicker">Demo Scenario</span>
      <h1>{html.escape(title or "标准演示报告")}</h1>
      <p>{_render_inline(goal or "本页面用于客户演示：快速查看内置样例从接入、分析、可视化到交付产物的完整闭环。")}</p>
      <div class="demo-pills">
        <span>标准样例</span>
        <span>本地确定性计划</span>
        <span>Run {_render_inline(run_id)}</span>
      </div>
    </div>
    <div class="demo-metrics">{metric_html}</div>
  </section>
"""


def _demo_flow(context: RunContext | None) -> str:
    plan_steps = len(context.agent_plan) if context else 0
    validation = context.validation_results[-1] if context and context.validation_results else {}
    evidence_nodes = validation.get("evidence_nodes", len(context.evidence.nodes) if context else 0)
    items = [
        ("接入样例", "加载内置规则库和 last_avg 遥测数据。"),
        ("生成计划", f"展示 Agent 固定演示计划，共 {plan_steps or '多'} 个步骤。"),
        ("调度工具", "按顺序执行画像、规则校验、数据质量、规则命中和关联挖掘。"),
        ("交付结果", f"输出报告、可视化和证据链，当前证据 {evidence_nodes} 节点。"),
    ]
    nodes = "".join(
        "<article>"
        f"<span>{index}</span>"
        f"<strong>{html.escape(title)}</strong>"
        f"<p>{html.escape(detail)}</p>"
        "</article>"
        for index, (title, detail) in enumerate(items, 1)
    )
    return f'<section class="demo-flow" aria-label="Demo 演示流程">{nodes}</section>'


def _demo_nav(items: list[tuple[str, str]]) -> str:
    if not items:
        return ""
    priority_titles = {"MVP 验收摘要", "可视化看板", "数据适配摘要", "规则命中摘要", "结果校验", "证据链"}
    selected = [item for item in items if item[1] in priority_titles]
    if len(selected) < 4:
        selected = items[:8]
    links = "".join(f'<a href="#{html.escape(section_id)}">{html.escape(title)}</a>' for section_id, title in selected)
    return f'<nav class="demo-jump-nav" aria-label="Demo 报告导航">{links}</nav>'


def _markdown_to_demo_html(markdown: str, context: RunContext | None = None) -> str:
    """Render the built-in demo report as a guided presentation-style page."""
    visualization_html = _visualization_fragment(context)
    body_lines = []
    list_type: str | None = None
    section_open = False
    section_index = 0
    toc_items: list[tuple[str, str]] = []
    intro_lines: list[str] = []
    report_title = "标准演示报告"
    before_first_section = True

    def close_list() -> None:
        nonlocal list_type
        if list_type:
            body_lines.append(f"</{list_type}>")
            list_type = None

    def open_list(next_type: str) -> None:
        nonlocal list_type
        if list_type == next_type:
            return
        close_list()
        body_lines.append(f"<{next_type}>")
        list_type = next_type

    for line in markdown.splitlines():
        if line.startswith("# "):
            close_list()
            if section_open:
                body_lines.append("</section>")
                section_open = False
            report_title = line[2:]
        elif line.startswith("## "):
            before_first_section = False
            close_list()
            if section_open:
                body_lines.append("</section>")
            section_open = True
            section_index += 1
            title = line[3:]
            section_id = f"demo-section-{section_index}"
            toc_items.append((section_id, title))
            focus_class = " demo-focus-section" if title in {"MVP 验收摘要", "可视化看板", "规则命中摘要", "结果校验"} else ""
            body_lines.append(
                f'<section id="{section_id}" class="demo-section{focus_class}">'
                '<div class="demo-section-head">'
                f'<span>演示 {section_index}</span>'
                f"<h2>{html.escape(title)}</h2>"
                "</div>"
            )
            if title == "可视化看板" and visualization_html:
                body_lines.append(visualization_html)
        elif line.startswith("### "):
            close_list()
            body_lines.append(f"<h3>{html.escape(line[4:])}</h3>")
        elif before_first_section:
            if line.startswith("- "):
                intro_lines.append(line[2:])
            elif line.strip():
                intro_lines.append(line.strip())
        elif line.startswith("- "):
            open_list("ul")
            body_lines.append(f"<li>{_render_inline(line[2:])}</li>")
        elif (ordered_item := _ordered_item_text(line)) is not None:
            open_list("ol")
            body_lines.append(f"<li>{_render_inline(ordered_item)}</li>")
        elif line.strip() == "":
            close_list()
        else:
            close_list()
            body_lines.append(f"<p>{_render_inline(line)}</p>")
    close_list()
    if section_open:
        body_lines.append("</section>")

    report_title = _display_report_title(report_title, context)
    hero = _demo_report_hero(report_title, intro_lines, context)
    flow = _demo_flow(context)
    jump_nav = _demo_nav(toc_items)
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(report_title or "标准演示报告")}</title>
  <style id="runtime-artifact-polish">
    :root {{ color-scheme: light; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif; }}
    * {{ box-sizing: border-box; }}
    html {{ scroll-behavior: smooth; }}
    body {{ margin: 0; background: #f4f7f8; color: #17212b; }}
    main.demo-report-main {{ max-width: 1360px; margin: 0 auto; padding: 28px 28px 60px; }}
    h1, h2, h3, p {{ margin: 0; letter-spacing: 0; }}
    h1 {{ font-size: 34px; line-height: 1.16; }}
    h2 {{ font-size: 19px; line-height: 1.25; }}
    h3 {{ font-size: 15px; margin: 16px 0 8px; }}
    p, li {{ font-size: 14px; line-height: 1.68; }}
    a {{ color: #2563a8; text-decoration: none; font-weight: 750; }}
    a:hover {{ text-decoration: underline; }}
    code {{ background: #eef3f7; border: 1px solid #d6dde6; padding: 2px 5px; border-radius: 4px; overflow-wrap: anywhere; }}
    nav:first-child {{ margin-bottom: 16px; display: flex; gap: 9px; flex-wrap: wrap; align-items: center; }}
    nav:first-child a, nav:first-child span {{ border: 1px solid #d6dde6; background: rgba(255,255,255,.95); color: #243447; padding: 8px 10px; border-radius: 6px; text-decoration: none; font-weight: 700; }}
    .demo-report-hero {{ display: grid; grid-template-columns: minmax(0, 1fr) 480px; gap: 18px; border: 1px solid #cfdde4; background: #ffffff; border-radius: 8px; padding: 24px; margin-bottom: 14px; box-shadow: 0 10px 24px rgba(23, 33, 43, .055); }}
    .demo-hero-copy p {{ color: #38485a; max-width: 780px; margin-top: 9px; }}
    .demo-kicker {{ display: inline-block; color: #a15c00; font-size: 12px; font-weight: 850; text-transform: uppercase; margin-bottom: 6px; }}
    .demo-pills {{ display: flex; flex-wrap: wrap; gap: 8px; margin-top: 14px; }}
    .demo-pills span {{ display: inline-flex; align-items: center; min-height: 30px; border: 1px solid #d6dde6; background: #f8fbfc; border-radius: 6px; padding: 5px 9px; color: #344457; font-size: 13px; font-weight: 750; }}
    .demo-metrics {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; }}
    .demo-metric {{ border: 1px solid #d6dde6; background: #f8fbfc; border-radius: 8px; padding: 13px; min-width: 0; }}
    .demo-metric span {{ display: block; color: #617080; font-size: 13px; line-height: 1.5; }}
    .demo-metric strong {{ display: block; color: #17212b; font-size: 22px; line-height: 1.15; margin: 4px 0; overflow-wrap: anywhere; }}
    .demo-flow {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 10px; margin-bottom: 14px; }}
    .demo-flow article {{ border: 1px solid #d6dde6; background: #ffffff; border-radius: 8px; padding: 14px; min-height: 126px; }}
    .demo-flow span {{ display: inline-grid; place-items: center; width: 28px; height: 28px; border-radius: 6px; background: #dff2ef; color: #0b5f59; font-weight: 850; margin-bottom: 9px; }}
    .demo-flow strong {{ display: block; margin-bottom: 5px; }}
    .demo-flow p {{ color: #617080; font-size: 13px; }}
    .demo-jump-nav {{ position: sticky; top: 0; z-index: 2; display: flex; gap: 8px; flex-wrap: wrap; align-items: center; border: 1px solid #d6dde6; background: rgba(255,255,255,.96); border-radius: 8px; padding: 10px; margin: 0 0 14px; backdrop-filter: blur(8px); }}
    .demo-jump-nav a {{ border: 1px solid #d6dde6; background: #f8fbfc; color: #344457; border-radius: 6px; padding: 7px 9px; font-size: 13px; text-decoration: none; }}
    .demo-jump-nav a:hover {{ background: #eef6f4; border-color: #b8d8d4; color: #0b5f59; }}
    .demo-report-content {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; align-items: start; }}
    .demo-section {{ border: 1px solid #d6dde6; background: #ffffff; border-radius: 8px; padding: 17px; scroll-margin-top: 84px; min-width: 0; }}
    .demo-focus-section {{ grid-column: 1 / -1; border-top: 4px solid #0f766e; }}
    .demo-section-head {{ display: flex; gap: 10px; align-items: center; padding-bottom: 12px; border-bottom: 1px solid #e5ebf1; margin-bottom: 13px; }}
    .demo-section-head span {{ display: inline-grid; place-items: center; min-width: 58px; height: 28px; border-radius: 6px; background: #fff4dc; color: #7d4c00; font-size: 12px; font-weight: 850; }}
    .demo-section ul, .demo-section ol {{ display: grid; gap: 8px; margin: 0; padding: 0; }}
    .demo-section ul {{ list-style: none; }}
    .demo-section ol {{ list-style-position: inside; }}
    .demo-section li {{ color: #344457; margin: 0; }}
    .demo-section ul li {{ position: relative; padding-left: 18px; }}
    .demo-section ul li::before {{ content: ""; position: absolute; left: 2px; top: .7em; width: 7px; height: 7px; border-radius: 50%; background: #0f766e; }}
    .demo-section p + p, .demo-section p + ul, .demo-section p + ol {{ margin-top: 8px; }}
    @media (max-width: 1120px) {{ .demo-report-hero {{ grid-template-columns: 1fr; }} .demo-flow {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }} .demo-report-content {{ grid-template-columns: 1fr; }} }}
    @media (max-width: 720px) {{ main.demo-report-main {{ padding: 22px 14px 44px; }} h1 {{ font-size: 26px; }} .demo-report-hero, .demo-section {{ padding: 15px; }} .demo-metrics, .demo-flow {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body><main class="demo-report-main">{hero}{flow}{jump_nav}<article class="demo-report-content">{''.join(body_lines)}</article></main></body></html>"""


def _visualization_fragment(context: RunContext | None) -> str:
    if context is None:
        return ""
    visualization = _latest_tool_result(context.tool_results, "build_visualizations")
    if not isinstance(visualization, dict):
        return ""
    fragment_raw = visualization.get("html_fragment_path")
    if not fragment_raw:
        return ""
    fragment_path = Path(str(fragment_raw))
    try:
        resolved_fragment = fragment_path.resolve()
        resolved_run = context.run_dir.resolve()
        if not str(resolved_fragment).startswith(str(resolved_run)):
            return ""
        if not resolved_fragment.exists():
            return ""
        return resolved_fragment.read_text(encoding="utf-8")
    except Exception:
        return ""


def write_report_files(context: RunContext, final_summary: str = "") -> dict[str, Any]:
    """Write both Markdown and HTML reports under the run artifacts directory.

    Markdown 便于审阅/归档，HTML 便于 Web 直接展示。两个文件都写在
    runtime/<run_id>/artifacts/ 下。
    """
    markdown = build_markdown_report(context, final_summary)
    artifacts_dir = context.run_dir / "artifacts"
    markdown_path = artifacts_dir / "report.md"
    html_path = artifacts_dir / "report.html"
    markdown_path.write_text(markdown, encoding="utf-8")
    html_path.write_text(markdown_to_html(markdown, context=context), encoding="utf-8")
    context.report_markdown_path = markdown_path
    context.report_html_path = html_path
    context.save_manifest()
    return {"report_markdown": str(markdown_path), "report_html": str(html_path), "summary": "报告文件已生成。"}
