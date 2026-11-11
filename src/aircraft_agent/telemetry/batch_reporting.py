"""Pure rendering of the batch overview; no execution or dataset adaptation."""
from typing import Any

def build_batch_markdown(manifest: dict[str, Any], adaptation_summary: dict[str, int]) -> str:
    items = manifest.get("items") or []
    completed = len(items)
    passed = len([item for item in items if item.get("validation_status") == "passed"])
    needs_review = len([item for item in items if item.get("validation_status") == "needs_review"])
    failed = len([item for item in items if item.get("status") == "failed"])
    lines = [
        "# 批量验收汇总报告",
        "",
        f"- 批次 ID: `{manifest.get('batch_id')}`",
        f"- 批次状态: {_cn_status(str(manifest.get('status', 'unknown')))}",
        f"- 验收目标: {manifest.get('goal')}",
        f"- 规则库: `{manifest.get('rule_path')}`",
        f"- 数据集目录: `{manifest.get('dataset_dir')}`",
        f"- 完成进度: {completed}/{manifest.get('dataset_total', 0)}",
        f"- 通过: {passed}；需复核: {needs_review}；失败: {failed}",
        f"- 规则稳定性分析: `/batches/{manifest.get('batch_id')}/rule-analysis`",
        f"- 数据适配审查: `/batches/{manifest.get('batch_id')}/dataset-adaptation`",
        f"- 映射复核辅助: `/batches/{manifest.get('batch_id')}/schema-mapping-review`",
        f"- 人工复核意见: `/batches/{manifest.get('batch_id')}/review-decisions`",
        f"- 映射回归验证: `/batches/{manifest.get('batch_id')}/mapping-regression`",
        f"- 泛化门禁汇总: `/batches/{manifest.get('batch_id')}/generalization-readiness`",
        "",
        "## 批量结论",
    ]
    if items and completed == manifest.get("dataset_total") and failed == 0 and needs_review == 0:
        lines.append("- 当前批次全部通过，可作为多数据集稳定性验收样例。")
    elif items:
        lines.append("- 当前批次已形成可复核结果，但仍存在需复核或失败数据集。")
    else:
        lines.append("- 当前批次尚未产生数据集结果。")
    lines.append("")
    lines.append("## 数据适配汇总")
    if adaptation_summary["datasets_with_adaptation"]:
        lines.append(
            f"- 适配证据覆盖: {adaptation_summary['datasets_with_adaptation']}/{completed} 个已完成数据集。"
        )
        lines.append(
            f"- 可直接执行规则合计: {adaptation_summary['ready_rules']}；"
            f"需映射确认: {adaptation_summary['needs_mapping_rules']}；"
            f"不适用: {adaptation_summary['not_applicable_rules']}；"
            f"需复核/停用: {adaptation_summary['review_or_disabled_rules']}。"
        )
        lines.append(
            f"- 数据集层面: 全部规则直接适配 {adaptation_summary['all_ready_datasets']} 个，"
            f"存在映射确认 {adaptation_summary['needs_mapping_datasets']} 个，"
            f"存在不适用规则 {adaptation_summary['not_applicable_datasets']} 个。"
        )
    elif items:
        lines.append("- 当前批次没有找到数据适配审查产物。若这是旧批次，需要重新运行批量任务才能形成第 6 步适配证据。")
    else:
        lines.append("- 当前批次尚未产生数据适配结果。")
    lines.append("")
    lines.append("## 数据集结果")
    for item in items:
        adaptation = item.get("dataset_adaptation") or {}
        adaptation_text = (
            f"适配 {adaptation.get('ready_count', 0)}/{adaptation.get('rule_count', 0)} 可直接执行，"
            f"需映射 {adaptation.get('needs_mapping_count', 0)}，不适用 {adaptation.get('not_applicable_count', 0)}"
            if adaptation
            else "适配审查未记录"
        )
        lines.append(
            f"- `{item.get('dataset')}`: {_cn_status(str(item.get('validation_status', item.get('status', 'unknown'))))}；"
            f"规则 {item.get('rules_evaluated', 0)}/{item.get('rules_total', 0)} 可执行，"
            f"命中 {item.get('rules_matched', 0)} 条，关联规则 {item.get('association_rules', 0)} 条，"
            f"Schema 复核项 {item.get('schema_review_items', 0)}；{adaptation_text}；报告: `{item.get('report_url')}`"
        )
    lines.append("")
    lines.append("## 规则稳定性线索")
    rule_hits: dict[str, dict[str, Any]] = {}
    for item in items:
        for rule in item.get("top_rules") or []:
            key = str(rule.get("rule_id"))
            record = rule_hits.setdefault(key, {"name": rule.get("name"), "datasets": 0, "max_rate": 0.0})
            record["datasets"] += 1
            record["max_rate"] = max(float(record["max_rate"]), float(rule.get("match_rate") or 0))
    if rule_hits:
        for rule_id, record in sorted(rule_hits.items()):
            lines.append(
                f"- `{rule_id}` {record.get('name')}: 在 {record.get('datasets')} 个数据集中进入命中 Top，最高命中率 {float(record.get('max_rate') or 0):.1%}。"
            )
    else:
        lines.append("- 暂无可汇总的规则命中线索。")
    lines.append("")
    lines.append("## 后续复核建议")
    lines.append("- 对高频命中规则检查阈值是否过宽，尤其是电池余量、电压和垂直速度类规则。")
    lines.append("- 对需复核数据集先查看单次规则校验页，再决定是否调整规则库或数据口径。")
    lines.append("- FP-growth 结果作为共现线索，不应替代专家规则结论。")
    return "\n".join(lines) + "\n"


def _cn_status(status: str) -> str:
    return {
        "running": "运行中",
        "complete": "已完成",
        "failed": "失败",
        "passed": "通过",
        "needs_review": "需复核",
        "not_run": "未运行",
        "unknown": "未知",
    }.get(status, status)
