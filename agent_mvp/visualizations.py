"""Static visualization generation for report artifacts.

可视化工具只读取 RunContext 中已经记录的工具结果，不重新读取原始数据，也不
重新执行分析。它生成静态 HTML/CSS 图表，便于本地部署、离线查看和证据链
归档。
"""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any

from .runtime import RunContext
from .schema_metadata import field_display_name, label_expression, latest_schema_metadata
from .utils import write_json


def build_visualization_artifacts(context: RunContext, max_items: int = 8) -> dict[str, Any]:
    """Create static visualization artifacts for the current run."""
    max_items = max(3, min(int(max_items or 8), 12))
    data = _collect_visualization_data(context, max_items=max_items)
    fragment = _render_dashboard_fragment(data)
    full_html = _render_full_dashboard_html(context, fragment)

    artifacts_dir = context.run_dir / "artifacts"
    fragment_path = artifacts_dir / "visualizations_fragment.html"
    dashboard_path = artifacts_dir / "visualizations.html"
    data_path = artifacts_dir / "visualizations.json"

    fragment_path.write_text(fragment, encoding="utf-8")
    dashboard_path.write_text(full_html, encoding="utf-8")
    write_json(data_path, data)

    chart_count = sum(1 for chart in data["charts"] if chart["rows"])
    return {
        "html_fragment_path": str(fragment_path),
        "dashboard_html_path": str(dashboard_path),
        "data_path": str(data_path),
        "chart_count": chart_count,
        "metrics": data["metrics"],
        "summary": f"已生成 {chart_count} 个报告可视化图表。",
    }


def _collect_visualization_data(context: RunContext, max_items: int) -> dict[str, Any]:
    schema = latest_schema_metadata(context)
    profile = next(iter(context.dataset_profiles.values()), {})
    rule_execution = _latest_tool_result(context.tool_results, "apply_rules")
    outlier_result = _latest_tool_result(context.tool_results, "detect_outliers")
    association_result = _latest_tool_result(context.tool_results, "mine_association_rules")
    correlation_result = _latest_tool_result(context.tool_results, "compute_correlations")
    validation = context.validation_results[-1] if context.validation_results else {}

    rule_results = rule_execution.get("rule_results", []) if isinstance(rule_execution, dict) else []
    evaluated_rules = [item for item in rule_results if item.get("status") == "evaluated"]
    matched_rules = [item for item in evaluated_rules if int(item.get("matched_rows") or 0) > 0]
    matched_rules.sort(key=lambda item: (item.get("matched_rows", 0), item.get("match_rate", 0)), reverse=True)

    outliers = outlier_result.get("outliers", []) if isinstance(outlier_result, dict) else []
    positive_outliers = [item for item in outliers if int(item.get("outlier_count") or 0) > 0]
    positive_outliers.sort(key=lambda item: (item.get("outlier_rate", 0), item.get("outlier_count", 0)), reverse=True)

    association_rules = association_result.get("rules", []) if isinstance(association_result, dict) else []
    association_rules = sorted(
        association_rules,
        key=lambda item: (float(item.get("lift") or 0), float(item.get("confidence") or 0)),
        reverse=True,
    )

    correlations = correlation_result.get("correlations", []) if isinstance(correlation_result, dict) else []
    correlations = sorted(correlations, key=lambda item: float(item.get("abs_correlation") or 0), reverse=True)

    metrics = {
        "rows": profile.get("rows", 0),
        "columns": profile.get("columns", 0),
        "rules_total": len(context.rules),
        "rules_evaluated": len(evaluated_rules),
        "rules_matched": len(matched_rules),
        "association_rules": len(association_result.get("rules", [])) if isinstance(association_result, dict) else 0,
        "validation_evidence_nodes": validation.get("evidence_nodes", len(context.evidence.nodes)),
        "validation_status": validation.get("status", "not_run"),
        "association_engine": association_result.get("engine", "未运行") if isinstance(association_result, dict) else "未运行",
    }

    return {
        "run_id": context.run_id,
        "metrics": metrics,
        "charts": [
            {
                "id": "rule_match_rate",
                "title": "规则命中率 Top",
                "subtitle": "按命中行数排序，展示规则触发强度。",
                "unit": "%",
                "kind": "percent",
                "accent": "green",
                "rows": [
                    {
                        "label": label_expression(f"{item.get('rule_id')} {item.get('name')}", schema),
                        "value": round(float(item.get("match_rate") or 0) * 100, 2),
                        "note": f"{item.get('matched_rows')} 行",
                    }
                    for item in matched_rules[:max_items]
                ],
            },
            {
                "id": "outlier_rate",
                "title": "数据质量离群提示",
                "subtitle": "基础统计离群值提示，仅作为数据质量线索。",
                "unit": "%",
                "kind": "percent",
                "accent": "amber",
                "rows": [
                    {
                        "label": field_display_name(str(item.get("column")), schema),
                        "value": round(float(item.get("outlier_rate") or 0) * 100, 2),
                        "note": f"{item.get('outlier_count')} 行",
                    }
                    for item in positive_outliers[:max_items]
                ],
            },
            {
                "id": "association_lift",
                "title": "FP-growth 关联强度",
                "subtitle": "按提升度 lift 排序，越高表示共现强度越突出。",
                "unit": "lift",
                "kind": "number",
                "accent": "blue",
                "rows": [
                    {
                        "label": f"{label_expression(item.get('antecedent'), schema)} -> {label_expression(item.get('consequent'), schema)}",
                        "value": round(float(item.get("lift") or 0), 3),
                        "note": f"置信度 {round(float(item.get('confidence') or 0) * 100, 1)}%",
                    }
                    for item in association_rules[:max_items]
                ],
            },
            {
                "id": "correlation_abs",
                "title": "数值字段相关性",
                "subtitle": "Pearson |r| 排序，只表示线性同步变化。",
                "unit": "|r|",
                "kind": "number",
                "accent": "slate",
                "rows": [
                    {
                        "label": f"{field_display_name(str(item.get('field_a')), schema)} / {field_display_name(str(item.get('field_b')), schema)}",
                        "value": round(float(item.get("abs_correlation") or 0), 3),
                        "note": f"r={round(float(item.get('correlation') or 0), 3)}",
                    }
                    for item in correlations[:max_items]
                ],
            },
        ],
    }


def _render_dashboard_fragment(data: dict[str, Any]) -> str:
    metrics = data["metrics"]
    cards = [
        ("数据规模", f"{metrics.get('rows', 0)} 行", f"{metrics.get('columns', 0)} 列"),
        ("规则执行", f"{metrics.get('rules_evaluated', 0)}/{metrics.get('rules_total', 0)}", f"命中 {metrics.get('rules_matched', 0)} 条"),
        ("关联规则", str(metrics.get("association_rules", 0)), str(metrics.get("association_engine", "未运行"))),
        ("校验状态", _cn_status(str(metrics.get("validation_status", "not_run"))), f"证据 {metrics.get('validation_evidence_nodes', 0)} 节点"),
    ]
    return f"""<style>
  .viz-dashboard {{ display: grid; gap: 18px; margin: 12px 0 8px; }}
  .viz-metrics {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; }}
  .viz-metric {{ border: 1px solid #d6dde6; background: #ffffff; border-radius: 8px; padding: 14px; box-shadow: 0 1px 0 rgba(23, 33, 43, 0.02); }}
  .viz-metric span {{ display: block; color: #5d6875; font-size: 13px; }}
  .viz-metric strong {{ display: block; color: #17212b; font-size: 22px; line-height: 1.25; margin: 4px 0; }}
  .viz-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; }}
  .viz-card {{ border: 1px solid #d6dde6; background: #ffffff; border-radius: 8px; padding: 16px; min-width: 0; box-shadow: 0 1px 0 rgba(23, 33, 43, 0.02); }}
  .viz-card h3 {{ margin: 0; font-size: 16px; letter-spacing: 0; }}
  .viz-subtitle {{ color: #5d6875; margin: 5px 0 14px; font-size: 13px; line-height: 1.5; }}
  .viz-bars {{ display: grid; gap: 10px; }}
  .viz-row {{ display: grid; grid-template-columns: 28px minmax(120px, 0.95fr) minmax(150px, 1.25fr) minmax(92px, auto); gap: 9px; align-items: center; }}
  .viz-rank {{ display: inline-grid; place-items: center; width: 24px; height: 24px; border-radius: 6px; background: #edf3f7; color: #334155; font-size: 12px; font-weight: 800; }}
  .viz-label {{ overflow: hidden; overflow-wrap: anywhere; max-height: 2.8em; color: #243447; font-size: 12px; line-height: 1.35; }}
  .viz-track {{ height: 12px; background: #edf2f7; border-radius: 999px; overflow: hidden; border: 1px solid #dfe7ee; }}
  .viz-fill {{ display: block; height: 100%; border-radius: inherit; }}
  .viz-fill.green {{ background: #0f766e; }}
  .viz-fill.amber {{ background: #b36b00; }}
  .viz-fill.blue {{ background: #2757a4; }}
  .viz-fill.slate {{ background: #58677a; }}
  .viz-value {{ color: #17212b; font-weight: 750; font-size: 12px; text-align: right; }}
  .viz-empty {{ border: 1px dashed #b8c5d2; color: #5d6875; border-radius: 8px; padding: 16px; background: #fbfcfd; }}
  @media (max-width: 960px) {{ .viz-metrics {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }} .viz-grid {{ grid-template-columns: 1fr; }} }}
  @media (max-width: 640px) {{ .viz-metrics {{ grid-template-columns: 1fr; }} .viz-row {{ grid-template-columns: 28px minmax(0, 1fr); gap: 6px; }} .viz-track, .viz-value {{ grid-column: 2; }} .viz-value {{ text-align: left; }} }}
</style>
<section class="viz-dashboard" aria-label="报告可视化看板">
  <div class="viz-metrics">
    {''.join(_render_metric_card(*card) for card in cards)}
  </div>
  <div class="viz-grid">
    {''.join(_render_chart(chart) for chart in data["charts"])}
  </div>
</section>"""


def _render_metric_card(title: str, value: str, note: str) -> str:
    return (
        '<div class="viz-metric">'
        f"<span>{html.escape(title)}</span>"
        f"<strong>{html.escape(str(value))}</strong>"
        f"<span>{html.escape(str(note))}</span>"
        "</div>"
    )


def _render_chart(chart: dict[str, Any]) -> str:
    rows = chart.get("rows") or []
    if not rows:
        body = '<div class="viz-empty">当前没有可展示的数据。</div>'
    else:
        max_value = max(float(row.get("value") or 0) for row in rows) or 1.0
        if chart.get("kind") == "percent":
            max_value = max(max_value, 100.0)
        body = '<div class="viz-bars">' + "".join(
            _render_bar_row(row, max_value, chart, index + 1) for index, row in enumerate(rows)
        ) + "</div>"
    return (
        '<article class="viz-card">'
        f"<h3>{html.escape(str(chart.get('title', '')))}</h3>"
        f"<p class=\"viz-subtitle\">{html.escape(str(chart.get('subtitle', '')))}</p>"
        f"{body}"
        "</article>"
    )


def _render_bar_row(row: dict[str, Any], max_value: float, chart: dict[str, Any], rank: int) -> str:
    value = float(row.get("value") or 0)
    width = max(2.0, min(100.0, value / max_value * 100.0))
    unit = str(chart.get("unit", ""))
    value_label = f"{value:.1f}%" if unit == "%" else f"{value:g} {unit}".strip()
    if row.get("note"):
        value_label = f"{value_label} · {row.get('note')}"
    label = str(row.get("label", ""))
    accent = html.escape(str(chart.get("accent", "green")))
    return (
        '<div class="viz-row">'
        f'<span class="viz-rank">{rank}</span>'
        f'<span class="viz-label" title="{html.escape(label)}">{html.escape(label)}</span>'
        f'<span class="viz-track"><span class="viz-fill {accent}" style="width:{width:.1f}%"></span></span>'
        f'<span class="viz-value">{html.escape(value_label)}</span>'
        "</div>"
    )


def _render_full_dashboard_html(context: RunContext, fragment: str) -> str:
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Agent MVP 可视化看板</title>
  <style>
    :root {{ color-scheme: light; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", sans-serif; }}
    body {{ margin: 0; background: linear-gradient(180deg, #e8f1f4 0, #f5f7fa 260px, #f8fafc 100%); color: #17212b; }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 28px 20px 56px; }}
    header {{ display: grid; gap: 8px; margin-bottom: 18px; padding-bottom: 16px; border-bottom: 1px solid #d6dde6; }}
    h1 {{ font-size: 26px; margin: 0; letter-spacing: 0; }}
    p {{ color: #5d6875; line-height: 1.65; margin: 0; }}
    a {{ color: #2757a4; text-decoration: none; font-weight: 650; }}
    code {{ background: #eef3f7; border: 1px solid #d6dde6; padding: 2px 5px; border-radius: 4px; }}
  </style>
</head>
<body>
<main>
  <header>
    <h1>Agent MVP 可视化看板</h1>
    <p>Run ID: <code>{html.escape(context.run_id)}</code>。本页面由 <code>build_visualizations</code> 工具生成。</p>
  </header>
  {fragment}
</main>
</body>
</html>"""


def _latest_tool_result(tool_results: list[dict[str, Any]], tool_name: str) -> Any:
    for item in reversed(tool_results):
        if item.get("tool") == tool_name:
            return item.get("result")
    return None


def _cn_status(status: str) -> str:
    mapping = {
        "created": "已创建",
        "running": "运行中",
        "complete": "已完成",
        "completed": "已完成",
        "failed": "失败",
        "passed": "通过",
        "needs_review": "需复核",
        "not_run": "未运行",
    }
    return mapping.get(status, status)
