"""Batch acceptance runner for repeated telemetry validation.

批量验收层用于回答一个关键问题：同一套规则库和同一套 Agent 工具链，
能否在多个数据集上稳定生成可复核结果。它不引入新的算法，只复用
RunStore、AgentRunner 和 ToolExecutor，把多个普通 run 汇总成一个批次报告。
"""

from __future__ import annotations

import json
from statistics import mean, median
import uuid
from pathlib import Path
from typing import Any

from .agent import AgentRunner
from .config import MODEL_RUN_MODE_LOCAL, Settings, build_run_settings
from .demo_data import DEMO_RULE_PATH, PROJECT_ROOT
from .reporting import markdown_to_html
from .review_decisions import REVIEW_DECISIONS_NAME, load_review_decisions
from .runtime import RunStore
from .tools import ToolExecutor
from .utils import read_json, utc_now_iso, write_json


BATCH_ROOT_NAME = "batches"
BATCH_MANIFEST_NAME = "batch_manifest.json"
BATCH_GOAL = "批量验收飞行遥测数据，验证 Schema 语义映射、数据适配、规则库、FP-growth 关联挖掘、报告和证据链的稳定性。"
RULE_ANALYSIS_JSON_NAME = "batch_rule_analysis.json"
RULE_ANALYSIS_MD_NAME = "batch_rule_analysis.md"
RULE_ANALYSIS_HTML_NAME = "batch_rule_analysis.html"
RULE_COMPARISON_JSON_PREFIX = "rule_version_comparison"
RULE_COMPARISON_MD_PREFIX = "rule_version_comparison"
RULE_COMPARISON_HTML_PREFIX = "rule_version_comparison"
KEY_HIT_SAMPLES_JSON_NAME = "key_hit_samples.json"
KEY_HIT_SAMPLES_MD_NAME = "key_hit_samples.md"
KEY_HIT_SAMPLES_HTML_NAME = "key_hit_samples.html"
BATCH_DATASET_ADAPTATION_JSON_NAME = "batch_dataset_adaptation.json"
BATCH_DATASET_ADAPTATION_MD_NAME = "batch_dataset_adaptation.md"
BATCH_DATASET_ADAPTATION_HTML_NAME = "batch_dataset_adaptation.html"
BATCH_SCHEMA_MAPPING_REVIEW_JSON_NAME = "batch_schema_mapping_review.json"
BATCH_SCHEMA_MAPPING_REVIEW_MD_NAME = "batch_schema_mapping_review.md"
BATCH_SCHEMA_MAPPING_REVIEW_HTML_NAME = "batch_schema_mapping_review.html"
BATCH_REVIEW_DECISIONS_JSON_NAME = "batch_review_decisions.json"
BATCH_REVIEW_DECISIONS_MD_NAME = "batch_review_decisions.md"
BATCH_REVIEW_DECISIONS_HTML_NAME = "batch_review_decisions.html"
BATCH_GENERALIZATION_JSON_NAME = "batch_generalization_readiness.json"
BATCH_GENERALIZATION_MD_NAME = "batch_generalization_readiness.md"
BATCH_GENERALIZATION_HTML_NAME = "batch_generalization_readiness.html"
BATCH_MAPPING_REGRESSION_JSON_NAME = "batch_mapping_regression.json"
BATCH_MAPPING_REGRESSION_MD_NAME = "batch_mapping_regression.md"
BATCH_MAPPING_REGRESSION_HTML_NAME = "batch_mapping_regression.html"
DEFAULT_HIT_SAMPLE_RULE_IDS = ["R001", "R002", "R005", "R007"]
RULE_CONTEXT_FIELDS = {
    "R001": [
        "1_battery_remaining_avg",
        "1_voltage_battery_avg",
        "1_current_battery_avg",
        "168_speed_avg",
        "33_alt_avg",
        "33_vz_avg",
    ],
    "R002": [
        "1_voltage_battery_avg",
        "1_battery_remaining_avg",
        "1_current_battery_avg",
        "168_speed_avg",
        "33_alt_avg",
        "33_vz_avg",
    ],
    "R005": [
        "29_temperature_avg",
        "29_press_abs_avg",
        "29_press_diff_avg",
        "33_alt_avg",
        "168_speed_avg",
    ],
    "R007": [
        "168_speed_avg",
        "33_alt_avg",
        "33_vz_avg",
        "33_vx_avg",
        "33_vy_avg",
        "241_vibration_z_avg",
    ],
}

SCHEMA_MAPPING_REVIEW_STATUS_LABELS = {
    "ready_for_human_review": "待人工复核",
    "needs_human_review": "需人工复核",
    "has_confirm_suggestions": "有确认建议",
    "already_resolved": "已处理",
    "suggest_confirm_candidate": "可考虑确认",
    "manual_review_required": "需专家复核",
    "not_applicable": "不适用",
    "already_safe": "已安全处理",
    "missing_review": "缺少复核证据",
}


def default_batch_dataset_dir() -> Path:
    """Return the default dataset directory used by the MVP batch acceptance."""
    return PROJECT_ROOT / "dataset" / "last_avg"


def customer_variant_dataset_dir() -> Path:
    """Return the built-in customer-schema variant dataset directory."""
    return PROJECT_ROOT / "dataset" / "customer_variants"


def named_batch_dataset_dirs() -> dict[str, Path]:
    """Known dataset directories exposed by CLI/Web forms."""
    return {
        "last_avg": default_batch_dataset_dir(),
        "customer_variants": customer_variant_dataset_dir(),
    }


def discover_batch_datasets(dataset_dir: Path | None = None, max_datasets: int | None = None) -> list[Path]:
    """Find CSV/XLSX datasets for batch validation in stable filename order."""
    root = dataset_dir or default_batch_dataset_dir()
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Batch dataset directory not found: {root}")
    candidates = sorted(
        [path for path in root.iterdir() if path.is_file() and path.suffix.lower() in {".csv", ".xlsx"}],
        key=lambda path: path.name,
    )
    if max_datasets and max_datasets > 0:
        return candidates[:max_datasets]
    return candidates


def create_batch_job(
    settings: Settings,
    *,
    goal: str = BATCH_GOAL,
    rule_path: Path | None = None,
    dataset_dir: Path | None = None,
    max_datasets: int | None = None,
) -> dict[str, Any]:
    """Create an empty batch manifest that can be filled by a background task."""
    selected_rule = rule_path or DEMO_RULE_PATH
    datasets = discover_batch_datasets(dataset_dir, max_datasets=max_datasets)
    batch_id = str(uuid.uuid4())
    batch_dir = _batch_dir(settings.storage_dir, batch_id)
    batch_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "batch_id": batch_id,
        "status": "running",
        "created_at": utc_now_iso(),
        "updated_at": utc_now_iso(),
        "goal": goal,
        "rule_path": str(selected_rule),
        "dataset_dir": str(dataset_dir or default_batch_dataset_dir()),
        "dataset_total": len(datasets),
        "dataset_paths": [str(path) for path in datasets],
        "completed": 0,
        "failed": 0,
        "passed": 0,
        "needs_review": 0,
        "items": [],
        "error": None,
        "report_markdown_path": str(batch_dir / "batch_report.md"),
        "report_html_path": str(batch_dir / "batch_report.html"),
    }
    _save_batch_manifest(settings.storage_dir, manifest)
    return manifest


def run_batch_job(settings: Settings, batch_id: str) -> dict[str, Any]:
    """Execute all datasets recorded in an existing batch manifest."""
    manifest = get_batch_manifest(settings.storage_dir, batch_id)
    try:
        rule_path = Path(str(manifest["rule_path"]))
        dataset_paths = [Path(str(path)) for path in manifest.get("dataset_paths", [])]
        local_settings, model_config = build_run_settings(settings, mode=MODEL_RUN_MODE_LOCAL)
        store = RunStore(settings.storage_dir)
        items = []
        for index, dataset_path in enumerate(dataset_paths, start=1):
            item = _run_one_dataset(
                store=store,
                run_settings=local_settings,
                model_config=model_config,
                goal=str(manifest.get("goal") or BATCH_GOAL),
                rule_path=rule_path,
                dataset_path=dataset_path,
            )
            item["index"] = index
            items.append(item)
            _update_manifest_progress(settings.storage_dir, manifest, items, status="running")
        _update_manifest_progress(settings.storage_dir, manifest, items, status="complete")
        return get_batch_manifest(settings.storage_dir, batch_id)
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = str(exc)
        manifest["updated_at"] = utc_now_iso()
        _save_batch_manifest(settings.storage_dir, manifest)
        _write_batch_report(settings.storage_dir, manifest)
        raise


def get_batch_manifest(storage_dir: Path, batch_id: str) -> dict[str, Any]:
    """Read one batch manifest by id."""
    path = _batch_manifest_path(storage_dir, batch_id)
    if not path.exists():
        raise FileNotFoundError(f"Batch not found: {batch_id}")
    return read_json(path)


def build_batch_rule_analysis(storage_dir: Path, batch_id: str) -> dict[str, Any]:
    """Build and persist per-rule stability analysis for one batch.

    新批次会在 batch item 中保存完整规则执行结果；旧批次则从每个 run 的
    run_events.jsonl 追溯 apply_rules 输出，因此历史批次也能补生成分析报告。
    """
    manifest = get_batch_manifest(storage_dir, batch_id)
    analysis = _collect_rule_analysis(storage_dir, manifest)
    batch_dir = _batch_dir(storage_dir, batch_id)
    json_path = batch_dir / RULE_ANALYSIS_JSON_NAME
    markdown_path = batch_dir / RULE_ANALYSIS_MD_NAME
    html_path = batch_dir / RULE_ANALYSIS_HTML_NAME
    markdown = _build_rule_analysis_markdown(analysis)
    write_json(json_path, analysis)
    markdown_path.write_text(markdown, encoding="utf-8")
    html_path.write_text(markdown_to_html(markdown), encoding="utf-8")
    manifest["rule_analysis_json_path"] = str(json_path)
    manifest["rule_analysis_markdown_path"] = str(markdown_path)
    manifest["rule_analysis_html_path"] = str(html_path)
    _save_batch_manifest(storage_dir, manifest)
    return analysis


def compare_batch_rule_versions(storage_dir: Path, base_batch_id: str, candidate_batch_id: str) -> dict[str, Any]:
    """Compare two completed batch runs that use different rule library versions.

    对比报告以“批次”为单位，而不是直接比较两份 YAML 文件。这样可以同时看到
    规则文本变化和跨数据集命中分布变化，适合做规则库迭代验收。
    """
    base_analysis = build_batch_rule_analysis(storage_dir, base_batch_id)
    candidate_analysis = build_batch_rule_analysis(storage_dir, candidate_batch_id)
    base_manifest = get_batch_manifest(storage_dir, base_batch_id)
    candidate_manifest = get_batch_manifest(storage_dir, candidate_batch_id)
    comparison = _collect_rule_version_comparison(base_manifest, candidate_manifest, base_analysis, candidate_analysis)
    candidate_dir = _batch_dir(storage_dir, candidate_batch_id)
    suffix = base_batch_id[:8]
    json_path = candidate_dir / f"{RULE_COMPARISON_JSON_PREFIX}_{suffix}.json"
    markdown_path = candidate_dir / f"{RULE_COMPARISON_MD_PREFIX}_{suffix}.md"
    html_path = candidate_dir / f"{RULE_COMPARISON_HTML_PREFIX}_{suffix}.html"
    markdown = _build_rule_version_comparison_markdown(comparison)
    write_json(json_path, comparison)
    markdown_path.write_text(markdown, encoding="utf-8")
    html_path.write_text(markdown_to_html(markdown), encoding="utf-8")
    candidate_manifest["latest_rule_comparison"] = {
        "base_batch_id": base_batch_id,
        "candidate_batch_id": candidate_batch_id,
        "json_path": str(json_path),
        "markdown_path": str(markdown_path),
        "html_path": str(html_path),
        "generated_at": comparison["generated_at"],
    }
    _save_batch_manifest(storage_dir, candidate_manifest)
    return comparison


def build_batch_hit_samples(
    storage_dir: Path,
    batch_id: str,
    *,
    rule_ids: list[str] | None = None,
    datasets_per_rule: int = 5,
    samples_per_dataset: int = 5,
) -> dict[str, Any]:
    """Export representative matched rows for expert rule review.

    批量稳定性报告回答“规则命中分布是否稳定”；本函数进一步回答“专家应该看
    哪些真实样例”。它优先从 run_events.jsonl 读取完整 apply_rules 输出，
    因为历史 batch manifest 为了轻量化没有保留 sample_indices。
    """
    manifest = get_batch_manifest(storage_dir, batch_id)
    selected_rule_ids = rule_ids or DEFAULT_HIT_SAMPLE_RULE_IDS
    samples = _collect_batch_hit_samples(
        storage_dir,
        manifest,
        rule_ids=selected_rule_ids,
        datasets_per_rule=datasets_per_rule,
        samples_per_dataset=samples_per_dataset,
    )
    batch_dir = _batch_dir(storage_dir, batch_id)
    json_path = batch_dir / KEY_HIT_SAMPLES_JSON_NAME
    markdown_path = batch_dir / KEY_HIT_SAMPLES_MD_NAME
    html_path = batch_dir / KEY_HIT_SAMPLES_HTML_NAME
    markdown = _build_hit_samples_markdown(samples)
    write_json(json_path, samples)
    markdown_path.write_text(markdown, encoding="utf-8")
    html_path.write_text(markdown_to_html(markdown), encoding="utf-8")
    manifest["key_hit_samples_json_path"] = str(json_path)
    manifest["key_hit_samples_markdown_path"] = str(markdown_path)
    manifest["key_hit_samples_html_path"] = str(html_path)
    manifest["key_hit_samples_generated_at"] = samples["generated_at"]
    _save_batch_manifest(storage_dir, manifest)
    return samples


def build_batch_dataset_adaptation(storage_dir: Path, batch_id: str) -> dict[str, Any]:
    """Aggregate per-run dataset adaptation reviews for one batch.

    单次 run 的 review_dataset_adaptation 回答“这一份数据能不能套用规则库”；
    批量层进一步回答“这套规则库在多份数据上是否普遍适配”。历史批次如果没有
    dataset_adaptation 产物，会被标记为缺少适配证据，而不是补造结论。
    """
    manifest = get_batch_manifest(storage_dir, batch_id)
    adaptation = _collect_batch_dataset_adaptation(storage_dir, manifest)
    batch_dir = _batch_dir(storage_dir, batch_id)
    json_path = batch_dir / BATCH_DATASET_ADAPTATION_JSON_NAME
    markdown_path = batch_dir / BATCH_DATASET_ADAPTATION_MD_NAME
    html_path = batch_dir / BATCH_DATASET_ADAPTATION_HTML_NAME
    markdown = _build_batch_dataset_adaptation_markdown(adaptation)
    write_json(json_path, adaptation)
    markdown_path.write_text(markdown, encoding="utf-8")
    html_path.write_text(markdown_to_html(markdown), encoding="utf-8")
    manifest["dataset_adaptation_json_path"] = str(json_path)
    manifest["dataset_adaptation_markdown_path"] = str(markdown_path)
    manifest["dataset_adaptation_html_path"] = str(html_path)
    manifest["dataset_adaptation_generated_at"] = adaptation["generated_at"]
    _save_batch_manifest(storage_dir, manifest)
    return adaptation


def build_batch_schema_mapping_review(storage_dir: Path, batch_id: str) -> dict[str, Any]:
    """Aggregate mapping-candidate review suggestions across one batch.

    单次 run 的 review_schema_mapping_candidates 只回答“这份数据里哪些候选
    映射值得人工看”；批量层进一步回答“哪些规则字段在多个数据结构下反复
    出现同类复核风险”。缺少产物的历史 run 标记为证据缺失，不补造结论。
    """
    manifest = get_batch_manifest(storage_dir, batch_id)
    review = _collect_batch_schema_mapping_review(storage_dir, manifest)
    batch_dir = _batch_dir(storage_dir, batch_id)
    json_path = batch_dir / BATCH_SCHEMA_MAPPING_REVIEW_JSON_NAME
    markdown_path = batch_dir / BATCH_SCHEMA_MAPPING_REVIEW_MD_NAME
    html_path = batch_dir / BATCH_SCHEMA_MAPPING_REVIEW_HTML_NAME
    markdown = _build_batch_schema_mapping_review_markdown(review)
    write_json(json_path, review)
    markdown_path.write_text(markdown, encoding="utf-8")
    html_path.write_text(markdown_to_html(markdown), encoding="utf-8")
    manifest["schema_mapping_review_json_path"] = str(json_path)
    manifest["schema_mapping_review_markdown_path"] = str(markdown_path)
    manifest["schema_mapping_review_html_path"] = str(html_path)
    manifest["schema_mapping_review_generated_at"] = review["generated_at"]
    _save_batch_manifest(storage_dir, manifest)
    return review


def build_batch_review_decisions(storage_dir: Path, batch_id: str) -> dict[str, Any]:
    """Aggregate run-local human review decisions inside one batch.

    该报告只回答“本批次内已经有哪些人工确认/驳回证据”。它不会把批内结论
    写入 ``schema_mapping_overrides.yaml``，也不会替代跨批次推广审核。
    """
    manifest = get_batch_manifest(storage_dir, batch_id)
    review = _collect_batch_review_decisions(storage_dir, manifest)
    batch_dir = _batch_dir(storage_dir, batch_id)
    json_path = batch_dir / BATCH_REVIEW_DECISIONS_JSON_NAME
    markdown_path = batch_dir / BATCH_REVIEW_DECISIONS_MD_NAME
    html_path = batch_dir / BATCH_REVIEW_DECISIONS_HTML_NAME
    markdown = _build_batch_review_decisions_markdown(review)
    write_json(json_path, review)
    markdown_path.write_text(markdown, encoding="utf-8")
    html_path.write_text(markdown_to_html(markdown), encoding="utf-8")
    manifest["review_decisions_json_path"] = str(json_path)
    manifest["review_decisions_markdown_path"] = str(markdown_path)
    manifest["review_decisions_html_path"] = str(html_path)
    manifest["review_decisions_generated_at"] = review["generated_at"]
    _save_batch_manifest(storage_dir, manifest)
    return review


def build_batch_generalization_readiness(storage_dir: Path, batch_id: str) -> dict[str, Any]:
    """Aggregate per-run generalization gates for one batch.

    单次 run 的 review_generalization_readiness 判断“这一份数据能否允许自动
    诊断解释”；批量层回答“这套流程在多个数据结构下是否稳定暴露泛化风险”。
    缺少门禁产物会被标记为证据缺失，不会补造通过结论。
    """
    manifest = get_batch_manifest(storage_dir, batch_id)
    readiness = _collect_batch_generalization_readiness(storage_dir, manifest)
    batch_dir = _batch_dir(storage_dir, batch_id)
    json_path = batch_dir / BATCH_GENERALIZATION_JSON_NAME
    markdown_path = batch_dir / BATCH_GENERALIZATION_MD_NAME
    html_path = batch_dir / BATCH_GENERALIZATION_HTML_NAME
    markdown = _build_batch_generalization_markdown(readiness)
    write_json(json_path, readiness)
    markdown_path.write_text(markdown, encoding="utf-8")
    html_path.write_text(markdown_to_html(markdown), encoding="utf-8")
    manifest["generalization_readiness_json_path"] = str(json_path)
    manifest["generalization_readiness_markdown_path"] = str(markdown_path)
    manifest["generalization_readiness_html_path"] = str(html_path)
    manifest["generalization_readiness_generated_at"] = readiness["generated_at"]
    _save_batch_manifest(storage_dir, manifest)
    return readiness


def build_batch_mapping_regression(storage_dir: Path, batch_id: str) -> dict[str, Any]:
    """Aggregate mapping regression validation across a batch."""
    manifest = get_batch_manifest(storage_dir, batch_id)
    regression = _collect_batch_mapping_regression(storage_dir, manifest)
    batch_dir = _batch_dir(storage_dir, batch_id)
    json_path = batch_dir / BATCH_MAPPING_REGRESSION_JSON_NAME
    markdown_path = batch_dir / BATCH_MAPPING_REGRESSION_MD_NAME
    html_path = batch_dir / BATCH_MAPPING_REGRESSION_HTML_NAME
    markdown = _build_batch_mapping_regression_markdown(regression)
    write_json(json_path, regression)
    markdown_path.write_text(markdown, encoding="utf-8")
    html_path.write_text(markdown_to_html(markdown), encoding="utf-8")
    manifest["mapping_regression_json_path"] = str(json_path)
    manifest["mapping_regression_markdown_path"] = str(markdown_path)
    manifest["mapping_regression_html_path"] = str(html_path)
    manifest["mapping_regression_generated_at"] = regression["generated_at"]
    _save_batch_manifest(storage_dir, manifest)
    return regression


def list_batch_manifests(storage_dir: Path, limit: int = 8) -> list[dict[str, Any]]:
    """Return newest batch manifests for the front-end overview."""
    root = storage_dir / BATCH_ROOT_NAME
    if not root.exists():
        return []
    manifests = []
    for path in sorted(root.glob(f"*/{BATCH_MANIFEST_NAME}"), key=lambda item: item.stat().st_mtime, reverse=True):
        try:
            manifests.append(read_json(path))
        except Exception:
            continue
        if len(manifests) >= limit:
            break
    return manifests


def _run_one_dataset(
    *,
    store: RunStore,
    run_settings: Settings,
    model_config: dict[str, Any],
    goal: str,
    rule_path: Path,
    dataset_path: Path,
) -> dict[str, Any]:
    context = store.create_run(goal)
    context.model_config = model_config
    context.status = "running"
    context.save_manifest()
    store.save_bytes(
        context,
        data=rule_path.read_bytes(),
        original_name=str(rule_path.relative_to(PROJECT_ROOT)) if rule_path.is_relative_to(PROJECT_ROOT) else str(rule_path),
        kind="rule_library",
        media_type="application/yaml" if rule_path.suffix.lower() in {".yaml", ".yml"} else None,
    )
    store.save_bytes(
        context,
        data=dataset_path.read_bytes(),
        original_name=str(dataset_path.relative_to(PROJECT_ROOT)) if dataset_path.is_relative_to(PROJECT_ROOT) else str(dataset_path),
        kind="dataset",
        media_type="text/csv" if dataset_path.suffix.lower() == ".csv" else None,
    )
    try:
        result = AgentRunner(context, ToolExecutor(context), run_settings).run()
        context.status = "complete"
        context.emit_event(
            "run_completed",
            title="运行已完成",
            message="批量验收子任务已完成。",
            payload={"report_html_path": str(context.report_html_path) if context.report_html_path else None},
        )
        context.save_manifest()
        return _collect_run_summary(context, dataset_path, result)
    except Exception as exc:
        context.status = "failed"
        context.error = str(exc)
        context.save_manifest()
        return {
            "dataset": dataset_path.name,
            "dataset_path": str(dataset_path),
            "run_id": context.run_id,
            "status": "failed",
            "validation_status": "failed",
            "error": str(exc),
            "report_url": f"/runs/{context.run_id}",
        }


def _collect_run_summary(context, dataset_path: Path, result: dict[str, Any]) -> dict[str, Any]:
    profile = next(iter(context.dataset_profiles.values()), {})
    validation = context.validation_results[-1] if context.validation_results else {}
    rule_execution = _latest_tool_result(context.tool_results, "apply_rules")
    association = _latest_tool_result(context.tool_results, "mine_association_rules")
    schema_review = _latest_tool_result(context.tool_results, "review_schema_metadata")
    dataset_adaptation = _latest_tool_result(context.tool_results, "review_dataset_adaptation")
    visualizations = _latest_tool_result(context.tool_results, "build_visualizations")
    rule_results = rule_execution.get("rule_results", []) if isinstance(rule_execution, dict) else []
    adaptation_rules = dataset_adaptation.get("rules", []) if isinstance(dataset_adaptation, dict) else []
    evaluated = [item for item in rule_results if item.get("status") == "evaluated"]
    matched = [item for item in evaluated if int(item.get("matched_rows") or 0) > 0]
    top_rules = sorted(matched, key=lambda item: (item.get("matched_rows", 0), item.get("match_rate", 0)), reverse=True)[:5]
    return {
        "dataset": dataset_path.name,
        "dataset_path": str(dataset_path),
        "run_id": context.run_id,
        "status": context.status,
        "validation_status": validation.get("status", "not_run"),
        "validation_summary": validation.get("summary", ""),
        "warnings": validation.get("warnings", []),
        "rows": profile.get("rows", 0),
        "columns": profile.get("columns", 0),
        "rules_total": len(rule_results) or len(context.rules),
        "rules_evaluated": len(evaluated),
        "rules_matched": len(matched),
        "association_rules": len(association.get("rules", [])) if isinstance(association, dict) else 0,
        "association_engine": association.get("engine", "未运行") if isinstance(association, dict) else "未运行",
        "schema_review_items": len(schema_review.get("review_items", [])) if isinstance(schema_review, dict) else 0,
        "low_confidence_fields": schema_review.get("low_confidence_count", 0) if isinstance(schema_review, dict) else 0,
        "dataset_adaptation": _compact_dataset_adaptation(dataset_adaptation),
        "dataset_adaptation_rules": [_compact_adaptation_rule(rule) for rule in adaptation_rules],
        "schema_mapping_review_url": f"/runs/{context.run_id}/schema-mapping-review",
        "chart_count": visualizations.get("chart_count", 0) if isinstance(visualizations, dict) else 0,
        "evidence_nodes": len(context.evidence.nodes),
        "report_url": f"/runs/{context.run_id}",
        "visualizations_url": f"/runs/{context.run_id}/visualizations",
        "dataset_adaptation_url": f"/runs/{context.run_id}/dataset-adaptation",
        "mode": result.get("mode", "offline"),
        "top_rules": [
            {
                "rule_id": item.get("rule_id"),
                "name": item.get("name"),
                "matched_rows": item.get("matched_rows", 0),
                "match_rate": item.get("match_rate", 0),
            }
            for item in top_rules
        ],
        "rule_results": [
            {
                "rule_id": item.get("rule_id"),
                "name": item.get("name"),
                "status": item.get("status"),
                "severity": item.get("severity"),
                "matched_rows": item.get("matched_rows", 0),
                "match_rate": item.get("match_rate", 0),
                "message": item.get("message", ""),
            }
            for item in rule_results
        ],
    }


def _update_manifest_progress(storage_dir: Path, manifest: dict[str, Any], items: list[dict[str, Any]], *, status: str) -> None:
    manifest["status"] = status
    manifest["updated_at"] = utc_now_iso()
    manifest["items"] = items
    manifest["completed"] = len([item for item in items if item.get("status") == "complete"])
    manifest["failed"] = len([item for item in items if item.get("status") == "failed"])
    manifest["passed"] = len([item for item in items if item.get("validation_status") == "passed"])
    manifest["needs_review"] = len([item for item in items if item.get("validation_status") == "needs_review"])
    _save_batch_manifest(storage_dir, manifest)
    _write_batch_report(storage_dir, manifest)
    if items:
        build_batch_rule_analysis(storage_dir, str(manifest["batch_id"]))
        build_batch_dataset_adaptation(storage_dir, str(manifest["batch_id"]))
        build_batch_schema_mapping_review(storage_dir, str(manifest["batch_id"]))
        build_batch_review_decisions(storage_dir, str(manifest["batch_id"]))
        build_batch_mapping_regression(storage_dir, str(manifest["batch_id"]))
        build_batch_generalization_readiness(storage_dir, str(manifest["batch_id"]))


def _write_batch_report(storage_dir: Path, manifest: dict[str, Any]) -> None:
    batch_dir = _batch_dir(storage_dir, str(manifest["batch_id"]))
    markdown_path = batch_dir / "batch_report.md"
    html_path = batch_dir / "batch_report.html"
    markdown = _build_batch_markdown(manifest)
    markdown_path.write_text(markdown, encoding="utf-8")
    html_path.write_text(markdown_to_html(markdown), encoding="utf-8")
    manifest["report_markdown_path"] = str(markdown_path)
    manifest["report_html_path"] = str(html_path)
    _save_batch_manifest(storage_dir, manifest)


def _build_batch_markdown(manifest: dict[str, Any]) -> str:
    items = manifest.get("items") or []
    completed = len(items)
    passed = len([item for item in items if item.get("validation_status") == "passed"])
    needs_review = len([item for item in items if item.get("validation_status") == "needs_review"])
    failed = len([item for item in items if item.get("status") == "failed"])
    adaptation_summary = _batch_adaptation_summary_from_items(items)
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


def _latest_tool_result(tool_results: list[dict[str, Any]], tool_name: str) -> Any:
    for item in reversed(tool_results):
        if item.get("tool") == tool_name:
            return item.get("result")
    return None


def _compact_dataset_adaptation(adaptation: Any) -> dict[str, Any]:
    if not isinstance(adaptation, dict):
        return {}
    mapping_candidates = adaptation.get("mapping_candidates") or []
    return {
        "dataset_kind_hint": adaptation.get("dataset_kind_hint", "unknown"),
        "field_count": adaptation.get("field_count", 0),
        "rule_count": adaptation.get("rule_count", 0),
        "required_field_count": adaptation.get("required_field_count", 0),
        "mapped_field_count": adaptation.get("mapped_field_count", 0),
        "ready_count": adaptation.get("ready_count", 0),
        "confirmed_mapping_count": adaptation.get("confirmed_mapping_count", 0),
        "confirmed_field_mapping_count": adaptation.get("confirmed_field_mapping_count", 0),
        "needs_mapping_count": adaptation.get("needs_mapping_count", 0),
        "needs_review_count": adaptation.get("needs_review_count", 0),
        "not_applicable_count": adaptation.get("not_applicable_count", 0),
        "disabled_count": adaptation.get("disabled_count", 0),
        "mapping_candidates_count": len(mapping_candidates),
        "summary": adaptation.get("summary", ""),
    }


def _compact_adaptation_rule(rule: dict[str, Any]) -> dict[str, Any]:
    return {
        "rule_id": rule.get("rule_id"),
        "name": rule.get("name"),
        "status": rule.get("status"),
        "required_fields": rule.get("required_fields") or [],
        "field_mappings": [
            {
                "rule_field": mapping.get("rule_field"),
                "status": mapping.get("status"),
                "candidate_field": mapping.get("candidate_field"),
                "score": mapping.get("score"),
                "reason": mapping.get("reason"),
                "match_method": mapping.get("match_method"),
            }
            for mapping in rule.get("field_mappings") or []
        ],
        "issues": rule.get("issues") or [],
        "recommendation": rule.get("recommendation", ""),
    }


def _batch_adaptation_summary_from_items(items: list[dict[str, Any]]) -> dict[str, int]:
    summary = {
        "datasets_with_adaptation": 0,
        "all_ready_datasets": 0,
        "confirmed_mapping_datasets": 0,
        "needs_mapping_datasets": 0,
        "not_applicable_datasets": 0,
        "ready_rules": 0,
        "confirmed_mapping_rules": 0,
        "needs_mapping_rules": 0,
        "not_applicable_rules": 0,
        "review_or_disabled_rules": 0,
    }
    for item in items:
        adaptation = item.get("dataset_adaptation") or {}
        if not adaptation:
            continue
        summary["datasets_with_adaptation"] += 1
        ready = int(adaptation.get("ready_count") or 0)
        confirmed_mapping = int(adaptation.get("confirmed_mapping_count") or 0)
        needs_mapping = int(adaptation.get("needs_mapping_count") or 0)
        not_applicable = int(adaptation.get("not_applicable_count") or 0)
        review_or_disabled = int(adaptation.get("needs_review_count") or 0) + int(adaptation.get("disabled_count") or 0)
        summary["ready_rules"] += ready
        summary["confirmed_mapping_rules"] += confirmed_mapping
        summary["needs_mapping_rules"] += needs_mapping
        summary["not_applicable_rules"] += not_applicable
        summary["review_or_disabled_rules"] += review_or_disabled
        if confirmed_mapping == 0 and needs_mapping == 0 and not_applicable == 0 and review_or_disabled == 0 and int(adaptation.get("rule_count") or 0) > 0:
            summary["all_ready_datasets"] += 1
        if confirmed_mapping:
            summary["confirmed_mapping_datasets"] += 1
        if needs_mapping:
            summary["needs_mapping_datasets"] += 1
        if not_applicable:
            summary["not_applicable_datasets"] += 1
    return summary


def _collect_batch_dataset_adaptation(storage_dir: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    items = manifest.get("items") or []
    datasets = []
    rule_map: dict[str, dict[str, Any]] = {}
    missing_adaptation = 0
    for item in items:
        run_id = str(item.get("run_id") or "")
        adaptation = _load_dataset_adaptation(storage_dir, run_id)
        if not adaptation and item.get("dataset_adaptation"):
            adaptation = {
                **(item.get("dataset_adaptation") or {}),
                "rules": item.get("dataset_adaptation_rules") or [],
            }
        if not adaptation:
            missing_adaptation += 1
            datasets.append(
                {
                    "dataset": item.get("dataset"),
                    "run_id": run_id,
                    "report_url": item.get("report_url") or f"/runs/{run_id}",
                    "status": "missing_adaptation",
                    "summary": "未找到数据适配审查产物。",
                    "rule_count": 0,
                    "ready_count": 0,
                    "confirmed_mapping_count": 0,
                    "needs_mapping_count": 0,
                    "not_applicable_count": 0,
                    "needs_review_count": 0,
                    "disabled_count": 0,
                }
            )
            continue

        compact = _compact_dataset_adaptation(adaptation)
        blocking_count = (
            int(compact.get("confirmed_mapping_count") or 0)
            + int(compact.get("needs_mapping_count") or 0)
            + int(compact.get("not_applicable_count") or 0)
            + int(compact.get("needs_review_count") or 0)
            + int(compact.get("disabled_count") or 0)
        )
        status = "ready" if blocking_count == 0 and int(compact.get("rule_count") or 0) > 0 else "needs_review"
        datasets.append(
            {
                "dataset": item.get("dataset") or adaptation.get("source_file"),
                "run_id": run_id,
                "report_url": item.get("report_url") or f"/runs/{run_id}",
                "dataset_adaptation_url": item.get("dataset_adaptation_url") or f"/runs/{run_id}/dataset-adaptation",
                "status": status,
                **compact,
            }
        )
        for rule in adaptation.get("rules") or []:
            rule_id = str(rule.get("rule_id") or "UNKNOWN")
            record = rule_map.setdefault(
                rule_id,
                {
                    "rule_id": rule_id,
                    "name": rule.get("name") or rule_id,
                    "dataset_total": len(items),
                    "ready_datasets": 0,
                    "confirmed_mapping_datasets": 0,
                    "needs_mapping_datasets": 0,
                    "not_applicable_datasets": 0,
                    "needs_review_datasets": 0,
                    "disabled_datasets": 0,
                    "datasets": [],
                    "mapping_issues": [],
                },
            )
            status = str(rule.get("status") or "unknown")
            if status == "ready":
                record["ready_datasets"] += 1
            elif status == "mapping_confirmed":
                record["confirmed_mapping_datasets"] += 1
            elif status == "needs_mapping_confirmation":
                record["needs_mapping_datasets"] += 1
            elif status == "not_applicable":
                record["not_applicable_datasets"] += 1
            elif status == "disabled":
                record["disabled_datasets"] += 1
            else:
                record["needs_review_datasets"] += 1
            compact_rule = _compact_adaptation_rule(rule)
            record["datasets"].append(
                {
                    "dataset": item.get("dataset"),
                    "run_id": run_id,
                    "status": status,
                    "required_fields": compact_rule.get("required_fields") or [],
                    "field_mappings": compact_rule.get("field_mappings") or [],
                }
            )
            for mapping in compact_rule.get("field_mappings") or []:
                if mapping.get("status") == "exact":
                    continue
                record["mapping_issues"].append(
                    {
                        "dataset": item.get("dataset"),
                        "run_id": run_id,
                        **mapping,
                    }
                )

    rules = []
    for record in rule_map.values():
        label, recommendation = _adaptation_rule_quality(record, missing_adaptation)
        record["quality_label"] = label
        record["recommendation"] = recommendation
        rules.append(record)
    rules.sort(key=lambda row: (_adaptation_quality_rank(str(row.get("quality_label"))), row.get("rule_id", "")))
    summary = _batch_adaptation_summary_from_items(
        [
            {"dataset_adaptation": row}
            for row in datasets
            if row.get("status") != "missing_adaptation"
        ]
    )
    summary.update(
        {
            "dataset_total": len(items),
            "dataset_count": len(datasets),
            "missing_adaptation_count": missing_adaptation,
            "rule_count": len(rules),
            "quality_counts": _count_labels(rules),
        }
    )
    return {
        "batch_id": manifest.get("batch_id"),
        "status": manifest.get("status"),
        "rule_path": manifest.get("rule_path"),
        "dataset_total": manifest.get("dataset_total"),
        "generated_at": utc_now_iso(),
        "summary": summary,
        "datasets": datasets,
        "rules": rules,
    }


def _load_dataset_adaptation(storage_dir: Path, run_id: str) -> dict[str, Any]:
    if not run_id:
        return {}
    artifact_path = storage_dir / run_id / "artifacts" / "dataset_adaptation.json"
    if artifact_path.exists():
        try:
            return read_json(artifact_path)
        except Exception:
            return {}
    return _load_dataset_adaptation_from_run_events(storage_dir, run_id)


def _load_dataset_adaptation_from_run_events(storage_dir: Path, run_id: str) -> dict[str, Any]:
    path = storage_dir / run_id / "run_events.jsonl"
    if not path.exists():
        return {}
    latest: dict[str, Any] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        payload = event.get("payload") or {}
        if event.get("type") == "tool_finished" and payload.get("tool") == "review_dataset_adaptation":
            result = payload.get("result") or {}
            if isinstance(result, dict):
                latest = result
    return latest


def _collect_batch_generalization_readiness(storage_dir: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    items = manifest.get("items") or []
    datasets = []
    check_map: dict[str, dict[str, Any]] = {}
    metric_totals = {
        "semantic_coverage_values": [],
        "unresolved_rule_count": 0,
        "unparseable_rule_count": 0,
        "not_applicable_count": 0,
        "needs_mapping_count": 0,
        "confirmed_mapping_count": 0,
        "evaluated_rule_count": 0,
    }
    status_counts: dict[str, int] = {}
    missing_gate_count = 0
    auto_allowed_count = 0

    for item in items:
        run_id = str(item.get("run_id") or "")
        gate = _load_generalization_readiness(storage_dir, run_id)
        if not gate:
            missing_gate_count += 1
            status_counts["missing_gate"] = status_counts.get("missing_gate", 0) + 1
            datasets.append(
                {
                    "dataset": item.get("dataset"),
                    "run_id": run_id,
                    "report_url": item.get("report_url") or f"/runs/{run_id}",
                    "generalization_url": f"/runs/{run_id}/generalization-readiness",
                    "status": "missing_gate",
                    "status_label": "缺少门禁证据",
                    "auto_diagnosis_allowed": False,
                    "verdict": "未找到泛化验收门禁产物，不能作为通过证据。",
                    "metrics": {},
                    "blocking_check_count": 0,
                    "recommendations": ["重新运行该数据集，补齐泛化验收门禁产物。"],
                }
            )
            continue

        status = str(gate.get("status") or "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
        metrics = gate.get("metrics") if isinstance(gate.get("metrics"), dict) else {}
        semantic_coverage = float(metrics.get("semantic_coverage") or 0)
        metric_totals["semantic_coverage_values"].append(semantic_coverage)
        for field in [
            "unresolved_rule_count",
            "unparseable_rule_count",
            "not_applicable_count",
            "needs_mapping_count",
            "confirmed_mapping_count",
            "evaluated_rule_count",
        ]:
            metric_totals[field] += int(metrics.get(field) or 0)
        if gate.get("auto_diagnosis_allowed"):
            auto_allowed_count += 1

        blocking_checks = 0
        for check in gate.get("checks") or []:
            if not isinstance(check, dict):
                continue
            check_id = str(check.get("id") or check.get("name") or "unknown")
            record = check_map.setdefault(
                check_id,
                {
                    "id": check_id,
                    "name": str(check.get("name") or check_id),
                    "passed_count": 0,
                    "needs_review_count": 0,
                    "failed_count": 0,
                    "datasets": [],
                    "messages": [],
                },
            )
            check_status = str(check.get("status") or "unknown")
            if check_status == "passed":
                record["passed_count"] += 1
            elif check_status == "failed":
                record["failed_count"] += 1
                blocking_checks += 1
            else:
                record["needs_review_count"] += 1
                blocking_checks += 1
            message = str(check.get("message") or "").strip()
            if message and message not in record["messages"]:
                record["messages"].append(message)
            record["datasets"].append(
                {
                    "dataset": item.get("dataset"),
                    "run_id": run_id,
                    "status": check_status,
                    "evidence": check.get("evidence") or "",
                }
            )

        datasets.append(
            {
                "dataset": item.get("dataset"),
                "run_id": run_id,
                "report_url": item.get("report_url") or f"/runs/{run_id}",
                "generalization_url": f"/runs/{run_id}/generalization-readiness",
                "status": status,
                "status_label": _generalization_status_label(status),
                "auto_diagnosis_allowed": bool(gate.get("auto_diagnosis_allowed")),
                "verdict": gate.get("verdict") or "",
                "metrics": metrics,
                "blocking_check_count": blocking_checks,
                "recommendations": gate.get("recommendations") or [],
            }
        )

    checks = []
    for record in check_map.values():
        record["quality_label"], record["recommendation"] = _batch_gate_check_quality(record)
        record["messages"] = record["messages"][:5]
        checks.append(record)
    checks.sort(key=lambda row: (_gate_check_quality_rank(str(row.get("quality_label"))), row.get("id", "")))

    semantic_values = metric_totals.pop("semantic_coverage_values")
    datasets_with_gate = len(items) - missing_gate_count
    summary = {
        "dataset_total": len(items),
        "dataset_count": len(datasets),
        "datasets_with_gate": datasets_with_gate,
        "missing_gate_count": missing_gate_count,
        "passed_count": status_counts.get("passed", 0),
        "needs_review_count": status_counts.get("needs_review", 0),
        "failed_count": status_counts.get("failed", 0),
        "unknown_count": status_counts.get("unknown", 0),
        "auto_diagnosis_allowed_count": auto_allowed_count,
        "avg_semantic_coverage": round(mean(semantic_values), 6) if semantic_values else 0,
        "min_semantic_coverage": round(min(semantic_values), 6) if semantic_values else 0,
        "max_semantic_coverage": round(max(semantic_values), 6) if semantic_values else 0,
        "status_counts": status_counts,
        "check_count": len(checks),
        **metric_totals,
    }
    batch_status, verdict = _batch_generalization_verdict(summary)
    summary["batch_gate_status"] = batch_status
    summary["batch_gate_status_label"] = _batch_generalization_status_label(batch_status)
    summary["verdict"] = verdict

    return {
        "batch_id": manifest.get("batch_id"),
        "status": manifest.get("status"),
        "rule_path": manifest.get("rule_path"),
        "dataset_total": manifest.get("dataset_total"),
        "generated_at": utc_now_iso(),
        "anti_overfit_principle": "批量泛化门禁只汇总证据，不为了让某个数据集通过而改字段、改阈值或隐藏不适用规则。",
        "summary": summary,
        "datasets": datasets,
        "checks": checks,
    }


def _collect_batch_mapping_regression(storage_dir: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    items = manifest.get("items") or []
    datasets = []
    rule_map: dict[str, dict[str, Any]] = {}
    missing_validation_count = 0
    status_counts: dict[str, int] = {}

    for item in items:
        run_id = str(item.get("run_id") or "")
        validation = _load_mapping_regression_validation(storage_dir, run_id)
        if not validation:
            missing_validation_count += 1
            status_counts["missing_validation"] = status_counts.get("missing_validation", 0) + 1
            datasets.append(
                {
                    "dataset": item.get("dataset"),
                    "run_id": run_id,
                    "report_url": item.get("report_url") or f"/runs/{run_id}",
                    "mapping_regression_url": f"/runs/{run_id}/mapping-regression",
                    "status": "missing_validation",
                    "status_label": "缺少验证证据",
                    "mapped_execution_validated": False,
                    "rule_count": 0,
                    "regression_passed_count": 0,
                    "needs_confirmation_count": 0,
                    "needs_review_count": 0,
                    "blocked_count": 0,
                    "failed_count": 0,
                }
            )
            continue

        status = str(validation.get("status") or "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
        datasets.append(
            {
                "dataset": item.get("dataset"),
                "run_id": run_id,
                "report_url": item.get("report_url") or f"/runs/{run_id}",
                "mapping_regression_url": f"/runs/{run_id}/mapping-regression",
                "status": status,
                "status_label": _mapping_regression_status_label(status),
                "mapped_execution_validated": bool(validation.get("mapped_execution_validated")),
                "rule_count": validation.get("rule_count", 0),
                "regression_passed_count": validation.get("regression_passed_count", 0),
                "no_mapping_count": validation.get("no_mapping_count", 0),
                "needs_confirmation_count": validation.get("needs_confirmation_count", 0),
                "needs_review_count": validation.get("needs_review_count", 0),
                "blocked_count": validation.get("blocked_count", 0),
                "failed_count": validation.get("failed_count", 0),
            }
        )

        for rule in validation.get("rules") or []:
            rule_id = str(rule.get("rule_id") or "UNKNOWN")
            record = rule_map.setdefault(
                rule_id,
                {
                    "rule_id": rule_id,
                    "name": rule.get("name") or rule_id,
                    "dataset_total": len(items),
                    "passed_datasets": 0,
                    "regression_passed_datasets": 0,
                    "needs_confirmation_datasets": 0,
                    "needs_review_datasets": 0,
                    "blocked_or_failed_datasets": 0,
                    "datasets": [],
                    "issues": [],
                },
            )
            rule_status = str(rule.get("status") or "unknown")
            if rule_status in {"passed_no_mapping", "regression_passed"}:
                record["passed_datasets"] += 1
            if rule_status == "regression_passed":
                record["regression_passed_datasets"] += 1
            elif rule_status == "needs_confirmation":
                record["needs_confirmation_datasets"] += 1
            elif rule_status == "needs_review":
                record["needs_review_datasets"] += 1
            elif rule_status in {"failed", "blocked"}:
                record["blocked_or_failed_datasets"] += 1
            record["datasets"].append(
                {
                    "dataset": item.get("dataset"),
                    "run_id": run_id,
                    "status": rule_status,
                    "matched_rows": rule.get("matched_rows", 0),
                    "match_rate": rule.get("match_rate", 0),
                    "rewrite_pairs": rule.get("rewrite_pairs") or [],
                }
            )
            for issue in rule.get("issues") or []:
                record["issues"].append({"dataset": item.get("dataset"), "run_id": run_id, "issue": issue})

    rules = []
    for record in rule_map.values():
        record["quality_label"], record["recommendation"] = _mapping_regression_rule_quality(record, missing_validation_count)
        rules.append(record)
    rules.sort(key=lambda row: (_mapping_regression_quality_rank(str(row.get("quality_label"))), row.get("rule_id", "")))

    datasets_with_validation = len(items) - missing_validation_count
    summary = {
        "dataset_total": len(items),
        "dataset_count": len(datasets),
        "datasets_with_validation": datasets_with_validation,
        "missing_validation_count": missing_validation_count,
        "passed_count": status_counts.get("passed", 0),
        "needs_review_count": status_counts.get("needs_review", 0),
        "failed_count": status_counts.get("failed", 0),
        "status_counts": status_counts,
        "rule_count": len(rules),
        "validated_mapping_rule_count": sum(1 for rule in rules if int(rule.get("regression_passed_datasets") or 0) > 0),
        "needs_confirmation_rule_count": sum(1 for rule in rules if int(rule.get("needs_confirmation_datasets") or 0) > 0),
        "needs_review_rule_count": sum(1 for rule in rules if int(rule.get("needs_review_datasets") or 0) > 0),
        "blocked_or_failed_rule_count": sum(1 for rule in rules if int(rule.get("blocked_or_failed_datasets") or 0) > 0),
        "quality_counts": _count_labels(rules),
    }
    batch_status, verdict = _batch_mapping_regression_verdict(summary)
    summary["batch_regression_status"] = batch_status
    summary["batch_regression_status_label"] = _batch_mapping_regression_status_label(batch_status)
    summary["verdict"] = verdict

    return {
        "batch_id": manifest.get("batch_id"),
        "status": manifest.get("status"),
        "rule_path": manifest.get("rule_path"),
        "dataset_total": manifest.get("dataset_total"),
        "generated_at": utc_now_iso(),
        "anti_overfit_principle": "批量映射回归只汇总已确认映射的验证证据；不为了提高通过率而新增别名、改规则或改阈值。",
        "summary": summary,
        "datasets": datasets,
        "rules": rules,
    }


def _collect_batch_schema_mapping_review(storage_dir: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    items = manifest.get("items") or []
    datasets = []
    mapping_map: dict[str, dict[str, Any]] = {}
    missing_review_count = 0
    status_counts: dict[str, int] = {}
    totals = {
        "confirm_suggestion_count": 0,
        "manual_review_count": 0,
        "not_applicable_count": 0,
        "already_safe_count": 0,
    }

    for item in items:
        run_id = str(item.get("run_id") or "")
        review = _load_schema_mapping_review(storage_dir, run_id)
        if not review:
            missing_review_count += 1
            status_counts["missing_review"] = status_counts.get("missing_review", 0) + 1
            datasets.append(
                {
                    "dataset": item.get("dataset"),
                    "run_id": run_id,
                    "report_url": item.get("report_url") or f"/runs/{run_id}",
                    "schema_mapping_review_url": f"/runs/{run_id}/schema-mapping-review",
                    "status": "missing_review",
                    "status_label": _schema_mapping_review_status_label("missing_review"),
                    "confirm_suggestion_count": 0,
                    "manual_review_count": 0,
                    "not_applicable_count": 0,
                    "already_safe_count": 0,
                }
            )
            continue

        status = str(review.get("status") or "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
        for field in totals:
            totals[field] += int(review.get(field) or 0)
        datasets.append(
            {
                "dataset": item.get("dataset"),
                "run_id": run_id,
                "report_url": item.get("report_url") or f"/runs/{run_id}",
                "schema_mapping_review_url": f"/runs/{run_id}/schema-mapping-review",
                "status": status,
                "status_label": _schema_mapping_review_status_label(status),
                "confirm_suggestion_count": review.get("confirm_suggestion_count", 0),
                "manual_review_count": review.get("manual_review_count", 0),
                "not_applicable_count": review.get("not_applicable_count", 0),
                "already_safe_count": review.get("already_safe_count", 0),
            }
        )

        for review_item in review.get("review_items") or []:
            if not isinstance(review_item, dict):
                continue
            rule_field = str(review_item.get("rule_field") or "")
            dataset_field = str(review_item.get("dataset_field") or "")
            key = f"{rule_field}::{dataset_field or 'NONE'}"
            record = mapping_map.setdefault(
                key,
                {
                    "key": key,
                    "rule_id": review_item.get("rule_id") or "",
                    "rule_name": review_item.get("rule_name") or "",
                    "rule_field": rule_field,
                    "dataset_field": dataset_field or None,
                    "dataset_total": len(items),
                    "suggest_confirm_datasets": 0,
                    "manual_review_datasets": 0,
                    "not_applicable_datasets": 0,
                    "already_safe_datasets": 0,
                    "datasets": [],
                    "risk_flags": [],
                    "match_methods": [],
                },
            )
            item_status = str(review_item.get("status") or "unknown")
            if item_status == "suggest_confirm_candidate":
                record["suggest_confirm_datasets"] += 1
            elif item_status == "manual_review_required":
                record["manual_review_datasets"] += 1
            elif item_status == "not_applicable":
                record["not_applicable_datasets"] += 1
            elif item_status == "already_safe":
                record["already_safe_datasets"] += 1
            method = str(review_item.get("match_method") or "")
            if method and method not in record["match_methods"]:
                record["match_methods"].append(method)
            for risk in review_item.get("risk_flags") or []:
                risk_text = str(risk)
                if risk_text and risk_text not in record["risk_flags"]:
                    record["risk_flags"].append(risk_text)
            record["datasets"].append(
                {
                    "dataset": item.get("dataset"),
                    "run_id": run_id,
                    "status": item_status,
                    "status_label": _schema_mapping_review_status_label(item_status),
                    "score": review_item.get("score", 0),
                    "risk_flags": review_item.get("risk_flags") or [],
                    "recommendation": review_item.get("recommendation") or "",
                }
            )

    mappings = []
    for record in mapping_map.values():
        record["quality_label"], record["recommendation"] = _schema_mapping_review_quality(record, missing_review_count)
        record["risk_flags"] = record["risk_flags"][:8]
        record["match_methods"] = record["match_methods"][:6]
        mappings.append(record)
    mappings.sort(key=lambda row: (_schema_mapping_review_quality_rank(str(row.get("quality_label"))), str(row.get("rule_field")), str(row.get("dataset_field") or "")))

    datasets_with_review = len(items) - missing_review_count
    summary = {
        "dataset_total": len(items),
        "dataset_count": len(datasets),
        "datasets_with_review": datasets_with_review,
        "missing_review_count": missing_review_count,
        "status_counts": status_counts,
        "mapping_candidate_count": len(mappings),
        "confirm_suggestion_count": totals["confirm_suggestion_count"],
        "manual_review_count": totals["manual_review_count"],
        "not_applicable_count": totals["not_applicable_count"],
        "already_safe_count": totals["already_safe_count"],
        "quality_counts": _count_labels(mappings),
    }
    batch_status, verdict = _batch_schema_mapping_review_verdict(summary)
    summary["batch_review_status"] = batch_status
    summary["batch_review_status_label"] = _batch_schema_mapping_review_status_label(batch_status)
    summary["verdict"] = verdict

    return {
        "batch_id": manifest.get("batch_id"),
        "status": manifest.get("status"),
        "rule_path": manifest.get("rule_path"),
        "dataset_total": manifest.get("dataset_total"),
        "generated_at": utc_now_iso(),
        "anti_overfit_principle": "批量映射复核辅助只汇总人工复核队列；不自动确认映射、不写全局配置、不把单数据集候选当作泛化结论。",
        "summary": summary,
        "datasets": datasets,
        "mappings": mappings,
    }


def _collect_batch_review_decisions(storage_dir: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    items = manifest.get("items") or []
    datasets = []
    mapping_map: dict[str, dict[str, Any]] = {}
    no_decision_count = 0
    missing_file_count = 0
    decision_event_count = 0
    effective_decision_count = 0

    for item in items:
        run_id = str(item.get("run_id") or "")
        run_dir = storage_dir / run_id
        run_manifest = _load_run_manifest(storage_dir, run_id)
        decisions = load_review_decisions(run_dir)
        decision_file_exists = (run_dir / REVIEW_DECISIONS_NAME).exists()
        events = _batch_rule_mapping_decision_events(decisions)
        latest = _latest_batch_mapping_decisions(events)
        regression_pairs = _batch_regression_passed_pairs(_load_mapping_regression_validation(storage_dir, run_id))
        dataset_signature = _batch_dataset_signature(run_manifest, item, run_id)
        dataset_name = item.get("dataset") or _batch_dataset_names(run_manifest) or run_id
        confirm_count = sum(1 for decision in latest.values() if decision.get("action") == "confirm")
        reject_count = sum(1 for decision in latest.values() if decision.get("action") == "reject")
        if not decision_file_exists:
            missing_file_count += 1
        if not latest:
            no_decision_count += 1
        decision_event_count += len(events)
        effective_decision_count += len(latest)
        datasets.append(
            {
                "dataset": dataset_name,
                "run_id": run_id,
                "report_url": item.get("report_url") or f"/runs/{run_id}",
                "schema_mapping_review_url": f"/runs/{run_id}/schema-mapping-review",
                "review_decisions_url": f"/runs/{run_id}/artifacts/review_decisions.json",
                "status": "has_decisions" if latest else "no_decisions",
                "status_label": "已有人工意见" if latest else "暂无人工意见",
                "decision_file_exists": decision_file_exists,
                "decision_event_count": len(events),
                "effective_decision_count": len(latest),
                "confirm_count": confirm_count,
                "reject_count": reject_count,
                "dataset_signature": dataset_signature,
            }
        )

        for decision in latest.values():
            key = _batch_mapping_decision_key(decision)
            if not key:
                continue
            record = mapping_map.setdefault(key, _empty_batch_decision_mapping(decision, len(items)))
            _merge_batch_decision_mapping(record, decision, item, dataset_signature, regression_pairs, run_id, dataset_name)

    mappings = []
    for record in mapping_map.values():
        record["confirmed_distinct_dataset_count"] = len(record.pop("_confirmed_dataset_signatures", set()))
        record["rejected_distinct_dataset_count"] = len(record.pop("_rejected_dataset_signatures", set()))
        record["regression_passed_distinct_dataset_count"] = len(record.pop("_regression_dataset_signatures", set()))
        record["notes"] = record["notes"][:8]
        record["datasets"] = record["datasets"][:12]
        record["status"], record["status_label"], record["recommendation"] = _batch_review_decision_mapping_status(record)
        mappings.append(record)
    mappings.sort(key=lambda row: (_batch_review_decision_rank(str(row.get("status"))), str(row.get("rule_field")), str(row.get("dataset_field"))))

    status_counts: dict[str, int] = {}
    for mapping in mappings:
        status = str(mapping.get("status") or "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
    summary = {
        "dataset_total": len(items),
        "dataset_count": len(datasets),
        "datasets_with_decisions": len(items) - no_decision_count,
        "no_decision_count": no_decision_count,
        "missing_decision_file_count": missing_file_count,
        "decision_event_count": decision_event_count,
        "effective_mapping_decision_count": effective_decision_count,
        "mapping_count": len(mappings),
        "confirm_count": sum(int(mapping.get("confirm_count") or 0) for mapping in mappings),
        "reject_count": sum(int(mapping.get("reject_count") or 0) for mapping in mappings),
        "ready_for_promotion_review_count": status_counts.get("ready_for_promotion_review", 0),
        "conflict_review_count": status_counts.get("conflict_review", 0),
        "needs_regression_evidence_count": status_counts.get("needs_regression_evidence", 0),
        "needs_more_batch_evidence_count": status_counts.get("needs_more_batch_evidence", 0),
        "rejected_only_count": status_counts.get("rejected_only", 0),
        "status_counts": status_counts,
    }
    batch_status, verdict = _batch_review_decisions_verdict(summary)
    summary["batch_review_decision_status"] = batch_status
    summary["batch_review_decision_status_label"] = _batch_review_decisions_status_label(batch_status)
    summary["verdict"] = verdict

    return {
        "batch_id": manifest.get("batch_id"),
        "status": manifest.get("status"),
        "rule_path": manifest.get("rule_path"),
        "dataset_total": manifest.get("dataset_total"),
        "generated_at": utc_now_iso(),
        "anti_overfit_principle": "批量人工复核意见只汇总本批次 run-local 决策；不自动推广到全局配置、不修改规则阈值、不把单批次结论当作最终泛化结论。",
        "summary": summary,
        "datasets": datasets,
        "mappings": mappings,
    }


def _load_run_manifest(storage_dir: Path, run_id: str) -> dict[str, Any]:
    path = storage_dir / run_id / "manifest.json"
    if not path.exists():
        return {}
    try:
        data = read_json(path)
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _batch_rule_mapping_decision_events(decisions: dict[str, Any]) -> list[dict[str, Any]]:
    events = [
        item
        for item in decisions.get("decisions") or []
        if isinstance(item, dict) and item.get("type") == "rule_field_mapping"
    ]
    if events:
        return events
    fallback = []
    for record in (decisions.get("confirmed_rule_mappings") or {}).values():
        if isinstance(record, dict):
            fallback.append({**record, "type": "rule_field_mapping", "action": "confirm"})
    for record in decisions.get("rejected_rule_mappings") or []:
        if isinstance(record, dict):
            fallback.append({**record, "type": "rule_field_mapping", "action": "reject"})
    return fallback


def _latest_batch_mapping_decisions(events: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    latest = {}
    for event in events:
        key = _batch_mapping_decision_key(event)
        if key:
            latest[key] = event
    return latest


def _batch_mapping_decision_key(decision: dict[str, Any]) -> str:
    rule_field = str(decision.get("rule_field") or "").strip()
    dataset_field = str(decision.get("dataset_field") or "").strip()
    if not rule_field or not dataset_field:
        return ""
    return f"{rule_field}::{dataset_field}"


def _empty_batch_decision_mapping(decision: dict[str, Any], dataset_total: int) -> dict[str, Any]:
    return {
        "key": _batch_mapping_decision_key(decision),
        "rule_field": str(decision.get("rule_field") or ""),
        "dataset_field": str(decision.get("dataset_field") or ""),
        "semantic_role": str(decision.get("semantic_role") or ""),
        "display_name": str(decision.get("display_name") or ""),
        "unit": str(decision.get("unit") or "unknown"),
        "statistic": str(decision.get("statistic") or "raw"),
        "dataset_total": dataset_total,
        "confirm_count": 0,
        "reject_count": 0,
        "regression_passed_count": 0,
        "confirmed_distinct_dataset_count": 0,
        "rejected_distinct_dataset_count": 0,
        "regression_passed_distinct_dataset_count": 0,
        "_confirmed_dataset_signatures": set(),
        "_rejected_dataset_signatures": set(),
        "_regression_dataset_signatures": set(),
        "datasets": [],
        "notes": [],
    }


def _merge_batch_decision_mapping(
    record: dict[str, Any],
    decision: dict[str, Any],
    item: dict[str, Any],
    dataset_signature: str,
    regression_pairs: set[tuple[str, str]],
    run_id: str,
    dataset_name: Any,
) -> None:
    action = "confirm" if decision.get("action") == "confirm" else "reject"
    pair = (str(decision.get("rule_field") or ""), str(decision.get("dataset_field") or ""))
    regression_passed = action == "confirm" and pair in regression_pairs
    if action == "confirm":
        record["confirm_count"] += 1
        record["_confirmed_dataset_signatures"].add(dataset_signature)
        if regression_passed:
            record["regression_passed_count"] += 1
            record["_regression_dataset_signatures"].add(dataset_signature)
    else:
        record["reject_count"] += 1
        record["_rejected_dataset_signatures"].add(dataset_signature)
    note = str(decision.get("note") or "").strip()
    if note and note not in record["notes"]:
        record["notes"].append(note)
    record["datasets"].append(
        {
            "dataset": dataset_name,
            "run_id": run_id,
            "report_url": item.get("report_url") or f"/runs/{run_id}",
            "schema_mapping_review_url": f"/runs/{run_id}/schema-mapping-review",
            "action": action,
            "action_label": "确认" if action == "confirm" else "驳回",
            "reviewer": decision.get("reviewer") or decision.get("confirmed_by") or decision.get("rejected_by") or "",
            "note": note,
            "created_at": decision.get("created_at") or decision.get("confirmed_at") or decision.get("rejected_at") or "",
            "dataset_signature": dataset_signature,
            "regression_passed": regression_passed,
        }
    )


def _batch_regression_passed_pairs(validation: dict[str, Any]) -> set[tuple[str, str]]:
    pairs = set()
    for rule in validation.get("rules") or []:
        if not isinstance(rule, dict) or rule.get("status") != "regression_passed":
            continue
        for pair in rule.get("rewrite_pairs") or []:
            if not isinstance(pair, dict):
                continue
            rule_field = str(pair.get("rule_field") or "")
            dataset_field = str(pair.get("dataset_field") or "")
            if rule_field and dataset_field:
                pairs.add((rule_field, dataset_field))
    return pairs


def _batch_dataset_signature(manifest: dict[str, Any], item: dict[str, Any], run_id: str) -> str:
    file_hashes = []
    for file_data in (manifest.get("files") or {}).values():
        if not isinstance(file_data, dict) or file_data.get("kind") != "dataset":
            continue
        sha = str(file_data.get("sha256") or "")
        if sha:
            file_hashes.append(sha)
    if file_hashes:
        return "|".join(sorted(file_hashes))
    return str(item.get("dataset") or run_id)


def _batch_dataset_names(manifest: dict[str, Any]) -> str:
    names = []
    for file_data in (manifest.get("files") or {}).values():
        if isinstance(file_data, dict) and file_data.get("kind") == "dataset":
            name = str(file_data.get("original_name") or "")
            if name:
                names.append(name)
    return ", ".join(names)


def _batch_review_decision_mapping_status(record: dict[str, Any]) -> tuple[str, str, str]:
    confirm_count = int(record.get("confirm_count") or 0)
    reject_count = int(record.get("reject_count") or 0)
    distinct_confirm = int(record.get("confirmed_distinct_dataset_count") or 0)
    regression_passed = int(record.get("regression_passed_count") or 0)
    if confirm_count and reject_count:
        return "conflict_review", "确认/驳回冲突", "同一映射在本批次内同时出现确认和驳回，先复核统计口径、单位和数据上下文。"
    if reject_count and not confirm_count:
        return "rejected_only", "仅驳回", "当前批次只有驳回证据，不应进入全局配置。"
    if confirm_count >= 2 and distinct_confirm >= 2 and regression_passed >= 2:
        return "ready_for_promotion_review", "可进入推广审核", "批内确认和映射回归证据较完整，可进入跨批次复核沉淀，但仍不自动写全局配置。"
    if confirm_count >= 2 and distinct_confirm >= 2:
        return "needs_regression_evidence", "缺少回归证据", "已有跨数据确认，但映射回归通过证据不足，先补验证再考虑推广审核。"
    if confirm_count:
        return "needs_more_batch_evidence", "批内证据不足", "已有人工确认，但确认次数或不同数据证据不足，暂不进入推广审核。"
    return "no_effective_decision", "无生效意见", "当前映射没有可用的人工确认或驳回证据。"


def _batch_review_decision_rank(status: str) -> int:
    return {
        "conflict_review": 0,
        "needs_regression_evidence": 1,
        "needs_more_batch_evidence": 2,
        "ready_for_promotion_review": 3,
        "rejected_only": 4,
        "no_effective_decision": 5,
    }.get(status, 9)


def _batch_review_decisions_verdict(summary: dict[str, Any]) -> tuple[str, str]:
    if int(summary.get("conflict_review_count") or 0):
        return "conflict_review", "当前批次存在人工确认与驳回冲突，应先解释冲突原因，不能进入全局配置。"
    if int(summary.get("needs_regression_evidence_count") or 0):
        return "needs_regression_evidence", "当前批次已有跨数据人工确认，但映射回归证据不足。"
    if int(summary.get("ready_for_promotion_review_count") or 0):
        return "has_promotion_review_candidates", "当前批次存在可进入推广审核的映射，但仍需跨批次沉淀后人工写入全局配置。"
    if int(summary.get("needs_more_batch_evidence_count") or 0):
        return "needs_more_batch_evidence", "当前批次已有人工意见，但确认次数或不同数据证据不足。"
    if int(summary.get("rejected_only_count") or 0):
        return "rejected_only", "当前批次主要是驳回意见，不应推广。"
    return "no_decisions", "当前批次尚未收集到人工确认或驳回意见。"


def _batch_review_decisions_status_label(status: str) -> str:
    return {
        "conflict_review": "存在冲突",
        "needs_regression_evidence": "缺少回归证据",
        "has_promotion_review_candidates": "有推广审核候选",
        "needs_more_batch_evidence": "证据不足",
        "rejected_only": "仅驳回",
        "no_decisions": "暂无人工意见",
    }.get(status, status or "未知")


def _load_schema_mapping_review(storage_dir: Path, run_id: str) -> dict[str, Any]:
    if not run_id:
        return {}
    artifact_path = storage_dir / run_id / "artifacts" / "schema_mapping_review.json"
    if artifact_path.exists():
        try:
            data = read_json(artifact_path)
        except Exception:
            data = {}
        if isinstance(data, dict):
            return data
    path = storage_dir / run_id / "run_events.jsonl"
    if not path.exists():
        return {}
    latest: dict[str, Any] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        payload = event.get("payload") or {}
        if event.get("type") == "tool_finished" and payload.get("tool") == "review_schema_mapping_candidates":
            result = payload.get("result") or {}
            if isinstance(result, dict):
                latest = result
    return latest


def _schema_mapping_review_quality(record: dict[str, Any], missing_review_count: int) -> tuple[str, str]:
    if int(record.get("not_applicable_datasets") or 0):
        return "不适用", "至少一个数据集没有可靠候选字段，应保持不适用，不要强行套规则。"
    if int(record.get("manual_review_datasets") or 0):
        return "需专家复核", "存在单位、统计口径、歧义或弱语义证据，不能自动确认映射。"
    if missing_review_count:
        return "缺少复核证据", "部分历史 run 缺少映射复核辅助产物，需要重跑批次补齐证据。"
    if int(record.get("suggest_confirm_datasets") or 0):
        return "可考虑确认", "候选具备较清晰语义证据，可进入人工确认队列；确认后仍需映射回归验证。"
    if int(record.get("already_safe_datasets") or 0):
        return "已安全处理", "字段已精确匹配或已确认，本清单无需进一步处理。"
    return "无候选", "当前批次未形成可处理候选。"


def _schema_mapping_review_quality_rank(label: str) -> int:
    return {
        "不适用": 0,
        "需专家复核": 1,
        "缺少复核证据": 2,
        "可考虑确认": 3,
        "已安全处理": 4,
        "无候选": 5,
    }.get(label, 9)


def _batch_schema_mapping_review_verdict(summary: dict[str, Any]) -> tuple[str, str]:
    if int(summary.get("missing_review_count") or 0):
        return "evidence_incomplete", "当前批次存在缺少映射复核辅助的 run，不能作为正式复核队列。"
    if int(summary.get("manual_review_count") or 0) or int(summary.get("not_applicable_count") or 0):
        return "needs_review", "当前批次稳定暴露候选映射风险，应先人工复核语义、单位和统计口径。"
    if int(summary.get("confirm_suggestion_count") or 0):
        return "has_confirm_suggestions", "当前批次存在可考虑人工确认的候选，但仍不能自动写入全局配置。"
    if int(summary.get("datasets_with_review") or 0):
        return "resolved", "当前批次映射复核辅助没有发现待处理候选。"
    return "not_run", "当前批次尚未形成映射复核辅助结果。"


def _batch_schema_mapping_review_status_label(status: str) -> str:
    return {
        "needs_review": "批量需复核",
        "has_confirm_suggestions": "有确认建议",
        "evidence_incomplete": "证据不完整",
        "resolved": "已处理",
        "not_run": "未运行",
    }.get(status, status or "未知")


def _schema_mapping_review_status_label(status: str) -> str:
    return SCHEMA_MAPPING_REVIEW_STATUS_LABELS.get(status, status or "未知")


def _load_mapping_regression_validation(storage_dir: Path, run_id: str) -> dict[str, Any]:
    if not run_id:
        return {}
    artifact_path = storage_dir / run_id / "artifacts" / "mapping_regression_validation.json"
    if artifact_path.exists():
        try:
            data = read_json(artifact_path)
        except Exception:
            data = {}
        if isinstance(data, dict):
            return data
    return _load_mapping_regression_from_run_events(storage_dir, run_id)


def _load_mapping_regression_from_run_events(storage_dir: Path, run_id: str) -> dict[str, Any]:
    path = storage_dir / run_id / "run_events.jsonl"
    if not path.exists():
        return {}
    latest: dict[str, Any] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        payload = event.get("payload") or {}
        if event.get("type") == "tool_finished" and payload.get("tool") == "validate_mapping_regression":
            result = payload.get("result") or {}
            if isinstance(result, dict):
                latest = result
    return latest


def _mapping_regression_rule_quality(rule: dict[str, Any], missing_validation_count: int) -> tuple[str, str]:
    if int(rule.get("blocked_or_failed_datasets") or 0):
        return "验证失败", "至少一个数据集的映射重写无法执行，不能推广该映射。"
    if int(rule.get("needs_confirmation_datasets") or 0):
        return "需映射确认", "仍有数据集只有候选映射，先确认或驳回后再验证。"
    if int(rule.get("needs_review_datasets") or 0):
        return "需复核", "映射可执行性或单位/统计口径仍需专家复核。"
    if missing_validation_count:
        return "缺少验证证据", "部分历史 run 缺少映射回归验证，需要重跑批次补齐证据。"
    if int(rule.get("regression_passed_datasets") or 0):
        return "映射回归稳定", "该规则的已确认映射在当前批次中完成技术执行验证。"
    return "无需映射", "当前批次中该规则无需字段重写或没有映射回归需求。"


def _mapping_regression_quality_rank(label: str) -> int:
    return {
        "验证失败": 0,
        "需映射确认": 1,
        "需复核": 2,
        "缺少验证证据": 3,
        "映射回归稳定": 4,
        "无需映射": 5,
    }.get(label, 9)


def _batch_mapping_regression_verdict(summary: dict[str, Any]) -> tuple[str, str]:
    if int(summary.get("missing_validation_count") or 0):
        return "evidence_incomplete", "当前批次存在缺少映射回归验证的 run，不能作为全局映射推广证据。"
    if int(summary.get("failed_count") or 0) or int(summary.get("blocked_or_failed_rule_count") or 0):
        return "blocked", "当前批次存在映射回归失败或阻断项，应先修复字段映射或规则条件。"
    if int(summary.get("needs_confirmation_rule_count") or 0) or int(summary.get("needs_review_rule_count") or 0):
        return "needs_review", "当前批次仍有映射需要确认或复核，不能自动推广。"
    if int(summary.get("datasets_with_validation") or 0):
        return "passed", "当前批次映射回归验证证据完整，未发现阻断项；仍需专家审核后才能推广。"
    return "not_run", "当前批次尚未形成映射回归验证结果。"


def _batch_mapping_regression_status_label(status: str) -> str:
    return {
        "passed": "批量映射回归通过",
        "needs_review": "批量需复核",
        "blocked": "批量阻断",
        "evidence_incomplete": "证据不完整",
        "not_run": "未运行",
    }.get(status, status or "未知")


def _mapping_regression_status_label(status: str) -> str:
    return {
        "passed": "通过",
        "needs_review": "需复核",
        "failed": "失败",
        "blocked": "阻断",
        "passed_no_mapping": "无需映射已通过",
        "regression_passed": "映射回归通过",
        "needs_confirmation": "需映射确认",
        "missing_validation": "缺少验证证据",
    }.get(status, status or "未知")


def _build_batch_schema_mapping_review_markdown(review: dict[str, Any]) -> str:
    summary = review.get("summary") or {}
    quality_counts = summary.get("quality_counts") or {}
    status_counts = summary.get("status_counts") or {}
    lines = [
        "# 批量映射候选复核辅助",
        "",
        f"- 批次 ID: `{review.get('batch_id')}`",
        f"- 批次状态: {_cn_status(str(review.get('status', 'unknown')))}",
        f"- 规则库: `{review.get('rule_path')}`",
        f"- 复核证据覆盖: {summary.get('datasets_with_review', 0)}/{summary.get('dataset_count', 0)}",
        f"- 批量复核状态: {_batch_schema_mapping_review_status_label(str(summary.get('batch_review_status')))}",
        f"- 状态分布: " + ("；".join(f"{_schema_mapping_review_status_label(key)} {value}" for key, value in sorted(status_counts.items())) or "无"),
        f"- 候选质量标签: " + ("；".join(f"{key} {value}" for key, value in sorted(quality_counts.items())) or "无"),
        f"- 反过拟合原则: {review.get('anti_overfit_principle')}",
        "",
        "## 批量结论",
        f"- {summary.get('verdict')}",
        "",
        "## 数据集复核概览",
    ]
    datasets = review.get("datasets") or []
    if not datasets:
        lines.append("- 暂无数据集映射复核记录。")
    for item in datasets:
        if item.get("status") == "missing_review":
            lines.append(f"- `{item.get('dataset')}`: 缺少映射复核辅助；报告: `{item.get('report_url')}`")
            continue
        lines.append(
            f"- `{item.get('dataset')}`: {item.get('status_label')}；"
            f"可考虑确认 {item.get('confirm_suggestion_count', 0)}，"
            f"需专家复核 {item.get('manual_review_count', 0)}，"
            f"不适用 {item.get('not_applicable_count', 0)}，"
            f"已安全处理 {item.get('already_safe_count', 0)}；"
            f"复核页: `{item.get('schema_mapping_review_url')}`"
        )

    lines.extend(["", "## 候选映射复核队列"])
    mappings = review.get("mappings") or []
    if not mappings:
        lines.append("- 暂无候选映射复核队列。")
    for item in mappings:
        lines.append(
            f"- `{item.get('rule_field')}` -> `{item.get('dataset_field') or '无候选'}`: {item.get('quality_label')}；"
            f"可考虑确认 {item.get('suggest_confirm_datasets', 0)}，"
            f"需专家复核 {item.get('manual_review_datasets', 0)}，"
            f"不适用 {item.get('not_applicable_datasets', 0)}，"
            f"已安全处理 {item.get('already_safe_datasets', 0)}。"
        )
        lines.append(f"  建议: {item.get('recommendation')}")
        if item.get("risk_flags"):
            lines.append(f"  风险: {'；'.join(str(flag) for flag in item.get('risk_flags') or [])}")
        for dataset in (item.get("datasets") or [])[:5]:
            lines.append(
                f"  - `{dataset.get('dataset')}`: {_schema_mapping_review_status_label(str(dataset.get('status')))}；"
                f"score={dataset.get('score')}；{'; '.join(str(flag) for flag in dataset.get('risk_flags') or []) or '无风险标签'}"
            )

    lines.extend(
        [
            "",
            "## 使用边界",
            "- 本报告只汇总映射候选复核队列，不自动确认映射。",
            "- 可考虑确认也必须由人在 Schema 语义映射页点击确认；确认后仍需映射执行计划和回归验证。",
            "- 统计口径、单位或缺字段风险不应通过降低阈值或强行补别名解决。",
        ]
    )
    return "\n".join(lines) + "\n"


def _build_batch_review_decisions_markdown(review: dict[str, Any]) -> str:
    summary = review.get("summary") or {}
    status_counts = summary.get("status_counts") or {}
    lines = [
        "# 批量人工复核意见汇总",
        "",
        f"- 批次 ID: `{review.get('batch_id')}`",
        f"- 批次状态: {_cn_status(str(review.get('status', 'unknown')))}",
        f"- 规则库: `{review.get('rule_path')}`",
        f"- 有人工意见数据集: {summary.get('datasets_with_decisions', 0)}/{summary.get('dataset_count', 0)}",
        f"- 原始复核事件数: {summary.get('decision_event_count', 0)}",
        f"- 生效映射意见数: {summary.get('effective_mapping_decision_count', 0)}",
        f"- 批量意见状态: {_batch_review_decisions_status_label(str(summary.get('batch_review_decision_status')))}",
        f"- 映射状态分布: " + ("；".join(f"{_batch_review_mapping_status_label(key)} {value}" for key, value in sorted(status_counts.items())) or "无"),
        f"- 反过拟合原则: {review.get('anti_overfit_principle')}",
        "",
        "## 批量结论",
        f"- {summary.get('verdict')}",
        "",
        "## 数据集意见概览",
    ]
    datasets = review.get("datasets") or []
    if not datasets:
        lines.append("- 暂无数据集记录。")
    for item in datasets:
        lines.append(
            f"- `{item.get('dataset')}`: {item.get('status_label')}；"
            f"确认 {item.get('confirm_count', 0)}，驳回 {item.get('reject_count', 0)}，"
            f"生效意见 {item.get('effective_decision_count', 0)}；"
            f"复核页: `{item.get('schema_mapping_review_url')}`"
        )

    lines.extend(["", "## 映射意见队列"])
    mappings = review.get("mappings") or []
    if not mappings:
        lines.append("- 暂无可汇总的人工复核意见。")
    for item in mappings:
        lines.append(
            f"- {_batch_review_mapping_status_label(str(item.get('status')))}: "
            f"`{item.get('rule_field')}` -> `{item.get('dataset_field')}`；"
            f"确认 {item.get('confirm_count', 0)} 次，驳回 {item.get('reject_count', 0)} 次，"
            f"确认数据指纹 {item.get('confirmed_distinct_dataset_count', 0)} 个，"
            f"映射回归通过 {item.get('regression_passed_count', 0)} 次。"
        )
        lines.append(f"  建议: {item.get('recommendation')}")
        if item.get("notes"):
            lines.append(f"  备注: {'；'.join(str(note) for note in item.get('notes') or [])}")
        for dataset in item.get("datasets") or []:
            regression = "，映射回归已通过" if dataset.get("regression_passed") else ""
            lines.append(
                f"  - `{dataset.get('dataset')}`: {dataset.get('action_label')}；"
                f"{dataset.get('note') or '无备注'}{regression}；"
                f"run: `{dataset.get('run_id')}`"
            )

    lines.extend(
        [
            "",
            "## 使用边界",
            "- 本报告只汇总本批次内的 run-local 人工意见，不自动写入 schema_mapping_overrides.yaml。",
            "- “可进入推广审核”只代表可以交给跨运行复核沉淀流程继续判断，不代表已经是全局知识。",
            "- 如果存在确认/驳回冲突，应优先回看原始数据、单位、统计窗口和映射回归结果。",
        ]
    )
    return "\n".join(lines) + "\n"


def _batch_review_mapping_status_label(status: str) -> str:
    return {
        "ready_for_promotion_review": "可进入推广审核",
        "conflict_review": "确认/驳回冲突",
        "needs_regression_evidence": "缺少回归证据",
        "needs_more_batch_evidence": "批内证据不足",
        "rejected_only": "仅驳回",
        "no_effective_decision": "无生效意见",
    }.get(status, status or "未知")


def _build_batch_mapping_regression_markdown(regression: dict[str, Any]) -> str:
    summary = regression.get("summary") or {}
    quality_counts = summary.get("quality_counts") or {}
    status_counts = summary.get("status_counts") or {}
    lines = [
        "# 批量映射回归验证",
        "",
        f"- 批次 ID: `{regression.get('batch_id')}`",
        f"- 批次状态: {_cn_status(str(regression.get('status', 'unknown')))}",
        f"- 规则库: `{regression.get('rule_path')}`",
        f"- 验证证据覆盖: {summary.get('datasets_with_validation', 0)}/{summary.get('dataset_count', 0)}",
        f"- 批量验证状态: {_batch_mapping_regression_status_label(str(summary.get('batch_regression_status')))}",
        f"- 状态分布: " + ("；".join(f"{_mapping_regression_status_label(key)} {value}" for key, value in sorted(status_counts.items())) or "无"),
        f"- 规则质量标签: " + ("；".join(f"{key} {value}" for key, value in sorted(quality_counts.items())) or "无"),
        f"- 反过拟合原则: {regression.get('anti_overfit_principle')}",
        "",
        "## 批量结论",
        f"- {summary.get('verdict')}",
        "",
        "## 数据集验证概览",
    ]
    datasets = regression.get("datasets") or []
    if not datasets:
        lines.append("- 暂无数据集映射回归记录。")
    for item in datasets:
        if item.get("status") == "missing_validation":
            lines.append(f"- `{item.get('dataset')}`: 缺少映射回归验证；报告: `{item.get('report_url')}`")
            continue
        lines.append(
            f"- `{item.get('dataset')}`: {item.get('status_label')}；"
            f"映射重写通过 {item.get('regression_passed_count', 0)}，无需映射 {item.get('no_mapping_count', 0)}，"
            f"需确认 {item.get('needs_confirmation_count', 0)}，需复核 {item.get('needs_review_count', 0)}，"
            f"阻断/失败 {int(item.get('blocked_count') or 0) + int(item.get('failed_count') or 0)}；"
            f"报告: `{item.get('report_url')}`"
        )

    lines.extend(["", "## 逐规则映射回归"])
    rules = regression.get("rules") or []
    if not rules:
        lines.append("- 暂无逐规则映射回归结果。")
    for rule in rules:
        lines.append(
            f"- `{rule.get('rule_id')}` {rule.get('name')}: {rule.get('quality_label')}；"
            f"通过 {rule.get('passed_datasets', 0)}/{rule.get('dataset_total', 0)}，"
            f"映射回归通过 {rule.get('regression_passed_datasets', 0)}，"
            f"需确认 {rule.get('needs_confirmation_datasets', 0)}，"
            f"需复核 {rule.get('needs_review_datasets', 0)}，"
            f"阻断/失败 {rule.get('blocked_or_failed_datasets', 0)}。"
        )
        lines.append(f"  建议: {rule.get('recommendation')}")
        for issue in (rule.get("issues") or [])[:5]:
            lines.append(f"  - `{issue.get('dataset')}`: {issue.get('issue')}")

    lines.extend(
        [
            "",
            "## 使用边界",
            "- 本报告只汇总映射重写技术验证，不证明阈值业务正确。",
            "- 批量通过也只是进入专家审核和配置候选，不自动写入 schema_mapping_overrides.yaml。",
            "- 缺少验证证据的历史 run 应重新运行，而不是按通过处理。",
        ]
    )
    return "\n".join(lines) + "\n"


def _load_generalization_readiness(storage_dir: Path, run_id: str) -> dict[str, Any]:
    if not run_id:
        return {}
    artifact_path = storage_dir / run_id / "artifacts" / "generalization_readiness.json"
    if artifact_path.exists():
        try:
            data = read_json(artifact_path)
        except Exception:
            data = {}
        if isinstance(data, dict):
            return data
    return _load_generalization_readiness_from_run_events(storage_dir, run_id)


def _load_generalization_readiness_from_run_events(storage_dir: Path, run_id: str) -> dict[str, Any]:
    path = storage_dir / run_id / "run_events.jsonl"
    if not path.exists():
        return {}
    latest: dict[str, Any] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        payload = event.get("payload") or {}
        if event.get("type") == "tool_finished" and payload.get("tool") == "review_generalization_readiness":
            result = payload.get("result") or {}
            if isinstance(result, dict):
                latest = result
    return latest


def _batch_gate_check_quality(check: dict[str, Any]) -> tuple[str, str]:
    if int(check.get("failed_count") or 0):
        return "存在失败", "至少一个数据集该检查失败，不能把该批次作为泛化通过证据。"
    if int(check.get("needs_review_count") or 0):
        return "需复核", "至少一个数据集该检查需要复核，应先定位字段映射、规则适用性或解析问题。"
    return "全部通过", "该检查在当前批次所有有门禁证据的数据集中通过。"


def _gate_check_quality_rank(label: str) -> int:
    return {"存在失败": 0, "需复核": 1, "全部通过": 2}.get(label, 9)


def _batch_generalization_verdict(summary: dict[str, Any]) -> tuple[str, str]:
    if int(summary.get("missing_gate_count") or 0):
        return "evidence_incomplete", "当前批次存在缺少泛化门禁证据的 run，不能给出正式泛化通过结论。"
    if int(summary.get("failed_count") or 0):
        return "blocked", "当前批次存在门禁失败数据集，应先修复规则解析、字段适配或数据质量问题。"
    if int(summary.get("needs_review_count") or 0):
        return "needs_review", "当前批次已稳定暴露泛化风险，但仍有数据集不允许自动诊断解释。"
    if int(summary.get("datasets_with_gate") or 0) and int(summary.get("auto_diagnosis_allowed_count") or 0) == int(
        summary.get("datasets_with_gate") or 0
    ):
        return "passed", "当前批次所有有门禁证据的数据集均通过泛化门禁，可进入下一轮规则解释或回归验收。"
    if int(summary.get("datasets_with_gate") or 0):
        return "needs_review", "当前批次存在不允许自动诊断的数据集，需要继续复核。"
    return "not_run", "当前批次尚未形成泛化门禁结果。"


def _build_batch_generalization_markdown(readiness: dict[str, Any]) -> str:
    summary = readiness.get("summary") or {}
    status_counts = summary.get("status_counts") or {}
    lines = [
        "# 批量泛化验收门禁",
        "",
        f"- 批次 ID: `{readiness.get('batch_id')}`",
        f"- 批次状态: {_cn_status(str(readiness.get('status', 'unknown')))}",
        f"- 规则库: `{readiness.get('rule_path')}`",
        f"- 数据集数量: {summary.get('dataset_total', 0)}",
        f"- 门禁证据覆盖: {summary.get('datasets_with_gate', 0)}/{summary.get('dataset_count', 0)}",
        f"- 批量门禁状态: {_batch_generalization_status_label(str(summary.get('batch_gate_status')))}",
        f"- 允许自动诊断: {summary.get('auto_diagnosis_allowed_count', 0)}/{summary.get('datasets_with_gate', 0)}",
        f"- 平均语义覆盖: {float(summary.get('avg_semantic_coverage') or 0):.1%}",
        f"- 状态分布: " + ("；".join(f"{_generalization_status_label(key)} {value}" for key, value in sorted(status_counts.items())) or "无"),
        f"- 反过拟合原则: {readiness.get('anti_overfit_principle')}",
        "",
        "## 批量结论",
        f"- {summary.get('verdict')}",
        "",
        "## 横向风险指标",
        f"- 未解决规则合计: {summary.get('unresolved_rule_count', 0)}",
        f"- 不可解析规则合计: {summary.get('unparseable_rule_count', 0)}",
        f"- 需映射确认规则合计: {summary.get('needs_mapping_count', 0)}",
        f"- 不适用规则合计: {summary.get('not_applicable_count', 0)}",
        f"- 已确认映射规则合计: {summary.get('confirmed_mapping_count', 0)}",
        f"- 已执行规则合计: {summary.get('evaluated_rule_count', 0)}",
        "",
        "## 数据集门禁概览",
    ]
    datasets = readiness.get("datasets") or []
    if not datasets:
        lines.append("- 暂无数据集门禁记录。")
    for item in datasets:
        metrics = item.get("metrics") or {}
        if item.get("status") == "missing_gate":
            lines.append(f"- `{item.get('dataset')}`: 缺少门禁证据；报告: `{item.get('report_url')}`")
            continue
        lines.append(
            f"- `{item.get('dataset')}`: {item.get('status_label')}；"
            f"允许自动诊断: {'是' if item.get('auto_diagnosis_allowed') else '否'}；"
            f"语义覆盖 {float(metrics.get('semantic_coverage') or 0):.1%}；"
            f"未解决规则 {metrics.get('unresolved_rule_count', 0)}；"
            f"不可解析 {metrics.get('unparseable_rule_count', 0)}；"
            f"阻断检查 {item.get('blocking_check_count', 0)}；"
            f"报告: `{item.get('report_url')}`"
        )

    lines.extend(["", "## 检查项横向分布"])
    checks = readiness.get("checks") or []
    if not checks:
        lines.append("- 暂无检查项汇总。")
    for check in checks:
        lines.append(
            f"- `{check.get('id')}` {check.get('name')}: {check.get('quality_label')}；"
            f"通过 {check.get('passed_count', 0)}，需复核 {check.get('needs_review_count', 0)}，失败 {check.get('failed_count', 0)}。"
        )
        lines.append(f"  建议: {check.get('recommendation')}")
        for message in check.get("messages") or []:
            lines.append(f"  - {message}")

    lines.extend(
        [
            "",
            "## 使用边界",
            "- 本报告只汇总泛化门禁证据，不调规则阈值、不改字段映射、不替代专家结论。",
            "- `needs_review` 是正确暴露风险，不是系统失败；面对未知客户数据时应优先保守。",
            "- 若批次缺少门禁证据，应重新运行该批次，而不是人工把缺失项视为通过。",
        ]
    )
    return "\n".join(lines) + "\n"


def _generalization_status_label(status: str) -> str:
    return {
        "passed": "通过",
        "needs_review": "需复核",
        "failed": "失败",
        "missing_gate": "缺少门禁证据",
        "unknown": "未知",
    }.get(status, status or "未知")


def _batch_generalization_status_label(status: str) -> str:
    return {
        "passed": "批量门禁通过",
        "needs_review": "批量需复核",
        "blocked": "批量阻断",
        "evidence_incomplete": "证据不完整",
        "not_run": "未运行",
    }.get(status, status or "未知")


def _adaptation_rule_quality(rule: dict[str, Any], missing_adaptation: int) -> tuple[str, str]:
    if int(rule.get("not_applicable_datasets") or 0):
        return "存在不适用", "部分数据集缺少必要字段或无法形成可信映射，不应强行套用该规则。"
    if int(rule.get("needs_mapping_datasets") or 0):
        return "需映射确认", "部分数据集只有字段候选映射，需人工确认字段语义、单位和统计口径。"
    if int(rule.get("confirmed_mapping_datasets") or 0):
        return "映射已确认", "部分数据集已存在确认映射，下一步应生成映射执行计划并做回归验证。"
    if int(rule.get("needs_review_datasets") or 0) or int(rule.get("disabled_datasets") or 0):
        return "需复核", "部分数据集的规则结构或启用状态需要人工复核。"
    if missing_adaptation:
        return "缺少适配证据", "部分历史运行没有数据适配产物，建议重新跑批次形成完整第 6 步证据。"
    return "适配稳定", "当前批次中该规则字段均为精确适配，可继续进入规则命中解释。"


def _adaptation_quality_rank(label: str) -> int:
    return {
        "存在不适用": 0,
        "需映射确认": 1,
        "映射已确认": 2,
        "需复核": 3,
        "缺少适配证据": 4,
        "适配稳定": 5,
    }.get(label, 9)


def _count_labels(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        label = str(row.get("quality_label") or "未知")
        counts[label] = counts.get(label, 0) + 1
    return counts


def _build_batch_dataset_adaptation_markdown(adaptation: dict[str, Any]) -> str:
    summary = adaptation.get("summary") or {}
    quality_counts = summary.get("quality_counts") or {}
    lines = [
        "# 批量数据适配审查",
        "",
        f"- 批次 ID: `{adaptation.get('batch_id')}`",
        f"- 批次状态: {_cn_status(str(adaptation.get('status', 'unknown')))}",
        f"- 规则库: `{adaptation.get('rule_path')}`",
        f"- 数据集数量: {summary.get('dataset_total', 0)}",
        f"- 适配证据覆盖: {summary.get('datasets_with_adaptation', 0)}/{summary.get('dataset_count', 0)}",
        f"- 缺少适配证据: {summary.get('missing_adaptation_count', 0)}",
        f"- 规则适配标签: " + ("；".join(f"{key} {value}" for key, value in sorted(quality_counts.items())) or "无"),
        "",
        "## 总体结论",
    ]
    if summary.get("missing_adaptation_count"):
        lines.append("- 当前批次存在没有第 6 步数据适配审查的历史 run，建议重新运行批次后再做正式泛化结论。")
    elif summary.get("not_applicable_rules"):
        lines.append("- 当前批次存在不适用规则，不应把现有规则库直接推广到所有数据集。")
    elif summary.get("needs_mapping_rules"):
        lines.append("- 当前批次存在需映射确认规则，必须先确认字段语义、单位和统计口径。")
    elif summary.get("confirmed_mapping_rules"):
        lines.append("- 当前批次存在已确认字段映射，建议下一步生成映射执行计划并做回归验证。")
    elif summary.get("datasets_with_adaptation"):
        lines.append("- 当前批次的数据适配审查未发现阻断项，可继续结合规则命中和专家复核解释结果。")
    else:
        lines.append("- 当前批次尚未形成数据适配审查结果。")

    lines.extend(
        [
            "",
            "## 数据集适配概览",
        ]
    )
    datasets = adaptation.get("datasets") or []
    if not datasets:
        lines.append("- 暂无数据集适配记录。")
    for item in datasets:
        if item.get("status") == "missing_adaptation":
            lines.append(f"- `{item.get('dataset')}`: 缺少适配证据；报告: `{item.get('report_url')}`")
            continue
        lines.append(
            f"- `{item.get('dataset')}`: {_adaptation_status_label(str(item.get('status')))}；"
            f"可直接执行 {item.get('ready_count', 0)}/{item.get('rule_count', 0)}，"
            f"已确认映射 {item.get('confirmed_mapping_count', 0)}，"
            f"需映射 {item.get('needs_mapping_count', 0)}，不适用 {item.get('not_applicable_count', 0)}，"
            f"规则字段覆盖 {item.get('mapped_field_count', 0)}/{item.get('required_field_count', 0)}；"
            f"报告: `{item.get('report_url')}`"
        )

    lines.extend(
        [
            "",
            "## 逐规则适配结论",
        ]
    )
    rules = adaptation.get("rules") or []
    if not rules:
        lines.append("- 暂无逐规则适配结果。")
    for rule in rules:
        lines.append(
            f"- `{rule.get('rule_id')}` {rule.get('name')}: {rule.get('quality_label')}；"
            f"精确适配 {rule.get('ready_datasets')}/{rule.get('dataset_total')}，"
            f"已确认映射 {rule.get('confirmed_mapping_datasets', 0)}，"
            f"需映射 {rule.get('needs_mapping_datasets')}，不适用 {rule.get('not_applicable_datasets')}，"
            f"需复核/停用 {int(rule.get('needs_review_datasets') or 0) + int(rule.get('disabled_datasets') or 0)}。"
        )
        lines.append(f"  建议: {rule.get('recommendation')}")
        issues = rule.get("mapping_issues") or []
        for issue in issues[:5]:
            candidate = issue.get("candidate_field") or "无候选"
            lines.append(
                f"  - `{issue.get('dataset')}`: {issue.get('rule_field')} -> {candidate} "
                f"({_field_mapping_status_label(str(issue.get('status')))}，score={issue.get('score')})"
            )

    lines.extend(
        [
            "",
            "## 使用边界",
            "- 本报告只判断规则字段与数据字段的适配情况，不评价规则阈值是否业务正确。",
            "- 语义候选映射必须经人工确认后，才能用于自动改写或正式诊断解释。",
            "- 当客户数据字段、单位或统计口径明显变化时，应优先查看本页，再查看规则命中率。",
        ]
    )
    return "\n".join(lines) + "\n"


def _collect_rule_analysis(storage_dir: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    items = manifest.get("items") or []
    dataset_total = len(items)
    rule_map: dict[str, dict[str, Any]] = {}
    for item in items:
        dataset = str(item.get("dataset", ""))
        run_id = str(item.get("run_id", ""))
        rule_results = item.get("rule_results") or _load_rule_results_from_run_events(storage_dir, run_id)
        for rule in rule_results:
            rule_id = str(rule.get("rule_id") or "UNKNOWN")
            record = rule_map.setdefault(
                rule_id,
                {
                    "rule_id": rule_id,
                    "name": rule.get("name") or rule_id,
                    "severity": rule.get("severity") or "",
                    "dataset_total": dataset_total,
                    "evaluated_datasets": 0,
                    "matched_datasets": 0,
                    "unparseable_datasets": 0,
                    "disabled_datasets": 0,
                    "total_matched_rows": 0,
                    "rates": [],
                    "datasets": [],
                },
            )
            status = str(rule.get("status", "unknown"))
            matched_rows = int(rule.get("matched_rows") or 0)
            match_rate = float(rule.get("match_rate") or 0)
            if status == "evaluated":
                record["evaluated_datasets"] += 1
                record["rates"].append(match_rate)
                if matched_rows > 0:
                    record["matched_datasets"] += 1
                record["total_matched_rows"] += matched_rows
            elif status == "unparseable":
                record["unparseable_datasets"] += 1
            elif status == "disabled":
                record["disabled_datasets"] += 1
            record["datasets"].append(
                {
                    "dataset": dataset,
                    "run_id": run_id,
                    "status": status,
                    "matched_rows": matched_rows,
                    "match_rate": match_rate,
                    "message": rule.get("message", ""),
                }
            )
    rules = []
    for record in rule_map.values():
        rates = [float(value) for value in record.pop("rates", [])]
        stats = _rule_rate_stats(rates)
        record.update(stats)
        record["matched_dataset_rate"] = round(record["matched_datasets"] / dataset_total, 6) if dataset_total else 0
        label, recommendation = _rule_quality(record)
        record["quality_label"] = label
        record["recommendation"] = recommendation
        rules.append(record)
    rules.sort(key=lambda item: (item.get("quality_rank", 99), -float(item.get("avg_match_rate") or 0), item.get("rule_id", "")))
    for item in rules:
        item.pop("quality_rank", None)
    return {
        "batch_id": manifest.get("batch_id"),
        "status": manifest.get("status"),
        "dataset_total": dataset_total,
        "generated_at": utc_now_iso(),
        "rules": rules,
        "summary": _rule_analysis_summary(rules),
    }


def _load_rule_results_from_run_events(storage_dir: Path, run_id: str) -> list[dict[str, Any]]:
    path = storage_dir / run_id / "run_events.jsonl"
    if not path.exists():
        return []
    latest: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        payload = event.get("payload") or {}
        if event.get("type") == "tool_finished" and payload.get("tool") == "apply_rules":
            result = payload.get("result") or {}
            latest = result.get("rule_results") or []
    return latest


def _rule_rate_stats(rates: list[float]) -> dict[str, Any]:
    if not rates:
        return {
            "min_match_rate": 0,
            "max_match_rate": 0,
            "avg_match_rate": 0,
            "median_match_rate": 0,
        }
    return {
        "min_match_rate": round(min(rates), 6),
        "max_match_rate": round(max(rates), 6),
        "avg_match_rate": round(mean(rates), 6),
        "median_match_rate": round(median(rates), 6),
    }


def _rule_quality(rule: dict[str, Any]) -> tuple[str, str]:
    avg_rate = float(rule.get("avg_match_rate") or 0)
    max_rate = float(rule.get("max_match_rate") or 0)
    matched_dataset_rate = float(rule.get("matched_dataset_rate") or 0)
    evaluated = int(rule.get("evaluated_datasets") or 0)
    dataset_total = int(rule.get("dataset_total") or 0)
    if int(rule.get("unparseable_datasets") or 0):
        rule["quality_rank"] = 0
        return "需修复", "存在无法解析的数据集结果，优先检查规则字段、条件表达式和数据口径。"
    if evaluated < dataset_total:
        rule["quality_rank"] = 1
        return "覆盖不足", "并非所有数据集都成功执行该规则，建议检查规则启用状态和字段覆盖。"
    if matched_dataset_rate >= 0.8 and avg_rate >= 0.5:
        rule["quality_rank"] = 2
        return "疑似过宽", "多数数据集高比例命中，建议专家复核阈值是否过宽或是否缺少飞行阶段约束。"
    if max_rate >= 0.9 and avg_rate >= 0.25:
        rule["quality_rank"] = 3
        return "局部高命中", "部分数据集命中率很高，建议结合单次报告检查是否为特定飞行段或阈值偏宽。"
    if 0 < avg_rate <= 0.05 and matched_dataset_rate >= 0.3:
        rule["quality_rank"] = 4
        return "稀疏事件", "命中比例较低但跨数据集出现，适合作为事件型规则保留并补充专家解释。"
    if matched_dataset_rate <= 0.2:
        rule["quality_rank"] = 5
        return "疑似过严", "多数数据集未触发，建议检查阈值是否过严或该规则是否只适用于少数场景。"
    rule["quality_rank"] = 6
    return "相对稳定", "命中分布未见明显异常，可继续保留并结合更多数据验证。"


def _rule_analysis_summary(rules: list[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for rule in rules:
        label = str(rule.get("quality_label", "未知"))
        counts[label] = counts.get(label, 0) + 1
    return {"rule_count": len(rules), "quality_counts": counts}


def _build_rule_analysis_markdown(analysis: dict[str, Any]) -> str:
    rules = analysis.get("rules") or []
    counts = (analysis.get("summary") or {}).get("quality_counts") or {}
    lines = [
        "# 批量规则稳定性分析",
        "",
        f"- 批次 ID: `{analysis.get('batch_id')}`",
        f"- 数据集数量: {analysis.get('dataset_total', 0)}",
        f"- 规则数量: {len(rules)}",
        f"- 质量标签: " + "；".join(f"{key} {value}" for key, value in sorted(counts.items())),
        "",
        "## 规则质量结论",
    ]
    if not rules:
        lines.append("- 暂未找到规则执行结果。")
    for rule in rules:
        lines.append(
            f"- `{rule.get('rule_id')}` {rule.get('name')}: {rule.get('quality_label')}；"
            f"平均命中率 {float(rule.get('avg_match_rate') or 0):.1%}，"
            f"中位命中率 {float(rule.get('median_match_rate') or 0):.1%}，"
            f"最高命中率 {float(rule.get('max_match_rate') or 0):.1%}，"
            f"命中数据集 {rule.get('matched_datasets')}/{rule.get('dataset_total')}。"
        )
        lines.append(f"  建议: {rule.get('recommendation')}")
    lines.append("")
    lines.append("## 逐规则数据集明细")
    for rule in rules:
        lines.append(f"### {rule.get('rule_id')} {rule.get('name')}")
        for item in rule.get("datasets") or []:
            lines.append(
                f"- `{item.get('dataset')}`: {item.get('status')}，"
                f"命中 {item.get('matched_rows')} 行，命中率 {float(item.get('match_rate') or 0):.1%}；"
                f"报告: `/runs/{item.get('run_id')}`"
            )
    lines.append("")
    lines.append("## 使用说明")
    lines.append("- “疑似过宽”不代表规则错误，只表示跨数据集命中率过高，需要专家确认阈值或适用阶段。")
    lines.append("- “稀疏事件”通常适合作为事件型规则保留，但需要补充业务解释。")
    lines.append("- 本分析只评估规则命中分布，不使用 `label` 或 `diagnosis` 做监督准确率评价。")
    return "\n".join(lines) + "\n"


def _collect_batch_hit_samples(
    storage_dir: Path,
    manifest: dict[str, Any],
    *,
    rule_ids: list[str],
    datasets_per_rule: int,
    samples_per_dataset: int,
) -> dict[str, Any]:
    candidates: dict[str, list[dict[str, Any]]] = {rule_id: [] for rule_id in rule_ids}
    for item in manifest.get("items") or []:
        run_id = str(item.get("run_id", ""))
        dataset_path = _dataset_path_for_batch_item(manifest, item)
        rule_results = _load_rule_results_from_run_events(storage_dir, run_id) or item.get("rule_results") or []
        for rule in rule_results:
            rule_id = str(rule.get("rule_id") or "")
            if rule_id not in candidates:
                continue
            if rule.get("status") != "evaluated" or int(rule.get("matched_rows") or 0) <= 0:
                continue
            candidates[rule_id].append(
                {
                    "dataset": item.get("dataset"),
                    "dataset_path": str(dataset_path) if dataset_path else "",
                    "run_id": run_id,
                    "report_url": item.get("report_url") or f"/runs/{run_id}",
                    "rule": rule,
                    "matched_rows": int(rule.get("matched_rows") or 0),
                    "match_rate": float(rule.get("match_rate") or 0),
                    "sample_indices": [int(index) for index in (rule.get("sample_indices") or [])[:samples_per_dataset]],
                }
            )

    rules = []
    frame_cache: dict[str, Any] = {}
    for rule_id in rule_ids:
        selected = sorted(
            candidates.get(rule_id, []),
            key=lambda row: (float(row.get("match_rate") or 0), int(row.get("matched_rows") or 0), str(row.get("dataset") or "")),
            reverse=True,
        )[: max(datasets_per_rule, 0)]
        rule_name = ""
        condition = ""
        recommendation = ""
        target_fields: list[str] = []
        dataset_records = []
        for row in selected:
            rule = row["rule"]
            rule_name = rule_name or str(rule.get("name") or rule_id)
            condition = condition or str(rule.get("condition") or "")
            recommendation = recommendation or str(rule.get("recommendation") or "")
            target_fields = target_fields or [str(field) for field in rule.get("target_fields") or []]
            dataset_path = Path(str(row.get("dataset_path") or ""))
            context_fields = _hit_sample_context_fields(rule_id, rule)
            samples = []
            if dataset_path.exists():
                df = frame_cache.get(str(dataset_path))
                if df is None:
                    df = _read_batch_dataset_frame(dataset_path)
                    frame_cache[str(dataset_path)] = df
                samples = _extract_hit_sample_rows(df, row["sample_indices"], context_fields)
            dataset_records.append(
                {
                    "dataset": row.get("dataset"),
                    "dataset_path": row.get("dataset_path"),
                    "run_id": row.get("run_id"),
                    "report_url": row.get("report_url"),
                    "matched_rows": row.get("matched_rows"),
                    "match_rate": row.get("match_rate"),
                    "sample_indices": row.get("sample_indices"),
                    "context_fields": context_fields,
                    "samples": samples,
                }
            )
        rules.append(
            {
                "rule_id": rule_id,
                "name": rule_name or rule_id,
                "condition": condition,
                "recommendation": recommendation,
                "target_fields": target_fields,
                "context_fields": _hit_sample_context_fields(rule_id, {"target_fields": target_fields, "condition": condition}),
                "selected_dataset_count": len(dataset_records),
                "datasets": dataset_records,
            }
        )
    return {
        "batch_id": manifest.get("batch_id"),
        "rule_path": manifest.get("rule_path"),
        "dataset_total": manifest.get("dataset_total"),
        "generated_at": utc_now_iso(),
        "rule_ids": rule_ids,
        "datasets_per_rule": datasets_per_rule,
        "samples_per_dataset": samples_per_dataset,
        "rules": rules,
        "summary": {
            "rule_count": len(rules),
            "dataset_sample_groups": sum(len(rule.get("datasets") or []) for rule in rules),
            "row_samples": sum(len(dataset.get("samples") or []) for rule in rules for dataset in rule.get("datasets") or []),
        },
    }


def _dataset_path_for_batch_item(manifest: dict[str, Any], item: dict[str, Any]) -> Path | None:
    if item.get("dataset_path"):
        return Path(str(item["dataset_path"]))
    dataset_name = item.get("dataset")
    dataset_dir = manifest.get("dataset_dir")
    if dataset_name and dataset_dir:
        return Path(str(dataset_dir)) / str(dataset_name)
    return None


def _read_batch_dataset_frame(path: Path):
    import pandas as pd

    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    return pd.read_csv(path)


def _hit_sample_context_fields(rule_id: str, rule: dict[str, Any]) -> list[str]:
    fields = list(RULE_CONTEXT_FIELDS.get(rule_id, []))
    for field in rule.get("target_fields") or []:
        if str(field) not in fields:
            fields.insert(0, str(field))
    condition = str(rule.get("condition") or "")
    for token in condition.replace("(", " ").replace(")", " ").replace("`", " ").split():
        if "_" in token and token not in fields and not token.replace(".", "", 1).isdigit():
            fields.insert(0, token)
    return fields[:10]


def _extract_hit_sample_rows(df, sample_indices: list[int], context_fields: list[str]) -> list[dict[str, Any]]:
    available_fields = [field for field in context_fields if field in df.columns]
    samples = []
    for index in sample_indices:
        try:
            row = df.loc[index]
        except Exception:
            try:
                row = df.iloc[int(index)]
            except Exception:
                continue
        values = {field: row.get(field) for field in available_fields}
        samples.append({"row_index": int(index), "values": values})
    return samples


def _build_hit_samples_markdown(samples: dict[str, Any]) -> str:
    summary = samples.get("summary") or {}
    lines = [
        "# 关键命中样例导出",
        "",
        f"- 批次 ID: `{samples.get('batch_id')}`",
        f"- 规则库: `{samples.get('rule_path')}`",
        f"- 数据集数量: {samples.get('dataset_total', 0)}",
        f"- 规则数量: {summary.get('rule_count', 0)}",
        f"- 数据集样例组: {summary.get('dataset_sample_groups', 0)}",
        f"- 行样例数量: {summary.get('row_samples', 0)}",
        "",
        "## 使用说明",
        "- 本导出面向专家复核，只展示每条规则的代表性命中样例和关键上下文字段。",
        "- 命中样例用于判断阈值和业务解释是否合理，不代表监督诊断准确率。",
        "- 若某条规则没有样例，通常表示当前批次未命中或历史运行未记录 sample_indices。",
        "",
        "## 规则样例",
    ]
    for rule in samples.get("rules") or []:
        lines.append(f"### {rule.get('rule_id')} {rule.get('name')}")
        if rule.get("condition"):
            lines.append(f"- 条件: `{rule.get('condition')}`")
        if rule.get("recommendation"):
            lines.append(f"- 建议动作: {rule.get('recommendation')}")
        lines.append(f"- 上下文字段: {', '.join(rule.get('context_fields') or []) or '无'}")
        datasets = rule.get("datasets") or []
        if not datasets:
            lines.append("- 暂无可导出的命中样例。")
            continue
        for dataset in datasets:
            lines.append(
                f"- 数据集 `{dataset.get('dataset')}`: 命中 {dataset.get('matched_rows')} 行，"
                f"命中率 {float(dataset.get('match_rate') or 0):.1%}；报告: `{dataset.get('report_url')}`"
            )
            for sample in dataset.get("samples") or []:
                fields = "；".join(f"{key}={_format_sample_value(value)}" for key, value in (sample.get("values") or {}).items())
                lines.append(f"  - 行 {sample.get('row_index')}: {fields or '无可展示字段'}")
    lines.append("")
    lines.append("## 后续复核建议")
    lines.append("- 专家优先查看 R001、R002 的高命中样例，确认 v2 阈值是否符合安全裕度。")
    lines.append("- 对 R005 的高命中数据集做单独解释，判断是否需要加入环境或传感器状态条件。")
    lines.append("- 对 R007 保持事件型规则定位，不应直接作为异常判定。")
    return "\n".join(lines) + "\n"


def _format_sample_value(value: Any) -> str:
    if value is None:
        return "空"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def _collect_rule_version_comparison(
    base_manifest: dict[str, Any],
    candidate_manifest: dict[str, Any],
    base_analysis: dict[str, Any],
    candidate_analysis: dict[str, Any],
) -> dict[str, Any]:
    base_rules = {str(rule.get("rule_id")): rule for rule in base_analysis.get("rules") or []}
    candidate_rules = {str(rule.get("rule_id")): rule for rule in candidate_analysis.get("rules") or []}
    base_rule_meta = _load_rule_library_metadata(Path(str(base_manifest.get("rule_path", ""))))
    candidate_rule_meta = _load_rule_library_metadata(Path(str(candidate_manifest.get("rule_path", ""))))
    records = []
    for rule_id in sorted(set(base_rules) | set(candidate_rules)):
        base_rule = base_rules.get(rule_id)
        candidate_rule = candidate_rules.get(rule_id)
        base_meta = base_rule_meta.get(rule_id, {})
        candidate_meta = candidate_rule_meta.get(rule_id, {})
        avg_delta = _rule_metric_delta(candidate_rule, base_rule, "avg_match_rate")
        max_delta = _rule_metric_delta(candidate_rule, base_rule, "max_match_rate")
        matched_delta = int((candidate_rule or {}).get("matched_datasets") or 0) - int((base_rule or {}).get("matched_datasets") or 0)
        judgement, recommendation = _rule_comparison_judgement(base_rule, candidate_rule, avg_delta)
        records.append(
            {
                "rule_id": rule_id,
                "name": (candidate_meta or base_meta).get("name")
                or (candidate_rule or base_rule or {}).get("name")
                or rule_id,
                "judgement": judgement,
                "recommendation": recommendation,
                "base": _comparison_rule_snapshot(base_rule, base_meta),
                "candidate": _comparison_rule_snapshot(candidate_rule, candidate_meta),
                "delta": {
                    "avg_match_rate": avg_delta,
                    "max_match_rate": max_delta,
                    "matched_datasets": matched_delta,
                    "quality_score": _quality_score((candidate_rule or {}).get("quality_label"))
                    - _quality_score((base_rule or {}).get("quality_label")),
                },
                "rule_change": _rule_change_summary(base_meta, candidate_meta),
            }
        )
    records.sort(key=lambda item: (_comparison_rank(str(item.get("judgement", ""))), item.get("rule_id", "")))
    return {
        "generated_at": utc_now_iso(),
        "base": _comparison_batch_snapshot(base_manifest, base_analysis),
        "candidate": _comparison_batch_snapshot(candidate_manifest, candidate_analysis),
        "summary": _rule_version_comparison_summary(records),
        "rules": records,
    }


def _comparison_batch_snapshot(manifest: dict[str, Any], analysis: dict[str, Any]) -> dict[str, Any]:
    return {
        "batch_id": manifest.get("batch_id"),
        "status": manifest.get("status"),
        "rule_path": manifest.get("rule_path"),
        "dataset_total": manifest.get("dataset_total"),
        "completed": manifest.get("completed"),
        "passed": manifest.get("passed"),
        "needs_review": manifest.get("needs_review"),
        "failed": manifest.get("failed"),
        "quality_counts": (analysis.get("summary") or {}).get("quality_counts") or {},
    }


def _comparison_rule_snapshot(rule: dict[str, Any] | None, meta: dict[str, Any]) -> dict[str, Any]:
    return {
        "exists": rule is not None or bool(meta),
        "name": meta.get("name") or (rule or {}).get("name"),
        "field": meta.get("field"),
        "operator": meta.get("operator"),
        "value": meta.get("value"),
        "severity": meta.get("severity") or (rule or {}).get("severity"),
        "condition": meta.get("condition"),
        "quality_label": (rule or {}).get("quality_label"),
        "avg_match_rate": (rule or {}).get("avg_match_rate"),
        "median_match_rate": (rule or {}).get("median_match_rate"),
        "max_match_rate": (rule or {}).get("max_match_rate"),
        "matched_datasets": (rule or {}).get("matched_datasets"),
        "dataset_total": (rule or {}).get("dataset_total"),
    }


def _load_rule_library_metadata(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        if path.suffix.lower() in {".yaml", ".yml"}:
            import yaml

            raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        elif path.suffix.lower() == ".json":
            raw = json.loads(path.read_text(encoding="utf-8"))
        else:
            return {}
    except Exception:
        return {}
    if isinstance(raw, dict) and isinstance(raw.get("rules"), list):
        items = raw["rules"]
    elif isinstance(raw, list):
        items = raw
    elif isinstance(raw, dict):
        items = [raw]
    else:
        items = []
    records: dict[str, dict[str, Any]] = {}
    for idx, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            continue
        rule_id = str(item.get("rule_id") or item.get("id") or f"R{idx:03d}")
        field = item.get("field")
        operator = item.get("operator") or item.get("op")
        value = item.get("value")
        condition = item.get("condition")
        if not condition and field is not None and operator and value is not None:
            condition = f"{field} {operator} {value!r}"
        records[rule_id] = {
            "rule_id": rule_id,
            "name": item.get("name") or item.get("title") or rule_id,
            "field": field,
            "operator": operator,
            "value": value,
            "severity": item.get("severity"),
            "condition": condition,
            "recommendation": item.get("recommendation"),
        }
    return records


def _rule_metric_delta(candidate_rule: dict[str, Any] | None, base_rule: dict[str, Any] | None, field: str) -> float | None:
    if candidate_rule is None or base_rule is None:
        return None
    return round(float(candidate_rule.get(field) or 0) - float(base_rule.get(field) or 0), 6)


def _rule_change_summary(base_meta: dict[str, Any], candidate_meta: dict[str, Any]) -> dict[str, Any]:
    fields = ["field", "operator", "value", "severity"]
    changes = {}
    for field in fields:
        before = base_meta.get(field)
        after = candidate_meta.get(field)
        if before != after:
            changes[field] = {"from": before, "to": after}
    return {
        "changed": bool(changes),
        "changes": changes,
        "base_condition": base_meta.get("condition"),
        "candidate_condition": candidate_meta.get("condition"),
    }


def _rule_comparison_judgement(
    base_rule: dict[str, Any] | None,
    candidate_rule: dict[str, Any] | None,
    avg_delta: float | None,
) -> tuple[str, str]:
    if base_rule is None and candidate_rule is not None:
        return "新增规则", "候选版本新增了该规则，建议先跑更多数据集确认命中分布是否稳定。"
    if base_rule is not None and candidate_rule is None:
        return "移除规则", "候选版本移除了该规则，建议确认该风险点是否已被其他规则覆盖。"
    base_label = str((base_rule or {}).get("quality_label") or "")
    candidate_label = str((candidate_rule or {}).get("quality_label") or "")
    score_delta = _quality_score(candidate_label) - _quality_score(base_label)
    if score_delta < 0:
        return "改善", "候选版本的规则质量标签更稳，建议作为下一轮专家复核优先版本。"
    if score_delta > 0:
        return "需复核", "候选版本的质量标签变差，建议暂缓替换并检查阈值或适用场景。"
    if avg_delta is not None and abs(avg_delta) >= 0.15:
        return "变化较大", "质量标签未变但平均命中率变化较大，建议结合逐数据集报告复核。"
    return "基本持平", "候选版本与基线版本表现接近，可继续保留观察。"


def _quality_score(label: Any) -> int:
    return {
        "相对稳定": 0,
        "稀疏事件": 1,
        "疑似过严": 2,
        "局部高命中": 2,
        "疑似过宽": 3,
        "覆盖不足": 4,
        "需修复": 5,
    }.get(str(label or ""), 3)


def _comparison_rank(label: str) -> int:
    return {
        "需复核": 0,
        "移除规则": 1,
        "新增规则": 2,
        "变化较大": 3,
        "改善": 4,
        "基本持平": 5,
    }.get(label, 9)


def _rule_version_comparison_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for record in records:
        judgement = str(record.get("judgement") or "未知")
        counts[judgement] = counts.get(judgement, 0) + 1
    return {
        "rule_count": len(records),
        "judgement_counts": counts,
        "changed_rule_count": len([record for record in records if (record.get("rule_change") or {}).get("changed")]),
    }


def _build_rule_version_comparison_markdown(comparison: dict[str, Any]) -> str:
    base = comparison.get("base") or {}
    candidate = comparison.get("candidate") or {}
    summary = comparison.get("summary") or {}
    rules = comparison.get("rules") or []
    judgement_counts = summary.get("judgement_counts") or {}
    lines = [
        "# 规则版本对比报告",
        "",
        f"- 生成时间: {comparison.get('generated_at')}",
        f"- 基线批次: `{base.get('batch_id')}`",
        f"- 候选批次: `{candidate.get('batch_id')}`",
        f"- 基线规则库: `{base.get('rule_path')}`",
        f"- 候选规则库: `{candidate.get('rule_path')}`",
        f"- 数据集规模: 基线 {base.get('completed')}/{base.get('dataset_total')}；候选 {candidate.get('completed')}/{candidate.get('dataset_total')}",
        f"- 规则变化数量: {summary.get('changed_rule_count', 0)}",
        f"- 对比结论分布: " + "；".join(f"{key} {value}" for key, value in sorted(judgement_counts.items())),
        "",
        "## 总体结论",
    ]
    if judgement_counts.get("需复核"):
        lines.append("- 候选版本存在需要优先复核的规则，不建议直接替换为正式规则库。")
    elif judgement_counts.get("改善"):
        lines.append("- 候选版本对部分规则的稳定性有改善，可作为下一轮专家复核和小范围验收版本。")
    else:
        lines.append("- 候选版本与基线版本整体接近，建议继续积累更多数据后再决定是否替换。")
    lines.append("")
    lines.append("## 逐规则对比")
    for rule in rules:
        base_rule = rule.get("base") or {}
        candidate_rule = rule.get("candidate") or {}
        delta = rule.get("delta") or {}
        change = rule.get("rule_change") or {}
        lines.append(f"### {rule.get('rule_id')} {rule.get('name')}")
        lines.append(f"- 对比判断: {rule.get('judgement')}。")
        lines.append(f"- 基线表现: {base_rule.get('quality_label') or '无'}，平均命中率 {_percent(base_rule.get('avg_match_rate'))}，最高命中率 {_percent(base_rule.get('max_match_rate'))}，命中数据集 {base_rule.get('matched_datasets')}/{base_rule.get('dataset_total')}。")
        lines.append(f"- 候选表现: {candidate_rule.get('quality_label') or '无'}，平均命中率 {_percent(candidate_rule.get('avg_match_rate'))}，最高命中率 {_percent(candidate_rule.get('max_match_rate'))}，命中数据集 {candidate_rule.get('matched_datasets')}/{candidate_rule.get('dataset_total')}。")
        lines.append(f"- 变化幅度: 平均命中率 {_percent_delta(delta.get('avg_match_rate'))}，最高命中率 {_percent_delta(delta.get('max_match_rate'))}，命中数据集 {delta.get('matched_datasets')}。")
        if change.get("changed"):
            change_parts = []
            for field, item in (change.get("changes") or {}).items():
                change_parts.append(f"{field}: {item.get('from')} -> {item.get('to')}")
            lines.append(f"- 规则配置变化: {'；'.join(change_parts)}。")
        else:
            lines.append("- 规则配置变化: 未变化。")
        lines.append(f"- 建议: {rule.get('recommendation')}")
    lines.append("")
    lines.append("## 使用边界")
    lines.append("- 本报告比较的是规则命中分布和规则配置变化，不代表监督学习准确率。")
    lines.append("- 如果两次批次的数据集数量或数据集集合不同，应先补跑同规模批次再做正式结论。")
    lines.append("- 候选规则库进入正式版本前，仍需要领域专家确认阈值、风险等级和适用飞行阶段。")
    return "\n".join(lines) + "\n"


def _percent(value: Any) -> str:
    if value is None:
        return "无"
    return f"{float(value):.1%}"


def _percent_delta(value: Any) -> str:
    if value is None:
        return "无"
    sign = "+" if float(value) > 0 else ""
    return f"{sign}{float(value):.1%}"


def _save_batch_manifest(storage_dir: Path, manifest: dict[str, Any]) -> None:
    write_json(_batch_manifest_path(storage_dir, str(manifest["batch_id"])), manifest)


def _batch_dir(storage_dir: Path, batch_id: str) -> Path:
    return storage_dir / BATCH_ROOT_NAME / batch_id


def _batch_manifest_path(storage_dir: Path, batch_id: str) -> Path:
    return _batch_dir(storage_dir, batch_id) / BATCH_MANIFEST_NAME


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


def _adaptation_status_label(status: str) -> str:
    return {
        "ready": "全部直接适配",
        "needs_review": "需复核",
        "missing_adaptation": "缺少适配证据",
    }.get(status, status)


def _field_mapping_status_label(status: str) -> str:
    return {
        "exact": "精确匹配",
        "confirmed_mapping": "已确认映射",
        "suffix_mismatch": "后缀口径不一致",
        "semantic_candidate": "语义候选",
        "missing": "缺失",
    }.get(status, status)
