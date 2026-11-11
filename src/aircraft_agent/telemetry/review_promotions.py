"""Aggregate run-level review decisions into global-promotion candidates.

The promotion report is deliberately conservative: it never writes the global
``schema_mapping_overrides.yaml`` file. It only turns repeated run-local human
review decisions into an auditable candidate list so a domain expert can decide
what deserves to become stable configuration.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any

import yaml

from aircraft_agent.telemetry.reporting import markdown_to_html
from aircraft_agent.telemetry.review_decisions import REVIEW_DECISIONS_NAME, load_review_decisions
from aircraft_agent.common.utils import read_json, utc_now_iso, write_json, write_text_atomic


PROMOTION_DIR_NAME = "review_promotions"
DEFAULT_MIN_CONFIRMATIONS = 2
DEFAULT_MIN_DISTINCT_DATASETS = 2


def build_review_promotion_report(
    storage_dir: Path,
    *,
    min_confirmations: int = DEFAULT_MIN_CONFIRMATIONS,
    min_distinct_datasets: int = DEFAULT_MIN_DISTINCT_DATASETS,
) -> dict[str, Any]:
    """Build and persist a report for review decisions that may be promoted.

    A mapping becomes ``ready_for_global_review`` only when it is confirmed in
    enough runs, appears across enough distinct dataset fingerprints, has enough
    mapping regression evidence, and has no rejected evidence. This keeps the
    workflow focused on generalization rather than fitting one current dataset.
    """
    storage_dir = Path(storage_dir)
    min_confirmations = max(1, int(min_confirmations or DEFAULT_MIN_CONFIRMATIONS))
    min_distinct_datasets = max(1, int(min_distinct_datasets or DEFAULT_MIN_DISTINCT_DATASETS))
    generated_at = utc_now_iso()

    run_records = _collect_run_review_records(storage_dir)
    aggregates: dict[str, dict[str, Any]] = {}
    decision_event_count = 0
    effective_decision_count = 0

    for run_record in run_records:
        decision_event_count += run_record["decision_event_count"]
        latest_decisions = _latest_rule_mapping_decisions(run_record["decisions"])
        effective_decision_count += len(latest_decisions)
        for decision in latest_decisions.values():
            key = _candidate_key(decision)
            if not key:
                continue
            record = aggregates.setdefault(key, _empty_candidate(decision))
            _merge_candidate_metadata(record, decision)
            _add_candidate_evidence(record, decision, run_record)

    candidates = [
        _finalize_candidate(record, min_confirmations=min_confirmations, min_distinct_datasets=min_distinct_datasets)
        for record in aggregates.values()
    ]
    candidates.sort(key=_candidate_sort_key)
    ready_candidates = [item for item in candidates if item["status"] == "ready_for_global_review"]
    promotion_yaml_preview = _promotion_yaml_preview(ready_candidates, generated_at)

    summary = {
        "scanned_run_count": len([path for path in _iter_possible_run_dirs(storage_dir)]),
        "reviewed_run_count": len(run_records),
        "decision_event_count": decision_event_count,
        "effective_mapping_decision_count": effective_decision_count,
        "candidate_count": len(candidates),
        "ready_for_global_review_count": len(ready_candidates),
        "conflict_review_count": sum(1 for item in candidates if item["status"] == "conflict_review"),
        "needs_more_evidence_count": sum(1 for item in candidates if item["status"] == "needs_more_evidence"),
        "needs_regression_evidence_count": sum(
            1 for item in candidates if item["status"] == "needs_regression_evidence"
        ),
        "rejected_only_count": sum(1 for item in candidates if item["status"] == "rejected_only"),
        "regression_passed_candidate_count": sum(
            1 for item in candidates if int(item.get("regression_passed_count") or 0) > 0
        ),
    }
    result = {
        "schema_id": "review_decision_promotion_report_v1",
        "generated_at": generated_at,
        "storage_dir": str(storage_dir),
        "min_confirmations": min_confirmations,
        "min_distinct_datasets": min_distinct_datasets,
        "principle": "只把跨运行、跨数据指纹稳定、映射回归通过且无驳回冲突的复核意见列为全局推广候选；不自动改写全局配置。",
        "summary": summary,
        "candidates": candidates,
        "promotion_yaml_preview": promotion_yaml_preview,
    }

    output_dir = storage_dir / PROMOTION_DIR_NAME
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "review_decision_promotions.json"
    markdown_path = output_dir / "review_decision_promotions.md"
    html_path = output_dir / "review_decision_promotions.html"
    markdown = render_review_promotion_markdown(result)

    write_json(json_path, result)
    write_text_atomic(markdown_path, markdown, encoding="utf-8")
    write_text_atomic(html_path, markdown_to_html(markdown), encoding="utf-8")
    result["paths"] = {
        "json": str(json_path),
        "markdown": str(markdown_path),
        "html": str(html_path),
    }
    write_json(json_path, result)
    return result


def render_review_promotion_markdown(report: dict[str, Any]) -> str:
    """Render the promotion candidate report as Chinese Markdown."""
    summary = report.get("summary") or {}
    lines = [
        "# 复核意见沉淀候选报告",
        "",
        f"- 生成时间: {report.get('generated_at')}",
        f"- 扫描目录: `{report.get('storage_dir')}`",
        f"- 最小确认次数: {report.get('min_confirmations')}",
        f"- 最少不同数据指纹: {report.get('min_distinct_datasets')}",
        f"- 原则: {report.get('principle')}",
        "",
        "## 总览",
        f"- 扫描运行数: {summary.get('scanned_run_count', 0)}",
        f"- 含复核意见运行数: {summary.get('reviewed_run_count', 0)}",
        f"- 原始复核事件数: {summary.get('decision_event_count', 0)}",
        f"- 生效映射决策数: {summary.get('effective_mapping_decision_count', 0)}",
        f"- 可推广复核候选: {summary.get('ready_for_global_review_count', 0)}",
        f"- 冲突待复核: {summary.get('conflict_review_count', 0)}",
        f"- 证据不足: {summary.get('needs_more_evidence_count', 0)}",
        f"- 缺少回归证据: {summary.get('needs_regression_evidence_count', 0)}",
        f"- 仅驳回项: {summary.get('rejected_only_count', 0)}",
        "",
        "## 判断口径",
        "- ready_for_global_review: 确认次数、不同数据指纹和映射回归通过证据均达到阈值，且没有驳回记录。",
        "- needs_more_evidence: 有确认记录，但跨数据证据还不足。",
        "- needs_regression_evidence: 人工确认已达到或接近阈值，但缺少跨数据映射回归通过证据。",
        "- conflict_review: 同一映射既有确认也有驳回，需要专家复核原因。",
        "- rejected_only: 目前只有驳回记录，不应进入全局配置。",
        "",
        "## 候选明细",
    ]
    candidates = report.get("candidates") or []
    if not candidates:
        lines.append("- 暂无可分析的复核意见。")
    for item in candidates:
        lines.append(
            "- "
            f"{_status_label(str(item.get('status')))}: "
            f"`{item.get('rule_field')}` -> `{item.get('dataset_field')}`；"
            f"确认 {item.get('confirm_count', 0)} 次，驳回 {item.get('reject_count', 0)} 次，"
            f"不同数据指纹 {item.get('confirmed_distinct_dataset_count', 0)} 个；"
            f"映射回归通过 {item.get('regression_passed_count', 0)} 次，"
            f"回归通过数据指纹 {item.get('regression_passed_distinct_dataset_count', 0)} 个。"
        )
        if item.get("notes"):
            lines.append(f"  备注: {'；'.join(str(note) for note in item.get('notes') or [])}")

    lines.extend(
        [
            "",
            "## YAML 预览",
            "以下内容只是待专家审核的候选片段，不会被系统自动写入全局配置。",
            "",
            "```yaml",
            str(report.get("promotion_yaml_preview") or "confirmed_rule_mappings: {}\n").rstrip(),
            "```",
        ]
    )
    return "\n".join(lines) + "\n"


def _collect_run_review_records(storage_dir: Path) -> list[dict[str, Any]]:
    records = []
    for run_dir in _iter_possible_run_dirs(storage_dir):
        if not (run_dir / REVIEW_DECISIONS_NAME).exists():
            continue
        decisions = load_review_decisions(run_dir)
        if not _has_rule_mapping_decisions(decisions):
            continue
        manifest = _read_manifest(run_dir)
        mapping_regression = _load_mapping_regression_validation(run_dir)
        records.append(
            {
                "run_id": run_dir.name,
                "run_dir": str(run_dir),
                "manifest": manifest,
                "user_goal": str(manifest.get("user_goal") or ""),
                "created_at": str(manifest.get("created_at") or ""),
                "dataset_signature": _dataset_signature(manifest, run_dir.name),
                "dataset_names": _dataset_names(manifest),
                "decisions": decisions,
                "decision_event_count": _rule_mapping_decision_event_count(decisions),
                "mapping_regression": mapping_regression,
                "mapping_regression_pairs": _mapping_regression_pairs(mapping_regression),
            }
        )
    return records


def _iter_possible_run_dirs(storage_dir: Path) -> list[Path]:
    if not storage_dir.exists():
        return []
    ignored = {"batches", PROMOTION_DIR_NAME, "__pycache__"}
    return sorted(path for path in storage_dir.iterdir() if path.is_dir() and path.name not in ignored)


def _read_manifest(run_dir: Path) -> dict[str, Any]:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        return {}
    try:
        manifest = read_json(manifest_path)
    except Exception:
        return {}
    return manifest if isinstance(manifest, dict) else {}


def _has_rule_mapping_decisions(decisions: dict[str, Any]) -> bool:
    return bool(_rule_mapping_decision_event_count(decisions))


def _rule_mapping_decision_event_count(decisions: dict[str, Any]) -> int:
    events = [
        item
        for item in decisions.get("decisions") or []
        if isinstance(item, dict) and item.get("type") == "rule_field_mapping"
    ]
    if events:
        return len(events)
    confirmed = decisions.get("confirmed_rule_mappings") or {}
    rejected = decisions.get("rejected_rule_mappings") or []
    return len(confirmed if isinstance(confirmed, dict) else {}) + len(rejected if isinstance(rejected, list) else [])


def _latest_rule_mapping_decisions(decisions: dict[str, Any]) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    events = [
        item
        for item in decisions.get("decisions") or []
        if isinstance(item, dict) and item.get("type") == "rule_field_mapping"
    ]
    if events:
        for event in events:
            action = "confirm" if event.get("action") == "confirm" else "reject"
            normalized = {**event, "action": action}
            key = _candidate_key(normalized)
            if key:
                latest[key] = normalized
        return latest

    for field, record in (decisions.get("confirmed_rule_mappings") or {}).items():
        if not isinstance(record, dict):
            continue
        event = {
            **record,
            "type": "rule_field_mapping",
            "action": "confirm",
            "rule_field": str(record.get("rule_field") or field),
        }
        key = _candidate_key(event)
        if key:
            latest[key] = event
    for record in decisions.get("rejected_rule_mappings") or []:
        if not isinstance(record, dict):
            continue
        event = {**record, "type": "rule_field_mapping", "action": "reject"}
        key = _candidate_key(event)
        if key:
            latest[key] = event
    return latest


def _candidate_key(decision: dict[str, Any]) -> str:
    rule_alias = _normalize_alias(str(decision.get("rule_field") or ""))
    dataset_field = str(decision.get("dataset_field") or "").strip()
    if not rule_alias or not dataset_field:
        return ""
    return f"{rule_alias}::{dataset_field}"


def _empty_candidate(decision: dict[str, Any]) -> dict[str, Any]:
    return {
        "rule_field_alias": _normalize_alias(str(decision.get("rule_field") or "")),
        "rule_field": str(decision.get("rule_field") or "").strip(),
        "dataset_field": str(decision.get("dataset_field") or "").strip(),
        "semantic_role": str(decision.get("semantic_role") or "").strip(),
        "display_name": str(decision.get("display_name") or "").strip(),
        "unit": str(decision.get("unit") or "unknown").strip() or "unknown",
        "statistic": str(decision.get("statistic") or "raw").strip() or "raw",
        "confirm_count": 0,
        "reject_count": 0,
        "confirmed_run_ids": [],
        "rejected_run_ids": [],
        "confirmed_dataset_signatures": set(),
        "rejected_dataset_signatures": set(),
        "regression_checked_count": 0,
        "regression_passed_count": 0,
        "regression_needs_review_count": 0,
        "missing_regression_count": 0,
        "regression_checked_run_ids": [],
        "regression_passed_run_ids": [],
        "regression_passed_dataset_signatures": set(),
        "regression_issues": [],
        "dataset_names": set(),
        "notes": [],
        "latest_decision_at": "",
    }


def _merge_candidate_metadata(record: dict[str, Any], decision: dict[str, Any]) -> None:
    for field in ["rule_field", "dataset_field", "semantic_role", "display_name", "unit", "statistic"]:
        value = str(decision.get(field) or "").strip()
        if value:
            record[field] = value
    created_at = str(decision.get("created_at") or decision.get("confirmed_at") or decision.get("rejected_at") or "")
    if created_at >= str(record.get("latest_decision_at") or ""):
        record["latest_decision_at"] = created_at
    note = str(decision.get("note") or "").strip()
    if note:
        record["notes"].append(note)


def _add_candidate_evidence(record: dict[str, Any], decision: dict[str, Any], run_record: dict[str, Any]) -> None:
    action = "confirm" if decision.get("action") == "confirm" else "reject"
    run_id = str(run_record.get("run_id") or "")
    dataset_signature = str(run_record.get("dataset_signature") or run_id)
    dataset_names = [str(name) for name in run_record.get("dataset_names") or [] if str(name)]
    if action == "confirm":
        record["confirm_count"] += 1
        record["confirmed_run_ids"].append(run_id)
        record["confirmed_dataset_signatures"].add(dataset_signature)
        _add_candidate_regression_evidence(record, decision, run_record, run_id, dataset_signature)
    else:
        record["reject_count"] += 1
        record["rejected_run_ids"].append(run_id)
        record["rejected_dataset_signatures"].add(dataset_signature)
    for name in dataset_names:
        record["dataset_names"].add(name)


def _add_candidate_regression_evidence(
    record: dict[str, Any],
    decision: dict[str, Any],
    run_record: dict[str, Any],
    run_id: str,
    dataset_signature: str,
) -> None:
    key = _candidate_key(decision)
    pair_evidence = (run_record.get("mapping_regression_pairs") or {}).get(key) or []
    if not pair_evidence:
        record["missing_regression_count"] += 1
        return

    record["regression_checked_count"] += 1
    record["regression_checked_run_ids"].append(run_id)
    if any(item.get("passed") for item in pair_evidence):
        record["regression_passed_count"] += 1
        record["regression_passed_run_ids"].append(run_id)
        record["regression_passed_dataset_signatures"].add(dataset_signature)
        return

    record["regression_needs_review_count"] += 1
    for item in pair_evidence:
        issues = [str(issue) for issue in item.get("issues") or [] if str(issue)]
        record["regression_issues"].append(
            {
                "run_id": run_id,
                "status": str(item.get("status") or "unknown"),
                "rule_id": str(item.get("rule_id") or ""),
                "rule_name": str(item.get("rule_name") or ""),
                "issues": issues[:5],
            }
        )


def _finalize_candidate(
    record: dict[str, Any],
    *,
    min_confirmations: int,
    min_distinct_datasets: int,
) -> dict[str, Any]:
    confirmed_distinct = len(record.get("confirmed_dataset_signatures") or set())
    regression_passed_distinct = len(record.get("regression_passed_dataset_signatures") or set())
    if record["confirm_count"] > 0 and record["reject_count"] > 0:
        status = "conflict_review"
        recommendation = "存在确认和驳回冲突，先复核字段语义、单位和统计口径差异。"
    elif (
        record["confirm_count"] >= min_confirmations
        and confirmed_distinct >= min_distinct_datasets
        and record["regression_passed_count"] >= min_confirmations
        and regression_passed_distinct >= min_distinct_datasets
        and record["regression_needs_review_count"] == 0
    ):
        status = "ready_for_global_review"
        recommendation = "满足跨运行、跨数据指纹和映射回归证据要求，可进入专家审核推广队列。"
    elif record["confirm_count"] >= min_confirmations and confirmed_distinct >= min_distinct_datasets:
        status = "needs_regression_evidence"
        recommendation = "人工确认已满足基础阈值，但映射回归通过证据不足或存在回归复核项，暂不推广。"
    elif record["confirm_count"] > 0:
        status = "needs_more_evidence"
        recommendation = "继续在更多数据结构或批量验收中观察，不建议现在写入全局配置。"
    else:
        status = "rejected_only"
        recommendation = "目前只有驳回证据，不应推广。"

    return {
        "key": f"{record['rule_field_alias']}::{record['dataset_field']}",
        "status": status,
        "status_label": _status_label(status),
        "recommendation": recommendation,
        "rule_field_alias": record["rule_field_alias"],
        "rule_field": record["rule_field"],
        "dataset_field": record["dataset_field"],
        "semantic_role": record["semantic_role"],
        "display_name": record["display_name"],
        "unit": record["unit"],
        "statistic": record["statistic"],
        "confirm_count": record["confirm_count"],
        "reject_count": record["reject_count"],
        "confirmed_distinct_dataset_count": confirmed_distinct,
        "rejected_distinct_dataset_count": len(record.get("rejected_dataset_signatures") or set()),
        "regression_checked_count": record["regression_checked_count"],
        "regression_passed_count": record["regression_passed_count"],
        "regression_needs_review_count": record["regression_needs_review_count"],
        "missing_regression_count": record["missing_regression_count"],
        "regression_passed_distinct_dataset_count": regression_passed_distinct,
        "confirmed_run_ids": sorted(set(record["confirmed_run_ids"])),
        "rejected_run_ids": sorted(set(record["rejected_run_ids"])),
        "regression_checked_run_ids": sorted(set(record["regression_checked_run_ids"])),
        "regression_passed_run_ids": sorted(set(record["regression_passed_run_ids"])),
        "regression_issues": _dedupe_regression_issues(record.get("regression_issues") or [])[:8],
        "dataset_names": sorted(record.get("dataset_names") or []),
        "notes": _dedupe_keep_order(record.get("notes") or [])[:8],
        "latest_decision_at": record.get("latest_decision_at") or "",
    }


def _candidate_sort_key(item: dict[str, Any]) -> tuple[int, int, int, str]:
    status_rank = {
        "ready_for_global_review": 0,
        "conflict_review": 1,
        "needs_regression_evidence": 2,
        "needs_more_evidence": 3,
        "rejected_only": 4,
    }.get(str(item.get("status")), 9)
    return (
        status_rank,
        -int(item.get("confirm_count") or 0),
        -int(item.get("confirmed_distinct_dataset_count") or 0),
        str(item.get("rule_field") or ""),
    )


def _promotion_yaml_preview(candidates: list[dict[str, Any]], generated_at: str) -> str:
    mappings = {}
    for item in candidates:
        mappings[str(item.get("rule_field") or item.get("rule_field_alias"))] = {
            "dataset_field": item.get("dataset_field") or "",
            "semantic_role": item.get("semantic_role") or "",
            "display_name": item.get("display_name") or "",
            "unit": item.get("unit") or "unknown",
            "statistic": item.get("statistic") or "raw",
            "confirmed_by": "cross_run_review_promotion",
            "confirmed_at": generated_at,
            "evidence_runs": item.get("confirmed_run_ids") or [],
            "evidence_distinct_dataset_count": item.get("confirmed_distinct_dataset_count") or 0,
            "regression_evidence_runs": item.get("regression_passed_run_ids") or [],
            "regression_evidence_distinct_dataset_count": item.get("regression_passed_distinct_dataset_count") or 0,
            "note": "跨运行复核和映射回归候选；仍需专家审核后再手动写入全局配置。",
        }
    return yaml.safe_dump({"confirmed_rule_mappings": mappings}, allow_unicode=True, sort_keys=False)


def _load_mapping_regression_validation(run_dir: Path) -> dict[str, Any]:
    artifact_path = run_dir / "artifacts" / "mapping_regression_validation.json"
    if artifact_path.exists():
        try:
            data = read_json(artifact_path)
        except Exception:
            data = {}
        if isinstance(data, dict):
            return data
    events_path = run_dir / "run_events.jsonl"
    if not events_path.exists():
        return {}
    latest: dict[str, Any] = {}
    for line in events_path.read_text(encoding="utf-8").splitlines():
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


def _mapping_regression_pairs(validation: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    pairs: dict[str, list[dict[str, Any]]] = {}
    if not isinstance(validation, dict):
        return pairs
    for rule in validation.get("rules") or []:
        if not isinstance(rule, dict):
            continue
        rule_status = str(rule.get("status") or "unknown")
        for pair in rule.get("rewrite_pairs") or []:
            if not isinstance(pair, dict):
                continue
            key = _candidate_key(pair)
            if not key:
                continue
            pairs.setdefault(key, []).append(
                {
                    "passed": rule_status == "regression_passed",
                    "status": rule_status,
                    "rule_id": str(rule.get("rule_id") or ""),
                    "rule_name": str(rule.get("name") or ""),
                    "issues": [str(issue) for issue in rule.get("issues") or [] if str(issue)],
                    "matched_rows": rule.get("matched_rows"),
                    "match_rate": rule.get("match_rate"),
                }
            )
    return pairs


def _dataset_signature(manifest: dict[str, Any], fallback: str) -> str:
    files = manifest.get("files") if isinstance(manifest, dict) else {}
    dataset_ids = [str(item) for item in manifest.get("dataset_file_ids") or []] if isinstance(manifest, dict) else []
    fingerprints = []
    if isinstance(files, dict):
        for dataset_id in dataset_ids:
            file_record = files.get(dataset_id)
            if not isinstance(file_record, dict):
                continue
            sha = str(file_record.get("sha256") or "").strip()
            name = str(file_record.get("original_name") or "").strip()
            size = str(file_record.get("size_bytes") or "").strip()
            fingerprints.append(sha or f"{name}:{size}")
    fingerprints = [item for item in fingerprints if item]
    if not fingerprints:
        return f"run:{fallback}"
    return "|".join(sorted(fingerprints))


def _dataset_names(manifest: dict[str, Any]) -> list[str]:
    files = manifest.get("files") if isinstance(manifest, dict) else {}
    dataset_ids = [str(item) for item in manifest.get("dataset_file_ids") or []] if isinstance(manifest, dict) else []
    names = []
    if isinstance(files, dict):
        for dataset_id in dataset_ids:
            file_record = files.get(dataset_id)
            if isinstance(file_record, dict) and file_record.get("original_name"):
                names.append(str(file_record.get("original_name")))
    return names


def _normalize_alias(value: str) -> str:
    normalized = str(value).lower()
    normalized = re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "_", normalized)
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized


def _status_label(status: str) -> str:
    return {
        "ready_for_global_review": "可进入全局审核",
        "needs_more_evidence": "证据不足",
        "needs_regression_evidence": "缺少回归证据",
        "conflict_review": "冲突待复核",
        "rejected_only": "仅有驳回",
    }.get(status, "未知")


def _dedupe_keep_order(values: list[Any]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def _dedupe_regression_issues(values: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    result = []
    for value in values:
        key = (
            str(value.get("run_id") or ""),
            str(value.get("status") or ""),
            str(value.get("rule_id") or ""),
            "|".join(str(issue) for issue in value.get("issues") or []),
        )
        if key in seen:
            continue
        seen.add(key)
        result.append(value)
    return result
