"""Run-level human review decisions.

Human review decisions are intentionally stored inside one run directory first.
They can influence refreshed artifacts for that run, but they do not
automatically become global schema overrides until they have been validated
across datasets.
"""

from __future__ import annotations

import re
import uuid
from pathlib import Path
from typing import Any

from .utils import read_json, utc_now_iso, write_json


REVIEW_DECISIONS_NAME = "review_decisions.json"


def review_decisions_path(run_dir: Path) -> Path:
    return run_dir / REVIEW_DECISIONS_NAME


def load_review_decisions(run_dir: Path) -> dict[str, Any]:
    """Load run-local human decisions in a stable shape."""
    path = review_decisions_path(run_dir)
    if not path.exists():
        return _empty_decisions(path)
    try:
        data = read_json(path)
    except Exception:
        return _empty_decisions(path)
    if not isinstance(data, dict):
        return _empty_decisions(path)
    data.setdefault("schema_id", "run_review_decisions_v1")
    data.setdefault("path", str(path))
    data.setdefault("decisions", [])
    data.setdefault("confirmed_rule_mappings", {})
    data.setdefault("rejected_rule_mappings", [])
    return data


def append_rule_mapping_decision(
    run_dir: Path,
    *,
    action: str,
    rule_field: str,
    dataset_field: str,
    semantic_role: str = "",
    display_name: str = "",
    unit: str = "",
    statistic: str = "",
    reviewer: str = "local_user",
    note: str = "",
) -> dict[str, Any]:
    """Append a confirm/reject decision for one rule-field mapping."""
    normalized_action = "confirm" if action == "confirm" else "reject"
    now = utc_now_iso()
    decisions = load_review_decisions(run_dir)
    decision = {
        "decision_id": str(uuid.uuid4()),
        "type": "rule_field_mapping",
        "action": normalized_action,
        "rule_field": str(rule_field).strip(),
        "dataset_field": str(dataset_field).strip(),
        "semantic_role": str(semantic_role or "").strip(),
        "display_name": str(display_name or "").strip(),
        "unit": str(unit or "").strip() or "unknown",
        "statistic": str(statistic or "").strip() or "raw",
        "reviewer": str(reviewer or "local_user").strip() or "local_user",
        "note": str(note or "").strip(),
        "created_at": now,
        "scope": "run",
    }
    if not decision["rule_field"] or not decision["dataset_field"]:
        raise ValueError("规则字段和数据字段不能为空。")

    existing = [item for item in decisions.get("decisions") or [] if isinstance(item, dict)]
    existing.append(decision)
    decisions["decisions"] = existing
    decisions["updated_at"] = now

    confirmed = dict(decisions.get("confirmed_rule_mappings") or {})
    rejected = [item for item in decisions.get("rejected_rule_mappings") or [] if isinstance(item, dict)]
    key = _normalize_alias(decision["rule_field"])

    if normalized_action == "confirm":
        confirmed[decision["rule_field"]] = {
            "rule_field": decision["rule_field"],
            "dataset_field": decision["dataset_field"],
            "semantic_role": decision["semantic_role"],
            "display_name": decision["display_name"],
            "unit": decision["unit"],
            "statistic": decision["statistic"],
            "confirmed_by": decision["reviewer"],
            "confirmed_at": now,
            "note": decision["note"],
            "source": "run_review_decision",
        }
        rejected = [
            item
            for item in rejected
            if not (
                _normalize_alias(str(item.get("rule_field") or "")) == key
                and str(item.get("dataset_field") or "") == decision["dataset_field"]
            )
        ]
    else:
        confirmed = {
            field: mapping
            for field, mapping in confirmed.items()
            if not (
                _normalize_alias(str(mapping.get("rule_field") or field)) == key
                and str(mapping.get("dataset_field") or "") == decision["dataset_field"]
            )
        }
        rejected.append(
            {
                "rule_field": decision["rule_field"],
                "dataset_field": decision["dataset_field"],
                "rejected_by": decision["reviewer"],
                "rejected_at": now,
                "note": decision["note"],
            }
        )

    decisions["confirmed_rule_mappings"] = confirmed
    decisions["rejected_rule_mappings"] = rejected
    write_json(review_decisions_path(run_dir), decisions)
    return decision


def review_decision_overrides(run_dir: Path) -> dict[str, Any]:
    """Return confirmed run-local decisions in schema-override shape."""
    decisions = load_review_decisions(run_dir)
    confirmed = {}
    for field, record in (decisions.get("confirmed_rule_mappings") or {}).items():
        if not isinstance(record, dict):
            continue
        rule_field = str(record.get("rule_field") or field).strip()
        dataset_field = str(record.get("dataset_field") or "").strip()
        if not rule_field or not dataset_field:
            continue
        confirmed[_normalize_alias(rule_field)] = {**record, "rule_field": rule_field, "dataset_field": dataset_field}
    return {
        "path": str(review_decisions_path(run_dir)),
        "confirmed_mappings": {},
        "confirmed_rule_mappings": confirmed,
        "raw": decisions,
    }


def rejected_rule_mapping_keys(run_dir: Path) -> set[tuple[str, str]]:
    """Return normalized (rule_field, dataset_field) pairs rejected in this run."""
    decisions = load_review_decisions(run_dir)
    rejected = set()
    for item in decisions.get("rejected_rule_mappings") or []:
        if not isinstance(item, dict):
            continue
        rule_field = str(item.get("rule_field") or "").strip()
        dataset_field = str(item.get("dataset_field") or "").strip()
        if rule_field and dataset_field:
            rejected.add((_normalize_alias(rule_field), dataset_field))
    return rejected


def merge_mapping_overrides(base: dict[str, Any], run_overrides: dict[str, Any]) -> dict[str, Any]:
    """Overlay run-local confirmations on top of stable config overrides."""
    merged = {
        "path": f"{base.get('path', '')};{run_overrides.get('path', '')}",
        "confirmed_mappings": dict(base.get("confirmed_mappings") or {}),
        "confirmed_rule_mappings": dict(base.get("confirmed_rule_mappings") or {}),
        "raw": {"base": base.get("raw") or {}, "run": run_overrides.get("raw") or {}},
    }
    merged["confirmed_mappings"].update(run_overrides.get("confirmed_mappings") or {})
    merged["confirmed_rule_mappings"].update(run_overrides.get("confirmed_rule_mappings") or {})
    return merged


def _empty_decisions(path: Path) -> dict[str, Any]:
    return {
        "schema_id": "run_review_decisions_v1",
        "path": str(path),
        "updated_at": "",
        "decisions": [],
        "confirmed_rule_mappings": {},
        "rejected_rule_mappings": [],
    }


def _normalize_alias(value: str) -> str:
    normalized = str(value).lower()
    normalized = re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "_", normalized)
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized
