"""Acceptance checks for invocation-bound structured report coverage."""
from __future__ import annotations

from copy import deepcopy
import json

import pytest

from aircraft_agent.agent.runtime import RunStore
from aircraft_agent.common.utils import hash_json
from aircraft_agent.evidence.chain import evidence_node_reference
from aircraft_agent.evidence.verification import verify_evidence_snapshot_bytes
from aircraft_agent.telemetry.coverage import (
    SCOPE_VERSION,
    build_report_coverage,
    collect_report_claims,
    verify_report_claims,
)
from aircraft_agent.tools.executor import ToolExecutor


@pytest.fixture
def context(tmp_path):
    return RunStore(tmp_path / "runs").create_run("结构化结论证据覆盖验收")


def profile(dataset="dataset-a", rows=12):
    return {
        "dataset_id": dataset,
        "profile": {
            "rows": rows,
            "columns": 2,
            "column_names": ["temperature", "pressure"],
            "column_profiles": [
                {"name": "temperature", "missing_rate": 0.0},
                {"name": "pressure", "missing_rate": 0.25},
            ],
        },
    }


def record(context, tool, result, arguments=None):
    arguments = deepcopy(arguments if arguments is not None else {"dataset_id": result.get("dataset_id")})
    node = context.evidence.add(
        type="tool_call", tool_name=tool, arguments=arguments,
        output=result, summary="Test structured output",
    )
    context.record_tool_result(tool, arguments, deepcopy(result), evidence_reference=evidence_node_reference(node))
    return context.tool_results[-1]


def checked_catalog(context, catalog=None, raw=None):
    catalog = catalog if catalog is not None else collect_report_claims(context)
    raw = raw if raw is not None else context.evidence.to_dict()
    raw, verification = verify_evidence_snapshot_bytes(json.dumps(raw).encode())
    coverage = verify_report_claims(catalog, raw, verification)
    return catalog, coverage


def claim_for(catalog, key, call_index=None):
    candidates = [claim for claim in catalog["claims"] if claim["key"] == key]
    if call_index is not None:
        call_id = f"{catalog['run_id']}:tool:{call_index}"
        candidates = [claim for claim in candidates if claim["required_sources"][0]["call_id"] == call_id]
    assert len(candidates) == 1
    return candidates[0]


def assert_full(coverage, denominator):
    assert coverage["reference_coverage"] == {"numerator": denominator, "denominator": denominator, "ratio": 1.0}
    assert coverage["verified_coverage"] == {
        "status": "assessable", "numerator": denominator,
        "denominator": denominator, "ratio": 1.0,
    }
    assert coverage["issues"] == []


def test_complete_profile_coverage_uses_frozen_scope_and_preserves_results(context):
    result = profile()
    record(context, "profile_dataset", result)
    bundle = build_report_coverage(context)
    catalog, coverage = bundle["catalog"], bundle["coverage"]
    assert_full(coverage, 5)
    assert catalog["scope_version"] == coverage["scope_version"] == SCOPE_VERSION
    assert catalog["run_id"] == coverage["run_id"] == context.run_id
    assert catalog["source_calls"][0]["result"] == result
    assert claim_for(catalog, "rows")["value"] == 12
    assert claim_for(catalog, "column_profiles")["value"] == [{"name": "pressure", "missing_rate": 0.25}]
    assert coverage["snapshot"]["phase"] == "report_render"
    assert coverage["snapshot"]["chain_hash"] == context.evidence.chain_hash()
    assert coverage["snapshot"]["node_count"] == 1
    assert "不等同于诊断准确率" in coverage["scope"]["interpretation"]
    assert any("自由文本" in item for item in coverage["scope"]["excluded"])


def test_two_datasets_partial_coverage_requires_every_aggregate_source(context):
    record(context, "profile_dataset", profile("dataset-a", 12))
    second = record(context, "profile_dataset", profile("dataset-b", 30))
    second.pop("evidence_reference")
    catalog, coverage = checked_catalog(context)
    assert coverage["reference_coverage"] == {"numerator": 4, "denominator": 9, "ratio": 4 / 9}
    assert coverage["verified_coverage"]["numerator"] == 4
    aggregate = claim_for(catalog, "total_rows")
    assert aggregate["value"] == 42
    assert len(aggregate["required_sources"]) == 2
    assert aggregate["validation"]["verified"] is False
    assert "missing_reference" in aggregate["validation"]["reasons"]
    assert {claim["dataset_id"] for claim in catalog["claims"]} == {"dataset-a", "dataset-b", None}


def test_all_missing_references_keep_the_full_denominator(context):
    item = record(context, "profile_dataset", profile())
    item.pop("evidence_reference")
    catalog, coverage = checked_catalog(context)
    assert len(catalog["claims"]) == 5
    assert coverage["reference_coverage"]["numerator"] == 0
    assert coverage["verified_coverage"]["ratio"] == 0
    assert all("missing_reference" in claim["validation"]["reasons"] for claim in catalog["claims"])


def test_multiple_sources_and_repeated_dataset_use_latest_rows_but_keep_all_calls(context):
    record(context, "profile_dataset", profile("dataset-a", 12))
    record(context, "profile_dataset", profile("dataset-b", 30))
    record(context, "profile_dataset", profile("dataset-a", 15))
    catalog, coverage = checked_catalog(context)
    assert_full(coverage, 13)
    aggregate = claim_for(catalog, "total_rows")
    assert aggregate["value"] == 45
    assert {source["call_id"] for source in aggregate["required_sources"]} == {
        f"{context.run_id}:tool:1", f"{context.run_id}:tool:2",
    }
    assert [claim_for(catalog, "rows", index)["value"] for index in range(3)] == [12, 30, 15]
    assert len({claim["claim_id"] for claim in catalog["claims"]}) == 13


def test_same_tool_same_arguments_and_output_cannot_substitute_another_invocation(context):
    first = record(context, "profile_dataset", profile())
    second = record(context, "profile_dataset", profile())
    first["evidence_reference"] = deepcopy(second["evidence_reference"])
    catalog, coverage = checked_catalog(context)
    assert "source_mismatch" in claim_for(catalog, "rows", 0)["validation"]["reasons"]
    assert claim_for(catalog, "rows", 1)["validation"]["verified"] is True
    assert coverage["reference_coverage"]["numerator"] == 9
    assert coverage["verified_coverage"]["numerator"] == 5


def test_non_scope_tool_calls_do_not_shift_invocation_matching(context):
    record(context, "check_missing_values", {"dataset_id": "dataset-a", "columns_with_missing": []})
    record(context, "profile_dataset", profile())
    _, coverage = checked_catalog(context)
    assert_full(coverage, 5)


def test_runner_failure_evidence_before_successful_tool_does_not_shift_invocation(context):
    # Both runner modes append a failure tool_call but do not record a tool_result.
    context.evidence.add(type="tool_call", tool_name="bad_tool", arguments={}, output={"error": "failed"}, summary="failed")
    store = RunStore(context.run_dir.parent)
    dataset = store.save_bytes(context, data=b"x,y\n1,2\n3,4\n", original_name="sample.csv", kind="dataset")
    ToolExecutor(context).execute("profile_dataset", {"dataset_id": dataset.file_id})
    _, coverage = checked_catalog(context)
    assert_full(coverage, 5)


def rule_result():
    return {"dataset_id": "dataset-a", "rule_results": [
        {"rule_id": "R1", "name": "温度", "status": "evaluated", "severity": "high", "condition": "temperature > 3",
         "matched_rows": 2, "match_rate": 0.2, "sample_indices": [1, 4, 5, 7, 8, 9]},
        {"rule_id": "R2", "name": "停用", "status": "disabled", "severity": "low", "message": "停用规则"},
        {"rule_id": "R3", "name": "未解析", "status": "unparseable", "severity": "medium", "message": "字段缺失"},
    ]}


@pytest.mark.parametrize("tool", ["apply_rules", "apply_rule_library"])
def test_rule_records_include_evaluated_disabled_unparseable_and_alias(context, tool):
    record(context, tool, rule_result())
    catalog, coverage = checked_catalog(context)
    assert_full(coverage, 7)
    assert claim_for(catalog, "evaluated_count")["value"] == 1
    assert claim_for(catalog, "unparseable_count")["value"] == 1
    assert claim_for(catalog, "matched_count")["value"] == 1
    assert claim_for(catalog, "rule_0")["value"]["sample_indices"] == [1, 4, 5, 7, 8]
    assert claim_for(catalog, "rule_2")["value"]["message"] == "字段缺失"
    assert catalog["source_calls"][0]["tool"] == tool


def test_association_thresholds_and_first_ten_records_match_original_indices(context):
    result = {
        "dataset_id": "dataset-a", "engine": "datamining_fpgrowth", "transactions": 100, "frequent_itemsets": 20,
        "parameters": {"min_support": 0.2, "effective_min_support": 0.2, "min_confidence": 0.7,
                       "min_lift": 1.2, "min_count": 2, "max_rules": 15},
        "rules": [{"antecedent": [f"x={i}"], "consequent": ["y=1"], "matched_rows": i,
                   "support": 0.2, "confidence": 0.7, "lift": 1.2} for i in range(12)],
    }
    record(context, "mine_association_rules", result)
    catalog, coverage = checked_catalog(context)
    assert_full(coverage, 20)
    assert claim_for(catalog, "count")["value"] == 12
    assert claim_for(catalog, "rule_9")["required_sources"][0]["result_path"] == ["rules", 9]
    assert not any(claim["key"] == "rule_10" for claim in catalog["claims"])


def validation_result():
    return {"status": "needs_review", "evidence_nodes": 0, "missing_tools": ["apply_rules"],
            "warnings": ["等待复核"], "errors": [], "chain_hash": "historical-validation-snapshot",
            "evidence_verification": {"valid": True, "verified": True}}


def test_validation_claims_describe_historical_tool_result_not_current_chain(context):
    result = validation_result()
    record(context, "validate_analysis", result, arguments={})
    context.validation_results.append(deepcopy(result))
    catalog, coverage = checked_catalog(context)
    assert_full(coverage, 8)
    assert claim_for(catalog, "chain_hash")["value"] == "historical-validation-snapshot"
    assert coverage["snapshot"]["chain_hash"] != claim_for(catalog, "chain_hash")["value"]


@pytest.mark.parametrize("kind", ["profile", "validation"])
def test_materialized_unrecorded_fact_is_counted_with_source_missing(context, kind):
    context.evidence.save()
    if kind == "profile":
        context.dataset_profiles["dataset-a"] = profile()["profile"]
    else:
        context.validation_results.append(validation_result())
    catalog, coverage = checked_catalog(context)
    assert len(catalog["claims"]) == (5 if kind == "profile" else 8)
    assert coverage["verified_coverage"]["numerator"] == 0
    assert all("source_missing" in claim["validation"]["reasons"] for claim in catalog["claims"])


@pytest.mark.parametrize("change,reason", [
    ("node_id", "node_missing"), ("node_hash", "hash_mismatch"),
    ("tool_name", "source_mismatch"), ("type", "source_mismatch"),
    ("missing_hash", "missing_reference"),
])
def test_invalid_reference_is_reported_without_shrinking_denominator(context, change, reason):
    item = record(context, "profile_dataset", profile())
    if change == "missing_hash":
        item["evidence_reference"].pop("node_hash")
    else:
        item["evidence_reference"][change] = "wrong"
    catalog, coverage = checked_catalog(context)
    assert coverage["verified_coverage"]["denominator"] == 5
    assert coverage["verified_coverage"]["numerator"] == 0
    assert all(reason in claim["validation"]["reasons"] for claim in catalog["claims"])


@pytest.mark.parametrize("node_type,tool", [("model_note", "record_evidence"), ("tool_call", "write_report"), ("tool_call", "detect_outliers")])
def test_valid_unrelated_node_cannot_replace_claim_source(context, node_type, tool):
    item = record(context, "profile_dataset", profile())
    node = context.evidence.add(type=node_type, tool_name=tool, arguments=item["arguments"], output=item["result"], summary="unrelated")
    item["evidence_reference"] = evidence_node_reference(node)
    catalog, coverage = checked_catalog(context)
    assert coverage["snapshot_verification"]["status"] == "pass"
    assert coverage["verified_coverage"]["numerator"] == 0
    assert "source_mismatch" in claim_for(catalog, "rows")["validation"]["reasons"]


def test_changed_source_result_fails_output_hash_even_if_claim_matches_it(context):
    item = record(context, "profile_dataset", profile())
    item["result"]["profile"]["rows"] = 999
    catalog, coverage = checked_catalog(context)
    assert claim_for(catalog, "rows")["value"] == 999
    assert "hash_mismatch" in claim_for(catalog, "rows")["validation"]["reasons"]
    assert coverage["verified_coverage"]["numerator"] == 0


@pytest.mark.parametrize("field,value", [("value", 999), ("value", True), ("display_value", "999"), ("dataset_id", "other-dataset")])
def test_changed_claim_value_display_or_dataset_is_rejected(context, field, value):
    record(context, "profile_dataset", profile(rows=1))
    catalog = collect_report_claims(context)
    claim_for(catalog, "rows")[field] = value
    catalog, coverage = checked_catalog(context, catalog)
    assert "source_mismatch" in claim_for(catalog, "rows")["validation"]["reasons"]
    assert coverage["verified_coverage"]["numerator"] == 4


def test_changed_arguments_fail_exact_argument_and_input_hash_checks(context):
    item = record(context, "profile_dataset", profile())
    item["arguments"] = {"dataset_id": "other-dataset"}
    catalog, coverage = checked_catalog(context)
    assert {"source_mismatch", "hash_mismatch"} <= set(claim_for(catalog, "rows")["validation"]["reasons"])
    assert coverage["verified_coverage"]["numerator"] == 0


@pytest.mark.parametrize("kind", ["chain_mismatch", "legacy", "invalid_json", "missing_file"])
def test_unassessable_snapshot_returns_null_verified_numerator_and_ratio(context, kind):
    record(context, "profile_dataset", profile())
    if kind == "chain_mismatch":
        raw = context.evidence.to_dict()
        raw["chain_hash"] = "0" * 64
        context.evidence.path.write_text(json.dumps(raw))
    elif kind == "legacy":
        context.evidence.path.write_text(json.dumps(context.evidence.to_list()))
    elif kind == "invalid_json":
        context.evidence.path.write_text("{")
    else:
        context.evidence.path.unlink()
    bundle = build_report_coverage(context)
    coverage = bundle["coverage"]
    assert coverage["reference_coverage"]["numerator"] == 5
    assert coverage["verified_coverage"] == {"status": "unassessable", "numerator": None, "denominator": 5, "ratio": None}
    assert all("snapshot_unassessable" in claim["validation"]["reasons"] for claim in bundle["catalog"]["claims"])


def test_zero_denominator_is_null_and_free_text_never_creates_claims(context):
    context.evidence.save()
    context.final_text = "数据共 100 行，规则准确率 99%，建议自动发布。"
    record(context, "summarize_findings", {"summary": context.final_text}, {})
    catalog, coverage = checked_catalog(context)
    assert catalog["claims"] == []
    assert coverage["reference_coverage"] == {"numerator": 0, "denominator": 0, "ratio": None}
    assert coverage["verified_coverage"] == {"status": "assessable", "numerator": 0, "denominator": 0, "ratio": None}


@pytest.mark.parametrize("conflict", [False, True])
def test_duplicate_claim_ids_reject_every_duplicate_and_report_conflicts(context, conflict):
    record(context, "profile_dataset", profile())
    catalog = collect_report_claims(context)
    duplicate = deepcopy(claim_for(catalog, "rows"))
    if conflict:
        duplicate["value"] = 99
    catalog["claims"].append(duplicate)
    catalog, coverage = checked_catalog(context, catalog)
    duplicates = [claim for claim in catalog["claims"] if claim["key"] == "rows"]
    assert all("duplicate_claim_id" in claim["validation"]["reasons"] for claim in duplicates)
    assert all(("conflicting_claim" in claim["validation"]["reasons"]) is conflict for claim in duplicates)
    assert coverage["reference_coverage"]["numerator"] == 4
    assert coverage["verified_coverage"]["numerator"] == 4
    assert coverage["verified_coverage"]["denominator"] == 6


def test_duplicate_source_ids_cannot_choose_one_valid_record(context):
    record(context, "profile_dataset", profile())
    catalog = collect_report_claims(context)
    catalog["source_calls"].append(deepcopy(catalog["source_calls"][0]))
    catalog, coverage = checked_catalog(context, catalog)
    assert coverage["verified_coverage"]["numerator"] == 0
    assert all("duplicate_source_id" in claim["validation"]["reasons"] for claim in catalog["claims"])


@pytest.mark.parametrize("change", ["missing_source", "missing_field", "unknown_operation", "unknown_aggregation", "no_sources"])
def test_unverifiable_or_missing_source_claims_are_explicit(context, change):
    record(context, "profile_dataset", profile())
    catalog = collect_report_claims(context)
    claim = claim_for(catalog, "rows")
    if change == "missing_source":
        claim["required_sources"][0]["call_id"] = "missing"
    elif change == "missing_field":
        claim["required_sources"][0]["result_path"] = ["profile", "absent"]
    elif change == "unknown_operation":
        claim["required_sources"][0]["operation"] = "guess_with_model"
    elif change == "unknown_aggregation":
        claim["aggregation"] = "invent_summary"
    else:
        claim["required_sources"] = []
    catalog, coverage = checked_catalog(context, catalog)
    reasons = claim_for(catalog, "rows")["validation"]["reasons"]
    assert "unverifiable" in reasons
    if change in {"missing_source", "no_sources"}:
        assert "source_missing" in reasons
    assert coverage["verified_coverage"]["numerator"] == 4


def test_missing_field_differs_from_explicit_null_without_dropping_claim(context):
    result = profile()
    result["profile"]["rows"] = None
    del result["profile"]["columns"]
    record(context, "profile_dataset", result)
    catalog, coverage = checked_catalog(context)
    assert claim_for(catalog, "rows")["value_available"] is True
    assert claim_for(catalog, "rows")["validation"]["verified"] is True
    assert claim_for(catalog, "columns")["display_value"] == "未提供"
    assert "unverifiable" in claim_for(catalog, "columns")["validation"]["reasons"]
    assert coverage["verified_coverage"]["denominator"] == 5


def test_rechecking_does_not_change_frozen_catalog_or_coverage(context):
    record(context, "profile_dataset", profile())
    catalog, coverage = checked_catalog(context)
    first_catalog, first_coverage = deepcopy(catalog), deepcopy(coverage)
    _, second_coverage = checked_catalog(context, catalog)
    assert catalog == first_catalog
    assert second_coverage == first_coverage
    assert hash_json(catalog["source_calls"][0]["result"]) == context.evidence.nodes[0].output_hash
