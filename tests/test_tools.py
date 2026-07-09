from pathlib import Path
import json

from agent_mvp.review_decisions import append_rule_mapping_decision
from agent_mvp.runtime import RunStore
from agent_mvp.tools import ToolExecutor


def make_context(tmp_path: Path):
    store = RunStore(tmp_path / "runtime")
    context = store.create_run("test goal")
    store.save_bytes(
        context,
        data=b"R001: temperature > 80\nR002: status == WARN\n",
        original_name="rules.md",
        kind="rule_library",
    )
    store.save_bytes(
        context,
        data=(
            "aircraft_type,mission_phase,temperature,vibration,status\n"
            "A,climb,82,6.8,WARN\n"
            "A,climb,85,7.4,WARN\n"
            "B,cruise,65,2.1,OK\n"
            "C,landing,68,2.9,OK\n"
        ).encode("utf-8"),
        original_name="data.csv",
        kind="dataset",
    )
    return context


def test_tool_flow_records_evidence(tmp_path: Path) -> None:
    context = make_context(tmp_path)
    executor = ToolExecutor(context)

    profile = executor.execute("profile_dataset", {})
    rules = executor.execute("parse_rule_library", {})
    rule_validation = executor.execute("validate_rule_library", {})
    schema = executor.execute("suggest_schema_metadata", {})
    schema_review = executor.execute("review_schema_metadata", {})
    schema_mapping = executor.execute("review_schema_mapping", {})
    adaptation = executor.execute("review_dataset_adaptation", {})
    mapping_plan = executor.execute("build_mapping_execution_plan", {})
    mapping_regression = executor.execute("validate_mapping_regression", {})
    applied = executor.execute("apply_rules", {})
    generalization = executor.execute("review_generalization_readiness", {"min_semantic_coverage": 0.0})
    mined = executor.execute("mine_association_rules", {"min_support": 0.25, "min_confidence": 0.5, "min_lift": 0.8})
    visualizations = executor.execute("build_visualizations", {"max_items": 5})
    validation = executor.execute("validate_analysis", {"required_tools": ["profile_dataset", "parse_rule_library", "apply_rules", "mine_association_rules"]})
    report = executor.execute("write_report", {"final_summary": "done"})

    assert profile["profile"]["rows"] == 4
    assert len(rules["rules"]) == 2
    assert rule_validation["rule_count"] == 2
    assert Path(rule_validation["markdown_path"]).exists()
    assert schema["field_count"] == 5
    assert Path(schema["schema_yaml_path"]).exists()
    assert schema["schema"]["fields"]["temperature"]["display_name"]
    assert Path(schema_review["markdown_path"]).exists()
    assert schema_mapping["field_count"] == 5
    assert Path(schema_mapping["markdown_path"]).exists()
    assert adaptation["ready_count"] == 2
    assert Path(adaptation["markdown_path"]).exists()
    assert mapping_plan["status"] == "passed"
    assert Path(mapping_plan["markdown_path"]).exists()
    assert mapping_regression["status"] == "passed"
    assert mapping_regression["no_mapping_count"] == 2
    assert Path(mapping_regression["markdown_path"]).exists()
    assert applied["rule_results"][0]["matched_rows"] == 2
    assert generalization["status"] == "passed"
    assert generalization["auto_diagnosis_allowed"] is True
    assert Path(generalization["markdown_path"]).exists()
    assert "fpgrowth" in mined["engine"]
    assert "rules" in mined
    assert len(mined["rules"]) <= 15
    assert mined["parameters"]["min_count"] == 2
    assert mined["warnings"]
    assert all(rule["matched_rows"] >= 2 for rule in mined["rules"])
    assert Path(visualizations["dashboard_html_path"]).exists()
    assert visualizations["chart_count"] >= 1
    assert validation["status"] == "passed"
    assert context.validation_results
    assert Path(report["report_html"]).exists()
    report_html = Path(report["report_html"]).read_text(encoding="utf-8")
    assert 'class="report-cover"' in report_html
    assert 'class="report-toc"' in report_html
    assert 'class="section-heading"' in report_html
    assert 'class="demo-report-main"' not in report_html
    assert "<code>write_report</code>" in report_html
    assert len(context.evidence.nodes) >= 6
    event_lines = context.event_path.read_text(encoding="utf-8").splitlines()
    events = [json.loads(line) for line in event_lines]
    event_types = [event["type"] for event in events]
    assert "run_created" in event_types
    assert "file_ingested" in event_types
    assert "tool_started" in event_types
    assert "tool_finished" in event_types


def test_demo_report_uses_demo_shell(tmp_path: Path) -> None:
    context = make_context(tmp_path)
    context.run_kind = "demo"
    executor = ToolExecutor(context)

    executor.execute("profile_dataset", {})
    executor.execute("parse_rule_library", {})
    executor.execute("apply_rules", {})
    executor.execute("mine_association_rules", {"min_support": 0.25, "min_confidence": 0.5, "min_lift": 0.8})
    executor.execute("build_visualizations", {"max_items": 5})
    executor.execute("validate_analysis", {"required_tools": ["profile_dataset", "parse_rule_library", "apply_rules"]})
    report = executor.execute("write_report", {"final_summary": "demo done"})

    report_html = Path(report["report_html"]).read_text(encoding="utf-8")
    assert 'class="demo-report-main"' in report_html
    assert 'class="demo-flow"' in report_html
    assert 'class="demo-jump-nav"' in report_html
    assert 'class="report-layout"' not in report_html


def test_validate_analysis_accepts_rule_library_alias(tmp_path: Path) -> None:
    context = make_context(tmp_path)
    executor = ToolExecutor(context)

    executor.execute("parse_rule_library", {})
    executor.execute("apply_rule_library", {})
    validation = executor.execute("validate_analysis", {"required_tools": ["parse_rule_library", "apply_rules"]})

    assert validation["missing_tools"] == []
    assert "Rule execution result is not available." not in validation["warnings"]


def test_generalization_readiness_passes_for_exact_executable_rules(tmp_path: Path) -> None:
    context = make_context(tmp_path)
    executor = ToolExecutor(context)

    executor.execute("profile_dataset", {})
    executor.execute("parse_rule_library", {})
    executor.execute("review_schema_mapping", {})
    executor.execute("review_dataset_adaptation", {})
    plan = executor.execute("build_mapping_execution_plan", {})
    regression = executor.execute("validate_mapping_regression", {})
    executor.execute("apply_rules", {})
    gate = executor.execute("review_generalization_readiness", {"min_semantic_coverage": 0.0})

    assert plan["status"] == "passed"
    assert plan["auto_execution_allowed"] is True
    assert plan["executable_without_mapping_count"] == 2
    assert regression["status"] == "passed"
    assert regression["no_mapping_count"] == 2
    assert gate["status"] == "passed"
    assert gate["auto_diagnosis_allowed"] is True
    assert gate["metrics"]["unresolved_rule_count"] == 0
    assert Path(gate["markdown_path"]).exists()


def test_numeric_rule_with_chinese_suffix_is_evaluated(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "runtime")
    context = store.create_run("suffix rule")
    store.save_bytes(
        context,
        data=b"R001: temperature > 80 \xe5\xbc\x82\xe5\xb8\xb8\n",
        original_name="rules.md",
        kind="rule_library",
    )
    store.save_bytes(
        context,
        data=b"temperature,status\n82,WARN\n70,OK\n90,WARN\n",
        original_name="data.csv",
        kind="dataset",
    )
    executor = ToolExecutor(context)

    rules = executor.execute("parse_rule_library", {})
    applied = executor.execute("apply_rules", {})

    assert rules["rules"][0]["condition"] == "temperature > 80"
    assert applied["rule_results"][0]["status"] == "evaluated"
    assert applied["rule_results"][0]["matched_rows"] == 2


def test_missing_and_outlier_tools(tmp_path: Path) -> None:
    context = make_context(tmp_path)
    executor = ToolExecutor(context)

    missing = executor.execute("check_missing_values", {})
    outliers = executor.execute("detect_outliers", {"method": "iqr"})

    assert missing["columns_with_missing"] == []
    assert outliers["method"] == "iqr"


def test_profile_and_rule_validation_detect_dataset_suffix_mismatch(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "runtime")
    context = store.create_run("analyze telemetry")
    store.save_bytes(
        context,
        data=b"R001: 1_battery_remaining_avg < 20\n",
        original_name="rules.md",
        kind="rule_library",
    )
    store.save_bytes(
        context,
        data=b"1_battery_remaining_last,label\n18,0\n25,0\n",
        original_name="all_features_last_last.csv",
        kind="dataset",
    )
    executor = ToolExecutor(context)

    profile = executor.execute("profile_dataset", {})
    executor.execute("parse_rule_library", {})
    validation = executor.execute("validate_rule_library", {})
    adaptation = executor.execute("review_dataset_adaptation", {})
    applied = executor.execute("apply_rules", {})
    analysis_validation = executor.execute(
        "validate_analysis",
        {"required_tools": ["profile_dataset", "parse_rule_library", "validate_rule_library", "apply_rules"]},
    )

    assert profile["dataset_kind_hint"] == "last_last"
    assert profile["profile"]["column_suffix_counts"]["_last"] == 1
    assert validation["suffix_mismatches"]
    assert validation["suffix_mismatches"][0]["rule_field"] == "1_battery_remaining_avg"
    assert validation["suffix_mismatches"][0]["dataset_field"] == "1_battery_remaining_last"
    assert "字段后缀不匹配" in validation["rules"][0]["issues"][0]
    assert adaptation["needs_mapping_count"] == 1
    assert adaptation["rules"][0]["status"] == "needs_mapping_confirmation"
    assert adaptation["rules"][0]["field_mappings"][0]["candidate_field"] == "1_battery_remaining_last"
    assert applied["rule_results"][0]["status"] == "unparseable"
    assert analysis_validation["status"] == "needs_review"
    assert "报告不得作为规则诊断结论" in " ".join(analysis_validation["warnings"])


def test_dataset_adaptation_suggests_semantic_candidate(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "runtime")
    context = store.create_run("customer schema")
    store.save_bytes(
        context,
        data=b"R001: 1_battery_remaining_avg < 20\n",
        original_name="rules.md",
        kind="rule_library",
    )
    store.save_bytes(
        context,
        data=b"main_battery_remaining,label\n18,0\n25,0\n",
        original_name="customer_schema.csv",
        kind="dataset",
    )
    executor = ToolExecutor(context)

    executor.execute("profile_dataset", {})
    executor.execute("parse_rule_library", {})
    adaptation = executor.execute("review_dataset_adaptation", {})
    executor.execute("apply_rules", {})
    gate = executor.execute("review_generalization_readiness", {"min_semantic_coverage": 0.0})

    mapping = adaptation["rules"][0]["field_mappings"][0]
    assert adaptation["needs_mapping_count"] == 1
    assert adaptation["not_applicable_count"] == 0
    assert mapping["status"] == "semantic_candidate"
    assert mapping["candidate_field"] == "main_battery_remaining"
    assert gate["status"] in {"needs_review", "blocked"}
    assert gate["auto_diagnosis_allowed"] is False


def test_schema_mapping_uses_semantic_alias_for_customer_field(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "runtime")
    context = store.create_run("customer semantic alias")
    store.save_bytes(
        context,
        data=b"R001: 1_battery_remaining_avg < 20\n",
        original_name="rules.md",
        kind="rule_library",
    )
    store.save_bytes(
        context,
        data=b"battery_pct,label\n18,0\n25,0\n",
        original_name="customer_schema.csv",
        kind="dataset",
    )
    executor = ToolExecutor(context)

    executor.execute("profile_dataset", {})
    executor.execute("parse_rule_library", {})
    executor.execute("suggest_schema_metadata", {})
    schema_mapping = executor.execute("review_schema_mapping", {})
    adaptation = executor.execute("review_dataset_adaptation", {})

    battery_field = next(item for item in schema_mapping["fields"] if item["field"] == "battery_pct")
    mapping = schema_mapping["rule_fields"][0]
    adaptation_mapping = adaptation["rules"][0]["field_mappings"][0]

    assert battery_field["semantic_role"] == "main_battery_remaining"
    assert mapping["selected_field"] == "battery_pct"
    assert schema_mapping["candidate_mapping_count"] == 1
    assert adaptation["needs_mapping_count"] == 1
    assert adaptation_mapping["candidate_field"] == "battery_pct"
    assert adaptation_mapping["match_method"] == "semantic_role"


def test_schema_mapping_review_assistant_classifies_candidates_conservatively(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "runtime")
    context = store.create_run("mapping review assistant")
    store.save_bytes(
        context,
        data=b"R001: 1_battery_remaining_avg < 20\n",
        original_name="rules.md",
        kind="rule_library",
    )
    store.save_bytes(
        context,
        data=b"main_battery_remaining_avg,label\n18,0\n25,0\n",
        original_name="customer_schema.csv",
        kind="dataset",
    )
    executor = ToolExecutor(context)

    executor.execute("profile_dataset", {})
    executor.execute("parse_rule_library", {})
    executor.execute("suggest_schema_metadata", {})
    executor.execute("review_schema_mapping", {})
    executor.execute("review_dataset_adaptation", {})
    review = executor.execute("review_schema_mapping_candidates", {})

    assert review["confirm_suggestion_count"] == 1
    assert review["manual_review_count"] == 0
    assert review["review_items"][0]["status"] == "suggest_confirm_candidate"
    assert Path(review["markdown_path"]).exists()


def test_schema_mapping_review_assistant_blocks_suffix_mismatch(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "runtime")
    context = store.create_run("mapping suffix mismatch review")
    store.save_bytes(
        context,
        data=b"R001: 1_battery_remaining_avg < 20\n",
        original_name="rules.md",
        kind="rule_library",
    )
    store.save_bytes(
        context,
        data=b"1_battery_remaining_last,label\n18,0\n25,0\n",
        original_name="customer_last.csv",
        kind="dataset",
    )
    executor = ToolExecutor(context)

    executor.execute("profile_dataset", {})
    executor.execute("parse_rule_library", {})
    executor.execute("suggest_schema_metadata", {})
    executor.execute("review_schema_mapping", {})
    executor.execute("review_dataset_adaptation", {})
    review = executor.execute("review_schema_mapping_candidates", {})

    item = review["review_items"][0]
    assert review["confirm_suggestion_count"] == 0
    assert review["manual_review_count"] == 1
    assert item["status"] == "manual_review_required"
    assert "统计口径" in " ".join(item["risk_flags"])


def test_schema_mapping_uses_confirmed_mapping_overrides(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "runtime")
    context = store.create_run("confirmed customer mapping")
    store.save_bytes(
        context,
        data=b"R001: 1_battery_remaining_avg < 20\n",
        original_name="rules.md",
        kind="rule_library",
    )
    store.save_bytes(
        context,
        data=b"battery_pct,label\n18,0\n25,0\n",
        original_name="customer_schema.csv",
        kind="dataset",
    )
    overrides_path = tmp_path / "schema_mapping_overrides.yaml"
    overrides_path.write_text(
        """
confirmed_mappings:
  battery_pct:
    semantic_role: main_battery_remaining
    display_name: 主电池剩余电量
    group: power
    unit: percent
    statistic: raw
    confirmed_by: unit_test
    note: battery_pct 已由专家确认。
""",
        encoding="utf-8",
    )
    executor = ToolExecutor(context)

    executor.execute("profile_dataset", {})
    executor.execute("parse_rule_library", {})
    executor.execute("suggest_schema_metadata", {})
    schema_mapping = executor.execute("review_schema_mapping", {"overrides_path": str(overrides_path)})
    adaptation = executor.execute("review_dataset_adaptation", {})
    plan = executor.execute("build_mapping_execution_plan", {})
    regression = executor.execute("validate_mapping_regression", {})

    battery_field = next(item for item in schema_mapping["fields"] if item["field"] == "battery_pct")
    mapping = schema_mapping["rule_fields"][0]
    adaptation_rule = adaptation["rules"][0]
    adaptation_mapping = adaptation_rule["field_mappings"][0]

    assert battery_field["confirmation_status"] == "confirmed"
    assert battery_field["match_method"] == "confirmed_override"
    assert schema_mapping["confirmed_mapping_count"] == 1
    assert schema_mapping["candidate_mapping_count"] == 0
    assert mapping["status"] == "confirmed_mapping"
    assert mapping["selected_field"] == "battery_pct"
    assert mapping["candidates"][0]["requires_confirmation"] is False
    assert adaptation["confirmed_mapping_count"] == 1
    assert adaptation["needs_mapping_count"] == 0
    assert adaptation_rule["status"] == "mapping_confirmed"
    assert adaptation_mapping["status"] == "confirmed_mapping"
    assert adaptation_mapping["requires_confirmation"] is False
    assert plan["status"] == "needs_review"
    assert plan["ready_for_regression_count"] == 0
    assert plan["needs_review_count"] == 1
    assert plan["rules"][0]["rewritten_condition_preview"] == "battery_pct < 20"
    assert "统计口径不一致" in " ".join(plan["rules"][0]["issues"])
    assert regression["status"] == "needs_review"
    assert regression["needs_review_count"] == 1


def test_schema_mapping_uses_run_review_confirmation(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "runtime")
    context = store.create_run("run-local human review")
    store.save_bytes(
        context,
        data=b"R001: 1_battery_remaining_avg < 20\n",
        original_name="rules.md",
        kind="rule_library",
    )
    store.save_bytes(
        context,
        data=b"battery_pct,label\n18,0\n25,0\n",
        original_name="customer_schema.csv",
        kind="dataset",
    )
    append_rule_mapping_decision(
        context.run_dir,
        action="confirm",
        rule_field="1_battery_remaining_avg",
        dataset_field="battery_pct",
        semantic_role="main_battery_remaining",
        display_name="主电池剩余电量",
        unit="percent",
        statistic="raw",
        note="本次 run 人工确认。",
    )
    executor = ToolExecutor(context)

    executor.execute("profile_dataset", {})
    executor.execute("parse_rule_library", {})
    schema_mapping = executor.execute("review_schema_mapping", {})
    adaptation = executor.execute("review_dataset_adaptation", {})
    plan = executor.execute("build_mapping_execution_plan", {})
    regression = executor.execute("validate_mapping_regression", {})

    mapping = schema_mapping["rule_fields"][0]
    assert schema_mapping["confirmed_mapping_count"] == 1
    assert schema_mapping["candidate_mapping_count"] == 0
    assert mapping["status"] == "confirmed_mapping"
    assert mapping["selected_field"] == "battery_pct"
    assert adaptation["confirmed_mapping_count"] == 1
    assert adaptation["needs_mapping_count"] == 0
    assert adaptation["rules"][0]["status"] == "mapping_confirmed"
    assert plan["rules"][0]["rewritten_condition_preview"] == "battery_pct < 20"
    assert regression["status"] == "needs_review"
    assert regression["needs_review_count"] == 1


def test_schema_mapping_respects_run_review_rejection(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "runtime")
    context = store.create_run("run-local rejection")
    store.save_bytes(
        context,
        data=b"R001: 1_battery_remaining_avg < 20\n",
        original_name="rules.md",
        kind="rule_library",
    )
    store.save_bytes(
        context,
        data=b"battery_pct,label\n18,0\n25,0\n",
        original_name="customer_schema.csv",
        kind="dataset",
    )
    append_rule_mapping_decision(
        context.run_dir,
        action="reject",
        rule_field="1_battery_remaining_avg",
        dataset_field="battery_pct",
        note="该客户字段口径不一致。",
    )
    executor = ToolExecutor(context)

    executor.execute("profile_dataset", {})
    executor.execute("parse_rule_library", {})
    schema_mapping = executor.execute("review_schema_mapping", {})
    adaptation = executor.execute("review_dataset_adaptation", {})

    assert schema_mapping["candidate_mapping_count"] == 0
    assert schema_mapping["missing_mapping_count"] == 1
    assert schema_mapping["rule_fields"][0]["selected_field"] is None
    assert adaptation["not_applicable_count"] == 1
