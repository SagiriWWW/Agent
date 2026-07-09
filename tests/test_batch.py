from pathlib import Path

from agent_mvp.batch import (
    build_batch_dataset_adaptation,
    build_batch_generalization_readiness,
    build_batch_hit_samples,
    build_batch_mapping_regression,
    build_batch_review_decisions,
    build_batch_rule_analysis,
    build_batch_schema_mapping_review,
    compare_batch_rule_versions,
    create_batch_job,
    run_batch_job,
)
from agent_mvp.config import Settings
from agent_mvp.demo_data import PROJECT_ROOT
from agent_mvp.review_decisions import append_rule_mapping_decision
from agent_mvp.utils import write_json


def test_batch_job_creates_runs_and_summary_report(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "datasets"
    dataset_dir.mkdir()
    rule_path = tmp_path / "rules.yaml"
    candidate_rule_path = tmp_path / "rules_v2.yaml"
    rule_path.write_text(
        "rules:\n"
        "  - rule_id: R001\n"
        "    name: 温度偏高\n"
        "    field: temperature\n"
        "    operator: '>'\n"
        "    value: 80\n"
        "    severity: high\n"
        "    recommendation: 检查温度趋势\n",
        encoding="utf-8",
    )
    candidate_rule_path.write_text(
        "rules:\n"
        "  - rule_id: R001\n"
        "    name: 温度偏高\n"
        "    field: temperature\n"
        "    operator: '>'\n"
        "    value: 90\n"
        "    severity: high\n"
        "    recommendation: 检查更高温度窗口\n",
        encoding="utf-8",
    )
    (dataset_dir / "flight_a.csv").write_text("temperature,status\n82,WARN\n70,OK\n", encoding="utf-8")
    (dataset_dir / "flight_b.csv").write_text("temperature,status\n75,OK\n91,WARN\n", encoding="utf-8")
    settings = Settings(
        storage_dir=tmp_path / "runtime",
        provider="local_demo",
        openai_api_key=None,
        openai_model="local",
        base_url=None,
        force_openai=False,
        max_tool_steps=10,
    )

    manifest = create_batch_job(settings, rule_path=rule_path, dataset_dir=dataset_dir, max_datasets=2)
    result = run_batch_job(settings, manifest["batch_id"])

    assert result["status"] == "complete"
    assert result["dataset_total"] == 2
    assert result["completed"] == 2
    assert result["passed"] == 2
    assert len(result["items"]) == 2
    assert all(item["rules_evaluated"] == 1 for item in result["items"])
    assert all(item["rule_results"] for item in result["items"])
    assert all(item["run_id"] for item in result["items"])
    assert Path(result["report_markdown_path"]).exists()
    assert Path(result["report_html_path"]).exists()
    report_text = Path(result["report_markdown_path"]).read_text(encoding="utf-8")
    assert "批量验收汇总报告" in report_text
    assert "数据适配汇总" in report_text
    assert "flight_a.csv" in report_text

    analysis = build_batch_rule_analysis(settings.storage_dir, result["batch_id"])
    adaptation = build_batch_dataset_adaptation(settings.storage_dir, result["batch_id"])
    mapping_review = build_batch_schema_mapping_review(settings.storage_dir, result["batch_id"])
    review_decisions = build_batch_review_decisions(settings.storage_dir, result["batch_id"])
    mapping_regression = build_batch_mapping_regression(settings.storage_dir, result["batch_id"])
    generalization = build_batch_generalization_readiness(settings.storage_dir, result["batch_id"])

    assert analysis["summary"]["rule_count"] == 1
    assert analysis["rules"][0]["rule_id"] == "R001"
    assert analysis["rules"][0]["evaluated_datasets"] == 2
    assert analysis["rules"][0]["matched_datasets"] == 2
    assert analysis["rules"][0]["quality_label"]
    refreshed = build_batch_rule_analysis(settings.storage_dir, result["batch_id"])
    assert Path(settings.storage_dir / "batches" / result["batch_id"] / "batch_rule_analysis.md").exists()
    assert refreshed["rules"][0]["avg_match_rate"] == analysis["rules"][0]["avg_match_rate"]
    assert adaptation["summary"]["datasets_with_adaptation"] == 2
    assert adaptation["summary"]["ready_rules"] == 2
    assert adaptation["rules"][0]["quality_label"] == "适配稳定"
    adaptation_path = settings.storage_dir / "batches" / result["batch_id"] / "batch_dataset_adaptation.md"
    assert adaptation_path.exists()
    assert "批量数据适配审查" in adaptation_path.read_text(encoding="utf-8")
    assert mapping_review["summary"]["datasets_with_review"] == 2
    assert mapping_review["summary"]["missing_review_count"] == 0
    assert mapping_review["summary"]["batch_review_status"] in {"resolved", "has_confirm_suggestions", "needs_review"}
    mapping_review_path = settings.storage_dir / "batches" / result["batch_id"] / "batch_schema_mapping_review.md"
    assert mapping_review_path.exists()
    assert "批量映射候选复核辅助" in mapping_review_path.read_text(encoding="utf-8")
    assert review_decisions["summary"]["batch_review_decision_status"] == "no_decisions"
    review_decisions_path = settings.storage_dir / "batches" / result["batch_id"] / "batch_review_decisions.md"
    assert review_decisions_path.exists()
    assert "批量人工复核意见汇总" in review_decisions_path.read_text(encoding="utf-8")
    assert mapping_regression["summary"]["datasets_with_validation"] == 2
    assert mapping_regression["summary"]["missing_validation_count"] == 0
    assert mapping_regression["summary"]["batch_regression_status"] == "passed"
    assert mapping_regression["datasets"][0]["mapping_regression_url"].startswith("/runs/")
    mapping_regression_path = settings.storage_dir / "batches" / result["batch_id"] / "batch_mapping_regression.md"
    assert mapping_regression_path.exists()
    assert "批量映射回归验证" in mapping_regression_path.read_text(encoding="utf-8")
    assert generalization["summary"]["datasets_with_gate"] == 2
    assert generalization["summary"]["missing_gate_count"] == 0
    assert generalization["summary"]["batch_gate_status"] in {"passed", "needs_review"}
    assert generalization["datasets"][0]["generalization_url"].startswith("/runs/")
    generalization_path = settings.storage_dir / "batches" / result["batch_id"] / "batch_generalization_readiness.md"
    assert generalization_path.exists()
    assert "批量泛化验收门禁" in generalization_path.read_text(encoding="utf-8")

    hit_samples = build_batch_hit_samples(
        settings.storage_dir,
        result["batch_id"],
        rule_ids=["R001"],
        datasets_per_rule=2,
        samples_per_dataset=2,
    )
    sample_rule = hit_samples["rules"][0]
    assert sample_rule["rule_id"] == "R001"
    assert sample_rule["selected_dataset_count"] == 2
    assert sample_rule["datasets"][0]["samples"]
    assert "temperature" in sample_rule["datasets"][0]["samples"][0]["values"]
    hit_sample_path = settings.storage_dir / "batches" / result["batch_id"] / "key_hit_samples.md"
    assert hit_sample_path.exists()
    hit_sample_text = hit_sample_path.read_text(encoding="utf-8")
    assert "关键命中样例导出" in hit_sample_text
    assert "temperature=" in hit_sample_text

    candidate_manifest = create_batch_job(settings, rule_path=candidate_rule_path, dataset_dir=dataset_dir, max_datasets=2)
    candidate_result = run_batch_job(settings, candidate_manifest["batch_id"])
    comparison = compare_batch_rule_versions(settings.storage_dir, result["batch_id"], candidate_result["batch_id"])

    assert comparison["summary"]["rule_count"] == 1
    assert comparison["summary"]["changed_rule_count"] == 1
    assert comparison["rules"][0]["rule_id"] == "R001"
    assert comparison["rules"][0]["rule_change"]["changes"]["value"] == {"from": 80, "to": 90}
    assert comparison["rules"][0]["delta"]["avg_match_rate"] < 0
    comparison_path = settings.storage_dir / "batches" / candidate_result["batch_id"] / f"rule_version_comparison_{result['batch_id'][:8]}.md"
    assert comparison_path.exists()
    assert "规则版本对比报告" in comparison_path.read_text(encoding="utf-8")


def test_customer_variant_batch_marks_mapping_review_instead_of_forcing_rules(tmp_path: Path) -> None:
    settings = Settings(
        storage_dir=tmp_path / "runtime",
        provider="local_demo",
        openai_api_key=None,
        openai_model="local",
        base_url=None,
        force_openai=False,
        max_tool_steps=10,
    )
    dataset_dir = PROJECT_ROOT / "dataset" / "customer_variants"
    rule_path = PROJECT_ROOT / "examples" / "rules_v2.yaml"

    manifest = create_batch_job(settings, rule_path=rule_path, dataset_dir=dataset_dir, max_datasets=3)
    result = run_batch_job(settings, manifest["batch_id"])
    adaptation = build_batch_dataset_adaptation(settings.storage_dir, result["batch_id"])
    mapping_review = build_batch_schema_mapping_review(settings.storage_dir, result["batch_id"])
    mapping_regression = build_batch_mapping_regression(settings.storage_dir, result["batch_id"])
    generalization = build_batch_generalization_readiness(settings.storage_dir, result["batch_id"])

    assert result["status"] == "complete"
    assert result["dataset_total"] == 3
    assert result["completed"] == 3
    assert result["passed"] == 0
    assert result["needs_review"] == 3
    assert result["dataset_dir"].endswith("dataset/customer_variants")
    assert adaptation["summary"]["datasets_with_adaptation"] == 3
    assert adaptation["summary"]["needs_mapping_rules"] > 0
    assert adaptation["summary"]["not_applicable_rules"] > 0
    assert adaptation["summary"]["ready_rules"] == 0
    assert any(rule["quality_label"] in {"需映射确认", "存在不适用"} for rule in adaptation["rules"])
    assert mapping_review["summary"]["datasets_with_review"] == 3
    assert mapping_review["summary"]["batch_review_status"] == "needs_review"
    assert mapping_review["summary"]["manual_review_count"] > 0
    assert any(item["quality_label"] in {"需专家复核", "不适用"} for item in mapping_review["mappings"])
    assert mapping_regression["summary"]["datasets_with_validation"] == 3
    assert mapping_regression["summary"]["batch_regression_status"] in {"needs_review", "blocked"}
    assert (
        mapping_regression["summary"]["needs_confirmation_rule_count"]
        + mapping_regression["summary"]["blocked_or_failed_rule_count"]
        > 0
    )
    assert generalization["summary"]["datasets_with_gate"] == 3
    assert generalization["summary"]["auto_diagnosis_allowed_count"] == 0
    assert generalization["summary"]["batch_gate_status"] == "needs_review"
    assert all(item["status"] == "needs_review" for item in generalization["datasets"])


def test_batch_review_decisions_aggregate_run_local_human_feedback(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "datasets"
    dataset_dir.mkdir()
    rule_path = tmp_path / "rules.yaml"
    rule_path.write_text(
        "rules:\n"
        "  - rule_id: R001\n"
        "    name: 电量偏低\n"
        "    field: battery_avg\n"
        "    operator: '<'\n"
        "    value: 20\n"
        "    severity: high\n"
        "    recommendation: 检查电池\n",
        encoding="utf-8",
    )
    (dataset_dir / "customer_a.csv").write_text("battery_pct\n18\n", encoding="utf-8")
    (dataset_dir / "customer_b.csv").write_text("battery_pct\n19\n", encoding="utf-8")
    settings = Settings(
        storage_dir=tmp_path / "runtime",
        provider="local_demo",
        openai_api_key=None,
        openai_model="local",
        base_url=None,
        force_openai=False,
        max_tool_steps=10,
    )

    manifest = create_batch_job(settings, rule_path=rule_path, dataset_dir=dataset_dir, max_datasets=2)
    result = run_batch_job(settings, manifest["batch_id"])
    for item in result["items"]:
        run_dir = settings.storage_dir / item["run_id"]
        append_rule_mapping_decision(
            run_dir,
            action="confirm",
            rule_field="battery_avg",
            dataset_field="battery_pct",
            semantic_role="main_battery_remaining",
            display_name="主电池剩余电量",
            unit="percent",
            statistic="raw",
            note="批内人工确认。",
        )
        write_json(
            run_dir / "artifacts" / "mapping_regression_validation.json",
            {
                "status": "passed",
                "mapped_execution_validated": True,
                "regression_passed_count": 1,
                "rules": [
                    {
                        "rule_id": "R001",
                        "name": "电量偏低",
                        "status": "regression_passed",
                        "rewrite_pairs": [{"rule_field": "battery_avg", "dataset_field": "battery_pct"}],
                        "issues": [],
                        "matched_rows": 1,
                        "match_rate": 1.0,
                    }
                ],
            },
        )

    review = build_batch_review_decisions(settings.storage_dir, result["batch_id"])
    mapping = review["mappings"][0]

    assert review["summary"]["datasets_with_decisions"] == 2
    assert review["summary"]["ready_for_promotion_review_count"] == 1
    assert review["summary"]["batch_review_decision_status"] == "has_promotion_review_candidates"
    assert mapping["status"] == "ready_for_promotion_review"
    assert mapping["confirm_count"] == 2
    assert mapping["regression_passed_count"] == 2
    assert mapping["confirmed_distinct_dataset_count"] == 2
    assert "不自动推广到全局配置" in review["anti_overfit_principle"]
