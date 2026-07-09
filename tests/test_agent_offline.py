from pathlib import Path

from agent_mvp.agent import AgentRunner
from agent_mvp.config import Settings
from agent_mvp.runtime import RunStore
from agent_mvp.tools import ToolExecutor


def test_offline_agent_creates_report_and_evidence(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "runtime")
    context = store.create_run("run all MVP steps")
    store.save_bytes(context, data=b"R001: 29_temperature_avg > 80\n", original_name="rules.md", kind="rule_library")
    dataset = "\n".join(
        [
            "29_temperature_avg,1_battery_remaining_avg,status",
            "82,24,WARN",
            "83,23,WARN",
            "84,22,WARN",
            "85,21,WARN",
            "86,20,WARN",
            "87,19,WARN",
            "88,18,WARN",
            "89,17,WARN",
            "90,16,WARN",
            "91,15,WARN",
            "68,78,OK",
            "65,82,OK",
        ]
    )
    store.save_bytes(
        context,
        data=dataset.encode("utf-8"),
        original_name="data.csv",
        kind="dataset",
    )
    settings = Settings(
        storage_dir=tmp_path / "runtime",
        provider="openai",
        openai_api_key=None,
        openai_model="gpt-5.5",
        base_url=None,
        force_openai=False,
        max_tool_steps=5,
    )

    result = AgentRunner(context, ToolExecutor(context), settings).run()

    assert result["mode"] == "offline"
    assert result["plan_steps"] == 21
    assert context.agent_plan
    assert [step["action"] for step in context.agent_plan] == [
        "profile_dataset",
        "parse_rule_library",
        "validate_rule_library",
        "suggest_schema_metadata",
        "review_schema_metadata",
        "review_schema_mapping",
        "review_dataset_adaptation",
        "review_schema_mapping_candidates",
        "build_mapping_execution_plan",
        "validate_mapping_regression",
        "check_missing_values",
        "detect_outliers",
        "apply_rules",
        "review_generalization_readiness",
        "check_association_backend",
        "mine_association_rules",
        "compute_correlations",
        "summarize_findings",
        "validate_analysis",
        "build_visualizations",
        "write_report",
    ]
    assert context.agent_trace
    assert context.validation_results
    assert context.validation_results[-1]["status"] == "passed"
    assert not context.validation_results[-1]["missing_tools"]
    assert context.report_html_path and context.report_html_path.exists()
    assert context.evidence.path.exists()
    artifacts_dir = context.run_dir / "artifacts"
    assert (artifacts_dir / "schema_mapping_review.json").exists()
    assert (artifacts_dir / "schema_mapping_review.md").exists()
    assert (artifacts_dir / "mapping_execution_plan.json").exists()
    assert (artifacts_dir / "mapping_execution_plan.md").exists()
    assert (artifacts_dir / "mapping_regression_validation.json").exists()
    assert (artifacts_dir / "mapping_regression_validation.md").exists()
    assert (artifacts_dir / "generalization_readiness.json").exists()
    assert (artifacts_dir / "generalization_readiness.md").exists()
    assert len(context.evidence.nodes) >= 5
