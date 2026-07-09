from pathlib import Path

from agent_mvp.review_decisions import append_rule_mapping_decision
from agent_mvp.review_promotions import build_review_promotion_report
from agent_mvp.runtime import RunStore
from agent_mvp.utils import read_json, write_json


def _create_reviewed_run(store: RunStore, goal: str, dataset_body: bytes):
    context = store.create_run(goal)
    store.save_bytes(
        context,
        data=dataset_body,
        original_name=f"{goal}.csv",
        kind="dataset",
    )
    return context


def _write_mapping_regression(context, *, rule_field: str, dataset_field: str, status: str = "regression_passed") -> None:
    write_json(
        context.run_dir / "artifacts" / "mapping_regression_validation.json",
        {
            "status": "passed" if status == "regression_passed" else "needs_review",
            "mapped_execution_validated": status == "regression_passed",
            "regression_passed_count": 1 if status == "regression_passed" else 0,
            "needs_review_count": 0 if status == "regression_passed" else 1,
            "rules": [
                {
                    "rule_id": "R001",
                    "name": "测试规则",
                    "status": status,
                    "rewrite_pairs": [{"rule_field": rule_field, "dataset_field": dataset_field}],
                    "issues": [] if status == "regression_passed" else ["测试中的回归复核项"],
                    "matched_rows": 1,
                    "match_rate": 1.0,
                }
            ],
        },
    )


def test_review_promotion_report_aggregates_cross_dataset_evidence(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "runtime")
    first = _create_reviewed_run(store, "customer_a", b"battery_pct,temp_c\n18,60\n")
    second = _create_reviewed_run(store, "customer_b", b"battery_pct,temp_c\n22,62\n")
    same_a = _create_reviewed_run(store, "repeat_a_1", b"altitude_ft\n1000\n")
    same_b = _create_reviewed_run(store, "repeat_a_2", b"altitude_ft\n1000\n")
    missing_regression_a = _create_reviewed_run(store, "missing_regression_a", b"speed_kn\n42\n")
    missing_regression_b = _create_reviewed_run(store, "missing_regression_b", b"speed_kn\n43\n")
    conflict_confirm = _create_reviewed_run(store, "conflict_confirm", b"temperature_c\n80\n")
    conflict_reject = _create_reviewed_run(store, "conflict_reject", b"temperature_c\n82\n")

    for context in [first, second]:
        append_rule_mapping_decision(
            context.run_dir,
            action="confirm",
            rule_field="1_battery_remaining_avg",
            dataset_field="battery_pct",
            semantic_role="main_battery_remaining",
            display_name="主电池剩余电量",
            unit="percent",
            statistic="raw",
            note="跨客户样本确认。",
        )
        _write_mapping_regression(context, rule_field="1_battery_remaining_avg", dataset_field="battery_pct")
    for context in [same_a, same_b]:
        append_rule_mapping_decision(
            context.run_dir,
            action="confirm",
            rule_field="altitude_avg",
            dataset_field="altitude_ft",
            semantic_role="altitude",
            display_name="飞行高度",
            unit="feet",
            statistic="raw",
            note="同一数据重复确认。",
        )
        _write_mapping_regression(context, rule_field="altitude_avg", dataset_field="altitude_ft")
    for context in [missing_regression_a, missing_regression_b]:
        append_rule_mapping_decision(
            context.run_dir,
            action="confirm",
            rule_field="speed_avg",
            dataset_field="speed_kn",
            semantic_role="ground_speed",
            display_name="地速",
            unit="knot",
            statistic="raw",
            note="有人工确认但尚未完成映射回归验证。",
        )
    append_rule_mapping_decision(
        conflict_confirm.run_dir,
        action="confirm",
        rule_field="temperature_avg",
        dataset_field="temperature_c",
        semantic_role="temperature",
        display_name="温度",
        unit="celsius",
        statistic="raw",
    )
    append_rule_mapping_decision(
        conflict_reject.run_dir,
        action="reject",
        rule_field="temperature_avg",
        dataset_field="temperature_c",
        semantic_role="temperature",
        display_name="温度",
        unit="celsius",
        statistic="raw",
        note="统计窗口不一致。",
    )

    report = build_review_promotion_report(store.storage_dir, min_confirmations=2, min_distinct_datasets=2)
    candidates = {item["rule_field"]: item for item in report["candidates"]}

    assert report["summary"]["ready_for_global_review_count"] == 1
    assert report["summary"]["needs_regression_evidence_count"] == 1
    assert candidates["1_battery_remaining_avg"]["status"] == "ready_for_global_review"
    assert candidates["1_battery_remaining_avg"]["confirmed_distinct_dataset_count"] == 2
    assert candidates["1_battery_remaining_avg"]["regression_passed_distinct_dataset_count"] == 2
    assert candidates["altitude_avg"]["status"] == "needs_more_evidence"
    assert candidates["altitude_avg"]["confirm_count"] == 2
    assert candidates["altitude_avg"]["confirmed_distinct_dataset_count"] == 1
    assert candidates["speed_avg"]["status"] == "needs_regression_evidence"
    assert candidates["speed_avg"]["missing_regression_count"] == 2
    assert candidates["temperature_avg"]["status"] == "conflict_review"
    assert "1_battery_remaining_avg" in report["promotion_yaml_preview"]
    assert "altitude_avg" not in report["promotion_yaml_preview"]
    assert "speed_avg" not in report["promotion_yaml_preview"]

    for path in report["paths"].values():
        assert Path(path).exists()
    persisted = read_json(Path(report["paths"]["json"]))
    assert persisted["summary"]["ready_for_global_review_count"] == 1
