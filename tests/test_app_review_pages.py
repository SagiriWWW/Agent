import json
from dataclasses import replace

from fastapi.testclient import TestClient

from agent_mvp import app as app_module
from agent_mvp.review_decisions import append_rule_mapping_decision
from agent_mvp.runtime import RunStore
from agent_mvp.utils import read_json, write_json


def test_review_pages_and_artifact_allowlist(tmp_path, monkeypatch) -> None:
    store = RunStore(tmp_path / "runtime")
    context = store.create_run("review pages")
    context.status = "complete"
    context.save_manifest()
    artifacts = context.run_dir / "artifacts"
    (artifacts / "rule_validation.json").write_text(
        json.dumps(
            {
                "rule_count": 1,
                "ready_count": 1,
                "needs_review_count": 0,
                "disabled_count": 0,
                "rules": [
                    {
                        "rule_id": "R001",
                        "name": "温度偏高",
                        "status": "ready",
                        "severity": "high",
                        "target_fields": ["temperature"],
                        "condition": "temperature > 80",
                        "recommendation": "检查温控系统",
                        "issues": [],
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (artifacts / "schema_review.json").write_text(
        json.dumps(
            {
                "field_count": 2,
                "low_confidence_count": 0,
                "ignored_count": 1,
                "rule_bound_count": 1,
                "review_items": [
                    {
                        "type": "ignored_field",
                        "field": "label",
                        "message": "字段被标记为忽略，需确认是否不参与分析。",
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (artifacts / "schema_mapping.json").write_text(
        json.dumps(
            {
                "source_file": "data.csv",
                "field_count": 2,
                "semantic_field_count": 1,
                "low_confidence_count": 0,
                "unmapped_field_count": 0,
                "required_field_count": 1,
                "exact_mapping_count": 0,
                "candidate_mapping_count": 1,
                "missing_mapping_count": 0,
                "ambiguous_mapping_count": 0,
                "fields": [
                    {
                        "field": "battery_pct",
                        "semantic_role": "main_battery_remaining",
                        "display_name": "主电池剩余电量",
                        "group": "power",
                        "unit": "percent",
                        "statistic": "raw",
                        "confidence": 0.98,
                        "reason": "来自稳定语义角色配置。",
                    }
                ],
                "rule_fields": [
                    {
                        "rule_field": "1_battery_remaining_avg",
                        "semantic_role": "main_battery_remaining",
                        "display_name": "主电池剩余电量均值",
                        "unit": "percent",
                        "statistic": "avg",
                        "status": "semantic_candidate",
                        "selected_field": "battery_pct",
                        "candidate_count": 1,
                        "ambiguous": False,
                        "candidates": [
                            {
                                "field": "battery_pct",
                                "score": 0.9,
                                "match_method": "semantic_role",
                                "unit": "percent",
                                "statistic": "raw",
                            }
                        ],
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (artifacts / "dataset_adaptation.json").write_text(
        json.dumps(
            {
                "source_file": "data.csv",
                "dataset_kind_hint": "unknown",
                "field_count": 2,
                "rule_count": 1,
                "ready_count": 0,
                "needs_mapping_count": 1,
                "not_applicable_count": 0,
                "needs_review_count": 0,
                "disabled_count": 0,
                "rules": [
                    {
                        "rule_id": "R001",
                        "name": "电量偏低",
                        "status": "needs_mapping_confirmation",
                        "field_mappings": [
                            {
                                "rule_field": "1_battery_remaining_avg",
                                "candidate_field": "1_battery_remaining_last",
                                "status": "suffix_mismatch",
                                "score": 0.85,
                            }
                        ],
                        "condition": "1_battery_remaining_avg < 20",
                        "recommendation": "先确认字段映射语义一致，再决定是否允许自动改写或执行该规则。",
                        "issues": ["存在非精确字段映射候选，需要人工确认后再执行。"],
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (artifacts / "schema_mapping_review.json").write_text(
        json.dumps(
            {
                "source_file": "data.csv",
                "status": "needs_human_review",
                "anti_overfit_principle": "只生成复核建议，不自动确认字段映射。",
                "confirm_suggestion_count": 0,
                "manual_review_count": 1,
                "not_applicable_count": 0,
                "already_safe_count": 0,
                "review_items": [
                    {
                        "rule_id": "R001",
                        "rule_name": "电量偏低",
                        "rule_field": "1_battery_remaining_avg",
                        "dataset_field": "1_battery_remaining_last",
                        "status": "manual_review_required",
                        "score": 0.85,
                        "match_method": "suffix_mismatch",
                        "rule_unit": "percent",
                        "candidate_unit": "percent",
                        "rule_statistic": "avg",
                        "candidate_statistic": "last",
                        "risk_flags": ["规则字段和候选字段存在统计口径后缀差异"],
                        "recommendation": "先确认 _avg/_last 统计口径是否业务等价；未确认前不要点击确认。",
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (artifacts / "mapping_execution_plan.json").write_text(
        json.dumps(
            {
                "status": "needs_review",
                "auto_execution_allowed": False,
                "anti_overfit_principle": "映射执行计划只描述字段重写预案，不自动修改规则条件、阈值或诊断结论。",
                "rule_count": 1,
                "executable_without_mapping_count": 0,
                "ready_for_regression_count": 0,
                "needs_confirmation_count": 1,
                "needs_review_count": 0,
                "blocked_count": 0,
                "regression_required_count": 0,
                "rules": [
                    {
                        "rule_id": "R001",
                        "name": "电量偏低",
                        "status": "needs_mapping_confirmation",
                        "execution_allowed_now": False,
                        "requires_regression_validation": False,
                        "condition": "1_battery_remaining_avg < 20",
                        "rewritten_condition_preview": "1_battery_remaining_avg < 20",
                        "rewrite_pairs": [],
                        "issues": ["字段映射仍是候选，不能进入执行计划。"],
                        "recommendation": "仍有候选映射未确认，先完成人工确认后再生成执行计划。",
                    }
                ],
                "field_plans": [
                    {
                        "rule_id": "R001",
                        "rule_field": "1_battery_remaining_avg",
                        "dataset_field": "1_battery_remaining_last",
                        "status": "needs_confirmation",
                        "rewrite_action": "wait_for_confirmation",
                        "unit_status": "unchecked",
                        "statistic_status": "unchecked",
                        "issues": ["字段映射仍是候选，不能进入执行计划。"],
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (artifacts / "mapping_regression_validation.json").write_text(
        json.dumps(
            {
                "status": "needs_review",
                "mapped_execution_validated": False,
                "anti_overfit_principle": "映射回归验证只证明重写预案可审查、可执行；不证明阈值业务正确。",
                "rule_count": 1,
                "passed_count": 0,
                "no_mapping_count": 0,
                "regression_passed_count": 0,
                "needs_confirmation_count": 1,
                "needs_review_count": 0,
                "blocked_count": 0,
                "failed_count": 0,
                "rules": [
                    {
                        "rule_id": "R001",
                        "name": "电量偏低",
                        "status": "needs_confirmation",
                        "condition": "1_battery_remaining_avg < 20",
                        "validated_condition": "1_battery_remaining_avg < 20",
                        "rewrite_pairs": [],
                        "matched_rows": 0,
                        "match_rate": 0,
                        "issues": ["仍有字段候选映射未确认，不能做回归验证。"],
                        "recommendation": "先在 Schema 语义映射页确认或驳回候选映射。",
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (artifacts / "generalization_readiness.json").write_text(
        json.dumps(
            {
                "status": "needs_review",
                "auto_diagnosis_allowed": False,
                "anti_overfit_principle": "不得为了让当前数据集通过而改阈值、改规则或把候选映射直接当作已确认映射。",
                "verdict": "系统已识别泛化风险并阻止直接自动诊断。",
                "metrics": {
                    "semantic_coverage": 0.5,
                    "field_count": 2,
                    "semantic_field_count": 1,
                    "rule_count": 1,
                    "ready_count": 0,
                    "confirmed_mapping_count": 0,
                    "needs_mapping_count": 1,
                    "not_applicable_count": 0,
                    "unresolved_rule_count": 1,
                    "unresolved_rule_ratio": 1.0,
                    "evaluated_rule_count": 0,
                    "unparseable_rule_count": 1,
                },
                "checks": [
                    {
                        "id": "candidate_mapping_requires_confirmation",
                        "name": "候选映射保持人工确认状态",
                        "status": "needs_review",
                        "message": "仍有规则只有候选映射，不能通过增加当前数据集专属别名来伪装成可执行。",
                        "evidence": "needs_mapping_count=1",
                    }
                ],
                "recommendations": ["把候选映射交给人工或第二分析 Agent 复核。"],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(app_module, "store", store)
    monkeypatch.setattr(app_module, "settings", replace(app_module.settings, storage_dir=store.storage_dir))
    client = TestClient(app_module.app)

    dashboard = client.get("/")
    demo_ui = client.get("/demo-ui")
    new_run_ui = client.get("/runs/new")
    batch_ui = client.get("/batch-ui")
    toolbox_ui = client.get("/toolbox-ui")
    mvp_doc = client.get("/docs/mvp")
    mvp_doc_markdown = client.get("/docs/mvp.md")
    rule_page = client.get(f"/runs/{context.run_id}/rule-validation")
    schema_page = client.get(f"/runs/{context.run_id}/schema-review")
    schema_mapping_page = client.get(f"/runs/{context.run_id}/schema-mapping")
    schema_mapping_review_page = client.get(f"/runs/{context.run_id}/schema-mapping-review")
    adaptation_page = client.get(f"/runs/{context.run_id}/dataset-adaptation")
    mapping_execution_page = client.get(f"/runs/{context.run_id}/mapping-execution-plan")
    mapping_regression_page = client.get(f"/runs/{context.run_id}/mapping-regression")
    generalization_page = client.get(f"/runs/{context.run_id}/generalization-readiness")
    progress_page = client.get(f"/runs/{context.run_id}/progress")
    events_response = client.get(f"/api/runs/{context.run_id}/events")
    run_json = client.get(f"/api/runs/{context.run_id}")
    artifact = client.get(f"/runs/{context.run_id}/artifacts/rule_validation.json")
    denied = client.get(f"/runs/{context.run_id}/artifacts/manifest.json")

    assert dashboard.status_code == 200
    assert "飞行遥测智能分析平台" in dashboard.text
    assert "/demo-ui" in dashboard.text
    assert "/runs/new" in dashboard.text
    assert demo_ui.status_code == 200
    assert "标准演示场景" in demo_ui.text
    assert 'action="/demo"' in demo_ui.text
    assert new_run_ui.status_code == 200
    assert "客户数据分析" in new_run_ui.text
    assert 'action="/runs"' in new_run_ui.text
    assert 'name="model_mode"' in new_run_ui.text
    assert 'name="model_provider"' in new_run_ui.text
    assert 'name="model_name"' in new_run_ui.text
    assert "外部模型优先" in new_run_ui.text
    assert 'accept=".md,.yaml,.yml' in new_run_ui.text
    assert 'accept=".csv,.xlsx' in new_run_ui.text
    assert batch_ui.status_code == 200
    assert "批量验收" in batch_ui.text
    assert 'action="/batches"' in batch_ui.text
    assert 'name="dataset_family"' in batch_ui.text
    assert 'value="customer_variants"' in batch_ui.text
    assert 'name="dataset_dir"' in batch_ui.text
    assert toolbox_ui.status_code == 200
    assert "分析能力中心" in toolbox_ui.text
    assert mvp_doc.status_code == 200
    assert "MVP 交付说明" in mvp_doc.text
    assert "/docs/mvp.md" in mvp_doc.text
    assert mvp_doc_markdown.status_code == 200
    assert "Agent 工具箱 MVP 验收说明" in mvp_doc_markdown.text
    assert rule_page.status_code == 200
    assert "规则库校验" in rule_page.text
    assert "检查温控系统" in rule_page.text
    assert schema_page.status_code == 200
    assert "Schema 复核" in schema_page.text
    assert "忽略字段" in schema_page.text
    assert schema_mapping_page.status_code == 200
    assert "Schema 语义映射" in schema_mapping_page.text
    assert "battery_pct" in schema_mapping_page.text
    assert schema_mapping_review_page.status_code == 200
    assert "映射候选复核辅助" in schema_mapping_review_page.text
    assert "需专家复核" in schema_mapping_review_page.text
    assert "人工复核意见" in schema_mapping_review_page.text
    assert f'action="/runs/{context.run_id}/schema-mapping/decisions"' in schema_mapping_review_page.text
    assert 'name="return_to" value="schema_mapping_review"' in schema_mapping_review_page.text
    assert adaptation_page.status_code == 200
    assert "数据适配审查" in adaptation_page.text
    assert "需映射确认" in adaptation_page.text
    assert mapping_execution_page.status_code == 200
    assert "映射执行计划" in mapping_execution_page.text
    assert "字段重写预案" in mapping_execution_page.text
    assert mapping_regression_page.status_code == 200
    assert "映射回归验证" in mapping_regression_page.text
    assert "需映射确认" in mapping_regression_page.text
    assert generalization_page.status_code == 200
    assert "泛化验收门禁" in generalization_page.text
    assert "候选映射保持人工确认状态" in generalization_page.text
    assert progress_page.status_code == 200
    assert "Agent 运行进度" in progress_page.text
    assert f"/api/runs/{context.run_id}" in progress_page.text
    assert "EventSource" in progress_page.text
    assert "/api/runs/${runId}/events" in progress_page.text
    assert events_response.status_code == 200
    assert "event: run_event" in events_response.text
    assert "run_created" in events_response.text
    assert run_json.status_code == 200
    assert run_json.json()["links"]["events"] == f"/api/runs/{context.run_id}/events"
    assert run_json.json()["links"]["schema_mapping"] == f"/runs/{context.run_id}/schema-mapping"
    assert run_json.json()["links"]["schema_mapping_review"] == f"/runs/{context.run_id}/schema-mapping-review"
    assert run_json.json()["links"]["dataset_adaptation"] == f"/runs/{context.run_id}/dataset-adaptation"
    assert run_json.json()["links"]["mapping_execution_plan"] == f"/runs/{context.run_id}/mapping-execution-plan"
    assert run_json.json()["links"]["mapping_regression_validation"] == f"/runs/{context.run_id}/mapping-regression"
    assert run_json.json()["links"]["generalization_readiness"] == f"/runs/{context.run_id}/generalization-readiness"
    assert run_json.json()["event_count"] >= 1
    assert artifact.status_code == 200
    assert denied.status_code == 404


def test_web_upload_rejects_unsupported_file_types(tmp_path, monkeypatch) -> None:
    store = RunStore(tmp_path / "runtime")
    monkeypatch.setattr(app_module, "store", store)
    monkeypatch.setattr(app_module, "settings", replace(app_module.settings, storage_dir=store.storage_dir))
    client = TestClient(app_module.app)

    invalid_rule = client.post(
        "/runs",
        data={"goal": "bad rule upload"},
        files=[
            ("rule_file", ("rules.txt", b"R001: temperature > 80\n", "text/plain")),
            ("dataset_files", ("data.csv", b"temperature\n82\n", "text/csv")),
        ],
        follow_redirects=False,
    )
    invalid_rule_id = invalid_rule.headers["location"].strip("/").split("/")[1]
    invalid_rule_manifest = read_json(store.get_run_dir(invalid_rule_id) / "manifest.json")

    invalid_dataset = client.post(
        "/runs",
        data={
            "goal": "bad dataset upload",
            "model_mode": "model_required",
            "model_provider": "bailian",
            "model_name": "qwen-plus",
        },
        files=[
            ("rule_file", ("rules.yaml", b"rules: []\n", "application/yaml")),
            ("dataset_files", ("data.xls", b"not really excel", "application/vnd.ms-excel")),
        ],
        follow_redirects=False,
    )
    invalid_dataset_id = invalid_dataset.headers["location"].strip("/").split("/")[1]
    invalid_dataset_manifest = read_json(store.get_run_dir(invalid_dataset_id) / "manifest.json")

    assert invalid_rule.status_code == 303
    assert invalid_rule_manifest["status"] == "failed"
    assert "规则库文件格式不支持" in invalid_rule_manifest["error"]
    assert invalid_dataset.status_code == 303
    assert invalid_dataset_manifest["status"] == "failed"
    assert "数据集文件格式不支持" in invalid_dataset_manifest["error"]
    assert invalid_dataset_manifest["model_config"]["mode"] == "model_required"
    assert invalid_dataset_manifest["model_config"]["provider"] == "bailian"
    assert invalid_dataset_manifest["model_config"]["model"] == "qwen-plus"
    assert invalid_dataset_manifest["model_config"]["force_model_api"] is True


def test_schema_mapping_decision_post_refreshes_artifacts(tmp_path, monkeypatch) -> None:
    store = RunStore(tmp_path / "runtime")
    context = store.create_run("review decision refresh")
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
    monkeypatch.setattr(app_module, "store", store)
    monkeypatch.setattr(app_module, "settings", replace(app_module.settings, storage_dir=store.storage_dir))
    client = TestClient(app_module.app)

    response = client.post(
        f"/runs/{context.run_id}/schema-mapping/decisions",
        data={
            "action": "confirm",
            "rule_field": "1_battery_remaining_avg",
            "dataset_field": "battery_pct",
            "semantic_role": "main_battery_remaining",
            "display_name": "主电池剩余电量",
            "unit": "percent",
            "statistic": "raw",
            "note": "本次 run 人工确认。",
            "return_to": "schema_mapping_review",
        },
        follow_redirects=False,
    )

    assert response.status_code == 303
    assert response.headers["location"] == f"/runs/{context.run_id}/schema-mapping-review"
    decisions = read_json(store.get_run_dir(context.run_id) / "review_decisions.json")
    schema_mapping = read_json(store.get_run_dir(context.run_id) / "artifacts" / "schema_mapping.json")
    adaptation = read_json(store.get_run_dir(context.run_id) / "artifacts" / "dataset_adaptation.json")
    page = client.get(f"/runs/{context.run_id}/schema-mapping")

    assert decisions["confirmed_rule_mappings"]["1_battery_remaining_avg"]["dataset_field"] == "battery_pct"
    assert schema_mapping["confirmed_mapping_count"] == 1
    assert schema_mapping["rule_fields"][0]["status"] == "confirmed_mapping"
    assert adaptation["confirmed_mapping_count"] == 1
    assert page.status_code == 200
    assert "人工复核意见" in page.text
    assert "本次 run 人工确认" in page.text


def test_review_promotions_page_and_api(tmp_path, monkeypatch) -> None:
    store = RunStore(tmp_path / "runtime")
    contexts = []
    for idx, value in enumerate([b"battery_pct\n18\n", b"battery_pct\n22\n"], 1):
        context = store.create_run(f"promotion run {idx}")
        store.save_bytes(context, data=value, original_name=f"customer_{idx}.csv", kind="dataset")
        append_rule_mapping_decision(
            context.run_dir,
            action="confirm",
            rule_field="1_battery_remaining_avg",
            dataset_field="battery_pct",
            semantic_role="main_battery_remaining",
            display_name="主电池剩余电量",
            unit="percent",
            statistic="raw",
            note="跨数据确认。",
        )
        write_json(
            context.run_dir / "artifacts" / "mapping_regression_validation.json",
            {
                "status": "passed",
                "mapped_execution_validated": True,
                "regression_passed_count": 1,
                "rules": [
                    {
                        "rule_id": "R001",
                        "name": "主电池剩余电量检查",
                        "status": "regression_passed",
                        "rewrite_pairs": [
                            {"rule_field": "1_battery_remaining_avg", "dataset_field": "battery_pct"}
                        ],
                        "issues": [],
                        "matched_rows": 1,
                        "match_rate": 1.0,
                    }
                ],
            },
        )
        contexts.append(context)
    monkeypatch.setattr(app_module, "store", store)
    monkeypatch.setattr(app_module, "settings", replace(app_module.settings, storage_dir=store.storage_dir))
    client = TestClient(app_module.app)

    page = client.get("/review-promotions")
    api = client.get("/api/review-promotions")

    assert page.status_code == 200
    assert "复核意见沉淀" in page.text
    assert "可进入全局审核" in page.text
    assert "battery_pct" in page.text
    assert api.status_code == 200
    assert api.json()["summary"]["ready_for_global_review_count"] == 1
    assert api.json()["candidates"][0]["status"] == "ready_for_global_review"
