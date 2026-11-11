import hashlib
import json

import pytest
from fastapi.testclient import TestClient

from aircraft_agent.agent.runtime import RunStore
from aircraft_agent.interface import app as app_module


@pytest.fixture
def progress_run(tmp_path, monkeypatch):
    store = RunStore(tmp_path / "runs")
    context = store.create_run("查看数据分析进度 <script>alert(1)</script>")
    context.agent_plan = [{"step": 1, "action": "profile_dataset", "thought": "检查数据字段"}]
    context.toolbox_manifest = [{"name": "profile_dataset", "label": "数据画像"}]
    context.save_manifest()
    monkeypatch.setattr(app_module, "store", store)
    return TestClient(app_module.app), context


def file_hashes(folder):
    return {str(path.relative_to(folder)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in folder.rglob("*") if path.is_file()}


@pytest.mark.parametrize("status,label", [
    ("created", "等待开始"), ("running", "正在运行"),
    ("complete", "运行已完成"), ("failed", "运行失败"),
])
def test_progress_has_readable_status_and_structured_sections(progress_run, status, label):
    client, context = progress_run
    context.status = status
    context.error = "数据无法读取" if status == "failed" else None
    context.save_manifest()
    before = file_hashes(context.run_dir)
    response = client.get(f"/runs/{context.run_id}/progress")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert label in response.text
    assert "执行步骤" in response.text
    assert "活动记录" in response.text
    assert 'id="run-progress-data"' in response.text
    assert 'id="events"' not in response.text
    assert "textContent += ev.data" not in response.text
    assert file_hashes(context.run_dir) == before


def test_progress_bootstrap_contains_display_data_and_escapes_script(progress_run):
    client, context = progress_run
    context.agent_trace = [{"step": 1, "action": "profile_dataset", "status": "complete", "observation": "共有50个字段"}]
    context.save_manifest()
    page = client.get(f"/runs/{context.run_id}/progress").text
    data = page.split('<script id="run-progress-data" type="application/json">', 1)[1].split("</script>", 1)[0]
    state = json.loads(data)
    assert state["agent_plan"][0]["step"] == 1
    assert state["agent_trace"][0]["observation"] == "共有50个字段"
    assert state["toolbox_manifest"][0]["label"] == "数据画像"
    assert "tool_results" not in state
    assert "查看数据分析进度 <script>" not in page
    assert "\\u003cscript\\u003e" in data


def test_completed_progress_links_to_report_and_evidence(progress_run):
    client, context = progress_run
    context.status = "complete"
    context.report_html_path = context.artifacts_dir / "report.html"
    context.report_html_path.write_text("<h1>分析报告</h1>", encoding="utf-8")
    context.save_manifest()
    page = client.get(f"/runs/{context.run_id}/progress").text
    assert f'href="/runs/{context.run_id}"' in page
    assert "查看报告" in page
    assert f"/runs/{context.run_id}/evidence" in page


def test_progress_unknown_run_is_not_a_fake_loading_page(progress_run):
    client, _ = progress_run
    assert client.get("/runs/missing-run/progress").status_code == 404
