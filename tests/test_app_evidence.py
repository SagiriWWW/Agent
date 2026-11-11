from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from aircraft_agent.agent.runtime import RunStore
from aircraft_agent.interface import app as app_module


def _snapshot(root: Path) -> dict:
    """Include directory entries, file bytes and modification times."""
    return {
        str(path.relative_to(root)): (
            path.is_dir(), path.stat().st_mtime_ns,
            None if path.is_dir() else path.read_bytes(),
        )
        for path in [root, *sorted(root.rglob("*"))]
    }


@pytest.fixture
def evidence_web(tmp_path, monkeypatch):
    run_store = RunStore(tmp_path / "runs")
    context = run_store.create_run("read-only evidence verification")
    for number in range(3):
        context.evidence.add(
            type="tool_call", tool_name="unit_tool", arguments={"number": number},
            output={"ok": True}, summary=f"node {number}",
        )
    report = context.run_dir / "artifacts" / "report.html"
    report.write_text("<!doctype html><html><head><title>Report</title></head><body><main>原始报告</main></body></html>", encoding="utf-8")
    context.report_html_path = report
    context.status = "complete"
    context.save_manifest()
    (context.run_dir / "uploads" / "original.csv").write_bytes(b"time,value\n1,2\n")
    monkeypatch.setattr(app_module, "store", run_store)
    monkeypatch.setattr(app_module, "settings", replace(app_module.settings, storage_dir=run_store.storage_dir))

    def forbidden(*args, **kwargs):
        pytest.fail("verification must not create or rehydrate a run, or initialize an Agent/EvidenceStore")

    monkeypatch.setattr(run_store, "create_run", forbidden)
    monkeypatch.setattr(app_module, "_rehydrate_run_context", forbidden)
    monkeypatch.setattr(app_module, "AgentRunner", forbidden)
    monkeypatch.setattr(app_module, "EvidenceStore", forbidden)
    return TestClient(app_module.app), context, run_store.storage_dir


def _urls(run_id: str) -> tuple[str, str]:
    return f"/api/runs/{run_id}/evidence/verify", f"/runs/{run_id}/evidence"


def test_evidence_valid_snapshot_and_existing_run_navigation_are_read_only(evidence_web):
    client, context, storage_dir = evidence_web
    before = _snapshot(storage_dir)
    api_url, page_url = _urls(context.run_id)
    response = client.get(api_url)
    result = response.json()
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    assert result["status"] == "pass"
    assert result["valid"] is True and result["verified"] is True
    assert result["schema_version"] == 2
    assert result["node_count"] == 3
    assert result["expected_chain_hash"] == result["actual_chain_hash"]
    assert result["first_mismatch_index"] is None
    assert result["first_mismatch_node_id"] is None
    assert result["index_base"] == 0

    page = client.get(page_url)
    assert page.status_code == 200
    assert page.headers["cache-control"] == "no-store"
    for content in (
        "PASS · 验证通过", "重新校验当前快照", "schema_version", "node_count",
        "valid", "verified", "expected_chain_hash", "actual_chain_hash",
        "first_mismatch_node_id", "first_mismatch_index", "从 0 开始",
        "name='viewport'", result["actual_chain_hash"], result["scope"],
    ):
        assert content in page.text
    for url in (f"/runs/{context.run_id}", f"/runs/{context.run_id}/progress"):
        navigation = client.get(url)
        assert navigation.status_code == 200
        assert f"href='{page_url}'" in navigation.text
    assert "原始报告" in client.get(f"/runs/{context.run_id}").text
    links = client.get(f"/api/runs/{context.run_id}").json()["links"]
    assert links["evidence_page"] == page_url
    assert links["evidence_verification"] == api_url
    assert _snapshot(storage_dir) == before


def test_evidence_without_report_still_has_reachable_navigation(evidence_web):
    client, context, storage_dir = evidence_web
    manifest_path = context.run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["report_html_path"] = None
    manifest_path.write_text(json.dumps(manifest))
    before = _snapshot(storage_dir)
    response = client.get(f"/runs/{context.run_id}")
    assert response.status_code == 200
    assert f"href='/runs/{context.run_id}/evidence'" in response.text
    assert _snapshot(storage_dir) == before


def test_evidence_rechecks_tampered_temporary_snapshot_and_ignores_manifest(evidence_web):
    client, context, storage_dir = evidence_web
    api_url, page_url = _urls(context.run_id)
    assert client.get(api_url).json()["status"] == "pass"
    chain = json.loads(context.evidence.path.read_text())
    chain["nodes"][1]["summary"] = "tampered temporary fixture"
    context.evidence.path.write_text(json.dumps(chain))
    before = _snapshot(storage_dir)
    result = client.get(api_url).json()
    assert result["status"] == "mismatch"
    assert result["valid"] is False and result["verified"] is True
    assert result["first_mismatch_index"] == 1
    assert result["first_mismatch_node_id"] == chain["nodes"][1]["node_id"]
    assert result["errors"][0]["kind"] == "node_hash_mismatch"
    assert result["expected_chain_hash"] != result["actual_chain_hash"]
    page = client.get(page_url)
    assert page.status_code == 200
    assert "FAIL · 完整性失配" in page.text
    assert "node_hash_mismatch" in page.text
    assert chain["nodes"][1]["node_id"] in page.text
    assert _snapshot(storage_dir) == before

    # Even an unreadable/stale manifest cannot replace the evidence result.
    (context.run_dir / "manifest.json").write_text("not JSON")
    before = _snapshot(storage_dir)
    assert client.get(api_url).json()["status"] == "mismatch"
    assert client.get(page_url).status_code == 200
    assert _snapshot(storage_dir) == before


def test_evidence_chain_level_mismatch_has_no_invented_node_position(evidence_web):
    client, context, storage_dir = evidence_web
    chain = json.loads(context.evidence.path.read_text())
    chain["chain_hash"] = "0" * 64
    context.evidence.path.write_text(json.dumps(chain))
    before = _snapshot(storage_dir)
    api_url, page_url = _urls(context.run_id)
    result = client.get(api_url).json()
    assert result["status"] == "mismatch"
    assert result["first_mismatch_index"] is None
    assert result["first_mismatch_node_id"] is None
    assert [error["kind"] for error in result["errors"]] == ["chain_hash_mismatch"]
    page = client.get(page_url).text
    assert "chain_hash_mismatch" in page
    assert "无可定位节点" in page
    assert "无可定位索引" in page
    assert _snapshot(storage_dir) == before


def test_evidence_missing_hash_is_not_repaired_by_loading(evidence_web):
    client, context, storage_dir = evidence_web
    chain = json.loads(context.evidence.path.read_text())
    del chain["nodes"][1]["node_hash"]
    context.evidence.path.write_text(json.dumps(chain))
    before = _snapshot(storage_dir)
    result = client.get(_urls(context.run_id)[0]).json()
    assert result["status"] == "mismatch"
    assert result["valid"] is False
    assert result["first_mismatch_index"] == 1
    assert result["errors"][0]["kind"] == "missing_node_hash"
    assert _snapshot(storage_dir) == before


@pytest.mark.parametrize("envelope", [False, True])
def test_evidence_legacy_is_readable_but_never_shown_as_verified(evidence_web, envelope):
    client, context, storage_dir = evidence_web
    nodes = json.loads(context.evidence.path.read_text())["nodes"]
    for node in nodes:
        for name in ("schema_version", "parent_hashes", "node_hash"):
            node.pop(name)
    raw = {"schema_version": 1, "evidence": nodes} if envelope else nodes
    context.evidence.path.write_text(json.dumps(raw))
    before = _snapshot(storage_dir)
    api_url, page_url = _urls(context.run_id)
    response = client.get(api_url)
    result = response.json()
    assert response.status_code == 200
    assert result["status"] == "legacy"
    assert result["schema_version"] == 1 and result["node_count"] == 3
    assert result["valid"] is True and result["verified"] is False
    page = client.get(page_url)
    assert page.status_code == 200
    assert "旧版可读 · 未验证" in page.text
    assert "PASS" not in page.text and "验证通过" not in page.text
    assert _snapshot(storage_dir) == before


@pytest.mark.parametrize(
    ("raw", "status"),
    [
        ('{"schema_version": 99, "nodes": []}', "unsupported_version"),
        ('{"schema_version": 2, "nodes": "wrong"}', "invalid_structure"),
        ('{"schema_version": 2, "nodes": [null]}', "invalid_structure"),
        ('{"schema_version": 2, "nodes": [{"schema_version": "wrong"}]}', "invalid_structure"),
        ("{}", "invalid_structure"),
        ("null", "invalid_structure"),
        ("not JSON", "invalid_json"),
    ],
)
def test_evidence_invalid_inputs_have_explicit_http_status_and_no_writes(evidence_web, raw, status):
    client, context, storage_dir = evidence_web
    context.evidence.path.write_text(raw)
    before = _snapshot(storage_dir)
    api_url, page_url = _urls(context.run_id)
    response = client.get(api_url)
    assert response.status_code == 422
    result = response.json()
    assert result["status"] == status
    assert result["valid"] is False and result["verified"] is False
    page = client.get(page_url)
    assert page.status_code == 422
    assert page.headers["cache-control"] == "no-store"
    assert "PASS" not in page.text and "验证通过" not in page.text
    assert _snapshot(storage_dir) == before


@pytest.mark.parametrize(
    "raw",
    [
        r'{"schema_version": "\ud800", "nodes": []}',
        r'{"schema_version": 1, "nodes": [], "chain_hash": "\ud800"}',
        r'{"\ud800": 1, "\ud800": 2}',
    ],
    ids=["schema-version", "legacy-chain-hash", "duplicate-key"],
)
def test_evidence_lone_unicode_surrogates_return_invalid_json_on_both_surfaces(evidence_web, raw):
    client, context, storage_dir = evidence_web
    context.evidence.path.write_text(raw, encoding="utf-8")
    before = _snapshot(storage_dir)
    api_url, page_url = _urls(context.run_id)
    response = client.get(api_url)
    assert response.status_code == 422
    assert response.json()["status"] == "invalid_json"
    assert response.json()["valid"] is False
    assert response.json()["verified"] is False
    page = client.get(page_url)
    assert page.status_code == 422
    assert "非法 JSON · 未验证" in page.text
    assert "invalid_json" in page.text
    assert _snapshot(storage_dir) == before


@pytest.mark.parametrize("missing_run", [False, True])
def test_evidence_missing_file_does_not_create_run_or_artifacts(evidence_web, missing_run):
    client, context, storage_dir = evidence_web
    context.evidence.path.unlink()
    before = _snapshot(storage_dir)
    api_url, page_url = _urls("never-created" if missing_run else context.run_id)
    response = client.get(api_url)
    assert response.status_code == 404
    assert response.json()["status"] == "file_not_found"
    page = client.get(page_url)
    assert page.status_code == 404
    assert "证据文件不存在" in page.text
    assert _snapshot(storage_dir) == before


def test_evidence_unreadable_file_type_returns_http_500_without_writes(evidence_web):
    client, context, storage_dir = evidence_web
    context.evidence.path.unlink()
    context.evidence.path.mkdir()
    before = _snapshot(storage_dir)
    api_url, page_url = _urls(context.run_id)
    response = client.get(api_url)
    assert response.status_code == 500
    assert response.headers["cache-control"] == "no-store"
    assert response.json()["status"] == "read_error"
    assert response.json()["valid"] is False
    assert response.json()["verified"] is False
    page = client.get(page_url)
    assert page.status_code == 500
    assert "读取失败 · 未验证" in page.text
    assert "PASS" not in page.text
    assert _snapshot(storage_dir) == before


@pytest.mark.parametrize("run_id", ["%2e%2e", "%5Coutside", "bad%20id", "a" * 129])
def test_evidence_rejects_invalid_run_path_boundaries(evidence_web, run_id):
    client, _, storage_dir = evidence_web
    before = _snapshot(storage_dir)
    api_url, page_url = _urls(run_id)
    response = client.get(api_url)
    assert response.status_code == 403
    assert response.json()["status"] == "path_rejected"
    assert client.get(page_url).status_code == 403
    assert _snapshot(storage_dir) == before


@pytest.mark.parametrize("target_inside", [False, True])
@pytest.mark.parametrize("link_kind", ["run", "file"])
def test_evidence_rejects_symlinks_even_to_another_configured_run(evidence_web, target_inside, link_kind, tmp_path):
    client, context, storage_dir = evidence_web
    target = storage_dir / "other-run" if target_inside else tmp_path / "outside"
    target.mkdir()
    target_file = target / "evidence_chain.json"
    target_file.write_bytes(context.evidence.path.read_bytes())
    external_before = _snapshot(target)
    if link_kind == "run":
        run_id = "linked-run"
        (storage_dir / run_id).symlink_to(target, target_is_directory=True)
    else:
        run_id = context.run_id
        context.evidence.path.unlink()
        context.evidence.path.symlink_to(target_file)
    before = _snapshot(storage_dir)
    api_url, page_url = _urls(run_id)
    response = client.get(api_url)
    assert response.status_code == 403
    assert response.json()["status"] == "path_rejected"
    assert client.get(page_url).status_code == 403
    assert _snapshot(storage_dir) == before
    assert _snapshot(target) == external_before


def test_evidence_page_escapes_untrusted_mismatch_node_and_error_data(evidence_web):
    client, context, _ = evidence_web
    chain = json.loads(context.evidence.path.read_text())
    malicious_id = '<script>alert("node")</script>'
    chain["nodes"][0]["node_id"] = malicious_id
    context.evidence.path.write_text(json.dumps(chain))
    response = client.get(_urls(context.run_id)[1])
    assert response.status_code == 200
    assert malicious_id not in response.text
    assert "&lt;script&gt;" in response.text
    assert "node_hash_mismatch" in response.text
