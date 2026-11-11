from __future__ import annotations

from pathlib import Path

from aircraft_agent.agent.runtime import RunStore
from aircraft_agent.common.utils import write_json
from aircraft_agent.evidence.chain import (
    EVIDENCE_SCHEMA_VERSION,
    EvidenceStore,
    compute_chain_hash,
    compute_node_hash,
    load_evidence_chain,
    verify_chain,
)
from aircraft_agent.telemetry.reporting import build_markdown_report


def _node(
    node_id: str,
    *,
    parent_ids: list[str] | None = None,
    parent_hashes: list[str] | None = None,
    summary: str = "done",
) -> dict:
    node = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "node_id": node_id,
        "parent_ids": parent_ids or [],
        "parent_hashes": parent_hashes or [],
        "type": "tool_call",
        "tool_name": "unit_tool",
        "input_hash": "input",
        "output_hash": "output",
        "arguments": {"b": 2, "a": {"y": 2, "x": 1}},
        "summary": summary,
        "timestamp": "2026-07-10T00:00:00Z",
        "model_response_id": None,
        "node_hash": "",
    }
    node["node_hash"] = compute_node_hash(node)
    return node


def test_v2_node_hash_is_stable_and_excludes_self_hash() -> None:
    node = _node("n1")
    reordered = {
        "node_hash": "ignored-different-value",
        "model_response_id": node["model_response_id"],
        "timestamp": node["timestamp"],
        "summary": node["summary"],
        "arguments": {"a": {"x": 1, "y": 2}, "b": 2},
        "output_hash": node["output_hash"],
        "input_hash": node["input_hash"],
        "tool_name": node["tool_name"],
        "type": node["type"],
        "parent_hashes": node["parent_hashes"],
        "parent_ids": node["parent_ids"],
        "node_id": node["node_id"],
        "schema_version": node["schema_version"],
    }

    assert compute_node_hash(node) == compute_node_hash(reordered)

    changed = dict(node)
    changed["summary"] = "tampered"
    assert compute_node_hash(node) != compute_node_hash(changed)


def test_v2_chain_hash_is_stable_for_same_nodes() -> None:
    first = _node("n1")
    second = _node("n2", parent_ids=["n1"], parent_hashes=[first["node_hash"]])
    nodes = [first, second]

    assert compute_chain_hash(nodes) == compute_chain_hash([dict(first), dict(second)])


def test_verify_chain_detects_first_tampered_middle_node(tmp_path: Path) -> None:
    store = EvidenceStore(tmp_path)
    store.add(type="file_ingest", tool_name=None, arguments={"file": "rules.md"}, output={"ok": True}, summary="ingest")
    store.add(type="tool_call", tool_name="parse", arguments={"file": "rules.md"}, output={"rules": 1}, summary="parse")
    store.add(type="tool_call", tool_name="report", arguments={}, output={"path": "report.html"}, summary="report")

    chain = load_evidence_chain(store.path)
    assert chain["schema_version"] == EVIDENCE_SCHEMA_VERSION
    assert chain["chain_hash"] == store.chain_hash()
    assert chain["nodes"][1]["parent_hashes"] == [chain["nodes"][0]["node_hash"]]
    assert verify_chain(chain).valid is True

    chain["nodes"][1]["summary"] = "tampered middle node"
    result = verify_chain(chain)

    assert result.verified is True
    assert result.valid is False
    assert result.first_mismatch_index == 1
    assert result.first_mismatch_node_id == chain["nodes"][1]["node_id"]
    assert result.actual_chain_hash != result.expected_chain_hash
    assert result.errors[0]["kind"] == "node_hash_mismatch"


def test_legacy_evidence_chain_can_be_read_without_v2_fields(tmp_path: Path) -> None:
    legacy_node = {
        "node_id": "legacy-1",
        "parent_ids": [],
        "type": "tool_call",
        "tool_name": "legacy_tool",
        "input_hash": "input",
        "output_hash": "output",
        "arguments": {"x": 1},
        "summary": "legacy",
        "timestamp": "2026-07-10T00:00:00Z",
        "model_response_id": None,
    }
    path = tmp_path / "evidence_chain.json"
    write_json(path, [legacy_node])

    chain = load_evidence_chain(path)
    assert chain["schema_version"] == 1
    assert chain["chain_hash"] is None
    assert chain["nodes"] == [legacy_node]

    result = verify_chain(path)
    assert result.valid is True
    assert result.verified is False
    assert result.node_count == 1

    store = EvidenceStore(tmp_path)
    assert len(store.nodes) == 1
    assert store.nodes[0].schema_version == 1


def test_report_shows_evidence_verification_failure_location(tmp_path: Path) -> None:
    run_store = RunStore(tmp_path / "runtime")
    context = run_store.create_run("show verification failure")
    context.evidence.add(
        type="tool_call",
        tool_name="unit_tool",
        arguments={"x": 1},
        output={"ok": True},
        summary="unit tool completed",
    )
    chain = load_evidence_chain(context.evidence.path)
    chain["nodes"][0]["summary"] = "tampered"
    write_json(context.evidence.path, chain)

    report = build_markdown_report(context)

    assert "verify_chain(): 失败" in report
    assert f"node_id=`{chain['nodes'][0]['node_id']}`" in report
    assert "node_hash_mismatch" in report
