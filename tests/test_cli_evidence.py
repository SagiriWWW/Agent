from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from aircraft_agent.evidence.chain import compute_chain_hash, compute_node_hash
from aircraft_agent.evidence.verification import EXIT_CODES
from aircraft_agent.interface import cli


def _chain() -> dict:
    nodes = []
    for index in range(3):
        node = {
            "schema_version": 2,
            "node_id": f"node-{index}",
            "parent_ids": [nodes[-1]["node_id"]] if nodes else [],
            "parent_hashes": [nodes[-1]["node_hash"]] if nodes else [],
            "type": "tool_call",
            "tool_name": "test_tool",
            "input_hash": "input",
            "output_hash": "output",
            "arguments": {"index": index},
            "summary": "test evidence",
            "timestamp": "2026-09-05T00:00:00Z",
            "model_response_id": None,
        }
        node["node_hash"] = compute_node_hash(node)
        nodes.append(node)
    return {"schema_version": 2, "chain_hash": compute_chain_hash(nodes), "nodes": nodes}


def _write(path: Path, raw: object) -> None:
    path.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")


def _run(monkeypatch: pytest.MonkeyPatch, *args: str) -> int:
    monkeypatch.setattr(sys, "argv", ["aircraft-agent", *args])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    return exc.value.code


@pytest.fixture(autouse=True)
def _forbid_execution(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*args, **kwargs):
        pytest.fail("verification must not configure or execute an Agent run")

    for name in ("get_settings", "RunStore", "AgentRunner", "ToolExecutor", "create_batch_job", "run_batch_job"):
        monkeypatch.setattr(cli, name, fail)


def test_cli_valid_json_is_read_only_and_needs_no_analysis_data(tmp_path, monkeypatch, capsys):
    run_dir = tmp_path / "existing-run"
    run_dir.mkdir()
    path = run_dir / "evidence_chain.json"
    raw = _chain()
    _write(path, raw)
    for name in ("manifest.json", "metadata.json", "events.jsonl", "report.html"):
        (run_dir / name).write_text(f"original {name}", encoding="utf-8")
    before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in run_dir.iterdir()}

    assert _run(monkeypatch, "--verify-evidence", str(path), "--json") == 0

    output = capsys.readouterr()
    result = json.loads(output.out)
    assert output.err == ""
    assert result["status"] == "pass"
    assert result["valid"] is True and result["verified"] is True
    assert result["schema_version"] == 2
    assert result["node_count"] == 3
    assert result["expected_chain_hash"] == raw["chain_hash"] == result["actual_chain_hash"]
    assert result["first_mismatch_index"] is None
    assert result["first_mismatch_node_id"] is None
    assert result["index_base"] == 0
    assert result["scope"]
    assert {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in run_dir.iterdir()} == before


def test_cli_human_output_and_verify_alias(tmp_path, monkeypatch, capsys):
    path = tmp_path / "evidence_chain.json"
    raw = _chain()
    raw["nodes"][1]["summary"] = "tampered temporary copy"
    _write(path, raw)

    assert _run(monkeypatch, "--verify", str(path)) == 1

    output = capsys.readouterr().out
    assert "校验失配" in output
    assert "版本: 2" in output
    assert "节点数: 3" in output
    assert "valid: false" in output and "verified: true" in output
    assert "预期 chain_hash:" in output and "实际 chain_hash:" in output
    assert "首个失配节点 ID: node-1" in output
    assert "首个失配节点索引（从 0 开始）: 1" in output
    assert "node_hash_mismatch" in output
    assert "校验边界:" in output


def test_cli_chain_level_failure_has_no_invented_node(tmp_path, monkeypatch, capsys):
    path = tmp_path / "evidence_chain.json"
    raw = _chain()
    raw["chain_hash"] = "0" * 64
    _write(path, raw)

    assert _run(monkeypatch, "--verify-evidence", str(path), "--json") == 1

    result = json.loads(capsys.readouterr().out)
    assert result["errors"][0]["kind"] == "chain_hash_mismatch"
    assert result["first_mismatch_index"] is None
    assert result["first_mismatch_node_id"] is None


def test_cli_missing_node_hash_is_not_repaired(tmp_path, monkeypatch, capsys):
    path = tmp_path / "evidence_chain.json"
    raw = _chain()
    del raw["nodes"][1]["node_hash"]
    _write(path, raw)
    before = (path.read_bytes(), path.stat().st_mtime_ns)

    assert _run(monkeypatch, "--verify-evidence", str(path), "--json") == 1

    result = json.loads(capsys.readouterr().out)
    assert result["errors"][0]["kind"] == "missing_node_hash"
    assert result["first_mismatch_index"] == 1
    assert (path.read_bytes(), path.stat().st_mtime_ns) == before


def test_cli_legacy_is_readable_but_never_success(tmp_path, monkeypatch, capsys):
    path = tmp_path / "evidence_chain.json"
    legacy = _chain()["nodes"]
    for node in legacy:
        for key in ("schema_version", "node_hash", "parent_hashes"):
            del node[key]
    _write(path, legacy)

    assert _run(monkeypatch, "--verify-evidence", str(path), "--json") == 3
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "legacy"
    assert result["valid"] is True and result["verified"] is False
    assert result["schema_version"] == 1 and result["node_count"] == 3
    assert _run(monkeypatch, "--verify-evidence", str(path)) == 3
    human = capsys.readouterr().out
    assert "旧版可读，但不可验证" in human
    assert "验证通过" not in human


@pytest.mark.parametrize(
    ("case", "status", "exit_code"),
    [
        ("missing", "file_not_found", 4),
        ("json", "invalid_json", 5),
        ("version", "unsupported_version", 6),
        ("structure", "invalid_structure", 7),
        ("directory", "read_error", 8),
    ],
)
def test_cli_input_failures_are_distinct(tmp_path, monkeypatch, capsys, case, status, exit_code):
    path = tmp_path / "evidence_chain.json"
    if case == "json":
        path.write_text("{invalid", encoding="utf-8")
    elif case == "version":
        _write(path, {**_chain(), "schema_version": 99})
    elif case == "structure":
        _write(path, {"schema_version": 2, "nodes": "invalid"})
    elif case == "directory":
        path.mkdir()

    assert _run(monkeypatch, "--verify-evidence", str(path), "--json") == exit_code

    result = json.loads(capsys.readouterr().out)
    assert result["status"] == status
    assert result["valid"] is False and result["verified"] is False
    assert result["errors"]


@pytest.mark.parametrize(
    "payload",
    [
        b'{"schema_version":"\\ud800","nodes":[]}',
        b'{"schema_version":1,"chain_hash":"\\ud800","nodes":[]}',
        b'{"schema_version":1,"nodes":[],"\\ud800":1,"\\ud800":2}',
    ],
)
def test_cli_isolated_unicode_surrogates_return_safe_json_error(tmp_path, monkeypatch, capsys, payload):
    path = tmp_path / "evidence_chain.json"
    path.write_bytes(payload)

    assert _run(monkeypatch, "--verify-evidence", str(path), "--json") == 5

    output = capsys.readouterr()
    output.out.encode("utf-8")
    result = json.loads(output.out)
    assert result["status"] == "invalid_json"
    assert result["valid"] is False and result["verified"] is False
    assert result["errors"][0]["kind"] == "invalid_json"
    assert output.err == ""
    assert path.read_bytes() == payload


@pytest.mark.parametrize(
    ("status", "exit_code"),
    [
        ("pass", 0),
        ("mismatch", 1),
        ("legacy", 3),
        ("file_not_found", 4),
        ("invalid_json", 5),
        ("unsupported_version", 6),
        ("invalid_structure", 7),
        ("read_error", 8),
        ("path_rejected", 9),
    ],
)
def test_cli_exit_code_contract(monkeypatch, capsys, status, exit_code):
    result = {"status": status, "valid": status in {"pass", "legacy"}, "verified": status in {"pass", "mismatch"}}
    monkeypatch.setattr(cli, "verify_evidence_file", lambda path: result)

    assert EXIT_CODES[status] == exit_code
    assert _run(monkeypatch, "--verify-evidence", "unused.json", "--json") == exit_code
    assert json.loads(capsys.readouterr().out) == result
    assert (exit_code == 0) is (status == "pass")


@pytest.mark.parametrize(
    "execution_args",
    [
        ["--demo"],
        ["--batch-last-avg"],
        ["--batch-dir", "last_avg"],
        ["--batch-dir", ""],
        ["--max-datasets", "0"],
        ["--max-datasets", "1"],
        ["--rule", "rules.json"],
        ["--rule", ""],
        ["--dataset", "input.csv"],
        ["--dataset", ""],
        ["--goal", "analyze"],
        ["--goal", ""],
    ],
)
def test_cli_verification_rejects_every_execution_option(monkeypatch, capsys, execution_args):
    def fail(path):
        pytest.fail("argument conflicts must be rejected before reading a file")

    monkeypatch.setattr(cli, "verify_evidence_file", fail)
    assert _run(monkeypatch, "--verify-evidence", "existing.json", *execution_args) == 2
    output = capsys.readouterr()
    assert output.out == ""
    assert "不能与执行参数" in output.err
    assert execution_args[0] in output.err


def test_cli_json_requires_verification_mode(monkeypatch, capsys):
    assert _run(monkeypatch, "--json", "--demo") == 2
    assert "--json 仅可与 --verify-evidence 一起使用" in capsys.readouterr().err


def test_cli_reads_current_snapshot_each_time(tmp_path, monkeypatch, capsys):
    path = tmp_path / "evidence_chain.json"
    raw = _chain()
    _write(path, raw)
    assert _run(monkeypatch, "--verify-evidence", str(path), "--json") == 0
    assert json.loads(capsys.readouterr().out)["status"] == "pass"

    raw["nodes"][1]["summary"] = "changed after first verification"
    _write(path, raw)
    before = path.read_bytes()
    assert _run(monkeypatch, "--verify-evidence", str(path), "--json") == 1
    result = json.loads(capsys.readouterr().out)
    assert result["first_mismatch_index"] == 1
    assert result["first_mismatch_node_id"] == "node-1"
    assert path.read_bytes() == before
