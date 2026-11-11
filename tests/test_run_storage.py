from __future__ import annotations

from pathlib import Path

import pytest

import aircraft_agent.common.utils as utils
from aircraft_agent.agent.runtime import RUN_STORAGE_SCHEMA_VERSION, RunStore
from aircraft_agent.common.utils import read_json, write_json
from aircraft_agent.evidence.chain import EVIDENCE_SCHEMA_VERSION, load_evidence_chain, verify_chain


def test_atomic_json_write_success(tmp_path: Path) -> None:
    path = tmp_path / "result.json"

    write_json(path, {"ok": True, "items": [1, 2, 3]})

    assert read_json(path) == {"ok": True, "items": [1, 2, 3]}
    assert not list(tmp_path.glob(".result.json.*.tmp"))


def test_atomic_json_write_failure_preserves_existing_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "result.json"
    path.write_text('{"ok": true}', encoding="utf-8")

    def fail_replace(src: Path, dst: Path) -> None:
        raise OSError("simulated replace failure")

    monkeypatch.setattr(utils.os, "replace", fail_replace)

    with pytest.raises(OSError):
        write_json(path, {"ok": False, "partial": "must not appear"})

    assert read_json(path) == {"ok": True}
    assert not list(tmp_path.glob(".result.json.*.tmp"))


def test_run_storage_persists_metadata_and_evidence_chain_v2(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "runs")
    context = store.create_run("verify durable run storage")

    stored = store.save_bytes(
        context,
        data=b"R001: temperature > 80\n",
        original_name="rules.md",
        kind="rule_library",
    )
    context.evidence.add(
        type="tool_call",
        tool_name="unit_tool",
        arguments={"file_id": stored.file_id},
        output={"ok": True},
        summary="unit tool completed",
    )
    context.save_manifest()

    chain = load_evidence_chain(context.evidence.path)
    manifest = read_json(context.manifest_path)
    metadata = read_json(context.metadata_path)

    assert context.uploads_dir.exists()
    assert context.artifacts_dir.exists()
    assert Path(stored.path).read_bytes() == b"R001: temperature > 80\n"
    assert chain["schema_version"] == EVIDENCE_SCHEMA_VERSION
    assert chain["chain_hash"] == context.evidence.chain_hash()
    assert verify_chain(context.evidence.path).valid is True
    assert manifest["storage_schema_version"] == RUN_STORAGE_SCHEMA_VERSION
    assert manifest["run_metadata_path"] == str(context.metadata_path)
    assert manifest["evidence_chain_path"] == str(context.evidence.path)
    assert manifest["chain_hash"] == context.evidence.chain_hash()
    assert manifest["evidence_verification"]["valid"] is True
    assert manifest["evidence_verification"]["verified"] is True
    assert metadata["storage_schema_version"] == RUN_STORAGE_SCHEMA_VERSION
    assert metadata["paths"]["manifest"] == str(context.manifest_path)
    assert metadata["paths"]["evidence_chain"] == str(context.evidence.path)
    assert metadata["evidence"]["schema_version"] == EVIDENCE_SCHEMA_VERSION
    assert metadata["evidence"]["node_count"] == len(context.evidence.nodes)
    assert metadata["evidence"]["verification"]["valid"] is True
