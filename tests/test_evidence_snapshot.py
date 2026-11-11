from __future__ import annotations

import json
from pathlib import Path

import pytest

from aircraft_agent.evidence import verification
from aircraft_agent.evidence.chain import EvidenceStore


def test_snapshot_and_verdict_use_one_read_and_parse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = EvidenceStore(tmp_path)
    store.add(type="tool_call", tool_name="profile_dataset", arguments={},
              output={"rows": 3}, summary="profile")
    expected = json.loads(store.path.read_bytes())
    original_open = verification.os.open
    original_loads = verification.json.loads
    original_verify = verification.verify_chain
    calls = {"open": 0, "parse": 0}

    def counted_open(*args, **kwargs):
        calls["open"] += 1
        return original_open(*args, **kwargs)

    def counted_loads(*args, **kwargs):
        calls["parse"] += 1
        return original_loads(*args, **kwargs)

    def replaced_file_during_verification(raw):
        store.path.write_bytes(b"changed after snapshot read")
        return original_verify(raw)

    monkeypatch.setattr(verification.os, "open", counted_open)
    monkeypatch.setattr(verification.json, "loads", counted_loads)
    monkeypatch.setattr(verification, "verify_chain", replaced_file_during_verification)

    raw, result = verification.read_evidence_snapshot(store.path)

    assert calls == {"open": 1, "parse": 1}
    assert raw == expected
    assert result["status"] == "pass"
    assert result["actual_chain_hash"] == expected["chain_hash"]
    assert store.path.read_bytes() == b"changed after snapshot read"


@pytest.mark.parametrize(
    ("payload", "raw", "status"),
    [
        (b"[]", [], "legacy"),
        (b'{"schema_version":2,"nodes":[],"chain_hash":"wrong"}',
         {"schema_version": 2, "nodes": [], "chain_hash": "wrong"}, "mismatch"),
        (b'{"schema_version":2}', {"schema_version": 2}, "invalid_structure"),
        (b'{"schema_version":2,"schema_version":1,"nodes":[]}', None, "invalid_json"),
        (b'{"schema_version":1,"nodes":[],"value":NaN}', None, "invalid_json"),
        (b"null", None, "invalid_structure"),
    ],
)
def test_snapshot_preserves_strict_status_and_never_repairs(
    tmp_path: Path, payload: bytes, raw: object, status: str,
) -> None:
    path = tmp_path / "evidence_chain.json"
    path.write_bytes(payload)
    before_mtime = path.stat().st_mtime_ns

    snapshot, result = verification.read_evidence_snapshot(path)

    assert snapshot == raw
    assert result["status"] == status
    assert verification.verify_evidence_file(path) == result
    assert verification.verify_evidence_snapshot_bytes(payload) == (snapshot, result)
    assert path.read_bytes() == payload
    assert path.stat().st_mtime_ns == before_mtime


def test_snapshot_read_failure_returns_no_contents(tmp_path: Path) -> None:
    path = tmp_path / "missing" / "evidence_chain.json"

    snapshot, result = verification.read_evidence_snapshot(path)

    assert snapshot is None
    assert result["status"] == "file_not_found"
    assert not path.parent.exists()
