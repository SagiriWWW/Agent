from __future__ import annotations

import errno
import json
import os
from pathlib import Path

import pytest

from aircraft_agent.agent.runtime import RunStore
from aircraft_agent.common import utils
from aircraft_agent.evidence import verification
from aircraft_agent.evidence.chain import EvidenceStore, compute_chain_hash, compute_node_hash
from aircraft_agent.evidence.verification import verify_evidence_file, verify_run_evidence


def _make_chain(directory: Path) -> tuple[Path, dict]:
    store = EvidenceStore(directory)
    for index in range(3):
        store.add(
            type="tool_call",
            tool_name=f"tool_{index}",
            arguments={"index": index},
            output={"ok": True},
            summary=f"step {index}",
        )
    return store.path, json.loads(store.path.read_text(encoding="utf-8"))


def _write(path: Path, raw: object) -> None:
    path.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")


def _snapshot(directory: Path) -> dict[str, tuple[int, bytes | None]]:
    return {
        str(path.relative_to(directory)): (
            path.stat().st_mtime_ns,
            path.read_bytes() if path.is_file() else None,
        )
        for path in [directory, *sorted(directory.rglob("*"))]
    }


def test_valid_v2_reports_matching_hashes_and_no_mismatch(tmp_path: Path) -> None:
    path, chain = _make_chain(tmp_path)

    result = verify_evidence_file(path)

    assert result["status"] == "pass"
    assert result["valid"] is True
    assert result["verified"] is True
    assert result["schema_version"] == 2
    assert result["node_count"] == 3
    assert result["expected_chain_hash"] == chain["chain_hash"]
    assert result["actual_chain_hash"] == chain["chain_hash"]
    assert result["first_mismatch_index"] is None
    assert result["first_mismatch_node_id"] is None
    assert result["errors"] == []
    assert result["index_base"] == 0
    assert "本地链内完整性" in result["scope"]


def test_empty_v2_with_correct_hash_is_valid(tmp_path: Path) -> None:
    path = tmp_path / "evidence_chain.json"
    _write(path, {"schema_version": 2, "nodes": [], "chain_hash": compute_chain_hash([])})

    result = verify_evidence_file(path)

    assert result["status"] == "pass"
    assert result["node_count"] == 0
    assert result["valid"] is True
    assert result["verified"] is True


def test_middle_node_tampering_locates_first_mismatch(tmp_path: Path) -> None:
    path, chain = _make_chain(tmp_path)
    chain["nodes"][1]["summary"] = "changed in temporary copy"
    _write(path, chain)

    result = verify_evidence_file(path)

    assert result["status"] == "mismatch"
    assert result["valid"] is False
    assert result["verified"] is True
    assert result["first_mismatch_index"] == 1
    assert result["first_mismatch_node_id"] == chain["nodes"][1]["node_id"]
    assert result["errors"][0]["kind"] == "node_hash_mismatch"
    assert result["expected_chain_hash"] != result["actual_chain_hash"]


@pytest.mark.parametrize("missing", [False, True], ids=["wrong", "missing"])
def test_chain_level_hash_error_does_not_invent_node_position(tmp_path: Path, missing: bool) -> None:
    path, chain = _make_chain(tmp_path)
    if missing:
        del chain["chain_hash"]
    else:
        chain["chain_hash"] = "wrong hash"
    _write(path, chain)

    result = verify_evidence_file(path)

    assert result["status"] == "mismatch"
    assert result["valid"] is False
    assert result["verified"] is True
    assert result["first_mismatch_index"] is None
    assert result["first_mismatch_node_id"] is None
    assert result["errors"][0]["kind"] == ("missing_chain_hash" if missing else "chain_hash_mismatch")


def test_missing_node_hash_is_detected_without_store_repair(tmp_path: Path) -> None:
    path, chain = _make_chain(tmp_path)
    del chain["nodes"][1]["node_hash"]
    _write(path, chain)
    before = path.read_bytes()

    result = verify_evidence_file(path)

    assert result["status"] == "mismatch"
    assert result["errors"][0]["kind"] == "missing_node_hash"
    assert result["first_mismatch_index"] == 1
    assert result["first_mismatch_node_id"] == chain["nodes"][1]["node_id"]
    assert path.read_bytes() == before


@pytest.mark.parametrize("kind", ["parent_hash_mismatch", "unknown_parent"])
def test_parent_link_integrity_is_checked_with_recomputed_node_hash(tmp_path: Path, kind: str) -> None:
    path, chain = _make_chain(tmp_path)
    chain["nodes"] = chain["nodes"][:2]
    node = chain["nodes"][1]
    if kind == "parent_hash_mismatch":
        node["parent_hashes"] = ["wrong parent hash"]
    else:
        node["parent_ids"] = ["nonexistent-parent"]
    node["node_hash"] = compute_node_hash(node)
    chain["chain_hash"] = compute_chain_hash(chain["nodes"])
    _write(path, chain)

    result = verify_evidence_file(path)

    assert result["status"] == "mismatch"
    assert result["first_mismatch_index"] == 1
    assert result["first_mismatch_node_id"] == node["node_id"]
    assert result["errors"][0]["kind"] == kind
    assert result["actual_chain_hash"] == result["expected_chain_hash"]


@pytest.mark.parametrize("envelope", [False, True], ids=["list", "envelope"])
def test_legacy_chain_is_readable_but_never_verified(tmp_path: Path, envelope: bool) -> None:
    path, chain = _make_chain(tmp_path)
    for node in chain["nodes"]:
        for field in ("schema_version", "node_hash", "parent_hashes"):
            del node[field]
    raw = {"schema_version": 1, "nodes": chain["nodes"]} if envelope else chain["nodes"]
    _write(path, raw)

    result = verify_evidence_file(path)

    assert result["status"] == "legacy"
    assert result["valid"] is True
    assert result["verified"] is False
    assert result["schema_version"] == 1
    assert result["node_count"] == 3
    assert result["actual_chain_hash"] is None


def test_v2_preserves_existing_store_compatibility_for_explicit_v1_node(tmp_path: Path) -> None:
    path, chain = _make_chain(tmp_path)
    node = chain["nodes"][0]
    node["schema_version"] = 1
    node["node_hash"] = compute_node_hash(node)
    chain["nodes"] = [node]
    chain["chain_hash"] = compute_chain_hash(chain["nodes"])
    _write(path, chain)

    assert verify_evidence_file(path)["status"] == "pass"


def test_optional_model_response_id_can_be_absent(tmp_path: Path) -> None:
    path, chain = _make_chain(tmp_path)
    del chain["nodes"][0]["model_response_id"]
    _write(path, chain)

    assert verify_evidence_file(path)["status"] == "pass"


@pytest.mark.parametrize("version", [-1, 0, 3, 999])
def test_unknown_envelope_version_is_unsupported(tmp_path: Path, version: int) -> None:
    path, chain = _make_chain(tmp_path)
    chain["schema_version"] = version
    _write(path, chain)

    result = verify_evidence_file(path)

    assert result["status"] == "unsupported_version"
    assert result["valid"] is False
    assert result["verified"] is False


@pytest.mark.parametrize("version", ["2", True, False, 2.0, 2.5, None])
def test_envelope_version_requires_an_integer(tmp_path: Path, version: object) -> None:
    path, chain = _make_chain(tmp_path)
    chain["schema_version"] = version
    _write(path, chain)

    assert verify_evidence_file(path)["status"] == "invalid_structure"


def test_unknown_node_version_is_unsupported(tmp_path: Path) -> None:
    path, chain = _make_chain(tmp_path)
    chain["nodes"][1]["schema_version"] = 3
    _write(path, chain)

    result = verify_evidence_file(path)

    assert result["status"] == "unsupported_version"
    assert result["valid"] is False
    assert result["verified"] is False


@pytest.mark.parametrize("raw", [None, 3, "chain", {}, {"schema_version": 2}, {"schema_version": 2, "nodes": None}, {"schema_version": 2, "nodes": {}}, {"schema_version": 2, "evidence": []}])
def test_invalid_envelope_never_normalizes_to_success(tmp_path: Path, raw: object) -> None:
    path = tmp_path / "evidence_chain.json"
    _write(path, raw)

    result = verify_evidence_file(path)

    assert result["status"] == "invalid_structure"
    assert result["valid"] is False
    assert result["verified"] is False


@pytest.mark.parametrize("bad_node", [None, [], "discarded", 42])
@pytest.mark.parametrize("legacy", [False, True])
def test_nonobject_nodes_are_not_silently_filtered(tmp_path: Path, bad_node: object, legacy: bool) -> None:
    path, chain = _make_chain(tmp_path)
    chain["nodes"].append(bad_node)
    _write(path, chain["nodes"] if legacy else chain)

    result = verify_evidence_file(path)

    assert result["status"] == "invalid_structure"
    assert result["valid"] is False
    assert result["verified"] is False


@pytest.mark.parametrize("field", ["schema_version", "node_id", "parent_ids", "parent_hashes", "type", "tool_name", "input_hash", "output_hash", "arguments", "summary", "timestamp"])
def test_required_v2_node_fields_cannot_be_filled_by_normalization(tmp_path: Path, field: str) -> None:
    path, chain = _make_chain(tmp_path)
    del chain["nodes"][0][field]
    _write(path, chain)

    assert verify_evidence_file(path)["status"] == "invalid_structure"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", True),
        ("schema_version", 2.0),
        ("schema_version", "2"),
        ("node_id", ""),
        ("node_id", 123),
        ("parent_ids", "parent"),
        ("parent_ids", [3]),
        ("parent_hashes", None),
        ("parent_hashes", [None]),
        ("type", None),
        ("tool_name", 23),
        ("input_hash", None),
        ("output_hash", []),
        ("arguments", []),
        ("summary", 3),
        ("timestamp", None),
        ("model_response_id", {}),
        ("node_hash", 3),
    ],
)
def test_invalid_node_types_are_rejected(tmp_path: Path, field: str, value: object) -> None:
    path, chain = _make_chain(tmp_path)
    chain["nodes"][0][field] = value
    _write(path, chain)

    assert verify_evidence_file(path)["status"] == "invalid_structure"


def test_duplicate_node_ids_are_invalid_even_with_consistent_hashes(tmp_path: Path) -> None:
    path, chain = _make_chain(tmp_path)
    node = chain["nodes"][0]
    chain["nodes"] = [node, dict(node)]
    chain["chain_hash"] = compute_chain_hash(chain["nodes"])
    _write(path, chain)

    assert verify_evidence_file(path)["status"] == "invalid_structure"


def test_invalid_chain_hash_type_is_rejected(tmp_path: Path) -> None:
    path, chain = _make_chain(tmp_path)
    chain["chain_hash"] = []
    _write(path, chain)

    assert verify_evidence_file(path)["status"] == "invalid_structure"


@pytest.mark.parametrize(
    "payload",
    [
        b"{",
        b"{\"schema_version\":1,\"schema_version\":2,\"nodes\":[]}",
        b"{\"schema_version\":1,\"nodes\":[],\"value\":NaN}",
        b"{\"schema_version\":1,\"nodes\":[],\"value\":Infinity}",
        b"{\"schema_version\":1,\"nodes\":[],\"value\":-Infinity}",
        b"{\"schema_version\":1,\"nodes\":[],\"value\":1e400}",
        b"\xff\xfe",
    ],
)
def test_invalid_or_ambiguous_json_is_rejected(tmp_path: Path, payload: bytes) -> None:
    path = tmp_path / "evidence_chain.json"
    path.write_bytes(payload)

    result = verify_evidence_file(path)

    assert result["status"] == "invalid_json"
    assert result["valid"] is False
    assert result["verified"] is False


def test_missing_file_is_reported_without_creating_paths(tmp_path: Path) -> None:
    path = tmp_path / "missing" / "evidence_chain.json"

    assert verify_evidence_file(path)["status"] == "file_not_found"
    assert verify_run_evidence(tmp_path / "runs", "missing-run")["status"] == "file_not_found"
    assert list(tmp_path.iterdir()) == []


def test_os_read_error_is_reported(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path, _ = _make_chain(tmp_path)

    def denied(*args, **kwargs):
        raise PermissionError("simulated unreadable snapshot")

    monkeypatch.setattr(verification.os, "open", denied)
    result = verify_evidence_file(path)

    assert result["status"] == "read_error"
    assert result["valid"] is False
    assert result["verified"] is False


@pytest.mark.parametrize("run_id", ["", ".", "..", "../outside", "run/../outside", "/absolute", "run\\outside", "bad\x00run"])
def test_run_path_traversal_is_rejected(tmp_path: Path, run_id: str) -> None:
    assert verify_run_evidence(tmp_path, run_id)["status"] == "path_rejected"


@pytest.mark.parametrize("inside", [False, True], ids=["outside", "inside"])
def test_run_directory_symlinks_are_rejected(tmp_path: Path, inside: bool) -> None:
    root = tmp_path / "runs"
    root.mkdir()
    target = root / "real-run" if inside else tmp_path / "outside"
    _make_chain(target)
    (root / "linked-run").symlink_to(target, target_is_directory=True)

    assert verify_run_evidence(root, "linked-run")["status"] == "path_rejected"


@pytest.mark.parametrize("inside", [False, True], ids=["outside", "inside"])
def test_evidence_file_symlinks_are_rejected(tmp_path: Path, inside: bool) -> None:
    root = tmp_path / "runs"
    run = root / "target-run"
    run.mkdir(parents=True)
    target_dir = run / "nested" if inside else tmp_path / "outside"
    target, _ = _make_chain(target_dir)
    path = run / "evidence_chain.json"
    path.symlink_to(target)

    assert verify_run_evidence(root, "target-run")["status"] == "path_rejected"
    # CLI follows the file explicitly chosen by its user; Web has a run boundary.
    assert verify_evidence_file(path)["status"] == "pass"


def test_configured_root_symlink_is_allowed_as_trusted_configuration(tmp_path: Path) -> None:
    root = tmp_path / "real-runs"
    _make_chain(root / "target-run")
    configured_root = tmp_path / "configured-runs"
    configured_root.symlink_to(root, target_is_directory=True)

    assert verify_run_evidence(configured_root, "target-run")["status"] == "pass"


def test_directory_in_place_of_evidence_is_not_read(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    path = root / "target-run" / "evidence_chain.json"
    path.mkdir(parents=True)

    assert verify_run_evidence(root, "target-run")["status"] == "read_error"
    assert verify_evidence_file(path)["status"] == "read_error"


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="requires POSIX FIFO support")
def test_fifo_is_rejected_without_blocking(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    run = root / "target-run"
    run.mkdir(parents=True)
    path = run / "evidence_chain.json"
    os.mkfifo(path)

    assert verify_run_evidence(root, "target-run")["status"] == "read_error"
    assert verify_evidence_file(path)["status"] == "read_error"


@pytest.mark.parametrize("surface", ["cli", "web"])
@pytest.mark.parametrize("scenario", ["regular_file", "directory", "fstat_error", "fdopen_error"])
def test_verification_closes_opened_descriptors_on_every_read_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, surface: str, scenario: str,
) -> None:
    root = tmp_path / "runs"
    run = root / "target-run"
    path = run / "evidence_chain.json"
    if scenario == "directory":
        path.mkdir(parents=True)
    else:
        _make_chain(run)

    original_open = os.open
    original_fstat = os.fstat
    opened: list[int] = []

    def recording_open(*args, **kwargs):
        fd = original_open(*args, **kwargs)
        opened.append(fd)
        return fd

    def fail_read(*args, **kwargs):
        raise OSError(errno.EIO, "simulated descriptor read failure")

    with monkeypatch.context() as patch:
        patch.setattr(verification.os, "open", recording_open)
        if scenario == "fstat_error":
            patch.setattr(verification.os, "fstat", fail_read)
        elif scenario == "fdopen_error":
            patch.setattr(verification.os, "fdopen", fail_read)
        result = verify_evidence_file(path) if surface == "cli" else verify_run_evidence(root, "target-run")

    assert result["status"] == ("pass" if scenario == "regular_file" else "read_error")
    assert len(opened) == (1 if surface == "cli" else 3)
    for fd in opened:
        with pytest.raises(OSError) as exc:
            original_fstat(fd)
        assert exc.value.errno == errno.EBADF


def test_verification_preserves_all_run_files_and_rechecks_current_snapshot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "runs"
    store = RunStore(root)
    context = store.create_run("read-only verification")
    for index in range(3):
        context.evidence.add(type="tool_call", tool_name="unit", arguments={}, output={}, summary=f"step {index}")
    context.save_manifest()
    (context.artifacts_dir / "report.md").write_text("existing report", encoding="utf-8")
    (context.uploads_dir / "input.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    before = _snapshot(context.run_dir)

    def forbidden(*args, **kwargs):
        pytest.fail("verification attempted to create or rewrite run state")

    monkeypatch.setattr(EvidenceStore, "__init__", forbidden)
    monkeypatch.setattr(RunStore, "create_run", forbidden)
    monkeypatch.setattr(utils, "write_json", forbidden)
    monkeypatch.setattr(utils, "atomic_write_bytes", forbidden)
    monkeypatch.setattr(utils, "write_text_atomic", forbidden)

    assert verify_evidence_file(context.evidence.path)["status"] == "pass"
    assert verify_run_evidence(root, context.run_id)["status"] == "pass"
    assert _snapshot(context.run_dir) == before

    # This run is a temporary fixture; the original project runs stay untouched.
    chain = json.loads(context.evidence.path.read_text(encoding="utf-8"))
    chain["nodes"][1]["summary"] = "tampered after first verification"
    _write(context.evidence.path, chain)
    after_tampering = _snapshot(context.run_dir)

    result = verify_run_evidence(root, context.run_id)

    assert result["status"] == "mismatch"
    assert result["first_mismatch_index"] == 1
    assert result["first_mismatch_node_id"] == chain["nodes"][1]["node_id"]
    assert _snapshot(context.run_dir) == after_tampering
