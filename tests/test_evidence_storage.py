from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import fcntl
import json
import multiprocessing
import os
from pathlib import Path
import threading

import pytest

from aircraft_agent.agent.runtime import RunStore
import aircraft_agent.common.utils as utils
from aircraft_agent.common.utils import stable_json, write_json
from aircraft_agent.evidence.chain import (
    EvidenceStore, compute_chain_hash, compute_node_hash, load_evidence_chain,
)
import aircraft_agent.evidence.storage as storage
from aircraft_agent.evidence.storage import EvidenceStorageError, check_evidence_storage
from aircraft_agent.evidence.verification import verify_evidence_file
from aircraft_agent.tools.executor import ToolExecutor


def _add(store: EvidenceStore, index: int = 0, **kwargs):
    return store.add(type="tool_call", tool_name="storage_test", arguments={"index": index},
                     output={"ok": True}, summary=f"证据 {index}", **kwargs)


def _new(path: Path, count: int = 1) -> EvidenceStore:
    store = EvidenceStore.create(path)
    for index in range(count):
        _add(store, index)
    return store


def _records(path: Path) -> list[dict]:
    return [json.loads(line) for line in (path / "evidence_chain.jsonl").read_bytes().splitlines()]


def _write_records(path: Path, records: list[dict]) -> None:
    (path / "evidence_chain.jsonl").write_bytes(
        b"".join(stable_json(record).encode("utf-8") + b"\n" for record in records)
    )


def _files(path: Path) -> dict:
    return {str(p.relative_to(path)): (p.read_bytes(), p.stat().st_mtime_ns)
            for p in path.rglob("*") if p.is_file()}


def _memory(store: EvidenceStore) -> tuple:
    return deepcopy(store.to_list()), list(store.tail_node_ids)


def _assert_blocked_unchanged(store: EvidenceStore, path: Path) -> None:
    before, memory = _files(path), _memory(store)
    for operation in (store.save, lambda: _add(store, 999)):
        with pytest.raises(EvidenceStorageError):
            operation()
        assert _files(path) == before
        assert _memory(store) == memory


def _process_append(path: str, barrier, queue) -> None:
    try:
        store = EvidenceStore(Path(path))
        before = _memory(store)
        barrier.wait(timeout=10)
        try:
            node = _add(store, 40)
            queue.put(("ok", node.node_id, len(store.nodes)))
        except EvidenceStorageError as exc:
            queue.put(("blocked", exc.state, _memory(store) == before))
    except BaseException as exc:
        queue.put(("unexpected", repr(exc)))


def _process_hold_lock(path: str, ready, release) -> None:
    with (Path(path) / ".evidence.lock").open("rb") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        ready.set()
        release.wait(timeout=10)


def _process_check_after_parent_unlock(path: str, ready, release, queue) -> None:
    ready.set()
    if not release.wait(timeout=10):
        queue.put("parent did not release lock")
        return
    queue.put(check_evidence_storage(Path(path))["status"])


def _process_try_inherited_store(store, queue) -> None:
    try:
        _add(store, 200)
        queue.put("unexpected success")
    except EvidenceStorageError as error:
        queue.put(error.state)


def test_append_preserves_history_and_repeated_save_changes_no_files(tmp_path):
    store = _new(tmp_path)
    log = tmp_path / "evidence_chain.jsonl"
    before = log.read_bytes()
    first = store.nodes[0].to_dict()
    _add(store, 1)
    assert log.read_bytes().startswith(before)
    records = _records(tmp_path)
    assert len(records) == 2
    assert records[0]["node"] == first
    assert [r["sequence"] for r in records] == [0, 1]
    assert all(r["storage_protocol_version"] == 1 and r["node"]["schema_version"] == 2 for r in records)
    snapshot = load_evidence_chain(store.path)
    assert set(snapshot) == {"schema_version", "chain_hash", "nodes"}
    assert [r["node"] for r in records] == snapshot["nodes"]
    assert compute_chain_hash([r["node"] for r in records]) == snapshot["chain_hash"] == store.chain_hash()
    files = _files(tmp_path)
    store.save()
    store.save()
    assert _files(tmp_path) == files
    result = check_evidence_storage(tmp_path)
    assert result["status"] == "consistent" and result["exit_code"] == 0
    assert result["consistent"] and result["comparable"] and not result["write_blocked"]
    assert result["snapshot"]["status"] == result["log"]["status"] == "pass"


def test_restart_continues_exact_prefix_and_recovers_nonadvancing_tail(tmp_path):
    store = _new(tmp_path)
    main = store.nodes[-1]
    side = _add(store, 1, advance_tail=False)
    assert side.node_id != main.node_id
    prefix = (tmp_path / "evidence_chain.jsonl").read_bytes()
    before = _files(tmp_path)
    restored = EvidenceStore(tmp_path)
    assert _files(tmp_path) == before
    assert restored.tail_node_ids == [main.node_id]
    assert _records(tmp_path)[-1]["tail_node_ids"] == [main.node_id]
    node = _add(restored, 2)
    assert node.parent_ids == [main.node_id]
    assert node.parent_hashes == [main.node_hash]
    assert (tmp_path / "evidence_chain.jsonl").read_bytes().startswith(prefix)
    assert len(_records(tmp_path)) == 3
    assert check_evidence_storage(tmp_path)["status"] == "consistent"


def test_empty_constructor_is_read_only_and_first_write_initializes(tmp_path):
    store = EvidenceStore(tmp_path)
    assert _files(tmp_path) == {}
    store.save()
    assert (tmp_path / "evidence_chain.jsonl").read_bytes() == b""
    assert check_evidence_storage(tmp_path)["status"] == "consistent"
    assert store.nodes == [] and store.tail_node_ids == []


def test_invalid_new_node_is_rejected_before_creating_transaction_intent(tmp_path):
    store = _new(tmp_path)
    before, memory = _files(tmp_path), _memory(store)
    with pytest.raises(EvidenceStorageError):
        store.add(type=23, tool_name="test", arguments={}, output={}, summary="invalid node type")
    assert _memory(store) == memory and _files(tmp_path) == before
    _add(store, 1)
    assert check_evidence_storage(tmp_path)["status"] == "consistent"


def test_stale_instance_cannot_overwrite_newer_commits(tmp_path):
    writer = _new(tmp_path)
    stale = EvidenceStore(tmp_path)
    original = _memory(stale)
    _add(writer, 1)
    before = _files(tmp_path)
    with pytest.raises(EvidenceStorageError) as error:
        _add(stale, 2)
    assert error.value.state == "stale_instance"
    assert _memory(stale) == original
    assert _files(tmp_path) == before
    assert len(_records(tmp_path)) == 2
    fresh = EvidenceStore(tmp_path)
    _add(fresh, 3)
    assert len(_records(tmp_path)) == 3


def test_stale_same_length_rehashed_content_is_not_accepted(tmp_path):
    store = _new(tmp_path)
    records = _records(tmp_path)
    records[0]["node"]["summary"] = "external coherent replacement"
    records[0]["node"]["node_hash"] = compute_node_hash(records[0]["node"])
    _write_records(tmp_path, records)
    nodes = [record["node"] for record in records]
    write_json(store.path, {"schema_version": 2, "chain_hash": compute_chain_hash(nodes), "nodes": nodes})
    assert check_evidence_storage(tmp_path)["status"] == "consistent"
    before, memory = _files(tmp_path), _memory(store)
    with pytest.raises(EvidenceStorageError) as error:
        store.save()
    assert error.value.state == "stale_instance"
    assert _files(tmp_path) == before and _memory(store) == memory


@pytest.mark.parametrize("mutation", ["summary", "hash", "tail", "append"])
def test_save_rejects_manual_mutation_without_touching_committed_files(tmp_path, mutation):
    store = _new(tmp_path)
    before = _files(tmp_path)
    if mutation == "summary":
        store.nodes[0].summary = "modified"
    elif mutation == "hash":
        store.nodes[0].node_hash = ""
    elif mutation == "tail":
        store.tail_node_ids = []
    else:
        store.nodes.append(deepcopy(store.nodes[0]))
    with pytest.raises(EvidenceStorageError):
        store.save()
    assert _files(tmp_path) == before


def test_same_instance_threads_preserve_all_successfully_committed_nodes(tmp_path):
    store = _new(tmp_path, 0)
    barrier = threading.Barrier(4)

    def append(index):
        barrier.wait(timeout=5)
        return _add(store, index).node_id

    with ThreadPoolExecutor(max_workers=4) as pool:
        ids = list(pool.map(append, range(4)))
    assert len(set(ids)) == len(store.nodes) == len(_records(tmp_path)) == 4
    assert check_evidence_storage(tmp_path)["status"] == "consistent"
    nodes = store.to_list()
    for previous, node in zip(nodes, nodes[1:]):
        assert node["parent_ids"] == [previous["node_id"]]


def test_separate_instances_in_threads_cannot_commit_from_one_stale_prefix(tmp_path):
    _new(tmp_path)
    stores = [EvidenceStore(tmp_path), EvidenceStore(tmp_path)]
    barrier = threading.Barrier(2)

    def append(store):
        before = _memory(store)
        barrier.wait(timeout=5)
        try:
            return ("ok", _add(store, 7).node_id)
        except EvidenceStorageError as exc:
            assert _memory(store) == before
            return ("blocked", exc.state)

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(append, stores))
    assert sorted(result[0] for result in results) == ["blocked", "ok"]
    assert len(_records(tmp_path)) == 2
    assert check_evidence_storage(tmp_path)["status"] == "consistent"


def test_forked_process_writers_cannot_commit_from_one_stale_prefix(tmp_path):
    _new(tmp_path)
    ctx = multiprocessing.get_context("fork")
    barrier, queue = ctx.Barrier(2), ctx.Queue()
    processes = [ctx.Process(target=_process_append, args=(str(tmp_path), barrier, queue)) for _ in range(2)]
    try:
        for process in processes:
            process.start()
        results = [queue.get(timeout=12) for _ in processes]
        for process in processes:
            process.join(timeout=5)
            assert process.exitcode == 0
        assert sorted(result[0] for result in results) == ["blocked", "ok"]
        assert next(result for result in results if result[0] == "blocked")[2] is True
        assert len(_records(tmp_path)) == 2
        assert check_evidence_storage(tmp_path)["status"] == "consistent"
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
        queue.close()


def test_read_only_check_reports_active_process_writer_without_creating_files(tmp_path):
    _new(tmp_path)
    before = _files(tmp_path)
    ctx = multiprocessing.get_context("fork")
    ready, release = ctx.Event(), ctx.Event()
    process = ctx.Process(target=_process_hold_lock, args=(str(tmp_path), ready, release))
    process.start()
    try:
        assert ready.wait(timeout=5)
        result = check_evidence_storage(tmp_path)
        assert result["status"] == "busy"
        assert result["comparable"] is False and result["consistent"] is False
        assert result["exit_code"] != 0
        assert _files(tmp_path) == before
    finally:
        release.set()
        process.join(timeout=5)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
    assert check_evidence_storage(tmp_path)["status"] == "consistent"


def test_fork_during_parent_lock_does_not_leave_child_with_inherited_mutex_or_fd(tmp_path):
    _new(tmp_path)
    before = _files(tmp_path)
    ctx = multiprocessing.get_context("fork")
    ready, release, queue = ctx.Event(), ctx.Event(), ctx.Queue()
    process = ctx.Process(target=_process_check_after_parent_unlock, args=(str(tmp_path), ready, release, queue))
    try:
        with storage.storage_lock(tmp_path, shared=False):
            process.start()
            assert ready.wait(timeout=5)
        release.set()
        assert queue.get(timeout=5) == "consistent"
        process.join(timeout=5)
        assert process.exitcode == 0
        assert _files(tmp_path) == before
    finally:
        release.set()
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
        queue.close()


def test_forked_child_must_not_continue_with_inherited_store_instance(tmp_path):
    store = _new(tmp_path)
    before, memory = _files(tmp_path), _memory(store)
    ctx = multiprocessing.get_context("fork")
    queue = ctx.Queue()
    process = ctx.Process(target=_process_try_inherited_store, args=(store, queue))
    process.start()
    try:
        assert queue.get(timeout=5) == "stale_instance"
        process.join(timeout=5)
        assert process.exitcode == 0
        assert _files(tmp_path) == before and _memory(store) == memory
        _add(store, 1)
        assert check_evidence_storage(tmp_path)["status"] == "consistent"
    finally:
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
        queue.close()


@pytest.mark.parametrize("case,status", [
    ("behind", "snapshot_behind"), ("ahead", "snapshot_ahead"),
    ("diverged", "diverged"), ("extra_field", "diverged"),
    ("snapshot_hash", "snapshot_corrupt"), ("missing_log", "missing_log"),
    ("missing_marker", "invalid_protocol"),
])
def test_two_file_mismatches_are_distinct_and_never_repaired(tmp_path, case, status):
    store = _new(tmp_path, 2)
    snapshot = load_evidence_chain(store.path)
    if case == "behind":
        snapshot["nodes"] = snapshot["nodes"][:1]
    elif case == "ahead":
        _write_records(tmp_path, _records(tmp_path)[:1])
    elif case == "diverged":
        snapshot["nodes"][1]["summary"] = "a different valid branch"
        snapshot["nodes"][1]["node_hash"] = compute_node_hash(snapshot["nodes"][1])
    elif case == "extra_field":
        snapshot["nodes"][1]["unhashed_extra"] = "must compare full content"
    elif case == "snapshot_hash":
        snapshot["nodes"][1]["node_hash"] = "0" * 64
    elif case == "missing_log":
        (tmp_path / "evidence_chain.jsonl").unlink()
    elif case == "missing_marker":
        (tmp_path / "evidence_storage.json").unlink()
    if case in {"behind", "diverged", "extra_field", "snapshot_hash"}:
        snapshot["chain_hash"] = compute_chain_hash(snapshot["nodes"])
        write_json(store.path, snapshot)
    before = _files(tmp_path)
    result = check_evidence_storage(tmp_path)
    assert result["status"] == status
    assert result["consistent"] is False and result["write_blocked"] is True
    assert result["exit_code"] != 0
    if case in {"behind", "ahead", "diverged", "extra_field"}:
        assert result["snapshot"]["status"] == result["log"]["status"] == "pass"
    assert _files(tmp_path) == before
    _assert_blocked_unchanged(store, tmp_path)
    with pytest.raises(EvidenceStorageError):
        EvidenceStore(tmp_path)
    assert _files(tmp_path) == before


@pytest.mark.parametrize("case", [
    "half_line", "complete_illegal", "middle", "duplicate_node", "sequence",
    "missing_hash", "bad_hash", "tail", "version", "blank_line", "duplicate_key",
    "missing_model_response", "unknown_node_field", "legacy_node",
    "nonfinite", "surrogate", "non_utf8",
])
def test_log_corruption_is_strict_and_preserves_all_suspect_bytes(tmp_path, case):
    store = _new(tmp_path, 3)
    path = tmp_path / "evidence_chain.jsonl"
    records, original = _records(tmp_path), path.read_bytes()
    if case == "half_line":
        path.write_bytes(original + b'{"storage_protocol_version":1')
    elif case == "complete_illegal":
        path.write_bytes(original + b'{"not":"a storage record"}\n')
    elif case == "middle":
        lines = original.splitlines(keepends=True)
        path.write_bytes(lines[0] + b'{invalid}\n' + lines[2])
    elif case == "blank_line":
        path.write_bytes(original + b"\n")
    elif case == "duplicate_key":
        path.write_bytes(original + b'{"x":1,"x":2}\n')
    elif case == "nonfinite":
        path.write_bytes(original + b'{"x":NaN}\n')
    elif case == "surrogate":
        path.write_bytes(original + b'{"x":"\\ud800"}\n')
    elif case == "non_utf8":
        path.write_bytes(original + b'\xff\n')
    else:
        if case == "duplicate_node":
            records[1]["node"] = deepcopy(records[0]["node"])
        elif case == "sequence":
            records[1]["sequence"] = 8
        elif case == "missing_hash":
            del records[1]["node"]["node_hash"]
        elif case == "bad_hash":
            records[1]["node"]["node_hash"] = "0" * 64
        elif case == "tail":
            records[1]["tail_node_ids"] = ["nonexistent"]
        elif case == "version":
            records[1]["storage_protocol_version"] = 77
        elif case == "missing_model_response":
            records[1]["node"].pop("model_response_id")
        elif case == "unknown_node_field":
            records[1]["node"]["unknown"] = "unhashed field"
        elif case == "legacy_node":
            records[1]["node"]["schema_version"] = 1
            records[1]["node"]["node_hash"] = compute_node_hash(records[1]["node"])
        _write_records(tmp_path, records)
    before = _files(tmp_path)
    result = check_evidence_storage(tmp_path)
    assert result["status"] == "log_corrupt"
    assert result["snapshot"]["status"] == "pass"
    assert result["consistent"] is False and result["write_blocked"] is True
    assert result["log"]["errors"]
    json.dumps(result, ensure_ascii=False).encode("utf-8")
    assert _files(tmp_path) == before
    _assert_blocked_unchanged(store, tmp_path)
    assert _files(tmp_path) == before


@pytest.mark.parametrize("version", [1, 2])
def test_old_runs_remain_readable_but_cannot_migrate_or_continue_writing(tmp_path, version):
    modern = _new(tmp_path / "modern")
    path = tmp_path / "old"
    path.mkdir()
    nodes = modern.to_list()
    if version == 1:
        for node in nodes:
            for field in ("schema_version", "node_hash", "parent_hashes"):
                node.pop(field)
        raw = nodes
    else:
        raw = modern.to_dict()
    write_json(path / "evidence_chain.json", raw)
    (path / "manifest.json").write_bytes(b"unchanged legacy metadata")
    before = _files(path)
    old = EvidenceStore(path)
    assert len(old.nodes) == 1
    if version == 1:
        assert old.nodes[0].node_hash == ""
    assert load_evidence_chain(old.path)["nodes"] == nodes
    result = check_evidence_storage(path)
    assert result["status"] == "legacy_no_log" and result["exit_code"] == 3
    assert result["consistent"] is False and result["comparable"] is False
    assert result["snapshot"]["status"] == ("legacy" if version == 1 else "pass")
    assert _files(path) == before
    _assert_blocked_unchanged(old, path)
    assert _files(path) == before


def test_pending_blocks_restart_and_cli_success_even_for_consistent_files(tmp_path):
    store = _new(tmp_path)
    (tmp_path / "evidence_pending.json").write_bytes(b"not valid JSON, still a stop marker")
    before = _files(tmp_path)
    result = check_evidence_storage(tmp_path)
    assert result["status"] == "consistent" and result["consistent"] is True
    assert result["write_blocked"] is True and result["exit_code"] == 8
    assert verify_evidence_file(store.path)["status"] == "pass"
    with pytest.raises(EvidenceStorageError):
        EvidenceStore(tmp_path)
    _assert_blocked_unchanged(store, tmp_path)
    assert _files(tmp_path) == before


def test_dangling_pending_symlink_is_preserved_and_never_treated_as_absent(tmp_path):
    store = _new(tmp_path)
    pending = tmp_path / "evidence_pending.json"
    absent_target = tmp_path / "missing-intent-target"
    pending.symlink_to(absent_target)
    before, memory = _files(tmp_path), _memory(store)
    link_stat = pending.lstat()
    result = check_evidence_storage(tmp_path)
    assert result["write_blocked"] is True and result["exit_code"] != 0
    with pytest.raises(EvidenceStorageError):
        EvidenceStore(tmp_path)
    with pytest.raises(EvidenceStorageError):
        _add(store, 1)
    assert pending.is_symlink() and pending.readlink() == absent_target
    assert pending.lstat().st_ino == link_stat.st_ino
    assert pending.lstat().st_mtime_ns == link_stat.st_mtime_ns
    assert not absent_target.exists()
    assert _files(tmp_path) == before and _memory(store) == memory


def test_missing_and_empty_run_checks_do_not_create_any_files(tmp_path):
    missing = tmp_path / "missing"
    assert check_evidence_storage(missing)["exit_code"] != 0
    assert not missing.exists()
    assert check_evidence_storage(tmp_path)["exit_code"] != 0
    assert _files(tmp_path) == {}


@pytest.mark.parametrize("case", ["missing_lock", "mode", "version", "duplicate_key", "surrogate", "log_without_marker"])
def test_protocol_errors_never_downgrade_or_modify_the_run(tmp_path, case):
    _new(tmp_path)
    marker = tmp_path / "evidence_storage.json"
    if case == "missing_lock":
        (tmp_path / ".evidence.lock").unlink()
    elif case == "mode":
        marker.write_text('{"storage_protocol_version":1,"mode":"legacy_snapshot"}')
    elif case == "version":
        marker.write_text('{"storage_protocol_version":2,"mode":"jsonl_snapshot"}')
    elif case == "duplicate_key":
        marker.write_text('{"storage_protocol_version":1,"mode":"jsonl_snapshot","mode":"jsonl_snapshot"}')
    elif case == "surrogate":
        marker.write_bytes(b'{"storage_protocol_version":1,"mode":"\\ud800"}')
    else:
        marker.unlink()
        (tmp_path / ".evidence.lock").unlink()
    before = _files(tmp_path)
    result = check_evidence_storage(tmp_path)
    assert result["status"] == "invalid_protocol"
    assert result["write_blocked"] is True and result["consistent"] is False
    json.dumps(result, ensure_ascii=False).encode("utf-8")
    assert _files(tmp_path) == before


@pytest.mark.parametrize("name", ["evidence_chain.jsonl", "evidence_storage.json", "evidence_pending.json"])
def test_duplicate_surrogate_keys_have_safe_utf8_storage_diagnostics(tmp_path, name):
    _new(tmp_path)
    (tmp_path / name).write_bytes(b'{"\\ud800":1,"\\ud800":2}\n')
    before = _files(tmp_path)
    result = check_evidence_storage(tmp_path)
    assert result["exit_code"] != 0 and result["write_blocked"] is True
    json.dumps(result, ensure_ascii=False).encode("utf-8")
    assert _files(tmp_path) == before


@pytest.mark.parametrize("value", [None, -1, True, "oversized"])
def test_invalid_write_return_never_advances_memory_or_snapshot(tmp_path, monkeypatch, value):
    store = _new(tmp_path)
    before, memory = _files(tmp_path), _memory(store)
    monkeypatch.setattr(storage, "_write_record", lambda file, data: len(data) + 1 if value == "oversized" else value)
    with pytest.raises(EvidenceStorageError) as error:
        _add(store, 1)
    assert error.value.state == "not_committed" and error.value.committed_count == 1
    assert _memory(store) == memory
    assert (tmp_path / "evidence_chain.jsonl").read_bytes() == before["evidence_chain.jsonl"][0]
    assert store.path.read_bytes() == before["evidence_chain.json"][0]
    assert check_evidence_storage(tmp_path)["write_blocked"] is True
    _assert_blocked_unchanged(store, tmp_path)


def test_checker_detects_initialization_between_lock_probe_and_file_reads(tmp_path, monkeypatch):
    original = storage._inspect_locked
    created_files = None

    def concurrent_initialize(path, *, allow_new_mode=True):
        nonlocal created_files
        if not allow_new_mode:
            _new(path)
            created_files = _files(path)
        return original(path, allow_new_mode=allow_new_mode)

    monkeypatch.setattr(storage, "_inspect_locked", concurrent_initialize)
    result = check_evidence_storage(tmp_path)
    assert result["status"] == "busy" and result["comparable"] is False
    assert result["consistent"] is False and result["exit_code"] != 0
    assert created_files is not None and _files(tmp_path) == created_files


@pytest.mark.parametrize("target", ["evidence_pending.json", "evidence_chain.json"])
def test_silently_incomplete_atomic_json_write_cannot_report_success(tmp_path, monkeypatch, target):
    store = _new(tmp_path)
    original = storage.write_json
    before, memory = _files(tmp_path), _memory(store)

    def silently_short_write(path, value):
        if Path(path).name == target:
            Path(path).write_bytes(b'{"incomplete":')
            return
        original(path, value)

    monkeypatch.setattr(storage, "write_json", silently_short_write)
    with pytest.raises(EvidenceStorageError) as error:
        _add(store, 1)
    if target == "evidence_pending.json":
        assert error.value.state == "not_committed"
        assert _memory(store) == memory
        assert (tmp_path / "evidence_chain.jsonl").read_bytes() == before["evidence_chain.jsonl"][0]
        assert store.path.read_bytes() == before["evidence_chain.json"][0]
    else:
        assert error.value.state == "log_committed"
        assert len(store.nodes) == len(_records(tmp_path)) == 2
        assert store.tail_node_ids == [store.nodes[-1].node_id]
    assert (tmp_path / "evidence_pending.json").exists()
    assert check_evidence_storage(tmp_path)["exit_code"] != 0
    _assert_blocked_unchanged(store, tmp_path)


def test_partial_writes_are_completed_before_snapshot_and_success(tmp_path, monkeypatch):
    store = _new(tmp_path)
    original = storage._write_record
    calls = []

    def short_write(file, data):
        calls.append(len(data))
        return original(file, data[:max(1, len(data) // 3)])

    monkeypatch.setattr(storage, "_write_record", short_write)
    node = _add(store, 1)
    assert len(calls) > 2
    assert _records(tmp_path)[-1]["node"]["node_id"] == node.node_id
    assert check_evidence_storage(tmp_path)["status"] == "consistent"


def test_caller_mutation_cannot_change_committed_memory_or_hash(tmp_path):
    store = _new(tmp_path, 0)
    arguments = {"nested": {"values": [1, 2]}, "path": tmp_path / "input.csv"}
    node = store.add(type="tool_call", tool_name="test", arguments=arguments,
                     output={"ok": True}, summary="detached arguments")
    memory, before, chain_hash = _memory(store), _files(tmp_path), store.chain_hash()
    arguments["nested"]["values"].append(3)
    arguments["path"] = "changed"
    assert _memory(store) == memory and store.chain_hash() == chain_hash
    assert node.arguments["nested"]["values"] == [1, 2]
    assert node.arguments["path"] == str(tmp_path / "input.csv")
    store.save()
    assert _files(tmp_path) == before
    assert check_evidence_storage(tmp_path)["status"] == "consistent"


@pytest.mark.parametrize("stage", ["write", "zero_write", "partial", "flush", "fsync"])
def test_log_write_failures_preserve_confirmed_memory_and_stop_future_writes(tmp_path, monkeypatch, stage):
    store = _new(tmp_path)
    before, memory = _files(tmp_path), _memory(store)
    original_write = storage._write_record
    original_flush = storage._flush_log
    calls = 0

    def fail_write(file, data):
        nonlocal calls
        calls += 1
        if stage == "zero_write":
            return 0
        if stage == "partial" and calls == 1:
            written = original_write(file, data[:17])
            original_flush(file)
            return written
        raise OSError(f"injected {stage} failure")

    def fail_flush(file):
        original_flush(file)
        raise OSError("injected flush failure after visible full record")

    def fail_fsync(file):
        raise OSError("injected log fsync failure")

    hook = "_flush_log" if stage == "flush" else "_fsync_log" if stage == "fsync" else "_write_record"
    function = fail_flush if stage == "flush" else fail_fsync if stage == "fsync" else fail_write
    monkeypatch.setattr(storage, hook, function)
    with pytest.raises(EvidenceStorageError) as error:
        _add(store, 1)
    assert error.value.state in {"not_committed", "uncertain"}
    assert error.value.committed_count == 1
    assert _memory(store) == memory
    assert store.path.read_bytes() == before["evidence_chain.json"][0]
    log = (tmp_path / "evidence_chain.jsonl").read_bytes()
    assert log.startswith(before["evidence_chain.jsonl"][0])
    if stage in {"write", "zero_write"}:
        assert log == before["evidence_chain.jsonl"][0]
    elif stage == "partial":
        assert log == before["evidence_chain.jsonl"][0] + log[-17:]
        assert not log.endswith(b"\n")
    else:
        assert len(_records(tmp_path)) == 2
    assert (tmp_path / "evidence_pending.json").exists()
    result = check_evidence_storage(tmp_path)
    assert result["write_blocked"] is True and result["exit_code"] != 0
    _assert_blocked_unchanged(store, tmp_path)
    with pytest.raises(EvidenceStorageError):
        EvidenceStore(tmp_path)


@pytest.mark.parametrize("after_replace", [False, True])
def test_snapshot_failure_keeps_log_commit_and_actual_memory_tail(tmp_path, monkeypatch, after_replace):
    store = _new(tmp_path)
    old_snapshot = store.path.read_bytes()
    original = storage.write_json

    def fail_snapshot(path, value):
        if Path(path).name == "evidence_chain.json":
            if after_replace:
                original(path, value)
            raise OSError("injected snapshot publish failure")
        original(path, value)

    monkeypatch.setattr(storage, "write_json", fail_snapshot)
    with pytest.raises(EvidenceStorageError) as error:
        _add(store, 1, advance_tail=False)
    assert error.value.state == "log_committed" and error.value.committed_count == 2
    assert len(store.nodes) == len(_records(tmp_path)) == 2
    assert store.tail_node_ids == [store.nodes[0].node_id]
    assert _records(tmp_path)[-1]["tail_node_ids"] == store.tail_node_ids
    result = check_evidence_storage(tmp_path)
    assert result["status"] == ("consistent" if after_replace else "snapshot_behind")
    if not after_replace:
        assert store.path.read_bytes() == old_snapshot
    assert result["write_blocked"] is True and result["exit_code"] == 8
    assert (tmp_path / "evidence_pending.json").exists()
    _assert_blocked_unchanged(store, tmp_path)
    with pytest.raises(EvidenceStorageError):
        EvidenceStore(tmp_path)


@pytest.mark.parametrize("fail_at", [1, 2, 3])
def test_each_append_directory_fsync_failure_preserves_phase_and_pending(tmp_path, monkeypatch, fail_at):
    store = _new(tmp_path)
    before, memory = _files(tmp_path), _memory(store)
    original = storage._sync_directory
    count = 0

    def fail_directory(path):
        nonlocal count
        count += 1
        if count == fail_at:
            raise OSError(f"injected directory fsync {fail_at}")
        original(path)

    monkeypatch.setattr(storage, "_sync_directory", fail_directory)
    with pytest.raises(EvidenceStorageError) as error:
        _add(store, 1)
    assert error.value.state == ("not_committed" if fail_at == 1 else "log_committed")
    expected = 1 if fail_at == 1 else 2
    assert error.value.committed_count == len(store.nodes) == len(_records(tmp_path)) == expected
    if fail_at == 1:
        assert _memory(store) == memory
        assert store.path.read_bytes() == before["evidence_chain.json"][0]
    else:
        assert store.tail_node_ids == [store.nodes[-1].node_id]
        assert len(load_evidence_chain(store.path)["nodes"]) == 2
    # After failed cleanup fsync, the implementation must re-establish the stop
    # marker; visible equal files alone cannot prove completion after restart.
    assert (tmp_path / "evidence_pending.json").exists()
    result = check_evidence_storage(tmp_path)
    assert result["status"] == "consistent" and result["exit_code"] == 8
    assert result["write_blocked"] is True
    _assert_blocked_unchanged(store, tmp_path)
    with pytest.raises(EvidenceStorageError):
        EvidenceStore(tmp_path)


@pytest.mark.parametrize("fail_at", [1, 2, 3, 4, 5, 6])
def test_each_initialization_directory_fsync_failure_cannot_be_retried_as_new(tmp_path, monkeypatch, fail_at):
    store = EvidenceStore(tmp_path)
    original = storage._sync_directory
    count = 0

    def fail_directory(path):
        nonlocal count
        count += 1
        if count == fail_at:
            raise OSError(f"injected initialization directory fsync {fail_at}")
        original(path)

    monkeypatch.setattr(storage, "_sync_directory", fail_directory)
    with pytest.raises(EvidenceStorageError) as error:
        store.save()
    assert error.value.state == "not_committed" and error.value.committed_count == 0
    assert store.nodes == [] and store.tail_node_ids == []
    assert (tmp_path / "evidence_pending.json").exists()
    assert check_evidence_storage(tmp_path)["exit_code"] != 0
    _assert_blocked_unchanged(store, tmp_path)
    before = _files(tmp_path)
    with pytest.raises(EvidenceStorageError):
        EvidenceStore.create(tmp_path)
    assert _files(tmp_path) == before


@pytest.mark.parametrize("failure", ["temp_fsync", "replace"])
def test_atomic_snapshot_actual_io_failures_keep_old_snapshot_and_new_log(tmp_path, monkeypatch, failure):
    store = _new(tmp_path)
    old_snapshot = store.path.read_bytes()
    original_write = storage.write_json
    original_fsync, original_replace = os.fsync, os.replace

    def fail_snapshot(path, value):
        if Path(path).name != "evidence_chain.json":
            return original_write(path, value)
        with monkeypatch.context() as isolated:
            if failure == "replace":
                def fail_replace(src, dst):
                    if Path(dst) == store.path:
                        raise OSError("injected atomic snapshot replace")
                    return original_replace(src, dst)
                isolated.setattr(utils.os, "replace", fail_replace)
            else:
                def fail_temp_fsync(fd):
                    if os.fstat(fd).st_mode & 0o170000 == 0o100000:
                        raise OSError("injected snapshot temp file fsync")
                    return original_fsync(fd)
                isolated.setattr(utils.os, "fsync", fail_temp_fsync)
            return original_write(path, value)

    monkeypatch.setattr(storage, "write_json", fail_snapshot)
    with pytest.raises(EvidenceStorageError) as error:
        _add(store, 1)
    assert error.value.state == "log_committed" and error.value.committed_count == 2
    assert len(store.nodes) == len(_records(tmp_path)) == 2
    assert store.path.read_bytes() == old_snapshot
    assert not list(tmp_path.glob(".evidence_chain.json.*.tmp"))
    assert (tmp_path / "evidence_pending.json").exists()
    assert check_evidence_storage(tmp_path)["status"] == "snapshot_behind"
    _assert_blocked_unchanged(store, tmp_path)


def test_pending_unlink_failure_does_not_roll_back_committed_memory(tmp_path, monkeypatch):
    store = _new(tmp_path)
    original = Path.unlink

    def fail_unlink(path, *args, **kwargs):
        if path.name == "evidence_pending.json":
            raise OSError("injected pending unlink failure")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_unlink)
    with pytest.raises(EvidenceStorageError) as error:
        _add(store, 1)
    assert error.value.state == "log_committed" and error.value.committed_count == 2
    assert len(store.nodes) == len(_records(tmp_path)) == 2
    assert store.tail_node_ids == [store.nodes[-1].node_id]
    assert (tmp_path / "evidence_pending.json").exists()
    result = check_evidence_storage(tmp_path)
    assert result["consistent"] is True and result["exit_code"] == 8
    _assert_blocked_unchanged(store, tmp_path)


@pytest.mark.parametrize("interrupt", [KeyboardInterrupt, SystemExit])
@pytest.mark.parametrize("stage", ["log_fsync", "snapshot", "cleanup_sync"])
def test_baseexception_interrupts_preserve_commit_state_and_restart_stop(tmp_path, monkeypatch, interrupt, stage):
    store = _new(tmp_path)
    memory = _memory(store)
    original_sync, original_write = storage._sync_directory, storage.write_json
    count = 0

    def interrupt_fsync(file):
        raise interrupt("interrupted before log fsync completed")

    def interrupt_snapshot(path, value):
        if Path(path).name == "evidence_chain.json":
            raise interrupt("interrupted before snapshot publication")
        return original_write(path, value)

    def interrupt_cleanup(path):
        nonlocal count
        count += 1
        if count == 3:
            raise interrupt("interrupted after pending unlink")
        original_sync(path)

    if stage == "log_fsync":
        monkeypatch.setattr(storage, "_fsync_log", interrupt_fsync)
    elif stage == "snapshot":
        monkeypatch.setattr(storage, "write_json", interrupt_snapshot)
    else:
        monkeypatch.setattr(storage, "_sync_directory", interrupt_cleanup)
    with pytest.raises(interrupt):
        _add(store, 1)
    if stage == "log_fsync":
        assert _memory(store) == memory
        assert len(_records(tmp_path)) == 2
    else:
        assert len(store.nodes) == len(_records(tmp_path)) == 2
        assert store.tail_node_ids == [store.nodes[-1].node_id]
    assert (tmp_path / "evidence_pending.json").exists()
    assert check_evidence_storage(tmp_path)["write_blocked"] is True
    _assert_blocked_unchanged(store, tmp_path)
    with pytest.raises(EvidenceStorageError):
        EvidenceStore(tmp_path)


def test_failed_stop_marker_restoration_explicitly_requires_operator_isolation(tmp_path, monkeypatch):
    store = _new(tmp_path)
    original_sync, original_write = storage._sync_directory, storage.write_json
    count = 0

    def fail_cleanup_sync(path):
        nonlocal count
        count += 1
        if count >= 3:
            raise OSError("medium failed during cleanup directory fsync")
        original_sync(path)

    def fail_restoring_pending(path, value):
        if Path(path).name == "evidence_pending.json" and count >= 3:
            raise OSError("medium cannot persist a stop marker")
        original_write(path, value)

    monkeypatch.setattr(storage, "_sync_directory", fail_cleanup_sync)
    monkeypatch.setattr(storage, "write_json", fail_restoring_pending)
    with pytest.raises(EvidenceStorageError) as error:
        _add(store, 1)
    assert error.value.state == "log_committed" and error.value.committed_count == 2
    assert len(store.nodes) == len(_records(tmp_path)) == 2
    assert not (tmp_path / "evidence_pending.json").exists()
    assert any("人工隔离" in note for note in error.value.__notes__)
    # The failed medium cannot encode durable quarantine. The local instance
    # stays blocked; a read-only content check has no record of this I/O error.
    assert check_evidence_storage(tmp_path)["consistent"] is True
    _assert_blocked_unchanged(store, tmp_path)


def test_tool_execution_does_not_report_success_after_evidence_commit_failure(tmp_path, monkeypatch):
    context = RunStore(tmp_path / "runs").create_run("storage error propagation")
    executor = ToolExecutor(context)
    executor.registry["storage_probe"] = lambda: {"summary": "tool computation finished"}
    original = storage._fsync_log

    def fail_fsync(file):
        raise OSError("injected evidence durability failure")

    monkeypatch.setattr(storage, "_fsync_log", fail_fsync)
    with pytest.raises(EvidenceStorageError):
        executor.execute("storage_probe", {})
    assert context.tool_results == []
    assert context.evidence.nodes == [] and context.evidence.tail_node_ids == []
    assert len(_records(context.run_dir)) == 1
    events = [json.loads(line) for line in context.event_path.read_text().splitlines()]
    assert not any(event["type"] == "tool_finished" for event in events)
    evidence_bytes = context.evidence.log_path.read_bytes()
    monkeypatch.setattr(storage, "_fsync_log", original)
    with pytest.raises(EvidenceStorageError):
        executor.execute("storage_probe", {})
    assert context.evidence.log_path.read_bytes() == evidence_bytes
    assert context.tool_results == [] and context.evidence.nodes == []
