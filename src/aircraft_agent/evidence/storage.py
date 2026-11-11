"""Versioned JSONL storage and read-only dual-file inspection.

The lock is cooperative and local: POSIX flock plus a per-directory thread
mutex. No claims are made about multi-host/cloud-sync writers or hostile edits.
JSONL fsync commits a node; the compatibility snapshot and durable removal of
an intent marker must also succeed before an operation reports full success.
"""
from __future__ import annotations

from contextlib import contextmanager
import errno
import os
from pathlib import Path
import stat
import threading
from typing import Any, BinaryIO, Iterator

from aircraft_agent.common.utils import stable_json, write_json

try:
    import fcntl
except ImportError:  # Read-only legacy snapshots remain usable on non-POSIX.
    fcntl = None  # type: ignore[assignment]

STORAGE_PROTOCOL_VERSION = 1
STORAGE_MODE = "jsonl_snapshot"
PROTOCOL_FILE = "evidence_storage.json"
LOG_FILE = "evidence_chain.jsonl"
SNAPSHOT_FILE = "evidence_chain.json"
LOCK_FILE = ".evidence.lock"
PENDING_FILE = "evidence_pending.json"
RECOVERY_FILE = "evidence_recovery.json"
RECOVERY_DIR = "evidence_recoveries"
PROTOCOL = {"storage_protocol_version": STORAGE_PROTOCOL_VERSION, "mode": STORAGE_MODE}
STORAGE_EXIT_CODES = {
    "consistent": 0, "snapshot_behind": 1, "snapshot_ahead": 1, "diverged": 1,
    "log_corrupt": 1, "snapshot_corrupt": 1, "missing_log": 1,
    "legacy_no_log": 3, "file_not_found": 4,
    "invalid_protocol": 8, "read_error": 8, "busy": 8, "pending": 8,
}
STORAGE_SCOPE = (
    "仅检查本次共享锁内可比较的兼容快照、JSONL 日志及存储协议/未完成事务标记。"
    "快照完整性不代表日志完整性；日志哈希按既有 Evidence schema v2 校验。"
    "日志追加约束仅覆盖遵守本协议的同机应用写入，不证明外部真实性、附件内容、"
    "操作系统级不可修改、外部不可抵赖或完整重写攻击可检测。"
    "本检查不创建文件、不迁移、不修复；不支持跨设备同步目录的并行写入。"
)
_THREAD_LOCKS: dict[str, threading.Lock] = {}
_THREAD_LOCKS_GUARD = threading.Lock()
_ACTIVE_LOCK_FDS: set[int] = set()


def _after_fork_child() -> None:
    # A child cannot inherit a Python mutex held by a vanished parent thread.
    # Close inherited flock descriptors too: they share the parent's open file
    # description and would otherwise keep its locks alive after parent exit.
    global _THREAD_LOCKS, _THREAD_LOCKS_GUARD, _ACTIVE_LOCK_FDS
    for fd in _ACTIVE_LOCK_FDS:
        try:
            os.close(fd)
        except OSError:
            pass
    _THREAD_LOCKS = {}
    _ACTIVE_LOCK_FDS = set()
    _THREAD_LOCKS_GUARD = threading.Lock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(before=lambda: _THREAD_LOCKS_GUARD.acquire(),
                        after_in_parent=lambda: _THREAD_LOCKS_GUARD.release(),
                        after_in_child=_after_fork_child)


def _diagnostic(value: Any) -> str:
    """Keep malformed input diagnostics safe for UTF-8 CLI/JSON rendering."""
    return str(value).encode("utf-8", errors="backslashreplace").decode("utf-8")


class EvidenceStorageError(RuntimeError):
    """A refused or incomplete operation, with its known commit outcome."""

    def __init__(self, message: str, *, state: str = "recovery_required", committed_count: int = 0) -> None:
        super().__init__(_diagnostic(message))
        self.state = state
        self.committed_count = committed_count
        self.recovery_required = state != "stale_instance"


def _thread_lock(run_dir: Path) -> threading.Lock:
    key = str(run_dir.resolve())
    with _THREAD_LOCKS_GUARD:
        return _THREAD_LOCKS.setdefault(key, threading.Lock())


@contextmanager
def storage_lock(run_dir: Path, *, shared: bool, create: bool = False,
                 blocking: bool = True) -> Iterator[None]:
    """Lock a permanent inode; readers never create or replace that inode."""
    if fcntl is None:
        raise EvidenceStorageError("存储协议需要 macOS/Linux POSIX flock 支持。")
    creator_pid = os.getpid()
    mutex = _thread_lock(run_dir)
    if not mutex.acquire(blocking=blocking):
        raise BlockingIOError(errno.EWOULDBLOCK, "evidence writer is busy")
    fd: int | None = None
    try:
        flags = os.O_RDONLY if shared else os.O_RDWR
        if create:
            run_dir.mkdir(parents=True, exist_ok=True)
            flags |= os.O_CREAT
        with _THREAD_LOCKS_GUARD:
            fd = os.open(run_dir / LOCK_FILE, flags | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600)
            _ACTIVE_LOCK_FDS.add(fd)
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise EvidenceStorageError("证据锁必须为永久普通文件。")
        operation = fcntl.LOCK_SH if shared else fcntl.LOCK_EX
        fcntl.flock(fd, operation | (0 if blocking else fcntl.LOCK_NB))
        if os.stat(run_dir / LOCK_FILE, follow_symlinks=False).st_ino != os.fstat(fd).st_ino:
            raise EvidenceStorageError("证据锁文件已替换；需要人工隔离并核对运行。")
        yield
    finally:
        if os.getpid() == creator_pid:
            if fd is not None:
                with _THREAD_LOCKS_GUARD:
                    _ACTIVE_LOCK_FDS.discard(fd)
                    os.close(fd)
            mutex.release()


def _entry_exists(path: Path) -> bool:
    # A dangling symlink is still an on-disk protocol entry, never an absent
    # marker that permits initialization or downgrading to a legacy run.
    try:
        path.lstat()
        return True
    except FileNotFoundError:
        return False


def _read_regular(path: Path) -> bytes:
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise OSError("evidence storage path must be a regular file")
        with os.fdopen(fd, "rb", closefd=False) as source:
            return source.read()
    finally:
        os.close(fd)


def _strict_json(data: bytes) -> Any:
    from aircraft_agent.evidence.verification import parse_strict_json_bytes
    return parse_strict_json_bytes(data)


def _envelope(nodes: list[dict[str, Any]]) -> dict[str, Any]:
    from aircraft_agent.evidence.chain import EVIDENCE_SCHEMA_VERSION, compute_chain_hash
    return {"schema_version": EVIDENCE_SCHEMA_VERSION, "chain_hash": compute_chain_hash(nodes), "nodes": nodes}


def _parse_log(data: bytes) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from aircraft_agent.evidence.chain import NODE_HASH_FIELDS, EVIDENCE_SCHEMA_VERSION
    from aircraft_agent.evidence.verification import _verify_raw_snapshot
    records: list[dict[str, Any]] = []
    try:
        tail: list[str] = []
        for index, line in enumerate(data.splitlines(keepends=True)):
            if not line.endswith(b"\n") or not line.strip():
                raise ValueError(f"日志第 {index} 条不是完整记录。")
            record = _strict_json(line)
            if not isinstance(record, dict) or set(record) != {"storage_protocol_version", "sequence", "node", "tail_node_ids"}:
                raise ValueError(f"日志第 {index} 条存储记录字段非法。")
            if type(record["storage_protocol_version"]) is not int or record["storage_protocol_version"] != STORAGE_PROTOCOL_VERSION:
                raise ValueError(f"日志第 {index} 条存储协议版本非法。")
            if type(record["sequence"]) is not int or record["sequence"] != index:
                raise ValueError(f"日志第 {index} 条顺序非法。")
            node = record["node"]
            if not isinstance(node, dict) or set(node) != {*NODE_HASH_FIELDS, "node_hash"}:
                raise ValueError(f"日志第 {index} 条必须包含完整且仅限 EvidenceNode.to_dict 的节点字段。")
            if type(node["schema_version"]) is not int or node["schema_version"] != EVIDENCE_SCHEMA_VERSION:
                raise ValueError(f"日志第 {index} 条节点必须为 Evidence schema v2。")
            next_tail = record["tail_node_ids"]
            if not isinstance(next_tail, list) or any(not isinstance(item, str) or not item for item in next_tail):
                raise ValueError(f"日志第 {index} 条 tail_node_ids 非法。")
            if next_tail != tail and next_tail != [node.get("node_id")]:
                raise ValueError(f"日志第 {index} 条主链尾不符合追加协议。")
            tail = next_tail
            records.append(record)
        result = _verify_raw_snapshot(_envelope([record["node"] for record in records]))
        if result["status"] != "pass":
            return records, {**result, "record_count": len(records)}
        return records, {**result, "record_count": len(records)}
    except (ValueError, UnicodeError, TypeError, RecursionError, OverflowError) as exc:
        return records, {"status": "invalid_structure", "valid": False, "verified": False,
                         "record_count": len(records), "node_count": len(records),
                         "errors": [{"kind": "log_corrupt", "index": len(records), "message": _diagnostic(exc)}]}


def _base_result(status: str, **fields: Any) -> dict[str, Any]:
    return {"status": status, "consistent": False, "comparable": False,
            "storage_protocol_version": None, "mode": None,
            "snapshot": None, "log": None, "write_state": "unknown", "write_blocked": True,
            "errors": [], "scope": STORAGE_SCOPE, "index_base": 0,
            "exit_code": STORAGE_EXIT_CODES[status], **fields}


def _inspect_locked(run_dir: Path, *, allow_new_mode: bool = True) -> tuple[dict[str, Any], list[dict[str, Any]], bytes]:
    from aircraft_agent.evidence.verification import verify_evidence_snapshot_bytes
    from aircraft_agent.evidence.recovery import inspect_recovery_state
    marker = run_dir / PROTOCOL_FILE
    pending_exists = _entry_exists(run_dir / PENDING_FILE)
    recovery, generation = inspect_recovery_state(run_dir)
    recovery_blocked = bool(recovery and recovery["write_blocked"])
    snapshot_raw = None
    try:
        snapshot_raw, snapshot = verify_evidence_snapshot_bytes(_read_regular(run_dir / SNAPSHOT_FILE))
    except FileNotFoundError:
        snapshot = {"status": "file_not_found", "valid": False, "verified": False, "node_count": None}
    if not _entry_exists(marker):
        if _entry_exists(run_dir / LOG_FILE) or pending_exists or recovery is not None:
            return _base_result("invalid_protocol", snapshot=snapshot,
                                errors=[{"kind": "unmarked_storage", "message": "存在日志或事务意图但缺少存储协议标记，禁止降级。"}]), [], b""
        status = "legacy_no_log" if _entry_exists(run_dir / SNAPSHOT_FILE) else "file_not_found"
        return _base_result(status, snapshot=snapshot, mode="legacy_snapshot",
                            write_state="legacy_read_only"), [], b""
    if not allow_new_mode:
        return _base_result("busy", snapshot=snapshot, errors=[{
            "kind": "busy", "message": "无锁探测期间出现新存储模式；本次不比较双文件，请复跑获取共享锁视图。",
        }]), [], b""
    try:
        protocol = _strict_json(_read_regular(marker))
        if not isinstance(protocol, dict) or set(protocol) != set(PROTOCOL) or stable_json(protocol) != stable_json(PROTOCOL):
            raise ValueError("未知或非法存储协议。")
    except (ValueError, UnicodeError, RecursionError) as exc:
        return _base_result("invalid_protocol", snapshot=snapshot,
                            errors=[{"kind": "invalid_protocol", "message": _diagnostic(exc)}]), [], b""
    info = {"snapshot": snapshot, **PROTOCOL}
    try:
        data = _read_regular(run_dir / LOG_FILE)
    except FileNotFoundError:
        return _base_result("missing_log", **info, write_state="recovery_required",
                            errors=[{"kind": "missing_log", "message": "新模式日志缺失，禁止降级或回填。"}]), [], b""
    records, log = _parse_log(data)
    info["log"] = log
    if log["status"] != "pass":
        status = "log_corrupt"
    elif snapshot["status"] != "pass":
        status = "snapshot_corrupt"
    else:
        snapshot_nodes = snapshot_raw["nodes"]
        log_nodes = [record["node"] for record in records]
        common = min(len(snapshot_nodes), len(log_nodes))
        if any(stable_json(snapshot_nodes[index]) != stable_json(log_nodes[index]) for index in range(common)):
            status = "diverged"
        elif len(snapshot_nodes) < len(log_nodes):
            status = "snapshot_behind"
        elif len(snapshot_nodes) > len(log_nodes):
            status = "snapshot_ahead"
        else:
            status = "consistent"
    pending = None
    errors = []
    if status != "consistent":
        detail: dict[str, Any] = {"kind": status, "message": "双文件未通过一致性检查；停止追加，保留全部原始文件，另行审核并恢复。"}
        if status in {"snapshot_behind", "snapshot_ahead", "diverged"}:
            detail.update(snapshot_node_count=len(snapshot_nodes), log_record_count=len(log_nodes))
        if status == "diverged":
            mismatch = next(index for index in range(common)
                            if stable_json(snapshot_nodes[index]) != stable_json(log_nodes[index]))
            detail.update(index=mismatch, snapshot_node_id=snapshot_nodes[mismatch]["node_id"],
                          log_node_id=log_nodes[mismatch]["node_id"])
        errors.append(detail)
    if pending_exists:
        try:
            pending = _strict_json(_read_regular(run_dir / PENDING_FILE))
        except (ValueError, UnicodeError, RecursionError) as exc:
            pending = {"invalid": _diagnostic(exc)}
        errors.append({"kind": "pending", "message": "存在未完成事务意图；内容一致也不能推定上次操作完整成功，需要人工恢复审核。"})
    consistent = status == "consistent"
    if recovery_blocked:
        errors.append({"kind": "recovery_pending", "message": recovery["reason"]})
    ready = consistent and not pending_exists and not recovery_blocked
    return _base_result(status, **info, comparable=log["status"] == snapshot["status"] == "pass",
                        consistent=consistent, pending=pending, recovery=recovery, errors=errors,
                        recovery_generation=generation.decode("utf-8", errors="backslashreplace"),
                        write_state="ready" if ready else "recovery_required", write_blocked=not ready,
                        exit_code=0 if ready else 8 if pending_exists or recovery_blocked else STORAGE_EXIT_CODES[status]), records, data


def check_evidence_storage(run_dir: Path) -> dict[str, Any]:
    """Inspect an existing run without creating, repairing or migrating files."""
    run_dir = Path(run_dir)
    try:
        if _entry_exists(run_dir / LOCK_FILE):
            with storage_lock(run_dir, shared=True, blocking=False):
                return _inspect_locked(run_dir)[0]
        if _entry_exists(run_dir / PROTOCOL_FILE):
            return _base_result("invalid_protocol", errors=[{"kind": "missing_lock", "message": "新模式永久锁缺失；无法取得可比较视图。"}])
        # A writer may initialize after the probes above. This branch can
        # inspect legacy snapshots only, never pronounce a new mode consistent.
        return _inspect_locked(run_dir, allow_new_mode=False)[0]
    except BlockingIOError:
        return _base_result("busy", errors=[{"kind": "busy", "message": "写入正在进行；本次未取得双文件一致视图，请稍后复跑。"}])
    except (OSError, EvidenceStorageError, ValueError, RuntimeError) as exc:
        return _base_result("read_error", errors=[{"kind": "read_error", "message": _diagnostic(exc)}])


verify_evidence_storage = check_evidence_storage


def _sync_directory(path: Path) -> None:
    """Strict directory fsync; storage must never swallow this failure."""
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _verify_published_json(path: Path, expected: dict[str, Any]) -> None:
    # Keep the existing atomic replacement helper, but also reject a silent
    # short write or unexpected publication before calling the transaction done.
    actual = _strict_json(_read_regular(path))
    if stable_json(actual) != stable_json(expected):
        raise OSError(f"原子发布内容与预期不符：{path.name}")


def _write_record(log: BinaryIO, data: bytes) -> int:
    return log.write(data)


def _flush_log(log: BinaryIO) -> None:
    log.flush()


def _fsync_log(log: BinaryIO) -> None:
    os.fsync(log.fileno())


class EvidenceStorage:
    """Internal transaction coordinator; EvidenceStore owns node construction."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = Path(run_dir)
        self.creator_pid = os.getpid()
        self.mode = "uninitialized"
        self.records: list[dict[str, Any]] = []
        self.log_bytes = b""
        self.recovery_generation = ""
        self.blocked: EvidenceStorageError | None = None
        if any(_entry_exists(self.run_dir / name) for name in (PROTOCOL_FILE, LOG_FILE, PENDING_FILE, RECOVERY_FILE, RECOVERY_DIR)):
            try:
                with storage_lock(self.run_dir, shared=True):
                    result, records, data = _inspect_locked(self.run_dir)
            except (OSError, ValueError) as exc:
                raise EvidenceStorageError(f"无法读取新模式的完整锁定视图：{exc}；停止追加并保留现场。") from exc
            self._require_ready(result)
            self.mode, self.records, self.log_bytes = STORAGE_MODE, records, data
            self.recovery_generation = result["recovery_generation"]
        elif _entry_exists(self.run_dir / SNAPSHOT_FILE):
            self.mode = "legacy_snapshot"

    @property
    def nodes(self) -> list[dict[str, Any]]:
        return [record["node"] for record in self.records]

    @property
    def tail(self) -> list[str]:
        return list(self.records[-1]["tail_node_ids"]) if self.records else []

    def _require_ready(self, result: dict[str, Any]) -> None:
        if result["write_blocked"]:
            raise EvidenceStorageError(f"证据存储状态 {result['status']} / {result['write_state']}：停止追加；保留全部文件并在独立恢复审核后处理。",
                                       committed_count=len(self.records))

    def _remember_failure(self, exc: BaseException, state: str) -> EvidenceStorageError:
        self.blocked = EvidenceStorageError(f"证据持久化未完整成功 ({state})：{exc}。停止写入，保留现场并人工核对事务意图与双文件。",
                                            state=state, committed_count=len(self.records))
        return self.blocked

    def _ensure_pending(self, intent: dict[str, Any]) -> None:
        write_json(self.run_dir / PENDING_FILE, intent)
        _sync_directory(self.run_dir)
        _verify_published_json(self.run_dir / PENDING_FILE, intent)

    def _restore_pending(self, intent: dict[str, Any]) -> None:
        if not _entry_exists(self.run_dir / PENDING_FILE):
            try:
                self._ensure_pending(intent)
            except BaseException:
                # If the medium cannot persist even a stop marker, no on-disk
                # protocol can guarantee restart quarantine. The exception
                # explicitly requires operator isolation before another writer.
                if self.blocked:
                    self.blocked.add_note("无法重新持久化停止标记；必须在重启或其他写入前人工隔离此运行目录。")

    def _initialize_locked(self) -> None:
        if any(_entry_exists(self.run_dir / name) for name in (PROTOCOL_FILE, LOG_FILE, SNAPSHOT_FILE, PENDING_FILE, RECOVERY_FILE, RECOVERY_DIR)):
            raise EvidenceStorageError("运行在实例构造后已出现证据文件；此陈旧实例不得初始化或降级。", state="stale_instance")
        intent = {**PROTOCOL, "operation": "initialize", "committed_count": 0}
        try:
            _sync_directory(self.run_dir)
            self._ensure_pending(intent)
            write_json(self.run_dir / PROTOCOL_FILE, PROTOCOL)
            _sync_directory(self.run_dir)
            _verify_published_json(self.run_dir / PROTOCOL_FILE, PROTOCOL)
            fd = os.open(self.run_dir / LOG_FILE, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
            _sync_directory(self.run_dir)
            write_json(self.run_dir / SNAPSHOT_FILE, _envelope([]))
            _sync_directory(self.run_dir)
            _verify_published_json(self.run_dir / SNAPSHOT_FILE, _envelope([]))
            (self.run_dir / PENDING_FILE).unlink()
            _sync_directory(self.run_dir)
            self.mode = STORAGE_MODE
        except BaseException as exc:
            error = self._remember_failure(exc, "not_committed")
            self._restore_pending(intent)
            if not isinstance(exc, Exception):
                exc.add_note(str(error))
                raise
            raise error from exc

    @contextmanager
    def writer(self) -> Iterator[None]:
        if self.creator_pid != os.getpid():
            raise EvidenceStorageError("fork 后必须重新构造 EvidenceStore，禁止复用父进程写入实例。", state="stale_instance", committed_count=len(self.records))
        if self.blocked:
            raise self.blocked
        if self.mode == "legacy_snapshot":
            raise EvidenceStorageError("旧运行只读：继续分析请创建新运行；本轮不迁移或回填历史日志。")
        try:
            with storage_lock(self.run_dir, shared=False, create=self.mode == "uninitialized"):
                if self.mode == "uninitialized":
                    self._initialize_locked()
                result, records, data = _inspect_locked(self.run_dir)
                self._require_ready(result)
                if (data != self.log_bytes or stable_json(records) != stable_json(self.records)
                        or result["recovery_generation"] != self.recovery_generation):
                    raise EvidenceStorageError("陈旧 EvidenceStore 实例：节点身份、顺序或完整已提交前缀已变化；重新加载一致运行后再写。",
                                               state="stale_instance", committed_count=len(self.records))
                yield
        except EvidenceStorageError:
            raise
        except Exception as exc:
            raise self._remember_failure(exc, "not_committed") from exc

    def validate_memory(self, nodes: list[dict[str, Any]], tail: list[str]) -> None:
        if stable_json(nodes) != stable_json(self.nodes) or stable_json(tail) != stable_json(self.tail):
            raise EvidenceStorageError("内存节点或主链尾已偏离已提交完整前缀；拒绝修改历史或手动插入节点。",
                                       state="stale_instance", committed_count=len(self.records))

    def commit(self, node: dict[str, Any], tail: list[str]) -> None:
        """Append exactly one validated record while the caller holds writer()."""
        record = {"storage_protocol_version": STORAGE_PROTOCOL_VERSION, "sequence": len(self.records),
                  "node": node, "tail_node_ids": tail}
        payload = (stable_json(record) + "\n").encode("utf-8")
        records, verdict = _parse_log(self.log_bytes + payload)
        if verdict["status"] != "pass":
            raise EvidenceStorageError(f"待写入节点未通过严格链校验：{verdict['errors']}", state="not_committed", committed_count=len(self.records))
        intent = {**PROTOCOL, "operation": "append", "committed_count": len(self.records),
                  "previous_chain_hash": _envelope(self.nodes)["chain_hash"], "record": record}
        committed = False
        before_bytes = self.log_bytes
        try:
            self._ensure_pending(intent)
            fd = os.open(self.run_dir / LOG_FILE, os.O_WRONLY | os.O_APPEND | os.O_NOFOLLOW)
            try:
                with os.fdopen(fd, "ab", buffering=0, closefd=False) as log:
                    offset = 0
                    while offset < len(payload):
                        written = _write_record(log, payload[offset:])
                        if type(written) is not int or written <= 0 or written > len(payload) - offset:
                            raise OSError("JSONL 写入没有进展或返回非法短写长度。")
                        offset += written
                    _flush_log(log)
                    _fsync_log(log)
                    committed = True
                    self.records, self.log_bytes = records, before_bytes + payload
            finally:
                os.close(fd)
            write_json(self.run_dir / SNAPSHOT_FILE, _envelope(self.nodes))
            _sync_directory(self.run_dir)
            _verify_published_json(self.run_dir / SNAPSHOT_FILE, _envelope(self.nodes))
            (self.run_dir / PENDING_FILE).unlink()
            _sync_directory(self.run_dir)
        except BaseException as exc:
            if committed:
                state = "log_committed"
            else:
                try:
                    state = "not_committed" if _read_regular(self.run_dir / LOG_FILE) == before_bytes else "uncertain"
                except Exception:
                    state = "uncertain"
            error = self._remember_failure(exc, state)
            self._restore_pending(intent)
            if not isinstance(exc, Exception):
                exc.add_note(str(error))
                raise
            raise error from exc
