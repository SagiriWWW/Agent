"""Read-only verification of raw evidence snapshots for CLI and Web.

The writer's compatibility loader deliberately fills defaults. This boundary
validates the raw shape first, then delegates all hashing to verify_chain().
"""

from __future__ import annotations

import errno
import json
import math
import os
from pathlib import Path
import re
import stat
from typing import Any

from aircraft_agent.evidence.chain import verify_chain


EXIT_CODES = {
    "pass": 0, "mismatch": 1, "legacy": 3, "file_not_found": 4,
    "invalid_json": 5, "unsupported_version": 6, "invalid_structure": 7,
    "read_error": 8, "path_rejected": 9,
}
HTTP_STATUS = {
    "pass": 200, "mismatch": 200, "legacy": 200, "file_not_found": 404,
    "invalid_json": 422, "unsupported_version": 422, "invalid_structure": 422,
    "read_error": 500, "path_rejected": 403,
}
SCOPE = (
    "结果仅证明本次读取快照的本地链内完整性，不代表外部真实性、独立标签正确性，"
    "也不代表运行附件内容已经逐项核验。完整重写并重算全部哈希无法由本地校验识别。"
    "校验不会修复、迁移或写回任何运行文件。"
)


def _result(status: str, **fields: Any) -> dict[str, Any]:
    return {
        "status": status, "valid": False, "verified": False,
        "schema_version": None, "node_count": None,
        "expected_chain_hash": None, "actual_chain_hash": None,
        "first_mismatch_index": None, "first_mismatch_node_id": None,
        "errors": [], "index_base": 0, "scope": SCOPE, **fields,
    }


def _error(status: str, message: str, **fields: Any) -> dict[str, Any]:
    message = message.encode("utf-8", errors="backslashreplace").decode("utf-8")
    return _result(status, errors=[{"kind": status, "message": message}], **fields)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    obj: dict[str, Any] = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError(f"重复 JSON 键：{key}")
        obj[key] = value
    return obj


def _finite_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("JSON 数值必须有限。")
    return number


def _reject_constant(value: str) -> None:
    raise ValueError(f"不支持非标准 JSON 数值：{value}")


def _check_unicode(raw: Any) -> None:
    # JSON escape sequences can decode to lone surrogates, even in a UTF-8
    # source. Reject these before any value reaches a CLI/HTTP response.
    pending = [raw]
    while pending:
        item = pending.pop()
        if isinstance(item, str):
            item.encode("utf-8")
        elif isinstance(item, dict):
            pending.extend(item.keys())
            pending.extend(item.values())
        elif isinstance(item, list):
            pending.extend(item)


def parse_strict_json_bytes(data: bytes) -> Any:
    """Decode JSON with the same strict rules for snapshots and storage records."""
    raw = json.loads(
        data.decode("utf-8"), object_pairs_hook=_unique_object,
        parse_float=_finite_float, parse_constant=_reject_constant,
    )
    _check_unicode(raw)
    return raw


def verify_evidence_snapshot_bytes(
    data: bytes,
) -> tuple[dict[str, Any] | list[Any] | None, dict[str, Any]]:
    """Parse once and verify the exact raw snapshot returned to the caller.

    Only a ``pass`` verification permits callers to treat the snapshot as
    hash-verifiable. Parsed legacy or malformed structures are retained for
    diagnostics without normalizing or repairing their contents.
    """
    try:
        raw = parse_strict_json_bytes(data)
    except (ValueError, UnicodeError, RecursionError) as exc:
        return None, _error("invalid_json", f"证据文件不是有效的 UTF-8 JSON：{exc}")

    snapshot = raw if isinstance(raw, (dict, list)) else None
    return snapshot, _verify_raw_snapshot(raw)


def _verify_bytes(data: bytes) -> dict[str, Any]:
    return verify_evidence_snapshot_bytes(data)[1]


def _verify_raw_snapshot(raw: Any) -> dict[str, Any]:
    if isinstance(raw, list):
        version, nodes, expected_hash = 1, raw, None
    elif isinstance(raw, dict):
        version = raw.get("schema_version", 1)
        nodes = raw.get("nodes", raw.get("evidence") if version == 1 else None)
        expected_hash = raw.get("chain_hash")
    else:
        return _error("invalid_structure", "证据链必须为对象或旧版节点列表。")

    info = {
        "schema_version": version,
        "node_count": len(nodes) if isinstance(nodes, list) else None,
        "expected_chain_hash": expected_hash if isinstance(expected_hash, str) else None,
    }
    if type(version) is not int:
        return _error("invalid_structure", "schema_version 必须为整数。", **info)
    if version not in (1, 2):
        return _error("unsupported_version", f"不支持证据链版本 {version}。", **info)
    if not isinstance(nodes, list):
        return _error("invalid_structure", "证据链必须显式包含 nodes 列表。", **info)
    if expected_hash is not None and not isinstance(expected_hash, str):
        return _error("invalid_structure", "chain_hash 必须为字符串或 null。", **info)

    seen: set[str] = set()
    for index, node in enumerate(nodes):
        message = _node_structure_error(node, version, seen)
        if message:
            node_id = node.get("node_id") if isinstance(node, dict) else None
            status = "unsupported_version" if message == "unsupported_version" else "invalid_structure"
            return _result(
                status, **info, first_mismatch_index=index,
                first_mismatch_node_id=node_id if isinstance(node_id, str) and node_id else None,
                errors=[{"kind": status, "index": index, "node_id": node_id,
                         "message": "不支持节点版本。" if status == "unsupported_version" else message}],
            )

    # Never pass through EvidenceStore/load_evidence_chain: missing stored hashes
    # must reach verify_chain unchanged, with its original mismatch semantics.
    try:
        result = verify_chain(raw).to_dict()
    except (ValueError, TypeError, OverflowError, RecursionError) as exc:
        return _error("invalid_structure", f"证据链无法校验：{exc}", **info)
    status = "legacy" if not result["verified"] else "pass" if result["valid"] else "mismatch"
    return _result(status, **result)


def _node_structure_error(node: Any, version: int, seen: set[str]) -> str | None:
    if not isinstance(node, dict):
        return "节点必须为对象，不能忽略非法节点。"
    node_id = node.get("node_id")
    if not isinstance(node_id, str) or not node_id:
        return "node_id 必须为非空字符串。"
    if node_id in seen:
        return "node_id 重复。"
    seen.add(node_id)
    node_version = node.get("schema_version", 1 if version == 1 else None)
    if type(node_version) is not int:
        return "节点 schema_version 必须为显式整数。"
    if node_version not in (1, 2):
        return "unsupported_version"
    # v1 may omit the v2 fields. Previously persisted v2 envelopes can contain
    # explicit v1 nodes; verify their stored hashes without rewriting them.
    required = ("parent_ids", "parent_hashes", "type", "tool_name", "input_hash",
                "output_hash", "arguments", "summary", "timestamp") if version == 2 else ()
    for field in required:
        if field not in node:
            return f"节点缺少字段：{field}。"
    for field in ("parent_ids", "parent_hashes"):
        if field in node and (not isinstance(node[field], list) or
                              any(not isinstance(item, str) or not item for item in node[field])):
            return f"{field} 必须为非空字符串的列表。"
    if len(set(node.get("parent_ids", []))) != len(node.get("parent_ids", [])):
        return "parent_ids 不能重复。"
    for field in ("type", "input_hash", "output_hash", "summary", "timestamp"):
        if field in node and not isinstance(node[field], str):
            return f"{field} 必须为字符串。"
    for field in ("tool_name", "model_response_id", "node_hash"):
        if field in node and node[field] is not None and not isinstance(node[field], str):
            return f"{field} 必须为字符串或 null。"
    if "arguments" in node and not isinstance(node["arguments"], dict):
        return "arguments 必须为对象。"
    return None


def _read_snapshot_fd(
    fd: int,
) -> tuple[dict[str, Any] | list[Any] | None, dict[str, Any]]:
    """Take ownership of an already opened descriptor and close it on all paths."""
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            return None, _error("read_error", "证据路径必须为普通文件。")
        with os.fdopen(fd, "rb", closefd=False) as source:
            return verify_evidence_snapshot_bytes(source.read())
    finally:
        os.close(fd)


def _read_fd(fd: int) -> dict[str, Any]:
    return _read_snapshot_fd(fd)[1]


def _io_error(exc: OSError, *, bounded: bool = False) -> dict[str, Any]:
    if bounded and exc.errno in (errno.ELOOP, errno.ENOTDIR):
        return _error("path_rejected", "运行或证据路径包含符号链接或非法目录。")
    if isinstance(exc, FileNotFoundError):
        return _error("file_not_found", "证据文件或运行目录不存在。")
    return _error("read_error", "无法读取证据文件，请检查文件类型和读取权限。")


def read_evidence_snapshot(
    path: Path,
) -> tuple[dict[str, Any] | list[Any] | None, dict[str, Any]]:
    """Read once and return raw contents plus their strict, read-only verdict."""
    try:
        return _read_snapshot_fd(os.open(path, os.O_RDONLY | os.O_NONBLOCK))
    except OSError as exc:
        return None, _io_error(exc)
    except ValueError:
        return None, _error("read_error", "证据文件路径无效。")


def verify_evidence_file(path: Path) -> dict[str, Any]:
    """Verify an explicitly chosen CLI file without reading app/model settings."""
    return read_evidence_snapshot(path)[1]


def verify_run_evidence(storage_dir: Path, run_id: str) -> dict[str, Any]:
    """Read only <configured root>/<run_id>/evidence_chain.json.

    The configured root is the trust boundary (it may itself be a configured
    symlink). Relative descriptor opens pin directories and O_NOFOLLOW rejects
    run/file symlinks, including internal links and replacements during lookup.
    """
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", run_id):
        return _error("path_rejected", "run_id 必须为 1–128 位字母、数字、下划线或连字符，且以字母或数字开头。")
    try:
        root = storage_dir.resolve(strict=True)
        root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            run_fd = os.open(run_id, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=root_fd)
            try:
                fd = os.open("evidence_chain.json", os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=run_fd)
                return _read_fd(fd)
            finally:
                os.close(run_fd)
        finally:
            os.close(root_fd)
    except OSError as exc:
        return _io_error(exc, bounded=True)
    except (ValueError, RuntimeError):
        return _error("path_rejected", "配置的运行根目录无效或包含循环符号链接。")
