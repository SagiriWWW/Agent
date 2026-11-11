"""Evidence-chain recording.

证据链是 MVP 的审计核心。每个节点记录输入参数 hash、输出 hash、摘要、
时间戳和父节点。新运行使用可追加 JSONL 和兼容 JSON 快照，存储协议独立
于节点 schema 与哈希算法；旧运行仍可读取且不会隐式迁移。
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from aircraft_agent.common.models import EvidenceNode, EvidenceVerificationResult, StoredFile
from aircraft_agent.common.utils import hash_json, read_json, stable_json, truncate_text, utc_now_iso


EVIDENCE_SCHEMA_VERSION = 2

NODE_HASH_FIELDS = (
    "schema_version",
    "node_id",
    "parent_ids",
    "parent_hashes",
    "type",
    "tool_name",
    "input_hash",
    "output_hash",
    "arguments",
    "summary",
    "timestamp",
    "model_response_id",
)


def load_evidence_chain(path: Path) -> dict[str, Any]:
    """Read v2 or legacy evidence_chain.json into a normalized envelope."""
    return normalize_evidence_chain(read_json(path))


def normalize_evidence_chain(raw: Any) -> dict[str, Any]:
    """Normalize legacy list files and v2 envelopes without requiring v2 fields."""
    if isinstance(raw, list):
        return {
            "schema_version": 1,
            "chain_hash": None,
            "nodes": [dict(node) for node in raw if isinstance(node, dict)],
        }
    if not isinstance(raw, dict):
        raise ValueError("evidence chain must be a JSON object or legacy node list")

    nodes = raw.get("nodes")
    if nodes is None and isinstance(raw.get("evidence"), list):
        nodes = raw.get("evidence")
    if nodes is None:
        nodes = []
    if not isinstance(nodes, list):
        raise ValueError("evidence chain nodes must be a list")

    return {
        "schema_version": int(raw.get("schema_version") or 1),
        "chain_hash": raw.get("chain_hash"),
        "nodes": [dict(node) for node in nodes if isinstance(node, dict)],
    }


def compute_node_hash(node: EvidenceNode | dict[str, Any]) -> str:
    """Compute a deterministic node hash, excluding node_hash itself."""
    data = _node_to_dict(node)
    payload = {field: _node_hash_value(data, field) for field in NODE_HASH_FIELDS}
    return hash_json(payload)


def compute_chain_hash(nodes: list[EvidenceNode] | list[dict[str, Any]]) -> str:
    """Compute a deterministic run-level hash from the ordered node contents."""
    node_hashes = [compute_node_hash(node) for node in nodes]
    return hash_json({"schema_version": EVIDENCE_SCHEMA_VERSION, "node_hashes": node_hashes})


def evidence_node_reference(node: EvidenceNode | dict[str, Any]) -> dict[str, Any]:
    """Return the stable fields needed to locate one evidence-chain node."""
    data = _node_to_dict(node)
    node_hash = data.get("node_hash") or compute_node_hash(data)
    return {
        "schema_version": int(data.get("schema_version") or EVIDENCE_SCHEMA_VERSION),
        "node_id": str(data.get("node_id") or ""),
        "node_hash": str(node_hash),
        "type": data.get("type"),
        "tool_name": data.get("tool_name"),
    }


def verify_chain(source: Path | dict[str, Any] | list[dict[str, Any]]) -> EvidenceVerificationResult:
    """Verify a v2 evidence chain and locate the first integrity mismatch.

    Legacy v1 list files are readable but not hash-verifiable, so they return
    verified=False while preserving node_count and schema_version.
    """
    raw = read_json(source) if isinstance(source, Path) else source
    chain = normalize_evidence_chain(raw)
    schema_version = int(chain["schema_version"])
    nodes = list(chain["nodes"])
    expected_chain_hash = chain.get("chain_hash")

    if schema_version < EVIDENCE_SCHEMA_VERSION:
        return EvidenceVerificationResult(
            valid=True,
            verified=False,
            schema_version=schema_version,
            node_count=len(nodes),
            expected_chain_hash=expected_chain_hash,
            actual_chain_hash=None,
        )

    errors: list[dict[str, Any]] = []
    first_mismatch_index: int | None = None
    first_mismatch_node_id: str | None = None
    actual_hash_by_id: dict[str, str] = {}
    actual_node_hashes: list[str] = []

    for index, node in enumerate(nodes):
        node_id = str(node.get("node_id") or "")
        stored_node_hash = str(node.get("node_hash") or "")
        actual_node_hash = compute_node_hash(node)
        actual_node_hashes.append(actual_node_hash)

        if not stored_node_hash:
            errors.append(
                {
                    "kind": "missing_node_hash",
                    "index": index,
                    "node_id": node_id,
                }
            )
            first_mismatch_index, first_mismatch_node_id = _first_mismatch(
                first_mismatch_index,
                first_mismatch_node_id,
                index,
                node_id,
            )
        elif stored_node_hash != actual_node_hash:
            errors.append(
                {
                    "kind": "node_hash_mismatch",
                    "index": index,
                    "node_id": node_id,
                    "expected": stored_node_hash,
                    "actual": actual_node_hash,
                }
            )
            first_mismatch_index, first_mismatch_node_id = _first_mismatch(
                first_mismatch_index,
                first_mismatch_node_id,
                index,
                node_id,
            )

        parent_ids = [str(parent_id) for parent_id in node.get("parent_ids") or []]
        parent_hashes = [str(parent_hash) for parent_hash in node.get("parent_hashes") or []]
        expected_parent_hashes: list[str] = []
        unknown_parent_ids: list[str] = []
        for parent_id in parent_ids:
            parent_hash = actual_hash_by_id.get(parent_id)
            if parent_hash is None:
                unknown_parent_ids.append(parent_id)
            else:
                expected_parent_hashes.append(parent_hash)

        if unknown_parent_ids:
            errors.append(
                {
                    "kind": "unknown_parent",
                    "index": index,
                    "node_id": node_id,
                    "parent_ids": unknown_parent_ids,
                }
            )
            first_mismatch_index, first_mismatch_node_id = _first_mismatch(
                first_mismatch_index,
                first_mismatch_node_id,
                index,
                node_id,
            )
        elif parent_hashes != expected_parent_hashes:
            errors.append(
                {
                    "kind": "parent_hash_mismatch",
                    "index": index,
                    "node_id": node_id,
                    "expected": parent_hashes,
                    "actual": expected_parent_hashes,
                }
            )
            first_mismatch_index, first_mismatch_node_id = _first_mismatch(
                first_mismatch_index,
                first_mismatch_node_id,
                index,
                node_id,
            )

        if node_id:
            actual_hash_by_id[node_id] = actual_node_hash

    actual_chain_hash = hash_json({"schema_version": EVIDENCE_SCHEMA_VERSION, "node_hashes": actual_node_hashes})
    if not expected_chain_hash:
        errors.append({"kind": "missing_chain_hash", "expected": None, "actual": actual_chain_hash})
    elif str(expected_chain_hash) != actual_chain_hash:
        errors.append(
            {
                "kind": "chain_hash_mismatch",
                "expected": str(expected_chain_hash),
                "actual": actual_chain_hash,
            }
        )

    return EvidenceVerificationResult(
        valid=not errors,
        verified=True,
        schema_version=schema_version,
        node_count=len(nodes),
        expected_chain_hash=str(expected_chain_hash) if expected_chain_hash else None,
        actual_chain_hash=actual_chain_hash,
        first_mismatch_index=first_mismatch_index,
        first_mismatch_node_id=first_mismatch_node_id,
        errors=errors,
    )


def _node_to_dict(node: EvidenceNode | dict[str, Any]) -> dict[str, Any]:
    if isinstance(node, EvidenceNode):
        return node.to_dict()
    return dict(node)


def _node_hash_value(data: dict[str, Any], field: str) -> Any:
    if field == "schema_version":
        return int(data.get(field) or EVIDENCE_SCHEMA_VERSION)
    if field in {"parent_ids", "parent_hashes"}:
        return [str(item) for item in data.get(field) or []]
    if field == "arguments":
        value = data.get(field) or {}
        return value if isinstance(value, dict) else {"value": value}
    if field == "model_response_id":
        return data.get(field)
    return data.get(field)


def _first_mismatch(
    current_index: int | None,
    current_node_id: str | None,
    index: int,
    node_id: str,
) -> tuple[int | None, str | None]:
    if current_index is not None:
        return current_index, current_node_id
    return index, node_id or None


class EvidenceStore:
    """Evidence recorder with explicit new-run storage and read-only legacy load.

    Construction never creates or migrates files. ``create`` (or the first
    write on a genuinely empty run) enables the JSONL storage protocol.
    """

    def __init__(self, run_dir: Path) -> None:
        from aircraft_agent.evidence.storage import EvidenceStorage, STORAGE_MODE
        self.run_dir = Path(run_dir)
        self.path = self.run_dir / "evidence_chain.json"
        self._storage = EvidenceStorage(self.run_dir)
        self.nodes: list[EvidenceNode] = []
        self.tail_node_ids: list[str] = []
        if self._storage.mode == STORAGE_MODE:
            self._sync_committed_memory()
        elif self.path.exists():
            self._load_existing()

    @classmethod
    def create(cls, run_dir: Path) -> "EvidenceStore":
        """Explicitly initialize a new run; existing evidence cannot be adopted."""
        from aircraft_agent.evidence.storage import EvidenceStorageError
        store = cls(run_dir)
        if store._storage.mode != "uninitialized":
            raise EvidenceStorageError("create 仅接受没有证据文件的新运行目录。")
        store.save()
        return store

    @property
    def storage_mode(self) -> str:
        return self._storage.mode

    @property
    def log_path(self) -> Path:
        return self.run_dir / "evidence_chain.jsonl"

    def _sync_committed_memory(self) -> None:
        committed = json.loads(stable_json(self._storage.nodes))
        self.nodes = [_evidence_node_from_dict(node) for node in committed]
        self.tail_node_ids = self._storage.tail

    def add(
        self,
        *,
        type: str,
        tool_name: str | None,
        arguments: dict[str, Any],
        output: Any,
        summary: str,
        parent_ids: list[str] | None = None,
        model_response_id: str | None = None,
        advance_tail: bool = True,
    ) -> EvidenceNode:
        """Create and persist one evidence node.

        parent_ids 默认接到当前 tail_node_ids，形成线性主链；需要表达并行
        或特殊依赖时可显式传 parent_ids。
        """
        with self._storage.writer():
            self._storage.validate_memory(self.to_list(), self.tail_node_ids)
            parents = list(parent_ids) if parent_ids is not None else list(self.tail_node_ids)
            parent_hashes = self._resolve_parent_hashes(parents)
            node = EvidenceNode(
                schema_version=EVIDENCE_SCHEMA_VERSION,
                node_id=str(uuid.uuid4()),
                parent_ids=parents,
                parent_hashes=parent_hashes,
                type=type,
                tool_name=tool_name,
                input_hash=hash_json(arguments),
                output_hash=hash_json(output),
                arguments=arguments,
                summary=truncate_text(summary, 500),
                timestamp=utc_now_iso(),
                model_response_id=model_response_id,
            )
            node.node_hash = compute_node_hash(node)
            # Detach mutable caller values using the existing serialization.
            # Published memory and persisted bytes then share the exact values.
            serialized = json.loads(stable_json(node.to_dict()))
            node = _evidence_node_from_dict(serialized)
            tail = [node.node_id] if advance_tail else list(self.tail_node_ids)
            try:
                self._storage.commit(serialized, tail)
            finally:
                # On snapshot failure the fsynced log node is committed; on
                # uncertain log durability only the known prefix is exposed.
                self._sync_committed_memory()
            return self.nodes[-1]

    def add_file_ingest(self, file: StoredFile) -> EvidenceNode:
        """Specialized evidence node for uploaded file ingestion.

        文件进入系统是证据链的起点，因此 parent_ids=[]。后续工具调用会默认接
        到当前 tail，形成“输入文件 -> 工具调用 -> 报告”的可追溯路径。
        """
        return self.add(
            type="file_ingest",
            tool_name=None,
            arguments={
                "file_id": file.file_id,
                "kind": file.kind,
                "original_name": file.original_name,
                "path": str(file.path),
            },
            output=file.to_dict(),
            summary=f"Ingested {file.kind} file {file.original_name} ({file.size_bytes} bytes).",
            parent_ids=[],
        )

    def save(self) -> None:
        """Initialize a new run or verify an unchanged, already committed view.

        New nodes must go through add(); save never rewrites historical bytes.
        """
        with self._storage.writer():
            self._storage.validate_memory(self.to_list(), self.tail_node_ids)

    def to_list(self) -> list[dict[str, Any]]:
        """Return a JSON-ready list for APIs/tests."""
        return [node.to_dict() for node in self.nodes]

    def to_dict(self) -> dict[str, Any]:
        """Return the v2 evidence-chain envelope."""
        return {
            "schema_version": EVIDENCE_SCHEMA_VERSION,
            "chain_hash": compute_chain_hash(self.nodes),
            "nodes": self.to_list(),
        }

    def chain_hash(self) -> str:
        """Return the current run-level chain hash."""
        return compute_chain_hash(self.nodes)

    def _resolve_parent_hashes(self, parent_ids: list[str]) -> list[str]:
        nodes_by_id = {node.node_id: node for node in self.nodes}
        hashes: list[str] = []
        for parent_id in parent_ids:
            parent = nodes_by_id.get(parent_id)
            if parent is None:
                raise ValueError(f"Unknown evidence parent_id: {parent_id}")
            hashes.append(parent.node_hash or compute_node_hash(parent))
        return hashes

    def _load_existing(self) -> None:
        chain = load_evidence_chain(self.path)
        self.nodes = [_evidence_node_from_dict(node) for node in chain["nodes"]]
        self.tail_node_ids = [self.nodes[-1].node_id] if self.nodes else []


def _evidence_node_from_dict(data: dict[str, Any]) -> EvidenceNode:
    return EvidenceNode(
        schema_version=int(data.get("schema_version") or 1),
        node_id=str(data.get("node_id") or ""),
        parent_ids=[str(parent_id) for parent_id in data.get("parent_ids") or []],
        parent_hashes=[str(parent_hash) for parent_hash in data.get("parent_hashes") or []],
        type=str(data.get("type") or ""),
        tool_name=data.get("tool_name"),
        input_hash=str(data.get("input_hash") or ""),
        output_hash=str(data.get("output_hash") or ""),
        arguments=dict(data.get("arguments") or {}),
        summary=str(data.get("summary") or ""),
        timestamp=str(data.get("timestamp") or ""),
        model_response_id=data.get("model_response_id"),
        node_hash=str(data.get("node_hash") or ""),
    )
