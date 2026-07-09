"""Evidence-chain recording.

证据链是 MVP 的审计核心。每个节点记录输入参数 hash、输出 hash、摘要、
时间戳和父节点。当前实现是轻量 JSON 文件；后续如果要做图查询或更强
审计，可以把 EvidenceStore 替换为数据库/图存储，外部调用点不必大改。
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from .models import EvidenceNode, StoredFile
from .utils import hash_json, truncate_text, utc_now_iso, write_json


class EvidenceStore:
    """Append-only evidence recorder for one run.

    EvidenceStore 是“证据链写入器”。所有文件进入、工具调用、Agent 计划和
    Agent 轨迹都会通过它追加节点。当前实现使用 JSON 文件，优点是简单可读；
    后续要做图查询时，可以把内部存储替换成数据库而不改外部调用方式。
    """

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.path = run_dir / "evidence_chain.json"
        # nodes 保存在内存中，每次 add 后立即写回 evidence_chain.json。
        self.nodes: list[EvidenceNode] = []
        # tail_node_ids 表示当前主链尾部。默认新节点接到尾部，形成线性链；
        # 如果传 parent_ids，就能表达更复杂的 DAG 依赖。
        self.tail_node_ids: list[str] = []

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
        parents = parent_ids if parent_ids is not None else list(self.tail_node_ids)
        node = EvidenceNode(
            node_id=str(uuid.uuid4()),
            parent_ids=parents,
            type=type,
            tool_name=tool_name,
            # arguments/output 可能包含 pandas/numpy 类型，hash_json 会先稳定序列化。
            input_hash=hash_json(arguments),
            output_hash=hash_json(output),
            arguments=arguments,
            summary=truncate_text(summary, 500),
            timestamp=utc_now_iso(),
            model_response_id=model_response_id,
        )
        self.nodes.append(node)
        if advance_tail:
            # 大多数节点会推进主链；少数旁路记录可以设置 advance_tail=False。
            self.tail_node_ids = [node.node_id]
        self.save()
        return node

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
        """Persist current in-memory nodes to evidence_chain.json."""
        write_json(self.path, [node.to_dict() for node in self.nodes])

    def to_list(self) -> list[dict[str, Any]]:
        """Return a JSON-ready list for APIs/tests."""
        return [node.to_dict() for node in self.nodes]
