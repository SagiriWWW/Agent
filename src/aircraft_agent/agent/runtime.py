"""Run-level state and filesystem storage helpers.

每次用户上传并运行 Agent 都会创建一个独立 run 目录：
outputs/runs/<run_id>/
  uploads/      原始上传文件
  artifacts/    报告等输出
  manifest.json 本次运行的轻量索引
  run_metadata.json 本次运行的存储元数据
  evidence_chain.json 审计证据链
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from aircraft_agent.evidence.chain import EVIDENCE_SCHEMA_VERSION, EvidenceStore, verify_chain
from aircraft_agent.common.models import RuleCard, StoredFile
from aircraft_agent.common.utils import atomic_write_bytes, safe_filename, sha256_file, to_jsonable, truncate_text, utc_now_iso, write_json


RUN_STORAGE_SCHEMA_VERSION = 1


@dataclass
class RunContext:
    """In-memory state shared by tools, agent runner, and report generation.

    RunContext 是一次 Agent 运行的“工作台”。上传文件、DataFrame 缓存、
    规则卡片、工具调用结果、Agent 计划、证据链路径都挂在这里。这样工具箱
    里的每个工具都不需要互相传一堆参数，只要拿到同一个 context 就能协作。
    """

    run_id: str
    run_dir: Path
    user_goal: str
    evidence: EvidenceStore
    # files 是 file_id -> StoredFile 的索引，所有上传文件都在这里登记。
    files: dict[str, StoredFile] = field(default_factory=dict)
    # dataset_file_ids/rule_file_ids 保留文件顺序；默认工具会使用第一个数据集
    # 和第一个规则库，后续可以扩展成多数据集选择。
    dataset_file_ids: list[str] = field(default_factory=list)
    rule_file_ids: list[str] = field(default_factory=list)
    # dataframes 是运行期缓存，避免每个工具重复读取同一个 CSV/XLSX。
    dataframes: dict[str, pd.DataFrame] = field(default_factory=dict)
    dataset_profiles: dict[str, dict[str, Any]] = field(default_factory=dict)
    rules: list[RuleCard] = field(default_factory=list)
    # tool_results 是报告和 summarize_findings 的主要来源，按执行顺序追加。
    tool_results: list[dict[str, Any]] = field(default_factory=list)
    # 以下四个字段用于展示“Agent 主结构”：工具箱清单、计划、执行轨迹和校验。
    toolbox_manifest: list[dict[str, Any]] = field(default_factory=list)
    agent_plan: list[dict[str, Any]] = field(default_factory=list)
    agent_trace: list[dict[str, Any]] = field(default_factory=list)
    validation_results: list[dict[str, Any]] = field(default_factory=list)
    model_config: dict[str, Any] = field(default_factory=dict)
    run_kind: str = "analysis"
    final_text: str = ""
    report_markdown_path: Path | None = None
    report_html_path: Path | None = None
    status: str = "created"
    error: str | None = None
    event_count: int = 0
    created_at: str = field(default_factory=utc_now_iso)

    @property
    def uploads_dir(self) -> Path:
        """Return the per-run upload directory."""
        return self.run_dir / "uploads"

    @property
    def artifacts_dir(self) -> Path:
        """Return the per-run artifacts directory."""
        return self.run_dir / "artifacts"

    @property
    def manifest_path(self) -> Path:
        """Return the run manifest path."""
        return self.run_dir / "manifest.json"

    @property
    def metadata_path(self) -> Path:
        """Return the run-level storage metadata path."""
        return self.run_dir / "run_metadata.json"

    @property
    def event_path(self) -> Path:
        """Return the append-only event log used by the live progress UI."""
        return self.run_dir / "run_events.jsonl"

    def add_file(self, file: StoredFile) -> None:
        """Attach an uploaded file to this run and record file-ingest evidence."""
        self.files[file.file_id] = file
        if file.kind == "dataset":
            self.dataset_file_ids.append(file.file_id)
        elif file.kind == "rule_library":
            self.rule_file_ids.append(file.file_id)
        # 文件进入系统时立即写证据链，确保后续报告能追溯到具体输入文件。
        self.evidence.add_file_ingest(file)
        self.save_manifest()
        self.emit_event(
            "file_ingested",
            title="文件已接入",
            message=f"已保存 {file.kind} 文件：{file.original_name}",
            payload={"file": file.to_dict()},
        )

    def emit_event(
        self,
        event_type: str,
        *,
        title: str = "",
        message: str = "",
        payload: dict[str, Any] | None = None,
        save: bool = True,
    ) -> dict[str, Any]:
        """Append one structured event for SSE/live progress consumers.

        run_events.jsonl 是运行过程协议的最小实现：每行一个 JSON 事件，既能
        被前端 SSE 读取，也方便后续外部系统或多 Agent 组件复用。
        """
        self.event_count += 1
        event = {
            "id": self.event_count,
            "run_id": self.run_id,
            "type": event_type,
            "title": title or event_type,
            "message": truncate_text(message or title or event_type, 800),
            "timestamp": utc_now_iso(),
            "payload": payload or {},
        }
        self.event_path.parent.mkdir(parents=True, exist_ok=True)
        with self.event_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(to_jsonable(event), ensure_ascii=False) + "\n")
            f.flush()
        if save:
            self.save_manifest()
        return event

    def record_tool_result(
        self,
        name: str,
        arguments: dict[str, Any],
        result: Any,
        *,
        evidence_reference: dict[str, Any] | None = None,
    ) -> None:
        """Keep ordered tool outputs for later summarization and reporting."""
        # 这里只保存结构化结果摘要，不保存 DataFrame 本体，避免 manifest 过大。
        item = {"tool": name, "arguments": arguments, "result": result}
        # Failed runner calls also create tool_call nodes, without a successful
        # tool_results entry. Pin the actual chain position while recording;
        # this in-memory locator does not change event or evidence formats.
        item["evidence_tool_call_index"] = sum(node.type == "tool_call" for node in self.evidence.nodes) - 1
        if evidence_reference:
            item["evidence_reference"] = evidence_reference
        self.tool_results.append(item)
        self.save_manifest()
        summary = result.get("summary") if isinstance(result, dict) else str(result)
        payload = {"tool": name, "arguments": arguments, "result": result}
        if evidence_reference:
            payload["evidence_reference"] = evidence_reference
        self.emit_event(
            "tool_finished",
            title=f"工具完成：{name}",
            message=f"{name} 已完成。{truncate_text(str(summary or ''), 500)}",
            payload=payload,
        )

    def record_agent_trace(
        self,
        *,
        step: int,
        thought: str,
        action: str,
        arguments: dict[str, Any],
        observation: str,
        status: str,
        evidence_reference: dict[str, Any] | None = None,
        related_evidence: list[dict[str, Any]] | None = None,
    ) -> None:
        """Keep the visible Thought-Action-Observation trace for the demo agent."""
        item = {
            "step": step,
            "thought": thought,
            "action": action,
            "arguments": arguments,
            "observation": observation,
            "status": status,
        }
        if evidence_reference:
            item["evidence_reference"] = evidence_reference
        if related_evidence:
            item["related_evidence"] = related_evidence
        self.agent_trace.append(item)
        self.save_manifest()
        event_type = "step_failed" if status == "failed" else "step_finished"
        payload = {
            "step": step,
            "thought": thought,
            "action": action,
            "arguments": arguments,
            "observation": observation,
            "status": status,
        }
        if evidence_reference:
            payload["evidence_reference"] = evidence_reference
        if related_evidence:
            payload["related_evidence"] = related_evidence
        self.emit_event(
            event_type,
            title=f"Agent 步骤 {step}：{action}",
            message=observation,
            payload=payload,
        )

    def file_summary(self) -> dict[str, Any]:
        """Return upload metadata safe to include in model prompts."""
        return {
            "rule_files": [self.files[file_id].to_dict() for file_id in self.rule_file_ids],
            "dataset_files": [self.files[file_id].to_dict() for file_id in self.dataset_file_ids],
        }

    def save_manifest(self) -> None:
        """Persist run metadata without storing bulky DataFrame contents.

        manifest.json 是“本次运行索引”，主要给 API 和报告页读取。证据链有自己
        的 evidence_chain.json；上传文件也单独存在 uploads/ 下。
        """
        updated_at = utc_now_iso()
        metadata = self.run_metadata(updated_at=updated_at)
        evidence_verification = metadata["evidence"].get("verification")
        write_json(self.metadata_path, metadata)
        write_json(
            self.manifest_path,
            {
                "storage_schema_version": RUN_STORAGE_SCHEMA_VERSION,
                "run_id": self.run_id,
                "created_at": self.created_at,
                "updated_at": updated_at,
                "status": self.status,
                "error": self.error,
                "user_goal": self.user_goal,
                "run_metadata_path": str(self.metadata_path),
                "run_dir": str(self.run_dir),
                "uploads_dir": str(self.uploads_dir),
                "artifacts_dir": str(self.artifacts_dir),
                "files": {k: v.to_dict() for k, v in self.files.items()},
                "dataset_file_ids": self.dataset_file_ids,
                "rule_file_ids": self.rule_file_ids,
                "toolbox_manifest": self.toolbox_manifest,
                "agent_plan": self.agent_plan,
                "agent_trace": self.agent_trace,
                "validation_results": self.validation_results,
                "model_config": self.model_config,
                "run_kind": self.run_kind,
                "report_markdown_path": str(self.report_markdown_path) if self.report_markdown_path else None,
                "report_html_path": str(self.report_html_path) if self.report_html_path else None,
                "event_log_path": str(self.event_path),
                "event_count": self.event_count,
                "evidence_chain_path": str(self.evidence.path),
                "evidence_schema_version": EVIDENCE_SCHEMA_VERSION,
                "chain_hash": self.evidence.chain_hash(),
                "evidence_verification": evidence_verification,
                "storage": metadata,
            },
        )

    def evidence_verification(self) -> dict[str, Any]:
        """Return a display-safe verification result for the persisted evidence chain."""
        try:
            source = self.evidence.path if self.evidence.path.exists() else self.evidence.to_dict()
            return verify_chain(source).to_dict()
        except Exception as exc:
            return {
                "valid": False,
                "verified": False,
                "schema_version": EVIDENCE_SCHEMA_VERSION,
                "node_count": len(self.evidence.nodes),
                "expected_chain_hash": self.evidence.chain_hash(),
                "actual_chain_hash": None,
                "first_mismatch_index": None,
                "first_mismatch_node_id": None,
                "errors": [{"kind": "verification_error", "message": str(exc)}],
            }

    def run_metadata(self, *, updated_at: str | None = None) -> dict[str, Any]:
        """Return the durable run-level storage record."""
        verification = self.evidence_verification()
        return {
            "storage_schema_version": RUN_STORAGE_SCHEMA_VERSION,
            "run_id": self.run_id,
            "created_at": self.created_at,
            "updated_at": updated_at or utc_now_iso(),
            "status": self.status,
            "error": self.error,
            "run_kind": self.run_kind,
            "user_goal": self.user_goal,
            "paths": {
                "run_dir": str(self.run_dir),
                "uploads_dir": str(self.uploads_dir),
                "artifacts_dir": str(self.artifacts_dir),
                "manifest": str(self.manifest_path),
                "run_metadata": str(self.metadata_path),
                "event_log": str(self.event_path),
                "evidence_chain": str(self.evidence.path),
                "evidence_log": str(self.evidence.log_path) if self.evidence.storage_mode == "jsonl_snapshot" else None,
            },
            "evidence": {
                "schema_version": EVIDENCE_SCHEMA_VERSION,
                "storage_protocol_version": 1 if self.evidence.storage_mode == "jsonl_snapshot" else None,
                "storage_mode": self.evidence.storage_mode,
                "node_count": len(self.evidence.nodes),
                "chain_hash": self.evidence.chain_hash(),
                "verification": verification,
            },
            "reports": {
                "markdown_path": str(self.report_markdown_path) if self.report_markdown_path else None,
                "html_path": str(self.report_html_path) if self.report_html_path else None,
            },
        }


class RunStore:
    """Creates run directories and writes uploaded files to disk.

    RunStore 只负责文件系统层面的事情：创建 run 目录、保存上传文件、返回
    某个 run 的路径。它不理解业务规则，也不执行分析。
    """

    def __init__(self, storage_dir: Path) -> None:
        self.storage_dir = storage_dir
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def create_run(self, user_goal: str) -> RunContext:
        """Initialize a new isolated run workspace."""
        run_id = str(uuid.uuid4())
        run_dir = self.storage_dir / run_id
        # 每次运行都拥有独立目录，方便删除、打包、审计，也避免不同用户/任务
        # 的上传文件和报告互相覆盖。
        (run_dir / "uploads").mkdir(parents=True, exist_ok=True)
        (run_dir / "artifacts").mkdir(parents=True, exist_ok=True)
        context = RunContext(
            run_id=run_id,
            run_dir=run_dir,
            user_goal=user_goal,
            evidence=EvidenceStore.create(run_dir),
        )
        context.save_manifest()
        context.emit_event(
            "run_created",
            title="运行已创建",
            message="已创建独立运行目录，等待文件接入和 Agent 调度。",
            payload={"user_goal": user_goal},
        )
        return context

    def save_bytes(
        self,
        context: RunContext,
        *,
        data: bytes,
        original_name: str,
        kind: str,
        media_type: str | None = None,
    ) -> StoredFile:
        """Save one uploaded file and compute its immutable SHA-256 digest."""
        file_id = str(uuid.uuid4())
        # 文件名带 UUID，既保留原始名字用于人工识别，又避免重名覆盖。
        filename = f"{file_id}_{safe_filename(original_name)}"
        path = context.uploads_dir / filename
        atomic_write_bytes(path, data)
        record = StoredFile(
            file_id=file_id,
            kind=kind,
            original_name=original_name,
            path=path,
            sha256=sha256_file(path),
            size_bytes=path.stat().st_size,
            media_type=media_type,
        )
        context.add_file(record)
        return record

    def get_run_dir(self, run_id: str) -> Path:
        return self.storage_dir / run_id
