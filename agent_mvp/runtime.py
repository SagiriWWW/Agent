"""Run-level state and filesystem storage helpers.

每次用户上传并运行 Agent 都会创建一个独立 run 目录：
runtime/<run_id>/
  uploads/      原始上传文件
  artifacts/    报告等输出
  manifest.json 本次运行的轻量索引
  evidence_chain.json 审计证据链
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .evidence import EvidenceStore
from .models import RuleCard, StoredFile
from .utils import safe_filename, sha256_file, to_jsonable, truncate_text, utc_now_iso, write_json


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
        if save:
            self.save_manifest()
        return event

    def record_tool_result(self, name: str, arguments: dict[str, Any], result: Any) -> None:
        """Keep ordered tool outputs for later summarization and reporting."""
        # 这里只保存结构化结果摘要，不保存 DataFrame 本体，避免 manifest 过大。
        self.tool_results.append({"tool": name, "arguments": arguments, "result": result})
        self.save_manifest()
        summary = result.get("summary") if isinstance(result, dict) else str(result)
        self.emit_event(
            "tool_finished",
            title=f"工具完成：{name}",
            message=f"{name} 已完成。{truncate_text(str(summary or ''), 500)}",
            payload={"tool": name, "arguments": arguments, "result": result},
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
    ) -> None:
        """Keep the visible Thought-Action-Observation trace for the demo agent."""
        self.agent_trace.append(
            {
                "step": step,
                "thought": thought,
                "action": action,
                "arguments": arguments,
                "observation": observation,
                "status": status,
            }
        )
        self.save_manifest()
        event_type = "step_failed" if status == "failed" else "step_finished"
        self.emit_event(
            event_type,
            title=f"Agent 步骤 {step}：{action}",
            message=observation,
            payload={
                "step": step,
                "thought": thought,
                "action": action,
                "arguments": arguments,
                "observation": observation,
                "status": status,
            },
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
        write_json(
            self.run_dir / "manifest.json",
            {
                "run_id": self.run_id,
                "created_at": utc_now_iso(),
                "status": self.status,
                "error": self.error,
                "user_goal": self.user_goal,
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
            },
        )


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
            evidence=EvidenceStore(run_dir),
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
        path = context.run_dir / "uploads" / filename
        path.write_bytes(data)
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
