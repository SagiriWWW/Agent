"""Shared dataclasses used across the MVP.

这些结构尽量保持简单、可 JSON 序列化，方便写入 manifest、报告和证据链。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class StoredFile:
    """Metadata for one uploaded file stored inside a run directory.

    每次运行都会把上传文件复制到 `runtime/<run_id>/uploads/`，然后用
    StoredFile 记录它的不可变元数据。这里不保存文件内容本身，只保存路径、
    哈希、大小和行列数等摘要，方便 manifest/report/evidence 复用。
    """

    # file_id 是系统内部生成的 UUID，用于避免用户上传同名文件时冲突。
    file_id: str
    # kind 用来区分规则库、数据集等类型；当前主要是 rule_library/dataset。
    kind: str
    # original_name 保留用户上传时的名字，报告里展示它比 UUID 更容易读。
    original_name: str
    # path 指向运行目录里的副本，不直接引用用户原始路径。
    path: Path
    # sha256 是证据链的基础：后续能证明报告对应的是哪一份原始输入。
    sha256: str
    size_bytes: int
    media_type: str | None = None
    # rows/columns 在真正读取数据集后回填；刚上传时可能还是 None。
    rows: int | None = None
    columns: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert Path to str so the object can be written into JSON files."""
        data = asdict(self)
        data["path"] = str(self.path)
        return data


@dataclass
class RuleCard:
    """Normalized representation of one rule from the rule library.

    规则库可能来自 md/txt/json/yaml/docx/pdf，格式差异很大。RuleCard 是
    统一后的中间结构：能自动执行的规则会放到 condition，暂时无法执行的
    规则也保留 description/source_ref，便于报告和人工复核。
    """

    rule_id: str
    name: str
    # 原始规则描述。即使 condition 抽取失败，也不会丢掉这段原文。
    description: str
    # 从规则文本中识别出的数据字段名，用于报告和可解释性。
    target_fields: list[str] = field(default_factory=list)
    # MVP 当前只支持简单比较表达式，如 temperature > 80。
    condition: str = ""
    # 预留给结构化规则中的阈值、上下限、枚举值等更复杂配置。
    threshold: dict[str, Any] = field(default_factory=dict)
    # 结构化规则元数据，用于报告排序、人工复核和后续决策建议。
    severity: str = "medium"
    recommendation: str = ""
    rationale: str = ""
    tags: list[str] = field(default_factory=list)
    review_required: bool = False
    enabled: bool = True
    # 规则来源，例如 rules.md:line:3，便于追溯。
    source_ref: str = ""
    # confidence 是解析可信度，不是业务规则准确率。
    confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EvidenceNode:
    """One auditable step in the evidence chain.

    一个 EvidenceNode 就是一条可审计记录。系统不会把大块原始数据塞进
    证据链，而是记录输入/输出的哈希和简短摘要，这样既能追溯，又不会让
    evidence_chain.json 变得巨大。
    """

    schema_version: int
    node_id: str
    # parent_ids 用来表达“本节点基于哪些上游节点”。当前主链是线性的，
    # 以后可以扩展成真正的 DAG。
    parent_ids: list[str]
    # parent_hashes 与 parent_ids 顺序一致。当前用于线性链校验，未来可直接
    # 扩展到 DAG 中的多父节点完整性校验。
    parent_hashes: list[str]
    type: str
    tool_name: str | None
    # input_hash/output_hash 分别对 arguments/output 做稳定 JSON 哈希。
    input_hash: str
    output_hash: str
    arguments: dict[str, Any]
    summary: str
    timestamp: str
    model_response_id: str | None = None
    node_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EvidenceVerificationResult:
    """Structured integrity result for evidence-chain verification."""

    valid: bool
    verified: bool
    schema_version: int
    node_count: int
    expected_chain_hash: str | None
    actual_chain_hash: str | None
    first_mismatch_index: int | None = None
    first_mismatch_node_id: str | None = None
    errors: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AgentTask:
    """Reserved task envelope for future async/background execution.

    目前 Web 端同步执行 Agent。等后续接任务队列、异步 worker 或多人协作时，
    AgentTask 可以作为任务提交/调度的统一数据结构。
    """

    task_id: str
    user_goal: str
    rule_file_ids: list[str]
    dataset_file_ids: list[str]
    allowed_tools: list[str]
    final_outputs: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
