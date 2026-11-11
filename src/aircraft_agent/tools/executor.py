"""Local tool implementations exposed to the LLM agent.

本文件是后续扩展 tool 的主要入口。新增工具时通常需要做三件事：
1. 在 tools/catalog.py 的 OPENAI_TOOL_SCHEMAS 声明工具名和 JSON 参数。
2. 在 ToolExecutor.registry 里把工具名映射到实际 Python 方法。
3. 新增一个同名方法，返回可 JSON 序列化的 dict，并带上简短 summary。

所有工具都在本地执行，模型只看到工具结果摘要和结构化结果；这能避免
把完整原始数据直接发给 OpenAI API。
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

import pandas as pd

from aircraft_agent.evidence.chain import evidence_node_reference, verify_chain
from aircraft_agent.telemetry.mining import datamining_backend_status, mine_with_fpgrowth
from aircraft_agent.common.models import RuleCard
from aircraft_agent.telemetry.parsers import parse_rule_library, profile_dataframe, read_dataset
from aircraft_agent.telemetry.reporting import write_report_files
from aircraft_agent.agent.runtime import RunContext
from aircraft_agent.telemetry.review_decisions import (
    merge_mapping_overrides,
    rejected_rule_mapping_keys,
    review_decision_overrides,
)
from aircraft_agent.telemetry.schema import (
    infer_field_semantics,
    load_schema_mapping_overrides,
    load_schema_semantics,
    schema_mapping_override_for_field,
    schema_mapping_override_for_rule_field,
    suggest_schema_metadata as build_schema_suggestions,
)
from aircraft_agent.common.utils import to_jsonable, truncate_text, write_json, write_text_atomic
from aircraft_agent.telemetry.visualizations import build_visualization_artifacts


from aircraft_agent.tools.catalog import (
    OPENAI_TOOL_SCHEMAS, TOOLBOX_CATALOG, TOOL_NAME_ALIASES, VALID_RULE_SEVERITIES,
)
from aircraft_agent.common.tool_contracts import TOOL_SCHEMA_VERSION, ToolFailure, versioned_tool


class ToolExecutor:
    """Dispatches LLM-requested tool calls to local Python functions.

    ToolExecutor 持有一次运行的 RunContext，因此工具可以共享已上传文件、
    DataFrame 缓存、规则卡片、工具结果和证据链。
    """

    def __init__(self, context: RunContext) -> None:
        self.context = context
        # 每次创建执行器时，把工具箱清单写进 context，这样报告/API 不需要重新
        # 依赖 tools.py 全局变量，也能展示本次运行可用工具。
        self.context.toolbox_manifest = self.toolbox_manifest()
        # registry 是模型工具名到本地函数的唯一映射表。
        # 新增工具时，记得同时更新 OPENAI_TOOL_SCHEMAS 和这里。
        self.registry = {
            "profile_dataset": self.profile_dataset,
            "parse_rule_library": self.parse_rule_library,
            "validate_rule_library": self.validate_rule_library,
            "suggest_schema_metadata": self.suggest_schema_metadata,
            "review_schema_metadata": self.review_schema_metadata,
            "review_schema_mapping": self.review_schema_mapping,
            "review_dataset_adaptation": self.review_dataset_adaptation,
            "review_schema_mapping_candidates": self.review_schema_mapping_candidates,
            "review_generalization_readiness": self.review_generalization_readiness,
            "build_mapping_execution_plan": self.build_mapping_execution_plan,
            "validate_mapping_regression": self.validate_mapping_regression,
            "check_missing_values": self.check_missing_values,
            "detect_outliers": self.detect_outliers,
            "apply_rules": self.apply_rules,
            "apply_rule_library": self.apply_rules,
            "check_association_backend": self.check_association_backend,
            "mine_association_rules": self.mine_association_rules,
            "compute_correlations": self.compute_correlations,
            "discover_frequent_patterns": self.discover_frequent_patterns,
            "build_visualizations": self.build_visualizations,
            "summarize_findings": self.summarize_findings,
            "validate_analysis": self.validate_analysis,
            "write_report": self.write_report,
            "record_evidence": self.record_evidence,
        }
        self.context.save_manifest()

    @property
    def tool_names(self) -> list[str]:
        return list(self.registry.keys())

    def toolbox_manifest(self) -> list[dict[str, Any]]:
        """Return the public toolbox manifest used by the Agent and demo UI."""
        # 返回浅拷贝，避免调用方误改全局 TOOLBOX_CATALOG。
        return [dict(item) for item in TOOLBOX_CATALOG]

    def execute(self, name: str, arguments: dict[str, Any] | None = None, *, record: bool = True) -> Any:
        """Run one local tool and optionally append tool output to evidence.

        record=True 是默认路径：每次工具调用都会进入 tool_results 和
        evidence_chain.json。测试或内部复用时可以传 record=False。
        """
        arguments = arguments or {}
        # 所有工具调用都必须经过 registry，避免模型或外部 API 任意调用 Python
        # 函数。这是 Agent 工具调用的安全边界之一。
        if record:
            self.context.emit_event(
                "tool_started",
                title=f"工具开始：{name}",
                message=f"开始执行本地工具 {name}。",
                payload={"tool": name, "arguments": arguments},
            )
        stage = "dispatch"
        try:
            if name not in self.registry:
                raise ValueError(f"Unknown tool: {name}")
            stage = "execution"
            result = self.registry[name](**arguments)
        except Exception as exc:
            if record:
                self.context.emit_event(
                    "tool_failed",
                    title=f"工具失败：{name}",
                    message=f"{name} 执行失败：{exc}",
                    payload={
                        "tool": name, "arguments": arguments, "error": str(exc),
                        "failure": ToolFailure.from_exception(
                            exc, stage=stage,
                            code="UNKNOWN_TOOL" if stage == "dispatch" else "TOOL_EXECUTION_FAILED",
                        ).model_dump(),
                    },
                )
            raise
        if record:
            node = self.context.evidence.add(
                type="tool_call",
                tool_name=name,
                arguments=arguments,
                output=result,
                summary=_result_summary(name, result),
            )
            self.context.record_tool_result(
                name,
                arguments,
                result,
                evidence_reference=evidence_node_reference(node),
            )
            # write_report 的报告内容包含证据链节点数；先记录工具节点，
            # 再重写一次报告，确保报告中的节点数和 JSON 文件一致。
            if name == "write_report":
                result = self.write_report(**arguments)
        return result

    def _dataset_id(self, dataset_id: str | None = None) -> str:
        """Resolve an optional dataset_id to a valid uploaded dataset file id."""
        if dataset_id and dataset_id in self.context.dataset_file_ids:
            return dataset_id
        if not self.context.dataset_file_ids:
            raise ValueError("No dataset file uploaded.")
        return self.context.dataset_file_ids[0]

    def _rule_file_id(self, rule_file_id: str | None = None) -> str:
        """Resolve an optional rule_file_id to a valid uploaded rule file id."""
        if rule_file_id and rule_file_id in self.context.rule_file_ids:
            return rule_file_id
        if not self.context.rule_file_ids:
            raise ValueError("No rule-library file uploaded.")
        return self.context.rule_file_ids[0]

    def _dataframe(self, dataset_id: str | None = None) -> tuple[str, pd.DataFrame]:
        """Load a dataset once, then reuse the cached DataFrame in this run."""
        file_id = self._dataset_id(dataset_id)
        if file_id not in self.context.dataframes:
            record = self.context.files[file_id]
            # 读取数据集后回填行列数，后续 manifest/report 就能展示数据规模。
            df = read_dataset(record.path)
            record.rows = int(len(df))
            record.columns = int(len(df.columns))
            self.context.dataframes[file_id] = df
            self.context.save_manifest()
        return file_id, self.context.dataframes[file_id]

    def _known_fields(self) -> list[str]:
        """Collect known dataset fields so natural-language rules can bind columns."""
        fields: list[str] = []
        for dataset_id in self.context.dataset_file_ids:
            try:
                # 已上传的所有数据集字段都会参与规则解析，帮助规则库识别目标字段。
                _, df = self._dataframe(dataset_id)
                fields.extend(str(c) for c in df.columns)
            except Exception:
                continue
        return sorted(set(fields))

    @versioned_tool
    def profile_dataset(self, dataset_id: str | None = None) -> dict[str, Any]:
        """Generate a compact dataset profile used by the model and report."""
        file_id, df = self._dataframe(dataset_id)
        record = self.context.files[file_id]
        # profile_dataframe 返回的是 JSON-safe 摘要，不会把整个 DataFrame 写进报告。
        profile = profile_dataframe(df)
        metadata = _dataset_metadata(record.original_name, [str(col) for col in df.columns])
        profile.update(metadata)
        self.context.dataset_profiles[file_id] = profile
        self.context.save_manifest()
        return {
            "dataset_id": file_id,
            "source_file": record.original_name,
            "dataset_kind_hint": metadata["dataset_kind_hint"],
            "profile": profile,
            "summary": f"已完成数据画像：{profile['rows']} 行，{profile['columns']} 列；数据口径提示：{metadata['dataset_kind_hint']}。",
        }

    @versioned_tool
    def parse_rule_library(self, rule_file_id: str | None = None) -> dict[str, Any]:
        """Parse uploaded rule text/JSON/YAML/etc. into normalized RuleCard items."""
        file_id = self._rule_file_id(rule_file_id)
        record = self.context.files[file_id]
        # known_fields 让规则解析器优先绑定真实数据字段，例如 temperature/status。
        rules = parse_rule_library(record.path, known_fields=self._known_fields())
        known = {rule.rule_id: rule for rule in self.context.rules}
        for rule in rules:
            # 同一 run 里多次解析时按 rule_id 去重/覆盖，避免重复追加。
            known[rule.rule_id] = rule
        self.context.rules = list(known.values())
        return {
            "rule_file_id": file_id,
            "rules": [rule.to_dict() for rule in rules],
            "summary": f"已解析 {len(rules)} 条规则卡片。",
        }

    def validate_rule_library(self, dataset_id: str | None = None, rule_file_id: str | None = None) -> dict[str, Any]:
        """Validate RuleCard metadata before using it as a stable rule library."""
        file_id, df = self._dataframe(dataset_id)
        if not self.context.rules:
            self.parse_rule_library(rule_file_id)
        known_fields = {str(col) for col in df.columns}
        records: list[dict[str, Any]] = []
        ready_count = 0
        needs_review_count = 0
        disabled_count = 0
        suffix_mismatches: list[dict[str, str]] = []
        for rule in self.context.rules:
            issues: list[str] = []
            missing_fields = []
            candidate_fields = _rule_field_candidates(rule)
            for field in candidate_fields:
                if field in known_fields:
                    continue
                alternate = _alternate_suffix_field(field, known_fields)
                if alternate:
                    issues.append(f"字段后缀不匹配：{field} 不存在，但数据集中存在 {alternate}")
                    suffix_mismatches.append({"rule_id": rule.rule_id, "rule_field": field, "dataset_field": alternate})
                else:
                    missing_fields.append(field)
            if missing_fields:
                issues.append(f"字段不存在：{', '.join(missing_fields)}")
            if not rule.condition:
                issues.append("缺少可执行 condition。")
            else:
                _, error = evaluate_rule_condition(df.head(min(len(df), 200)), rule)
                if error:
                    issues.append(f"condition 无法执行：{error}")
            severity = str(rule.severity or "medium").lower()
            if severity not in VALID_RULE_SEVERITIES:
                issues.append(f"严重程度不在允许范围：{rule.severity}")
            if not rule.recommendation:
                issues.append("缺少建议动作 recommendation。")
            if not rule.enabled:
                status = "disabled"
                disabled_count += 1
            elif issues or rule.review_required:
                status = "needs_review"
                needs_review_count += 1
            else:
                status = "ready"
                ready_count += 1
            records.append(
                {
                    "rule_id": rule.rule_id,
                    "name": rule.name,
                    "status": status,
                    "severity": severity,
                    "target_fields": rule.target_fields,
                    "condition": rule.condition,
                    "recommendation": rule.recommendation,
                    "review_required": rule.review_required,
                    "issues": issues,
                    "source_ref": rule.source_ref,
                }
            )

        artifacts_dir = self.context.run_dir / "artifacts"
        json_path = artifacts_dir / "rule_validation.json"
        md_path = artifacts_dir / "rule_validation.md"
        result = {
            "dataset_id": file_id,
            "rule_count": len(records),
            "ready_count": ready_count,
            "needs_review_count": needs_review_count,
            "disabled_count": disabled_count,
            "suffix_mismatches": suffix_mismatches,
            "rules": records,
            "json_path": str(json_path),
            "markdown_path": str(md_path),
            "summary": f"规则库校验完成：{ready_count} 条可执行，{needs_review_count} 条需复核，{disabled_count} 条已停用。",
        }
        write_json(json_path, result)
        write_text_atomic(md_path, _render_rule_validation_markdown(result), encoding="utf-8")
        return result

    def suggest_schema_metadata(self, max_fields: int = 120, config_path: str | None = None) -> dict[str, Any]:
        """Suggest field display names/groups as a reviewable schema draft."""
        if not self.context.dataset_profiles:
            self.profile_dataset()
        return build_schema_suggestions(self.context, max_fields=max_fields, config_path=config_path)

    def review_schema_metadata(self) -> dict[str, Any]:
        """Create a human review checklist for the current schema suggestion."""
        schema_result = _latest_tool_result(self.context.tool_results, "suggest_schema_metadata")
        if not isinstance(schema_result, dict) or not isinstance(schema_result.get("schema"), dict):
            schema_result = self.suggest_schema_metadata()
        schema = schema_result.get("schema") or {}
        fields = schema.get("fields") or {}
        low_confidence = [
            {"field": field, **info}
            for field, info in fields.items()
            if float(info.get("confidence", 0.0) or 0.0) < 0.8
        ]
        ignored_fields = [
            {"field": field, **info}
            for field, info in fields.items()
            if info.get("ignore")
        ]
        rule_bound_fields = list(schema_result.get("rule_bound_fields") or [])
        missing_rule_bound_fields = [field for field in rule_bound_fields if field not in fields]
        review_items = []
        for item in low_confidence:
            review_items.append({"type": "low_confidence_field", "field": item["field"], "message": "字段中文名或分组置信度较低，建议人工确认。"})
        for field in missing_rule_bound_fields:
            review_items.append({"type": "missing_rule_bound_field", "field": field, "message": "规则绑定字段没有出现在 Schema 草案中。"})
        for item in ignored_fields:
            review_items.append({"type": "ignored_field", "field": item["field"], "message": "字段被标记为忽略，需确认是否不参与分析。"})

        artifacts_dir = self.context.run_dir / "artifacts"
        json_path = artifacts_dir / "schema_review.json"
        md_path = artifacts_dir / "schema_review.md"
        result = {
            "field_count": len(fields),
            "low_confidence_count": len(low_confidence),
            "ignored_count": len(ignored_fields),
            "rule_bound_count": len(rule_bound_fields),
            "missing_rule_bound_fields": missing_rule_bound_fields,
            "review_items": review_items,
            "json_path": str(json_path),
            "markdown_path": str(md_path),
            "summary": f"Schema 复核清单已生成：{len(review_items)} 个复核项，其中 {len(low_confidence)} 个低置信度字段。",
        }
        write_json(json_path, result)
        write_text_atomic(md_path, _render_schema_review_markdown(result), encoding="utf-8")
        return result

    def review_schema_mapping(
        self,
        dataset_id: str | None = None,
        rule_file_id: str | None = None,
        semantics_path: str | None = None,
        overrides_path: str | None = None,
    ) -> dict[str, Any]:
        """Infer semantic field roles and rule-field mapping candidates.

        这个工具是跨客户数据泛化的入口：它把“字段叫什么”上升到“字段代表
        什么语义角色”。结果只作为候选和复核依据，不自动改写规则或数据。
        """
        file_id, df = self._dataframe(dataset_id)
        if not self.context.rules:
            self.parse_rule_library(rule_file_id)
        if not self.context.dataset_profiles:
            self.profile_dataset(file_id)

        schema_result = _latest_tool_result(self.context.tool_results, "suggest_schema_metadata")
        if not isinstance(schema_result, dict) or not isinstance(schema_result.get("schema"), dict):
            schema_result = self.suggest_schema_metadata()
        schema = schema_result.get("schema") or {}
        schema_fields = schema.get("fields") or {}
        semantics = load_schema_semantics(semantics_path)
        mapping_overrides = merge_mapping_overrides(
            load_schema_mapping_overrides(overrides_path),
            review_decision_overrides(self.context.run_dir),
        )
        rejected_mapping_keys = rejected_rule_mapping_keys(self.context.run_dir)
        profile = self.context.dataset_profiles.get(file_id) or {}
        profile_index = _column_profile_index(profile)
        known_fields = [str(col) for col in df.columns]
        known_field_set = set(known_fields)

        fields: list[dict[str, Any]] = []
        for field in known_fields:
            semantic = infer_field_semantics(field, semantics=semantics, schema_fields=schema_fields)
            semantic = _apply_confirmed_schema_override(
                semantic,
                schema_mapping_override_for_field(field, mapping_overrides),
                override_source=str(mapping_overrides.get("path") or ""),
            )
            column_profile = profile_index.get(field, {})
            semantic.update(
                {
                    "dtype": column_profile.get("dtype"),
                    "missing_rate": column_profile.get("missing_rate", 0),
                    "unique_count": column_profile.get("unique_count"),
                }
            )
            fields.append(semantic)

        required_fields = sorted({field for rule in self.context.rules for field in _rule_field_candidates(rule)})
        rule_fields: list[dict[str, Any]] = []
        mapping_records: list[dict[str, Any]] = []
        exact_mapping_count = 0
        candidate_mapping_count = 0
        confirmed_mapping_count = 0
        missing_mapping_count = 0
        ambiguous_mapping_count = 0

        for rule_field in required_fields:
            rule_semantic = infer_field_semantics(rule_field, semantics=semantics, schema_fields=schema_fields)
            confirmed_rule_mapping = schema_mapping_override_for_rule_field(rule_field, mapping_overrides)
            candidates = _semantic_rule_mapping_candidates(
                rule_field,
                rule_semantic,
                fields,
                known_field_set,
                confirmed_rule_mapping=confirmed_rule_mapping,
            )
            candidates = _filter_rejected_rule_mapping_candidates(rule_field, candidates, rejected_mapping_keys)
            status = "missing"
            selected = candidates[0] if candidates else None
            if rule_field in known_field_set:
                status = "exact"
                exact_mapping_count += 1
            elif selected:
                status = str(selected.get("status") or "semantic_candidate")
                if status == "confirmed_mapping":
                    confirmed_mapping_count += 1
                else:
                    candidate_mapping_count += 1
            else:
                missing_mapping_count += 1
            ambiguous = bool(
                selected
                and selected.get("status") != "confirmed_mapping"
                and len(candidates) > 1
                and float(candidates[0].get("score") or 0) - float(candidates[1].get("score") or 0) < 0.08
            )
            if ambiguous:
                ambiguous_mapping_count += 1
            rule_field_record = {
                "rule_field": rule_field,
                "semantic_role": rule_semantic.get("semantic_role", ""),
                "display_name": rule_semantic.get("display_name", ""),
                "unit": rule_semantic.get("unit", "unknown"),
                "statistic": rule_semantic.get("statistic", "raw"),
                "confidence": rule_semantic.get("confidence", 0),
                "status": status,
                "selected_field": selected.get("field") if selected else (rule_field if rule_field in known_field_set else None),
                "candidate_count": len(candidates),
                "ambiguous": ambiguous,
                "candidates": candidates,
            }
            rule_fields.append(rule_field_record)
            mapping_records.append(rule_field_record)

        low_confidence_fields = [item for item in fields if float(item.get("confidence") or 0) < 0.7 and not item.get("ignore")]
        unmapped_fields = [item for item in fields if not item.get("semantic_role") and not item.get("ignore")]

        artifacts_dir = self.context.run_dir / "artifacts"
        json_path = artifacts_dir / "schema_mapping.json"
        md_path = artifacts_dir / "schema_mapping.md"
        result = {
            "dataset_id": file_id,
            "source_file": self.context.files[file_id].original_name,
            "semantic_config_path": str(semantics_path or "configs/schema_semantics.yaml"),
            "mapping_overrides_path": str(mapping_overrides.get("path") or overrides_path or "configs/schema_mapping_overrides.yaml"),
            "review_decisions_path": str(self.context.run_dir / "review_decisions.json"),
            "field_count": len(fields),
            "semantic_field_count": sum(1 for item in fields if item.get("semantic_role")),
            "low_confidence_count": len(low_confidence_fields),
            "unmapped_field_count": len(unmapped_fields),
            "required_field_count": len(required_fields),
            "exact_mapping_count": exact_mapping_count,
            "candidate_mapping_count": candidate_mapping_count,
            "confirmed_mapping_count": confirmed_mapping_count,
            "missing_mapping_count": missing_mapping_count,
            "ambiguous_mapping_count": ambiguous_mapping_count,
            "fields": fields,
            "rule_fields": rule_fields,
            "mappings": mapping_records,
            "json_path": str(json_path),
            "markdown_path": str(md_path),
            "summary": (
                f"Schema 语义映射完成：{sum(1 for item in fields if item.get('semantic_role'))}/{len(fields)} 个字段匹配稳定语义角色，"
                f"{exact_mapping_count} 个规则字段精确匹配，{confirmed_mapping_count} 个规则字段已确认映射，"
                f"{candidate_mapping_count} 个规则字段存在候选映射，"
                f"{missing_mapping_count} 个规则字段缺少候选。"
            ),
        }
        write_json(json_path, result)
        write_text_atomic(md_path, _render_schema_mapping_markdown(result), encoding="utf-8")
        return result

    def review_dataset_adaptation(self, dataset_id: str | None = None, rule_file_id: str | None = None) -> dict[str, Any]:
        """Review whether current rules can be safely applied to this dataset schema."""
        file_id, df = self._dataframe(dataset_id)
        if not self.context.rules:
            self.parse_rule_library(rule_file_id)

        record = self.context.files[file_id]
        known_fields = [str(col) for col in df.columns]
        known_field_set = set(known_fields)
        metadata = _dataset_metadata(record.original_name, known_fields)
        schema_result = _latest_tool_result(self.context.tool_results, "suggest_schema_metadata")
        if not isinstance(schema_result, dict) or not isinstance(schema_result.get("schema"), dict):
            schema_result = self.suggest_schema_metadata()
        schema = schema_result.get("schema") or {}
        schema_fields = schema.get("fields") or {}
        schema_mapping = _latest_tool_result(self.context.tool_results, "review_schema_mapping")
        if not isinstance(schema_mapping, dict):
            schema_mapping = self.review_schema_mapping(dataset_id=file_id, rule_file_id=rule_file_id)
        semantic_mapping_index = _schema_mapping_index(schema_mapping)

        rules: list[dict[str, Any]] = []
        mapping_candidates: list[dict[str, Any]] = []
        confirmed_mapping_records: list[dict[str, Any]] = []
        unique_required_fields: set[str] = set()
        ready_count = 0
        needs_mapping_count = 0
        confirmed_mapping_count = 0
        needs_review_count = 0
        not_applicable_count = 0
        disabled_count = 0
        mapped_field_count = 0

        for rule in self.context.rules:
            required_fields = _rule_field_candidates(rule)
            unique_required_fields.update(required_fields)
            field_mappings = [
                _field_mapping_review(field, known_field_set, known_fields, schema_fields, semantic_mapping_index)
                for field in required_fields
            ]
            mapped_field_count += sum(1 for item in field_mappings if item.get("status") != "missing")
            for item in field_mappings:
                if item.get("status") == "confirmed_mapping":
                    confirmed_mapping_records.append({"rule_id": rule.rule_id, "rule_name": rule.name, **item})
                elif item.get("status") != "exact":
                    mapping_candidates.append({"rule_id": rule.rule_id, "rule_name": rule.name, **item})

            issues: list[str] = []
            if not required_fields:
                issues.append("规则未识别出明确字段，不能自动判断适用性。")
            if not rule.condition:
                issues.append("规则缺少可执行 condition。")
            if rule.review_required:
                issues.append("规则被标记为需要人工复核。")

            missing = [item for item in field_mappings if item.get("status") == "missing"]
            candidate = [item for item in field_mappings if item.get("status") in {"suffix_mismatch", "semantic_candidate"}]
            confirmed = [item for item in field_mappings if item.get("status") == "confirmed_mapping"]
            if missing:
                issues.append("存在数据集中找不到的规则字段。")
            if candidate:
                issues.append("存在非精确字段映射候选，需要人工确认后再执行。")

            if not rule.enabled:
                status = "disabled"
                recommendation = "规则已停用，不进入本次自动执行。"
                disabled_count += 1
            elif missing:
                status = "not_applicable"
                recommendation = "不要强行套用该规则；应补充字段映射、转换配置或建立面向新数据结构的新规则。"
                not_applicable_count += 1
            elif candidate:
                status = "needs_mapping_confirmation"
                recommendation = "先确认字段映射语义一致，再决定是否允许自动改写或执行该规则。"
                needs_mapping_count += 1
            elif confirmed:
                status = "mapping_confirmed"
                recommendation = "字段映射已由确认配置支持；下一步应生成映射执行计划并完成回归验证后再自动诊断。"
                confirmed_mapping_count += 1
            elif issues:
                status = "needs_review"
                recommendation = "先补齐规则结构或人工复核后再执行。"
                needs_review_count += 1
            else:
                status = "ready"
                recommendation = "字段精确匹配，可进入规则执行步骤。"
                ready_count += 1

            rules.append(
                {
                    "rule_id": rule.rule_id,
                    "name": rule.name,
                    "status": status,
                    "required_fields": required_fields,
                    "field_mappings": field_mappings,
                    "condition": rule.condition,
                    "severity": str(rule.severity or "medium").lower(),
                    "issues": issues,
                    "recommendation": recommendation,
                    "source_ref": rule.source_ref,
                }
            )

        artifacts_dir = self.context.run_dir / "artifacts"
        json_path = artifacts_dir / "dataset_adaptation.json"
        md_path = artifacts_dir / "dataset_adaptation.md"
        result = {
            "dataset_id": file_id,
            "source_file": record.original_name,
            "dataset_kind_hint": metadata["dataset_kind_hint"],
            "field_count": len(known_fields),
            "rule_count": len(rules),
            "required_field_count": len(unique_required_fields),
            "mapped_field_count": mapped_field_count,
            "ready_count": ready_count,
            "needs_mapping_count": needs_mapping_count,
            "confirmed_mapping_count": confirmed_mapping_count,
            "confirmed_field_mapping_count": len(confirmed_mapping_records),
            "needs_review_count": needs_review_count,
            "not_applicable_count": not_applicable_count,
            "disabled_count": disabled_count,
            "mapping_candidates": mapping_candidates,
            "confirmed_mappings": confirmed_mapping_records,
            "rules": rules,
            "json_path": str(json_path),
            "markdown_path": str(md_path),
            "summary": (
                f"数据适配审查完成：{ready_count} 条可直接执行，"
                f"{confirmed_mapping_count} 条使用已确认映射，{needs_mapping_count} 条需映射确认，{not_applicable_count} 条不适用，"
                f"{needs_review_count + disabled_count} 条需复核或停用。"
            ),
        }
        write_json(json_path, result)
        write_text_atomic(md_path, _render_dataset_adaptation_markdown(result), encoding="utf-8")
        return result

    def review_schema_mapping_candidates(
        self,
        dataset_id: str | None = None,
        rule_file_id: str | None = None,
        min_confirm_score: float = 0.86,
    ) -> dict[str, Any]:
        """Create a conservative checklist for human mapping review.

        这个工具只做复核辅助：它不会写入 ``review_decisions.json``，不会修改
        ``schema_mapping_overrides.yaml``，也不会把候选映射直接视为可执行规则。
        """
        file_id = self._dataset_id(dataset_id)
        schema_mapping = _latest_tool_result(self.context.tool_results, "review_schema_mapping")
        if not isinstance(schema_mapping, dict):
            schema_mapping = self.review_schema_mapping(dataset_id=file_id, rule_file_id=rule_file_id)
        adaptation = _latest_tool_result(self.context.tool_results, "review_dataset_adaptation")
        if not isinstance(adaptation, dict):
            adaptation = self.review_dataset_adaptation(dataset_id=file_id, rule_file_id=rule_file_id)

        schema_index = _schema_mapping_index(schema_mapping)
        min_confirm_score = max(0.0, min(float(min_confirm_score or 0.86), 1.0))
        review_items: list[dict[str, Any]] = []
        for rule in adaptation.get("rules") or []:
            for mapping in rule.get("field_mappings") or []:
                rule_field = str(mapping.get("rule_field") or "")
                schema_item = schema_index.get(rule_field, {})
                review_items.append(
                    _schema_mapping_candidate_review_item(
                        rule,
                        mapping,
                        schema_item,
                        min_confirm_score=min_confirm_score,
                    )
                )

        status_counts = Counter(str(item.get("status") or "unknown") for item in review_items)
        confirm_suggestion_count = int(status_counts.get("suggest_confirm_candidate", 0))
        manual_review_count = int(status_counts.get("manual_review_required", 0))
        not_applicable_count = int(status_counts.get("not_applicable", 0))
        already_safe_count = int(status_counts.get("already_safe", 0))
        overall_status = "ready_for_human_review"
        if not review_items:
            overall_status = "not_run"
        elif manual_review_count or not_applicable_count:
            overall_status = "needs_human_review"
        elif confirm_suggestion_count:
            overall_status = "has_confirm_suggestions"
        elif already_safe_count == len(review_items):
            overall_status = "already_resolved"

        artifacts_dir = self.context.run_dir / "artifacts"
        json_path = artifacts_dir / "schema_mapping_review.json"
        md_path = artifacts_dir / "schema_mapping_review.md"
        result = {
            "dataset_id": file_id,
            "source_file": schema_mapping.get("source_file") or adaptation.get("source_file") or "",
            "status": overall_status,
            "status_label": _schema_mapping_review_status_label(overall_status),
            "anti_overfit_principle": (
                "候选映射复核辅助只给出人工处理建议；不会自动确认字段映射、不会写入全局配置，"
                "也不会把单数据集可执行结果当作业务正确性。"
            ),
            "parameters": {"min_confirm_score": min_confirm_score},
            "review_item_count": len(review_items),
            "confirm_suggestion_count": confirm_suggestion_count,
            "manual_review_count": manual_review_count,
            "not_applicable_count": not_applicable_count,
            "already_safe_count": already_safe_count,
            "status_counts": dict(status_counts),
            "review_items": review_items,
            "json_path": str(json_path),
            "markdown_path": str(md_path),
            "summary": (
                f"映射候选复核辅助完成：{confirm_suggestion_count} 个可考虑确认，"
                f"{manual_review_count} 个需专家复核，{not_applicable_count} 个不适用，"
                f"{already_safe_count} 个已安全处理。"
            ),
        }
        write_json(json_path, result)
        write_text_atomic(md_path, _render_schema_mapping_review_markdown(result), encoding="utf-8")
        return result

    def build_mapping_execution_plan(
        self,
        allow_statistic_mismatch: bool = False,
        allow_unknown_unit: bool = False,
    ) -> dict[str, Any]:
        """Create a reviewable execution plan for confirmed field mappings.

        这一步仍然是“计划”，不是执行：它只说明哪些规则字段可以被映射到
        客户字段、哪些地方存在单位/统计口径风险，以及是否需要回归验证。
        """
        schema_mapping = _latest_tool_result(self.context.tool_results, "review_schema_mapping")
        if not isinstance(schema_mapping, dict):
            schema_mapping = self.review_schema_mapping()
        adaptation = _latest_tool_result(self.context.tool_results, "review_dataset_adaptation")
        if not isinstance(adaptation, dict):
            adaptation = self.review_dataset_adaptation()

        rule_plans: list[dict[str, Any]] = []
        field_plans_flat: list[dict[str, Any]] = []
        executable_without_mapping_count = 0
        ready_for_regression_count = 0
        needs_confirmation_count = 0
        needs_review_count = 0
        blocked_count = 0
        rewritten_rule_count = 0
        regression_required_count = 0

        for rule in adaptation.get("rules") or []:
            field_plans = [
                _mapping_execution_field_plan(
                    mapping,
                    allow_statistic_mismatch=bool(allow_statistic_mismatch),
                    allow_unknown_unit=bool(allow_unknown_unit),
                )
                for mapping in rule.get("field_mappings") or []
            ]
            for item in field_plans:
                field_plans_flat.append({"rule_id": rule.get("rule_id"), "rule_name": rule.get("name"), **item})

            rewrite_pairs = [
                {"rule_field": item.get("rule_field"), "dataset_field": item.get("dataset_field")}
                for item in field_plans
                if item.get("rewrite_action") == "rewrite_field" and item.get("dataset_field")
            ]
            condition = str(rule.get("condition") or "")
            rewritten_condition = _rewrite_condition_fields(condition, rewrite_pairs)
            field_statuses = {str(item.get("status")) for item in field_plans}
            issues = [issue for item in field_plans for issue in item.get("issues") or []]

            if not field_plans:
                status = "blocked"
                recommendation = "规则未识别出字段，不能生成映射执行计划。"
            elif "blocked" in field_statuses:
                status = "blocked"
                recommendation = "存在缺失或不适用字段，不能生成可执行映射。"
            elif "needs_confirmation" in field_statuses:
                status = "needs_mapping_confirmation"
                recommendation = "仍有候选映射未确认，先完成人工确认后再生成执行计划。"
            elif "needs_review" in field_statuses:
                status = "needs_review"
                recommendation = "字段关系已确认，但单位、统计口径或歧义风险需要复核。"
            elif rewrite_pairs:
                status = "ready_for_regression"
                recommendation = "可进入映射后规则回归验证；验证前不得作为自动诊断结论。"
            else:
                status = "executable_without_mapping"
                recommendation = "规则字段精确匹配，无需字段重写。"

            if status == "executable_without_mapping":
                executable_without_mapping_count += 1
            elif status == "ready_for_regression":
                ready_for_regression_count += 1
            elif status == "needs_mapping_confirmation":
                needs_confirmation_count += 1
            elif status == "needs_review":
                needs_review_count += 1
            else:
                blocked_count += 1
            if rewrite_pairs:
                rewritten_rule_count += 1
            if any(item.get("requires_regression_validation") for item in field_plans):
                regression_required_count += 1

            rule_plans.append(
                {
                    "rule_id": rule.get("rule_id"),
                    "name": rule.get("name"),
                    "source_status": rule.get("status"),
                    "status": status,
                    "execution_allowed_now": status == "executable_without_mapping",
                    "requires_regression_validation": any(item.get("requires_regression_validation") for item in field_plans),
                    "condition": condition,
                    "rewritten_condition_preview": rewritten_condition,
                    "rewrite_pairs": rewrite_pairs,
                    "field_plans": field_plans,
                    "issues": issues,
                    "recommendation": recommendation,
                }
            )

        auto_execution_allowed = (
            bool(rule_plans)
            and blocked_count == 0
            and needs_confirmation_count == 0
            and needs_review_count == 0
            and ready_for_regression_count == 0
            and regression_required_count == 0
        )
        artifacts_dir = self.context.run_dir / "artifacts"
        json_path = artifacts_dir / "mapping_execution_plan.json"
        md_path = artifacts_dir / "mapping_execution_plan.md"
        result = {
            "status": "passed" if auto_execution_allowed else ("blocked" if blocked_count else "needs_review"),
            "auto_execution_allowed": auto_execution_allowed,
            "anti_overfit_principle": "映射执行计划只描述字段重写预案，不自动修改规则条件、阈值或诊断结论。",
            "parameters": {
                "allow_statistic_mismatch": bool(allow_statistic_mismatch),
                "allow_unknown_unit": bool(allow_unknown_unit),
            },
            "rule_count": len(rule_plans),
            "executable_without_mapping_count": executable_without_mapping_count,
            "ready_for_regression_count": ready_for_regression_count,
            "needs_confirmation_count": needs_confirmation_count,
            "needs_review_count": needs_review_count,
            "blocked_count": blocked_count,
            "rewritten_rule_count": rewritten_rule_count,
            "regression_required_count": regression_required_count,
            "field_plan_count": len(field_plans_flat),
            "rules": rule_plans,
            "field_plans": field_plans_flat,
            "json_path": str(json_path),
            "markdown_path": str(md_path),
            "summary": (
                f"映射执行计划已生成：{executable_without_mapping_count} 条无需重写，"
                f"{ready_for_regression_count} 条可进入回归验证，{needs_confirmation_count} 条需确认，"
                f"{needs_review_count} 条需复核，{blocked_count} 条阻断。"
            ),
        }
        write_json(json_path, result)
        write_text_atomic(md_path, _render_mapping_execution_plan_markdown(result), encoding="utf-8")
        return result

    def validate_mapping_regression(
        self,
        dataset_id: str | None = None,
        require_strict_unit: bool = True,
        require_strict_statistic: bool = True,
    ) -> dict[str, Any]:
        """Validate confirmed mapping rewrites without changing rules.

        这一步只验证“已确认字段映射生成的重写预案是否能透明执行”。它不会
        把映射写入全局配置，不会修改原始规则，也不会因为当前数据命中率好看
        就认定阈值业务正确。
        """
        file_id, df = self._dataframe(dataset_id)
        plan = _latest_tool_result(self.context.tool_results, "build_mapping_execution_plan")
        if not isinstance(plan, dict):
            plan = self.build_mapping_execution_plan()

        rules: list[dict[str, Any]] = []
        passed_count = 0
        no_mapping_count = 0
        regression_passed_count = 0
        needs_confirmation_count = 0
        needs_review_count = 0
        blocked_count = 0
        failed_count = 0

        for rule in plan.get("rules") or []:
            status, record = _validate_mapping_regression_rule(
                df,
                rule,
                require_strict_unit=bool(require_strict_unit),
                require_strict_statistic=bool(require_strict_statistic),
            )
            rules.append(record)
            if status == "passed_no_mapping":
                passed_count += 1
                no_mapping_count += 1
            elif status == "regression_passed":
                passed_count += 1
                regression_passed_count += 1
            elif status == "needs_confirmation":
                needs_confirmation_count += 1
            elif status == "needs_review":
                needs_review_count += 1
            elif status == "failed":
                failed_count += 1
            else:
                blocked_count += 1

        mapped_execution_validated = (
            bool(rules)
            and failed_count == 0
            and blocked_count == 0
            and needs_confirmation_count == 0
            and needs_review_count == 0
        )
        status = "passed" if mapped_execution_validated else ("failed" if failed_count or blocked_count else "needs_review")
        artifacts_dir = self.context.run_dir / "artifacts"
        json_path = artifacts_dir / "mapping_regression_validation.json"
        md_path = artifacts_dir / "mapping_regression_validation.md"
        result = {
            "dataset_id": file_id,
            "status": status,
            "mapped_execution_validated": mapped_execution_validated,
            "anti_overfit_principle": "映射回归验证只证明重写预案可审查、可执行；不证明阈值业务正确，也不自动推广到全局配置。",
            "parameters": {
                "require_strict_unit": bool(require_strict_unit),
                "require_strict_statistic": bool(require_strict_statistic),
            },
            "rule_count": len(rules),
            "passed_count": passed_count,
            "no_mapping_count": no_mapping_count,
            "regression_passed_count": regression_passed_count,
            "needs_confirmation_count": needs_confirmation_count,
            "needs_review_count": needs_review_count,
            "blocked_count": blocked_count,
            "failed_count": failed_count,
            "rules": rules,
            "json_path": str(json_path),
            "markdown_path": str(md_path),
            "summary": (
                f"映射回归验证完成：{regression_passed_count} 条映射重写通过，"
                f"{no_mapping_count} 条无需映射，{needs_confirmation_count} 条需确认，"
                f"{needs_review_count} 条需复核，{blocked_count} 条阻断，{failed_count} 条失败。"
            ),
        }
        write_json(json_path, result)
        write_text_atomic(md_path, _render_mapping_regression_validation_markdown(result), encoding="utf-8")
        return result

    def review_generalization_readiness(
        self,
        min_semantic_coverage: float = 0.6,
        max_unresolved_rule_ratio: float = 0.35,
    ) -> dict[str, Any]:
        """Gate whether this run is safe to generalize without overfitting.

        这个工具不追求“当前数据集结果漂亮”，而是检查系统有没有正确承认
        字段结构差异、候选映射、不适用规则和不可解析规则。它给出门禁结论，
        但不修改规则、阈值、字段映射或执行结果。
        """
        schema_mapping = _latest_tool_result(self.context.tool_results, "review_schema_mapping")
        if not isinstance(schema_mapping, dict):
            schema_mapping = self.review_schema_mapping()
        adaptation = _latest_tool_result(self.context.tool_results, "review_dataset_adaptation")
        if not isinstance(adaptation, dict):
            adaptation = self.review_dataset_adaptation()
        mapping_execution_plan = _latest_tool_result(self.context.tool_results, "build_mapping_execution_plan")
        rule_execution = _latest_tool_result(self.context.tool_results, "apply_rules")
        validation = self.context.validation_results[-1] if self.context.validation_results else {}

        field_count = int(schema_mapping.get("field_count") or 0)
        semantic_field_count = int(schema_mapping.get("semantic_field_count") or 0)
        semantic_coverage = round(semantic_field_count / field_count, 4) if field_count else 0.0
        rule_count = int(adaptation.get("rule_count") or 0)
        ready_count = int(adaptation.get("ready_count") or 0)
        confirmed_mapping_count = int(adaptation.get("confirmed_mapping_count") or 0)
        needs_mapping_count = int(adaptation.get("needs_mapping_count") or 0)
        not_applicable_count = int(adaptation.get("not_applicable_count") or 0)
        needs_review_count = int(adaptation.get("needs_review_count") or 0) + int(adaptation.get("disabled_count") or 0)
        unresolved_rule_count = confirmed_mapping_count + needs_mapping_count + not_applicable_count + needs_review_count
        unresolved_rule_ratio = round(unresolved_rule_count / rule_count, 4) if rule_count else 0.0

        rule_results = rule_execution.get("rule_results") if isinstance(rule_execution, dict) else []
        evaluated_count = sum(1 for item in (rule_results or []) if item.get("status") == "evaluated")
        unparseable_count = sum(1 for item in (rule_results or []) if item.get("status") == "unparseable")
        rule_path_resolved = bool(rule_count and unresolved_rule_count == 0)
        plan_available = isinstance(mapping_execution_plan, dict)
        plan_regression_required = int(mapping_execution_plan.get("regression_required_count") or 0) if plan_available else 0
        plan_blocked = int(mapping_execution_plan.get("blocked_count") or 0) if plan_available else 0

        checks: list[dict[str, Any]] = []
        checks.append(
            _generalization_check(
                "schema_mapping_available",
                "Schema 语义映射已生成",
                bool(schema_mapping),
                "缺少 Schema 语义映射时，系统无法判断新字段是否只是名称变化还是结构变化。",
                evidence=f"semantic_field_count={semantic_field_count}, field_count={field_count}",
            )
        )
        checks.append(
            _generalization_check(
                "semantic_coverage",
                "字段语义覆盖率",
                semantic_coverage >= float(min_semantic_coverage) or rule_path_resolved,
                f"语义覆盖率 {semantic_coverage:.1%} 低于门槛 {float(min_semantic_coverage):.1%}，需要扩展通用语义角色或人工复核。",
                evidence=f"{semantic_field_count}/{field_count}, rule_path_resolved={rule_path_resolved}",
                status_if_failed="needs_review",
            )
        )
        checks.append(
            _generalization_check(
                "dataset_adaptation_available",
                "数据适配审查已生成",
                bool(adaptation),
                "缺少数据适配审查时，系统可能绕过字段适配直接解释规则命中。",
                evidence=f"rule_count={rule_count}",
            )
        )
        checks.append(
            _generalization_check(
                "unresolved_mappings_are_blocked",
                "未解决映射未被当作自动结论",
                unresolved_rule_count == 0 or validation.get("status") != "passed",
                "存在已确认映射、候选映射、不适用或需复核规则时，最终校验不能直接通过。",
                evidence=f"unresolved_rule_count={unresolved_rule_count}, validation={validation.get('status', 'not_run')}",
            )
        )
        checks.append(
            _generalization_check(
                "confirmed_mapping_requires_execution_plan",
                "已确认映射仍要求执行计划",
                confirmed_mapping_count == 0 or plan_available,
                "已确认映射只代表字段关系已沉淀，仍需生成映射执行计划和回归验证，不能直接等同于自动诊断。",
                evidence=f"confirmed_mapping_count={confirmed_mapping_count}, plan_available={plan_available}",
                status_if_failed="needs_review",
            )
        )
        if confirmed_mapping_count and plan_available:
            checks.append(
                _generalization_check(
                    "mapping_execution_plan_requires_regression",
                    "映射执行计划仍需回归验证",
                    plan_regression_required == 0 and plan_blocked == 0 and bool(mapping_execution_plan.get("auto_execution_allowed")),
                    "已确认映射已有执行计划，但计划仍要求回归验证或存在阻断项，不能直接自动诊断。",
                    evidence=(
                        f"regression_required={plan_regression_required}, "
                        f"blocked={plan_blocked}, auto_execution_allowed={mapping_execution_plan.get('auto_execution_allowed')}"
                    ),
                    status_if_failed="needs_review",
                )
            )
        checks.append(
            _generalization_check(
                "candidate_mapping_requires_confirmation",
                "候选映射保持人工确认状态",
                needs_mapping_count == 0,
                "仍有规则只有候选映射，不能通过增加当前数据集专属别名来伪装成可执行。",
                evidence=f"needs_mapping_count={needs_mapping_count}",
                status_if_failed="needs_review",
            )
        )
        checks.append(
            _generalization_check(
                "not_applicable_is_not_forced",
                "不适用规则未被强行套用",
                not_applicable_count == 0,
                "存在不适用规则是合理风险输出，应补充通用映射、转换配置或建立新规则，而不是围绕当前数据调阈值。",
                evidence=f"not_applicable_count={not_applicable_count}",
                status_if_failed="needs_review",
            )
        )
        if rule_results:
            checks.append(
                _generalization_check(
                    "rule_execution_matches_adaptation",
                    "规则执行与适配审查一致",
                    not (unparseable_count and ready_count == rule_count and rule_count > 0),
                    "适配审查称全部可直接执行，但规则执行仍存在不可解析项，需复核字段绑定或规则解析。",
                    evidence=f"evaluated={evaluated_count}, unparseable={unparseable_count}, ready={ready_count}/{rule_count}",
                )
            )
        else:
            checks.append(
                _generalization_check(
                    "rule_execution_available",
                    "规则执行结果已生成",
                    False,
                    "缺少规则执行结果，无法判断适配审查与实际执行是否一致。",
                    evidence="apply_rules not found",
                    status_if_failed="needs_review",
                )
            )

        blocking_checks = [item for item in checks if item.get("status") == "blocked"]
        review_checks = [item for item in checks if item.get("status") == "needs_review"]
        status = "blocked" if blocking_checks else ("needs_review" if review_checks else "passed")
        auto_diagnosis_allowed = status == "passed" and unresolved_rule_count == 0 and unparseable_count == 0
        if status == "passed":
            verdict = "未发现过拟合门禁风险；当前数据可进入自动诊断解释。"
        elif status == "needs_review":
            verdict = "系统已识别泛化风险并阻止直接自动诊断；应先复核映射、单位、统计口径或规则适用性。"
        else:
            verdict = "泛化门禁阻断；缺少关键证据或存在把未解决适配问题当作自动结论的风险。"

        recommendations = _generalization_recommendations(
            confirmed_mapping_count=confirmed_mapping_count,
            needs_mapping_count=needs_mapping_count,
            not_applicable_count=not_applicable_count,
            semantic_coverage=semantic_coverage,
            min_semantic_coverage=float(min_semantic_coverage),
            unresolved_rule_ratio=unresolved_rule_ratio,
            max_unresolved_rule_ratio=float(max_unresolved_rule_ratio),
        )
        artifacts_dir = self.context.run_dir / "artifacts"
        json_path = artifacts_dir / "generalization_readiness.json"
        md_path = artifacts_dir / "generalization_readiness.md"
        result = {
            "status": status,
            "status_label": _generalization_status_label(status),
            "auto_diagnosis_allowed": auto_diagnosis_allowed,
            "anti_overfit_principle": "不得为了让当前数据集通过而改阈值、改规则或把候选映射直接当作已确认映射。",
            "verdict": verdict,
            "metrics": {
                "semantic_coverage": semantic_coverage,
                "field_count": field_count,
                "semantic_field_count": semantic_field_count,
                "rule_count": rule_count,
                "ready_count": ready_count,
                "confirmed_mapping_count": confirmed_mapping_count,
                "mapping_execution_plan_available": plan_available,
                "mapping_execution_plan_regression_required_count": plan_regression_required,
                "needs_mapping_count": needs_mapping_count,
                "not_applicable_count": not_applicable_count,
                "needs_review_or_disabled_count": needs_review_count,
                "unresolved_rule_count": unresolved_rule_count,
                "unresolved_rule_ratio": unresolved_rule_ratio,
                "evaluated_rule_count": evaluated_count,
                "unparseable_rule_count": unparseable_count,
            },
            "thresholds": {
                "min_semantic_coverage": float(min_semantic_coverage),
                "max_unresolved_rule_ratio": float(max_unresolved_rule_ratio),
            },
            "checks": checks,
            "recommendations": recommendations,
            "json_path": str(json_path),
            "markdown_path": str(md_path),
            "summary": f"泛化验收门禁{_generalization_status_label(status)}：{len(blocking_checks)} 个阻断项，{len(review_checks)} 个复核项。",
        }
        write_json(json_path, result)
        write_text_atomic(md_path, _render_generalization_readiness_markdown(result), encoding="utf-8")
        return result

    def check_missing_values(self, dataset_id: str | None = None) -> dict[str, Any]:
        """Return missing-value diagnostics and simple treatment suggestions."""
        file_id, df = self._dataframe(dataset_id)
        rows = len(df)
        columns = []
        for col in df.columns:
            missing = int(df[col].isna().sum())
            if missing:
                rate = missing / rows if rows else 0.0
                recommendation = "复核源数据"
                # 这里的建议只是数据质量层面的提示，不自动修改原始数据。
                if rate < 0.05:
                    recommendation = "可考虑简单填补或逐行复核"
                elif rate > 0.4:
                    recommendation = "建议从自动决策字段中排除"
                columns.append(
                    {
                        "column": str(col),
                        "missing_count": missing,
                        "missing_rate": round(rate, 6),
                        "recommendation": recommendation,
                    }
                )
        return {
            "dataset_id": file_id,
            "columns_with_missing": columns,
            "summary": f"发现 {len(columns)} 个字段存在缺失值。",
        }

    def detect_outliers(
        self,
        dataset_id: str | None = None,
        method: str = "iqr",
        columns: list[str] | None = None,
    ) -> dict[str, Any]:
        """Detect numeric outliers with transparent statistical rules."""
        file_id, df = self._dataframe(dataset_id)
        # 只对数值列做基础离群值提示；文本/类别字段应由规则或频繁模式工具处理。
        numeric_cols = [str(c) for c in df.select_dtypes(include="number").columns]
        selected = [c for c in (columns or numeric_cols) if c in numeric_cols]
        results = []
        for col in selected:
            series = df[col].dropna()
            if series.empty:
                continue
            if method == "zscore":
                # z-score 适合近似正态分布；阈值固定为 3，便于审计。
                std = float(series.std() or 0)
                if std == 0:
                    mask = pd.Series(False, index=df.index)
                else:
                    z = (df[col] - float(series.mean())) / std
                    mask = z.abs() > 3
                bounds = {"z_abs": 3}
            else:
                # IQR 方法不要求正态分布，是 demo 默认的更稳妥选择。
                q1 = float(series.quantile(0.25))
                q3 = float(series.quantile(0.75))
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                mask = (df[col] < lower) | (df[col] > upper)
                bounds = {"lower": lower, "upper": upper}
            indices = [int(i) for i in df.index[mask.fillna(False)].tolist()[:20]]
            # sample_indices 只保留前 20 个，给专家复核入口，不在报告里塞所有行。
            results.append(
                {
                    "column": col,
                    "outlier_count": int(mask.fillna(False).sum()),
                    "outlier_rate": round(float(mask.fillna(False).mean()), 6) if len(df) else 0.0,
                    "bounds": to_jsonable(bounds),
                    "sample_indices": indices,
                }
            )
        return {"dataset_id": file_id, "method": method, "outliers": results, "summary": f"已检查 {len(selected)} 个数值字段的离群值。"}

    @versioned_tool
    def apply_rules(self, dataset_id: str | None = None, rule_file_id: str | None = None) -> dict[str, Any]:
        """Evaluate parsed RuleCard conditions against the selected dataset."""
        file_id, df = self._dataframe(dataset_id)
        if not self.context.rules:
            # 用户可能没有显式先调用 parse_rule_library。工具层自动补一次解析，
            # 让 apply_rules 可以独立调用。
            self.parse_rule_library(rule_file_id)
        results = []
        for rule in self.context.rules:
            if not rule.enabled:
                results.append(
                    {
                        "rule_id": rule.rule_id,
                        "name": rule.name,
                        "status": "disabled",
                        "message": "规则已在规则库中停用。",
                        "severity": rule.severity,
                        "target_fields": rule.target_fields,
                        "recommendation": rule.recommendation,
                        "review_required": rule.review_required,
                        "source_ref": rule.source_ref,
                    }
                )
                continue
            mask, error = evaluate_rule_condition(df, rule)
            if error:
                # 不可解析规则不能偷偷跳过；显式返回 unparseable 方便人工复核。
                results.append(
                    {
                        "rule_id": rule.rule_id,
                        "name": rule.name,
                        "status": "unparseable",
                        "message": error,
                        "failure": ToolFailure(code="RULE_UNPARSEABLE", message=error, stage="rule_evaluation").model_dump(),
                        "severity": rule.severity,
                        "target_fields": rule.target_fields,
                        "recommendation": rule.recommendation,
                        "review_required": rule.review_required,
                        "source_ref": rule.source_ref,
                    }
                )
                continue
            matched = int(mask.sum())
            total = int(len(df))
            # match_rate 是规则命中比例，不代表规则正确性；正确性仍需专家/测试集评估。
            results.append(
                {
                    "rule_id": rule.rule_id,
                    "name": rule.name,
                    "status": "evaluated",
                    "condition": rule.condition or rule.description,
                    "matched_rows": matched,
                    "match_rate": round(matched / total, 6) if total else 0.0,
                    "sample_indices": [int(i) for i in df.index[mask].tolist()[:20]],
                    "severity": rule.severity,
                    "target_fields": rule.target_fields,
                    "recommendation": rule.recommendation,
                    "review_required": rule.review_required,
                    "source_ref": rule.source_ref,
                }
            )
        return {
            "dataset_id": file_id,
            "rule_results": results,
            "summary": f"已执行 {sum(1 for r in results if r['status'] == 'evaluated')} 条规则；{sum(1 for r in results if r['status'] == 'unparseable')} 条规则无法解析。",
        }

    def mine_association_rules(
        self,
        dataset_id: str | None = None,
        columns: list[str] | None = None,
        min_support: float = 0.2,
        min_confidence: float = 0.7,
        min_lift: float = 1.2,
        min_count: int = 2,
        max_len: int = 3,
        max_antecedent: int = 2,
        max_rules: int = 15,
        max_rows: int = 5000,
    ) -> dict[str, Any]:
        """Mine association rules using FP-growth tree generation."""
        file_id, df = self._dataframe(dataset_id)
        # 关联规则挖掘不直接处理连续数值，而是先由 build_transactions 把每行
        # 转成 field=value 的事务集合，数值列会自动分箱。
        transactions = build_transactions(df, columns=columns, max_rows=max_rows)
        mined = mine_with_fpgrowth(
            transactions,
            min_support=min_support,
            min_confidence=min_confidence,
            min_lift=min_lift,
            max_len=max_len,
            max_antecedent=max_antecedent,
            max_rules=max_rules,
            min_count=min_count,
        )
        return {
            "dataset_id": file_id,
            "transactions": len(transactions),
            **mined,
        }

    def check_association_backend(self) -> dict[str, Any]:
        """Report which FP-growth backend the association-mining tool will use."""
        return datamining_backend_status()

    def discover_frequent_patterns(
        self,
        dataset_id: str | None = None,
        columns: list[str] | None = None,
        min_support: float = 0.1,
    ) -> dict[str, Any]:
        """Find high-support single-field patterns for quick exploratory analysis."""
        file_id, df = self._dataframe(dataset_id)
        transactions = build_transactions(df, columns=columns, max_rows=5000)
        n = len(transactions)
        item_counts = Counter(item for row in transactions for item in row)
        # frequent_patterns 是比 association_rules 更轻的探索工具，只看单项高频。
        patterns = [
            {"pattern": item, "support": round(count / n, 6), "matched_rows": int(count)}
            for item, count in item_counts.items()
            if n and count / n >= min_support
        ]
        patterns.sort(key=lambda p: (p["support"], p["matched_rows"]), reverse=True)
        return {"dataset_id": file_id, "patterns": patterns[:100], "summary": f"发现 {len(patterns[:100])} 个高频单字段模式。"}

    def compute_correlations(
        self,
        dataset_id: str | None = None,
        columns: list[str] | None = None,
        top_n: int = 20,
    ) -> dict[str, Any]:
        """Compute strongest numeric field correlations."""
        file_id, df = self._dataframe(dataset_id)
        numeric = df.select_dtypes(include="number")
        if columns:
            selected = [c for c in columns if c in numeric.columns]
            numeric = numeric[selected]
        if numeric.shape[1] < 2:
            return {"dataset_id": file_id, "correlations": [], "summary": "可用于相关性分析的数值字段少于 2 个。"}
        # Pearson 相关性只衡量线性关系。报告里应把它作为线索，而不是因果结论。
        corr = numeric.corr(numeric_only=True)
        pairs = []
        cols = list(corr.columns)
        for i, a in enumerate(cols):
            for b in cols[i + 1 :]:
                value = corr.loc[a, b]
                if pd.isna(value):
                    continue
                pairs.append({"field_a": str(a), "field_b": str(b), "correlation": round(float(value), 6), "abs_correlation": round(abs(float(value)), 6)})
        pairs.sort(key=lambda x: x["abs_correlation"], reverse=True)
        return {"dataset_id": file_id, "correlations": pairs[:top_n], "summary": f"已计算 {numeric.shape[1]} 个数值字段的 Pearson 相关性。"}

    def summarize_findings(self, focus: str | None = None) -> dict[str, Any]:
        """Compress accumulated tool summaries into a model/report-friendly list."""
        summaries = []
        for item in self.context.tool_results:
            # 优先使用工具主动返回的 summary；没有 summary 时才截断原始结果。
            result = item.get("result")
            if isinstance(result, dict) and result.get("summary"):
                summaries.append({"tool": item.get("tool"), "summary": result["summary"]})
            else:
                summaries.append({"tool": item.get("tool"), "summary": truncate_text(str(result), 200)})
        return {"focus": focus or "", "tool_summaries": summaries, "summary": f"已汇总 {len(summaries)} 个工具结果。"}

    def build_visualizations(self, max_items: int = 8) -> dict[str, Any]:
        """Generate static charts used by the final report and web UI."""
        return build_visualization_artifacts(self.context, max_items=max_items)

    @versioned_tool
    def validate_analysis(self, required_tools: list[str] | None = None, strict: bool = False) -> dict[str, Any]:
        """Validate that the Agent has enough tool-backed evidence to report.

        这是“结果校验层”的 MVP 版本。它不判断业务结论一定正确，而是检查：
        - 关键工具是否都执行过；
        - 规则命中是否有可执行结果；
        - 证据链节点数是否和工具调用基本匹配。

        strict=True 时，缺少必需工具会变成 errors；默认 strict=False 更适合 demo，
        允许生成报告但带告警。
        """
        required_tools = required_tools or [
            "profile_dataset",
            "parse_rule_library",
            "validate_rule_library",
            "suggest_schema_metadata",
            "review_schema_metadata",
            "review_schema_mapping",
            "review_dataset_adaptation",
            "build_mapping_execution_plan",
            "validate_mapping_regression",
            "review_generalization_readiness",
            "check_missing_values",
            "detect_outliers",
            "apply_rules",
            "check_association_backend",
            "mine_association_rules",
            "summarize_findings",
        ]
        completed = [str(item.get("tool")) for item in self.context.tool_results]
        canonical_completed = {_canonical_tool_name(name) for name in completed}
        missing = [name for name in required_tools if _canonical_tool_name(name) not in canonical_completed]
        warnings = []
        errors = []
        requires_review = False
        if missing:
            (errors if strict else warnings).append(f"缺少必需工具调用：{', '.join(missing)}")

        rule_results = _latest_tool_result(self.context.tool_results, "apply_rules")
        if isinstance(rule_results, dict):
            # 规则校验关注“有没有实际执行规则”和“有没有待人工复核规则”。
            results = rule_results.get("rule_results", [])
            evaluated = [r for r in results if r.get("status") == "evaluated"]
            unparseable = [r for r in results if r.get("status") == "unparseable"]
            if not evaluated:
                warnings.append("没有成功执行任何可计算规则。")
                requires_review = True
            if unparseable:
                warnings.append(f"{len(unparseable)} 条规则无法解析，已保留供人工复核。")
                requires_review = True
            if results and len(unparseable) == len(results):
                warnings.append("本次规则库与数据字段口径不匹配，报告不得作为规则诊断结论。")
                requires_review = True
        else:
            warnings.append("缺少规则执行结果。")
            requires_review = True

        adaptation = _latest_tool_result(self.context.tool_results, "review_dataset_adaptation")
        if isinstance(adaptation, dict):
            needs_mapping = int(adaptation.get("needs_mapping_count") or 0)
            confirmed_mapping = int(adaptation.get("confirmed_mapping_count") or 0)
            not_applicable = int(adaptation.get("not_applicable_count") or 0)
            mapping_plan = _latest_tool_result(self.context.tool_results, "build_mapping_execution_plan")
            if confirmed_mapping:
                if isinstance(mapping_plan, dict):
                    warnings.append(f"{confirmed_mapping} 条规则使用已确认字段映射，映射执行计划已生成，但需回归验证后再作为自动诊断依据。")
                else:
                    warnings.append(f"{confirmed_mapping} 条规则使用已确认字段映射，需生成映射执行计划并回归验证后再作为自动诊断依据。")
                requires_review = True
            if needs_mapping:
                warnings.append(f"{needs_mapping} 条规则只有字段映射候选，需确认后再作为自动诊断依据。")
                requires_review = True
            if not_applicable:
                warnings.append(f"{not_applicable} 条规则不适用于当前数据结构，不应强行套用。")
                requires_review = True
            if adaptation.get("rule_count") and not adaptation.get("ready_count") and not confirmed_mapping:
                warnings.append("数据适配审查未发现可直接执行规则，报告不得作为规则诊断结论。")
                requires_review = True

        mapping_regression = _latest_tool_result(self.context.tool_results, "validate_mapping_regression")
        if isinstance(mapping_regression, dict):
            regression_status = str(mapping_regression.get("status") or "unknown")
            if regression_status != "passed":
                warnings.append(f"映射回归验证{_mapping_regression_status_label(regression_status)}：{mapping_regression.get('summary', '')}")
                requires_review = True
        elif isinstance(_latest_tool_result(self.context.tool_results, "build_mapping_execution_plan"), dict):
            warnings.append("缺少映射回归验证结果，已确认映射不得直接进入自动诊断。")
            requires_review = True

        schema_mapping = _latest_tool_result(self.context.tool_results, "review_schema_mapping")
        if isinstance(schema_mapping, dict):
            missing_mapping = int(schema_mapping.get("missing_mapping_count") or 0)
            ambiguous_mapping = int(schema_mapping.get("ambiguous_mapping_count") or 0)
            if missing_mapping:
                warnings.append(f"{missing_mapping} 个规则字段缺少 Schema 语义映射候选，需补充别名或建立新规则。")
                requires_review = True
            if ambiguous_mapping:
                warnings.append(f"{ambiguous_mapping} 个规则字段存在多个相近候选，需人工确认映射。")
                requires_review = True

        generalization = _latest_tool_result(self.context.tool_results, "review_generalization_readiness")
        if isinstance(generalization, dict):
            gate_status = str(generalization.get("status") or "unknown")
            if gate_status != "passed":
                warnings.append(f"泛化验收门禁{_generalization_status_label(gate_status)}：{generalization.get('verdict', '')}")
                requires_review = True
            if not generalization.get("auto_diagnosis_allowed"):
                warnings.append("泛化门禁未允许自动诊断解释，报告应作为复核材料而不是最终诊断结论。")
                requires_review = True

        evidence_nodes = len(self.context.evidence.nodes)
        if evidence_nodes < len(completed):
            # 正常情况下，每个工具调用都会生成至少一个 evidence node。
            warnings.append("证据链节点数少于已完成工具调用数。")

        evidence_source = self.context.evidence.path if self.context.evidence.path.exists() else self.context.evidence.to_dict()
        try:
            evidence_verification = verify_chain(evidence_source).to_dict()
        except Exception as exc:
            evidence_verification = {
                "valid": False,
                "verified": False,
                "schema_version": self.context.evidence.to_dict().get("schema_version"),
                "node_count": evidence_nodes,
                "expected_chain_hash": self.context.evidence.chain_hash(),
                "actual_chain_hash": None,
                "first_mismatch_index": None,
                "first_mismatch_node_id": None,
                "errors": [{
                    "kind": "verification_error", "message": str(exc),
                    "failure": ToolFailure.from_exception(
                        exc, stage="evidence_verification", code="EVIDENCE_VERIFICATION_FAILED",
                    ).model_dump(),
                }],
            }
        if not evidence_verification.get("valid"):
            errors.append(f"证据链验证失败：{_verification_failure_summary(evidence_verification)}")

        status = "failed" if errors else ("needs_review" if requires_review else "passed")
        status_label = {"passed": "通过", "needs_review": "需复核", "failed": "失败"}.get(status, status)
        result = {
            "status": status,
            "required_tools": required_tools,
            "completed_tools": completed,
            "missing_tools": missing,
            "warnings": warnings,
            "errors": errors,
            "evidence_nodes": evidence_nodes,
            "evidence_schema_version": evidence_verification.get("schema_version"),
            "chain_hash": evidence_verification.get("expected_chain_hash") or self.context.evidence.chain_hash(),
            "evidence_verification": evidence_verification,
            "summary": f"结果校验{status_label}：{len(warnings)} 个告警，{len(errors)} 个错误。",
        }
        if status == "failed":
            result["failure"] = ToolFailure(
                code="ANALYSIS_VALIDATION_FAILED", message="; ".join(errors), stage="analysis_validation",
            ).model_dump()
        result["schema_version"] = TOOL_SCHEMA_VERSION
        self.context.validation_results.append(result)
        # validation_results 进入 manifest/report，便于前端展示校验状态。
        self.context.save_manifest()
        return result

    @versioned_tool
    def write_report(self, final_summary: str = "") -> dict[str, Any]:
        """Write report artifacts based on the current RunContext."""
        result = write_report_files(self.context, final_summary=final_summary)
        return result

    def record_evidence(self, summary: str, details: dict[str, Any] | None = None) -> dict[str, Any]:
        """Allow the model to add an explicit note to the evidence chain."""
        details = details or {}
        node = self.context.evidence.add(
            type="model_note",
            tool_name="record_evidence",
            arguments={"summary": summary, "details": details},
            output=details,
            summary=summary,
        )
        return {"evidence_reference": evidence_node_reference(node), "summary": "证据备注已记录。"}


def _render_rule_validation_markdown(result: dict[str, Any]) -> str:
    """Render rule validation results for human review."""
    lines = [
        "# 规则库校验清单",
        "",
        f"- 规则总数: {result.get('rule_count')}",
        f"- 可执行规则: {result.get('ready_count')}",
        f"- 需复核规则: {result.get('needs_review_count')}",
        f"- 已停用规则: {result.get('disabled_count')}",
        "",
        "## 规则明细",
    ]
    for item in result.get("rules") or []:
        issues = "；".join(item.get("issues") or []) or "无"
        lines.append(
            f"- `{item.get('rule_id')}` {item.get('name')}：状态={item.get('status')}，"
            f"严重程度={item.get('severity')}，字段={', '.join(item.get('target_fields') or []) or '未绑定'}，问题={issues}"
        )
        if item.get("recommendation"):
            lines.append(f"  建议动作: {item.get('recommendation')}")
    lines.append("")
    return "\n".join(lines)


def _render_schema_review_markdown(result: dict[str, Any]) -> str:
    """Render schema review checklist for human confirmation."""
    lines = [
        "# Schema 复核清单",
        "",
        f"- 字段总数: {result.get('field_count')}",
        f"- 低置信度字段数: {result.get('low_confidence_count')}",
        f"- 忽略字段数: {result.get('ignored_count')}",
        f"- 规则绑定字段数: {result.get('rule_bound_count')}",
        "",
        "## 复核项",
    ]
    review_items = result.get("review_items") or []
    if review_items:
        for item in review_items:
            lines.append(f"- `{item.get('field')}` [{item.get('type')}]: {item.get('message')}")
    else:
        lines.append("- 当前没有必须复核的字段项。")
    lines.append("")
    return "\n".join(lines)


def _render_schema_mapping_markdown(result: dict[str, Any]) -> str:
    """Render semantic schema mapping results for human review."""
    lines = [
        "# Schema 语义映射审查",
        "",
        "本报告用于判断新数据字段与稳定语义角色、规则字段之间的关系；系统只输出候选映射，不自动改写规则或阈值。",
        "",
        f"- 数据文件: `{result.get('source_file')}`",
        f"- 字段总数: {result.get('field_count')}",
        f"- 已匹配语义角色字段: {result.get('semantic_field_count')}",
        f"- 低置信度字段: {result.get('low_confidence_count')}",
        f"- 未映射业务字段: {result.get('unmapped_field_count')}",
        f"- 规则字段: {result.get('required_field_count')}",
        f"- 精确匹配: {result.get('exact_mapping_count')}",
        f"- 已确认映射: {result.get('confirmed_mapping_count', 0)}",
        f"- 候选映射: {result.get('candidate_mapping_count')}",
        f"- 缺少候选: {result.get('missing_mapping_count')}",
        f"- 存在歧义: {result.get('ambiguous_mapping_count')}",
        "",
        "## 字段语义角色",
    ]
    visible_fields = [item for item in result.get("fields") or [] if item.get("semantic_role") or not item.get("ignore")]
    if visible_fields:
        for item in visible_fields[:80]:
            role = item.get("semantic_role") or "未匹配"
            lines.append(
                f"- `{item.get('field')}` -> {role}；显示名: {item.get('display_name')}；"
                f"单位: {item.get('unit')}；统计口径: {item.get('statistic')}；置信度: {item.get('confidence')}。"
            )
    else:
        lines.append("- 当前没有可展示的字段语义记录。")
    lines.extend(["", "## 规则字段候选映射"])
    for item in result.get("rule_fields") or []:
        selected = item.get("selected_field") or "无候选"
        ambiguous = "；存在歧义，需人工确认" if item.get("ambiguous") else ""
        lines.append(
            f"- `{item.get('rule_field')}` -> `{selected}`；状态: {_field_mapping_status_label(str(item.get('status')))}；"
            f"语义角色: {item.get('semantic_role') or '未匹配'}；候选数: {item.get('candidate_count')}{ambiguous}。"
        )
        candidates = item.get("candidates") or []
        if candidates:
            parts = [
                f"{candidate.get('field')} score={candidate.get('score')} method={candidate.get('match_method')}"
                for candidate in candidates[:3]
            ]
            lines.append(f"  候选: {'; '.join(parts)}")
    if not result.get("rule_fields"):
        lines.append("- 当前规则库没有识别出规则字段。")
    lines.extend(
        [
            "",
            "## 使用边界",
            "- 语义角色匹配不代表单位、采样频率、统计窗口完全一致。",
            "- 候选映射需要人工确认后，才能进入自动字段改写或正式诊断解释。",
            "- 如果客户数据结构变化明显，应优先查看本报告，再查看规则命中率。",
            "",
        ]
    )
    return "\n".join(lines)


def _render_dataset_adaptation_markdown(result: dict[str, Any]) -> str:
    """Render dataset adaptation review for humans before rule execution."""
    lines = [
        "# 数据适配审查报告",
        "",
        "本报告用于判断当前规则库是否能安全迁移到本次数据集；系统只给出字段映射候选和规则适用性，不自动改写字段、规则或阈值。",
        "",
        f"- 数据文件: `{result.get('source_file')}`",
        f"- 数据口径提示: {result.get('dataset_kind_hint')}",
        f"- 数据字段数: {result.get('field_count')}",
        f"- 规则总数: {result.get('rule_count')}",
        f"- 可直接执行: {result.get('ready_count')}",
        f"- 已确认映射: {result.get('confirmed_mapping_count', 0)}",
        f"- 需映射确认: {result.get('needs_mapping_count')}",
        f"- 不适用: {result.get('not_applicable_count')}",
        f"- 需复核或停用: {int(result.get('needs_review_count') or 0) + int(result.get('disabled_count') or 0)}",
        "",
        "## 规则适用性",
    ]
    for item in result.get("rules") or []:
        status = str(item.get("status") or "unknown")
        mappings = []
        for mapping in item.get("field_mappings") or []:
            rule_field = mapping.get("rule_field")
            candidate = mapping.get("candidate_field")
            mapping_status = str(mapping.get("status") or "unknown")
            if candidate:
                score = mapping.get("score")
                score_text = f", score={score}" if score is not None else ""
                mappings.append(f"{rule_field} -> {candidate}({_field_mapping_status_label(mapping_status)}{score_text})")
            else:
                mappings.append(f"{rule_field}({_field_mapping_status_label(mapping_status)})")
        lines.append(
            f"- `{item.get('rule_id')}` {item.get('name')}: {_adaptation_status_label(status)}；"
            f"字段映射: {'; '.join(mappings) or '未识别字段'}"
        )
        issues = "；".join(item.get("issues") or [])
        if issues:
            lines.append(f"  问题: {issues}")
        if item.get("recommendation"):
            lines.append(f"  建议: {item.get('recommendation')}")
    if not result.get("rules"):
        lines.append("- 当前没有可审查的规则。")
    lines.append("")
    return "\n".join(lines)


def _render_schema_mapping_review_markdown(result: dict[str, Any]) -> str:
    """Render conservative mapping-candidate review suggestions."""
    lines = [
        "# 映射候选复核辅助清单",
        "",
        "本报告只辅助人工复核字段映射候选，不自动确认、不自动写入全局配置、不修改规则阈值。",
        "",
        f"- 数据文件: `{result.get('source_file')}`",
        f"- 复核状态: {_schema_mapping_review_status_label(str(result.get('status') or 'unknown'))}",
        f"- 可考虑确认: {result.get('confirm_suggestion_count', 0)}",
        f"- 需专家复核: {result.get('manual_review_count', 0)}",
        f"- 不适用: {result.get('not_applicable_count', 0)}",
        f"- 已安全处理: {result.get('already_safe_count', 0)}",
        f"- 原则: {result.get('anti_overfit_principle')}",
        "",
        "## 复核建议",
    ]
    items = result.get("review_items") or []
    if not items:
        lines.append("- 当前没有可复核的映射候选。")
    for item in items:
        flags = "；".join(str(flag) for flag in item.get("risk_flags") or []) or "无"
        lines.append(
            f"- `{item.get('rule_id')}` {item.get('rule_name')}: "
            f"`{item.get('rule_field')}` -> `{item.get('dataset_field') or '无'}`；"
            f"{_schema_mapping_review_status_label(str(item.get('status')))}；"
            f"score={item.get('score')}；方法={item.get('match_method') or '无'}。"
        )
        lines.append(f"  建议: {item.get('recommendation')}")
        lines.append(f"  风险: {flags}")
    lines.extend(
        [
            "",
            "## 使用边界",
            "- 可考虑确认表示候选具备较清晰的语义证据，但仍需要人工在 Schema 映射页确认。",
            "- 后缀口径不一致、单位不一致、token_overlap 或歧义候选不能自动确认。",
            "- 不适用表示当前数据中没有可靠字段，不能为了让规则通过而补造映射。",
            "",
        ]
    )
    return "\n".join(lines)


def _render_mapping_execution_plan_markdown(result: dict[str, Any]) -> str:
    """Render confirmed mapping execution plan without executing rewrites."""
    lines = [
        "# 映射执行计划",
        "",
        "本报告把已确认字段映射转成可审查的规则字段重写预案。它不会自动修改规则库、阈值或诊断结论；映射后规则必须先完成回归验证。",
        "",
        f"- 计划状态: {_mapping_execution_status_label(str(result.get('status') or 'unknown'))}",
        f"- 是否允许立即自动执行: {'是' if result.get('auto_execution_allowed') else '否'}",
        f"- 无需重写规则: {result.get('executable_without_mapping_count', 0)}",
        f"- 可进入回归验证: {result.get('ready_for_regression_count', 0)}",
        f"- 需映射确认: {result.get('needs_confirmation_count', 0)}",
        f"- 需复核: {result.get('needs_review_count', 0)}",
        f"- 阻断: {result.get('blocked_count', 0)}",
        "",
        "## 逐规则计划",
    ]
    for rule in result.get("rules") or []:
        lines.append(
            f"- `{rule.get('rule_id')}` {rule.get('name')}: {_mapping_execution_status_label(str(rule.get('status')))}；"
            f"立即执行: {'是' if rule.get('execution_allowed_now') else '否'}；"
            f"需回归验证: {'是' if rule.get('requires_regression_validation') else '否'}。"
        )
        if rule.get("rewrite_pairs"):
            pairs = "; ".join(f"{item.get('rule_field')} -> {item.get('dataset_field')}" for item in rule.get("rewrite_pairs") or [])
            lines.append(f"  字段重写预案: {pairs}")
        if rule.get("condition") != rule.get("rewritten_condition_preview"):
            lines.append(f"  条件预览: `{rule.get('condition')}` -> `{rule.get('rewritten_condition_preview')}`")
        issues = "；".join(rule.get("issues") or [])
        if issues:
            lines.append(f"  风险: {issues}")
        if rule.get("recommendation"):
            lines.append(f"  建议: {rule.get('recommendation')}")
    if not result.get("rules"):
        lines.append("- 当前没有可生成计划的规则。")

    lines.extend(["", "## 字段映射风险"])
    for item in result.get("field_plans") or []:
        lines.append(
            f"- `{item.get('rule_id')}` {item.get('rule_field')} -> {item.get('dataset_field') or '无'}；"
            f"状态: {_mapping_execution_status_label(str(item.get('status')))}；"
            f"单位: {item.get('unit_status')}；统计口径: {item.get('statistic_status')}；"
            f"动作: {item.get('rewrite_action')}。"
        )
    if not result.get("field_plans"):
        lines.append("- 当前没有字段级计划。")
    lines.extend(
        [
            "",
            "## 使用边界",
            "- 本计划不证明规则阈值在新客户数据上业务正确。",
            "- 统计口径不同（例如 avg 与 raw/last）必须经过复核或转换配置。",
            "- 已确认映射进入自动执行前，需要用多数据集回归验证。",
            "",
        ]
    )
    return "\n".join(lines)


def _render_mapping_regression_validation_markdown(result: dict[str, Any]) -> str:
    """Render mapping regression validation evidence."""
    lines = [
        "# 映射回归验证",
        "",
        "本报告验证已确认字段映射生成的规则重写预案是否能透明执行。它不修改规则库、不改阈值、不写全局配置，也不把单个数据集的命中率当作业务正确性证明。",
        "",
        f"- 验证状态: {_mapping_regression_status_label(str(result.get('status') or 'unknown'))}",
        f"- 映射执行是否已验证: {'是' if result.get('mapped_execution_validated') else '否'}",
        f"- 规则数量: {result.get('rule_count', 0)}",
        f"- 映射重写通过: {result.get('regression_passed_count', 0)}",
        f"- 无需映射: {result.get('no_mapping_count', 0)}",
        f"- 需确认: {result.get('needs_confirmation_count', 0)}",
        f"- 需复核: {result.get('needs_review_count', 0)}",
        f"- 阻断/失败: {int(result.get('blocked_count') or 0) + int(result.get('failed_count') or 0)}",
        f"- 反过拟合原则: {result.get('anti_overfit_principle')}",
        "",
        "## 逐规则验证",
    ]
    for rule in result.get("rules") or []:
        lines.append(
            f"- `{rule.get('rule_id')}` {rule.get('name')}: {_mapping_regression_status_label(str(rule.get('status')))}；"
            f"命中 {rule.get('matched_rows', 0)} 行，命中率 {float(rule.get('match_rate') or 0):.1%}。"
        )
        if rule.get("rewrite_pairs"):
            pairs = "; ".join(f"{item.get('rule_field')} -> {item.get('dataset_field')}" for item in rule.get("rewrite_pairs") or [])
            lines.append(f"  字段重写: {pairs}")
        if rule.get("condition") != rule.get("validated_condition"):
            lines.append(f"  条件验证: `{rule.get('condition')}` -> `{rule.get('validated_condition')}`")
        issues = "；".join(rule.get("issues") or [])
        if issues:
            lines.append(f"  风险: {issues}")
        if rule.get("recommendation"):
            lines.append(f"  建议: {rule.get('recommendation')}")
    if not result.get("rules"):
        lines.append("- 当前没有可验证的映射执行计划。")
    lines.extend(
        [
            "",
            "## 使用边界",
            "- 通过验证只表示字段重写预案可执行，不表示阈值、单位换算或业务诊断已经专家确认。",
            "- 当前数据命中率只作为回归证据，不应为了命中率好看而修改规则。",
            "- 真正进入全局配置前，应结合批量映射回归验证和人工复核意见沉淀。",
            "",
        ]
    )
    return "\n".join(lines)


def _render_generalization_readiness_markdown(result: dict[str, Any]) -> str:
    """Render the anti-overfitting generalization gate result."""
    metrics = result.get("metrics") or {}
    lines = [
        "# 泛化验收门禁",
        "",
        "本报告用于判断当前运行是否存在围绕单一数据集过拟合的风险。门禁目标不是让当前数据集结果更好看，而是确认系统能诚实区分可直接执行、已确认映射、需映射确认和不适用规则。",
        "",
        f"- 门禁状态: {_generalization_status_label(str(result.get('status') or 'unknown'))}",
        f"- 是否允许自动诊断解释: {'是' if result.get('auto_diagnosis_allowed') else '否'}",
        f"- 反过拟合原则: {result.get('anti_overfit_principle')}",
        f"- 结论: {result.get('verdict')}",
        "",
        "## 核心指标",
        f"- 字段语义覆盖率: {float(metrics.get('semantic_coverage') or 0):.1%} ({metrics.get('semantic_field_count', 0)}/{metrics.get('field_count', 0)})",
        f"- 规则适配: 可直接执行 {metrics.get('ready_count', 0)}，已确认映射 {metrics.get('confirmed_mapping_count', 0)}，需映射确认 {metrics.get('needs_mapping_count', 0)}，不适用 {metrics.get('not_applicable_count', 0)}。",
        f"- 未解决规则比例: {float(metrics.get('unresolved_rule_ratio') or 0):.1%}",
        f"- 规则执行: 成功执行 {metrics.get('evaluated_rule_count', 0)}，不可解析 {metrics.get('unparseable_rule_count', 0)}。",
        "",
        "## 门禁检查",
    ]
    for item in result.get("checks") or []:
        lines.append(
            f"- {_generalization_status_label(str(item.get('status')))}｜{item.get('name')}: {item.get('message')} 证据: `{item.get('evidence')}`"
        )
    lines.extend(["", "## 建议动作"])
    recommendations = result.get("recommendations") or []
    if recommendations:
        for item in recommendations:
            lines.append(f"- {item}")
    else:
        lines.append("- 当前没有额外建议。")
    lines.extend(
        [
            "",
            "## 使用边界",
            "- 本门禁不自动修改规则库、阈值、字段映射或原始数据。",
            "- 已确认映射只说明字段关系被沉淀，不代表单位换算、统计窗口和阈值已自动通过。",
            "- 如果新客户数据结构明显变化，应把本门禁作为解释规则命中前的前置审查。",
            "",
        ]
    )
    return "\n".join(lines)


def _result_summary(name: str, result: Any) -> str:
    if isinstance(result, dict) and result.get("summary"):
        return str(result["summary"])
    return f"{name} 已完成：{truncate_text(str(result), 300)}"


def _latest_tool_result(tool_results: list[dict[str, Any]], name: str) -> Any:
    expected = _canonical_tool_name(name)
    for item in reversed(tool_results):
        if _canonical_tool_name(str(item.get("tool"))) == expected:
            return item.get("result")
    return None


def _canonical_tool_name(name: str) -> str:
    return TOOL_NAME_ALIASES.get(name, name)


def _verification_failure_summary(result: dict[str, Any]) -> str:
    first_node = result.get("first_mismatch_node_id") or "unknown"
    first_index = result.get("first_mismatch_index")
    errors = result.get("errors") or []
    first_error = errors[0] if errors else {}
    kind = first_error.get("kind", "unknown")
    return (
        f"first_mismatch_index={first_index}, "
        f"first_mismatch_node_id={first_node}, "
        f"error={kind}"
    )


def _column_profile_index(profile: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(item.get("name")): item
        for item in profile.get("column_profiles", [])
        if isinstance(item, dict) and item.get("name") is not None
    }


def _apply_confirmed_schema_override(
    semantic: dict[str, Any],
    override: dict[str, Any] | None,
    *,
    override_source: str = "",
) -> dict[str, Any]:
    """Merge a human-confirmed semantic mapping into one field semantic record."""
    if not isinstance(override, dict):
        return semantic
    updated = dict(semantic)
    for key in ["semantic_role", "display_name", "group", "role", "unit", "statistic", "source_hint"]:
        value = override.get(key)
        if value is not None and value != "":
            updated[key] = value
    try:
        override_confidence = float(override.get("confidence") or 0.99)
    except (TypeError, ValueError):
        override_confidence = 0.99
    updated["confidence"] = round(max(float(updated.get("confidence") or 0.0), min(override_confidence, 1.0)), 3)
    updated["match_method"] = "confirmed_override"
    updated["confirmation_status"] = "confirmed"
    updated["requires_confirmation"] = False
    if override_source:
        updated["confirmation_source"] = override_source
    for key in ["confirmed_by", "confirmed_at", "note"]:
        if override.get(key) is not None and override.get(key) != "":
            output_key = "confirmation_note" if key == "note" else key
            updated[output_key] = override.get(key)
    note = str(override.get("note") or "").strip()
    updated["reason"] = "来自人工确认映射配置。" + (f" {note}" if note else "")
    return updated


def _semantic_rule_mapping_candidates(
    rule_field: str,
    rule_semantic: dict[str, Any],
    dataset_semantics: list[dict[str, Any]],
    known_field_set: set[str],
    confirmed_rule_mapping: dict[str, Any] | None = None,
    max_candidates: int = 5,
) -> list[dict[str, Any]]:
    if rule_field in known_field_set:
        return [
            {
                "field": rule_field,
                "status": "exact",
                "score": 1.0,
                "match_method": "exact",
                "semantic_role": rule_semantic.get("semantic_role", ""),
                "unit": rule_semantic.get("unit", "unknown"),
                "statistic": rule_semantic.get("statistic", "raw"),
                "requires_confirmation": False,
                "reason": "字段名精确匹配。",
            }
        ]

    candidates: list[dict[str, Any]] = []
    rule_role = str(rule_semantic.get("semantic_role") or "")
    rule_tokens = _field_semantic_tokens(rule_field)
    rule_unit = str(rule_semantic.get("unit") or "unknown")
    rule_statistic = str(rule_semantic.get("statistic") or "raw")
    dataset_semantic_index = {str(item.get("field") or ""): item for item in dataset_semantics if item.get("field")}

    if isinstance(confirmed_rule_mapping, dict):
        confirmed_field = str(confirmed_rule_mapping.get("dataset_field") or "").strip()
        if confirmed_field in known_field_set:
            field_semantic = dataset_semantic_index.get(confirmed_field, {})
            candidates.append(
                {
                    "field": confirmed_field,
                    "status": "confirmed_mapping",
                    "score": 1.0,
                    "match_method": "confirmed_rule_mapping",
                    "semantic_role": confirmed_rule_mapping.get("semantic_role") or field_semantic.get("semantic_role") or rule_role,
                    "display_name": confirmed_rule_mapping.get("display_name") or field_semantic.get("display_name", ""),
                    "unit": confirmed_rule_mapping.get("unit") or field_semantic.get("unit", "unknown"),
                    "statistic": confirmed_rule_mapping.get("statistic") or field_semantic.get("statistic", "raw"),
                    "requires_confirmation": False,
                    "confirmed_by": confirmed_rule_mapping.get("confirmed_by", ""),
                    "confirmed_at": confirmed_rule_mapping.get("confirmed_at", ""),
                    "confirmation_note": confirmed_rule_mapping.get("note", ""),
                    "reason": "来自人工确认的规则字段映射配置。",
                }
            )

    for item in dataset_semantics:
        field = str(item.get("field") or "")
        if not field:
            continue
        if any(candidate.get("field") == field for candidate in candidates):
            continue
        score = 0.0
        method = ""
        reason = ""
        if rule_role and item.get("semantic_role") == rule_role:
            score = 0.86
            method = "semantic_role"
            reason = "规则字段和数据字段匹配到同一稳定语义角色。"
            confirmed_field_semantic = item.get("confirmation_status") == "confirmed"
            candidate_stat = str(item.get("statistic") or "raw")
            if candidate_stat == rule_statistic:
                score += 0.06
            elif {candidate_stat, rule_statistic} <= {"avg", "last"}:
                score -= 0.08
                reason += " 但统计口径不同，需要确认。"
            candidate_unit = str(item.get("unit") or "unknown")
            if rule_unit != "unknown" and candidate_unit != "unknown":
                if candidate_unit == rule_unit:
                    score += 0.04
                else:
                    score -= 0.18
                    reason += " 单位线索不一致，需要确认。"
            if confirmed_field_semantic:
                score = max(score, 0.99)
                method = "confirmed_field_semantic"
                reason = "数据字段语义已由人工确认配置固化。"
        else:
            candidate_tokens = set(str(token) for token in item.get("tokens") or [])
            common = rule_tokens & candidate_tokens
            if common:
                precision = len(common) / max(len(candidate_tokens), 1)
                recall = len(common) / max(len(rule_tokens), 1)
                score = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
                method = "token_overlap"
                reason = "字段核心词相近，但未匹配到同一稳定语义角色。"
        if score < 0.6:
            continue
        is_confirmed = item.get("confirmation_status") == "confirmed" and method == "confirmed_field_semantic"
        candidates.append(
            {
                "field": field,
                "status": "confirmed_mapping" if is_confirmed else "semantic_candidate",
                "score": round(max(0.0, min(score, 0.99)), 3),
                "match_method": method,
                "semantic_role": item.get("semantic_role", ""),
                "display_name": item.get("display_name", ""),
                "unit": item.get("unit", "unknown"),
                "statistic": item.get("statistic", "raw"),
                "requires_confirmation": not is_confirmed,
                "confirmed_by": item.get("confirmed_by", ""),
                "confirmed_at": item.get("confirmed_at", ""),
                "confirmation_note": item.get("confirmation_note", ""),
                "reason": reason,
            }
        )
    candidates.sort(key=lambda item: (float(item.get("score") or 0), str(item.get("match_method")), str(item.get("field"))), reverse=True)
    return candidates[:max_candidates]


def _schema_mapping_index(schema_mapping: dict[str, Any]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for item in schema_mapping.get("rule_fields") or []:
        if isinstance(item, dict) and item.get("rule_field"):
            index[str(item["rule_field"])] = item
    return index


def _filter_rejected_rule_mapping_candidates(
    rule_field: str,
    candidates: list[dict[str, Any]],
    rejected_keys: set[tuple[str, str]],
) -> list[dict[str, Any]]:
    if not rejected_keys:
        return candidates
    normalized_rule_field = _normalize_mapping_key(rule_field)
    return [
        candidate
        for candidate in candidates
        if (normalized_rule_field, str(candidate.get("field") or "")) not in rejected_keys
    ]


def _normalize_mapping_key(value: str) -> str:
    normalized = str(value).lower()
    normalized = re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "_", normalized)
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized


def build_transactions(df: pd.DataFrame, columns: list[str] | None = None, max_rows: int = 5000) -> list[set[str]]:
    """Convert rows into item sets such as ``field=value`` for rule mining."""
    selected = [c for c in (columns or list(df.columns)) if c in df.columns]
    if not selected:
        selected = list(df.columns)
    selected = [c for c in selected if not _is_metadata_or_target_column(str(c))]
    if not selected:
        selected = [c for c in df.columns if not _is_metadata_or_target_column(str(c))]
    # 限制最多 30 列，避免宽表在关联规则挖掘阶段组合爆炸。
    selected = selected[:30]
    work = df[selected].head(max_rows).copy()
    selected = [col for col in selected if work[col].nunique(dropna=True) > 1]
    work = work[selected]
    for col in selected:
        if pd.api.types.is_numeric_dtype(work[col]) and work[col].nunique(dropna=True) > 6:
            # 连续数值字段先分箱，否则每个唯一数值都会变成低支持度 item。
            try:
                work[col] = pd.qcut(work[col], q=4, duplicates="drop").astype(str)
            except Exception:
                work[col] = work[col].astype(str)
        else:
            work[col] = work[col].astype(str)
    transactions: list[set[str]] = []
    for _, row in work.iterrows():
        items = set()
        for col in selected:
            value = row[col]
            if value and value.lower() not in {"nan", "none", "nat"}:
                # 每个单元格变成一个 item，如 status=WARN。
                items.add(f"{col}={truncate_text(str(value), 80)}")
        transactions.append(items)
    return transactions


def _is_metadata_or_target_column(name: str) -> bool:
    lower = name.lower()
    if lower in {"label", "diagnosis"}:
        return True
    return lower.startswith("00000_") or "time" in lower or "span" in lower


def _dataset_metadata(original_name: str, columns: list[str]) -> dict[str, Any]:
    """Infer lightweight dataset-kind hints from file name and column suffixes."""
    suffix_counts = {
        "_avg": sum(1 for column in columns if column.endswith("_avg")),
        "_last": sum(1 for column in columns if column.endswith("_last")),
    }
    name_lower = original_name.lower()
    if "last_avg" in name_lower or suffix_counts["_avg"] > suffix_counts["_last"]:
        hint = "last_avg"
    elif "last_last" in name_lower or suffix_counts["_last"] > suffix_counts["_avg"]:
        hint = "last_last"
    else:
        hint = "unknown"
    return {
        "source_file": original_name,
        "dataset_kind_hint": hint,
        "column_suffix_counts": suffix_counts,
    }


def _alternate_suffix_field(field: str, known_fields: set[str]) -> str | None:
    """Find an existing _avg/_last counterpart for a missing rule field."""
    if field.endswith("_avg"):
        candidate = f"{field[:-4]}_last"
        return candidate if candidate in known_fields else None
    if field.endswith("_last"):
        candidate = f"{field[:-5]}_avg"
        return candidate if candidate in known_fields else None
    return None


def _rule_field_candidates(rule: RuleCard) -> list[str]:
    """Collect explicit and text-mentioned telemetry fields from a rule."""
    candidates = list(rule.target_fields)
    for text in [rule.condition, rule.description]:
        if not text:
            continue
        candidates.extend(re.findall(r"\b[A-Za-z0-9]+(?:_[A-Za-z0-9]+)+\b", text))
    # 保序去重，避免同一字段重复报错。
    return list(dict.fromkeys(candidates))


def _field_mapping_review(
    field: str,
    known_field_set: set[str],
    known_fields: list[str],
    schema_fields: dict[str, Any],
    semantic_mapping_index: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Classify one rule field against the current dataset fields."""
    if field in known_field_set:
        return {
            "rule_field": field,
            "status": "exact",
            "dataset_field": field,
            "candidate_field": field,
            "score": 1.0,
            "display_name": _schema_display_name(schema_fields, field),
            "requires_confirmation": False,
            "reason": "字段名精确匹配。",
        }

    alternate = _alternate_suffix_field(field, known_field_set)
    if alternate:
        return {
            "rule_field": field,
            "status": "suffix_mismatch",
            "dataset_field": None,
            "candidate_field": alternate,
            "score": 0.85,
            "display_name": _schema_display_name(schema_fields, alternate),
            "requires_confirmation": True,
            "reason": "规则字段与数据字段仅 _avg/_last 后缀不同，可能代表统计口径不同。",
        }

    semantic_mapping = (semantic_mapping_index or {}).get(field)
    if semantic_mapping:
        candidates_from_mapping = semantic_mapping.get("candidates") or []
        selected_field = semantic_mapping.get("selected_field")
        if selected_field and candidates_from_mapping:
            best = candidates_from_mapping[0]
            mapping_status = str(best.get("status") or semantic_mapping.get("status") or "semantic_candidate")
            is_confirmed = mapping_status == "confirmed_mapping"
            return {
                "rule_field": field,
                "status": mapping_status,
                "dataset_field": selected_field if is_confirmed else None,
                "candidate_field": selected_field,
                "score": best.get("score", semantic_mapping.get("confidence", 0.0)),
                "display_name": best.get("display_name") or _schema_display_name(schema_fields, str(selected_field)),
                "requires_confirmation": not is_confirmed,
                "reason": best.get("reason") or ("字段映射已由确认配置支持。" if is_confirmed else "字段匹配到同一语义角色，但仍需确认单位和统计口径。"),
                "semantic_role": best.get("semantic_role") or semantic_mapping.get("semantic_role", ""),
                "match_method": best.get("match_method", "semantic_role"),
                "unit": semantic_mapping.get("unit", "unknown"),
                "statistic": semantic_mapping.get("statistic", "raw"),
                "candidate_unit": best.get("unit", "unknown"),
                "candidate_statistic": best.get("statistic", "raw"),
                "confirmed_by": best.get("confirmed_by", ""),
                "confirmed_at": best.get("confirmed_at", ""),
                "confirmation_note": best.get("confirmation_note", ""),
                "ambiguous": bool(semantic_mapping.get("ambiguous")),
                "candidates": candidates_from_mapping[:3],
            }

    candidates = _semantic_field_candidates(field, known_fields, schema_fields)
    if candidates:
        best = candidates[0]
        return {
            "rule_field": field,
            "status": "semantic_candidate",
            "dataset_field": None,
            "candidate_field": best["field"],
            "score": best["score"],
            "display_name": best.get("display_name", ""),
            "requires_confirmation": True,
            "reason": "字段名核心词相近，但不是精确匹配，需要人工确认语义和单位。",
            "candidates": candidates,
        }

    return {
        "rule_field": field,
        "status": "missing",
        "dataset_field": None,
        "candidate_field": None,
        "score": 0.0,
        "display_name": "",
        "requires_confirmation": True,
        "reason": "当前数据集中未找到可解释的字段映射候选。",
    }


def _schema_mapping_candidate_review_item(
    rule: dict[str, Any],
    mapping: dict[str, Any],
    schema_item: dict[str, Any],
    *,
    min_confirm_score: float,
) -> dict[str, Any]:
    """Classify one mapping candidate for human review without applying it."""
    source_status = str(mapping.get("status") or "unknown")
    rule_field = str(mapping.get("rule_field") or "")
    dataset_field = str(mapping.get("candidate_field") or mapping.get("dataset_field") or "")
    score = float(mapping.get("score") or 0.0)
    match_method = str(mapping.get("match_method") or "")
    rule_unit = str(mapping.get("unit") or "unknown")
    candidate_unit = str(mapping.get("candidate_unit") or mapping.get("unit") or "unknown")
    rule_statistic = str(mapping.get("statistic") or "raw")
    candidate_statistic = str(mapping.get("candidate_statistic") or mapping.get("statistic") or "raw")
    ambiguous = bool(mapping.get("ambiguous") or schema_item.get("ambiguous"))
    risk_flags: list[str] = []

    if source_status in {"exact", "confirmed_mapping"}:
        status = "already_safe"
        action = "none"
        recommendation = "字段已精确匹配或已由人工确认，无需在本清单中重复处理。"
    elif source_status == "missing" or not dataset_field:
        status = "not_applicable"
        action = "keep_not_applicable"
        risk_flags.append("当前数据没有可靠候选字段")
        recommendation = "保持不适用，不要为了让规则执行而补造字段映射。"
    elif source_status == "suffix_mismatch":
        status = "manual_review_required"
        action = "expert_review"
        risk_flags.append("规则字段和候选字段存在统计口径后缀差异")
        recommendation = "先确认 _avg/_last 等统计口径是否业务等价；未确认前不要点击确认。"
    else:
        if ambiguous:
            risk_flags.append("候选字段存在歧义")
        if match_method not in {"semantic_role", "confirmed_field_semantic"}:
            risk_flags.append("候选仅来自字段名相似或 token_overlap")
        if score < min_confirm_score:
            risk_flags.append(f"候选分数低于阈值 {min_confirm_score:g}")
        if _known_and_different(rule_unit, candidate_unit):
            risk_flags.append(f"单位线索不一致: {rule_unit} vs {candidate_unit}")
        if _known_and_different(rule_statistic, candidate_statistic):
            risk_flags.append(f"统计口径不一致: {rule_statistic} vs {candidate_statistic}")

        if risk_flags:
            status = "manual_review_required"
            action = "expert_review"
            recommendation = "先复核字段语义、单位、统计口径和业务阶段；不能自动确认该候选映射。"
        else:
            status = "suggest_confirm_candidate"
            action = "consider_confirm"
            recommendation = "语义角色、分数和口径线索较稳定，可进入人工确认候选；确认后仍需映射执行计划和回归验证。"

    return {
        "rule_id": rule.get("rule_id"),
        "rule_name": rule.get("name"),
        "rule_field": rule_field,
        "dataset_field": dataset_field or None,
        "source_status": source_status,
        "status": status,
        "status_label": _schema_mapping_review_status_label(status),
        "recommended_action": action,
        "score": round(score, 3),
        "match_method": match_method,
        "semantic_role": mapping.get("semantic_role") or schema_item.get("semantic_role") or "",
        "rule_unit": rule_unit,
        "candidate_unit": candidate_unit,
        "rule_statistic": rule_statistic,
        "candidate_statistic": candidate_statistic,
        "ambiguous": ambiguous,
        "risk_flags": risk_flags,
        "recommendation": recommendation,
        "reason": mapping.get("reason") or "",
        "source_ref": rule.get("source_ref"),
    }


def _known_and_different(left: str, right: str) -> bool:
    left_value = str(left or "unknown")
    right_value = str(right or "unknown")
    if left_value in {"", "unknown"} or right_value in {"", "unknown"}:
        return False
    return left_value != right_value


def _semantic_field_candidates(
    field: str,
    known_fields: list[str],
    schema_fields: dict[str, Any],
    max_candidates: int = 3,
) -> list[dict[str, Any]]:
    """Return conservative semantic candidates from field-name token overlap."""
    target_tokens = _field_semantic_tokens(field)
    if not target_tokens:
        return []
    candidates: list[dict[str, Any]] = []
    for candidate in known_fields:
        candidate_tokens = _field_semantic_tokens(candidate)
        if not candidate_tokens:
            continue
        common = target_tokens & candidate_tokens
        if not common:
            continue
        precision = len(common) / len(candidate_tokens)
        recall = len(common) / len(target_tokens)
        score = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
        if score < 0.6:
            continue
        candidates.append(
            {
                "field": candidate,
                "score": round(score, 3),
                "matched_tokens": sorted(common),
                "display_name": _schema_display_name(schema_fields, candidate),
            }
        )
    candidates.sort(key=lambda item: (item["score"], len(item["matched_tokens"]), item["field"]), reverse=True)
    return candidates[:max_candidates]


def _field_semantic_tokens(field: str) -> set[str]:
    """Normalize field names for cross-dataset candidate matching."""
    normalized = str(field).lower()
    normalized = re.sub(r"^\d+[_-]+", "", normalized)
    normalized = re.sub(r"(?:[_-](?:avg|last|mean|median|min|max|std|sum|count))+$", "", normalized)
    raw_tokens = re.split(r"[^0-9a-z\u4e00-\u9fff]+", normalized)
    generic = {
        "avg",
        "last",
        "mean",
        "median",
        "min",
        "max",
        "std",
        "sum",
        "count",
        "value",
        "values",
        "field",
        "fields",
        "data",
        "sensor",
        "metric",
    }
    return {token for token in raw_tokens if token and token not in generic and not token.isdigit()}


def _schema_display_name(schema_fields: dict[str, Any], field: str) -> str:
    info = schema_fields.get(field) if isinstance(schema_fields, dict) else None
    if isinstance(info, dict):
        return str(info.get("display_name") or "")
    return ""


def _mapping_execution_field_plan(
    mapping: dict[str, Any],
    *,
    allow_statistic_mismatch: bool = False,
    allow_unknown_unit: bool = False,
) -> dict[str, Any]:
    """Classify one field mapping for planned execution."""
    status = str(mapping.get("status") or "unknown")
    rule_field = str(mapping.get("rule_field") or "")
    dataset_field = str(mapping.get("candidate_field") or mapping.get("dataset_field") or "")
    issues: list[str] = []
    rewrite_action = "none"
    requires_regression = False
    rule_unit = str(mapping.get("unit") or "unknown")
    candidate_unit = str(mapping.get("candidate_unit") or mapping.get("unit") or "unknown")
    rule_statistic = str(mapping.get("statistic") or "raw")
    candidate_statistic = str(mapping.get("candidate_statistic") or mapping.get("statistic") or "raw")

    if status == "exact":
        return {
            "rule_field": rule_field,
            "dataset_field": dataset_field or rule_field,
            "source_status": status,
            "status": "executable_without_mapping",
            "rewrite_action": "no_rewrite",
            "unit_status": "not_required",
            "statistic_status": "not_required",
            "requires_regression_validation": False,
            "issues": [],
            "reason": mapping.get("reason") or "字段精确匹配。",
        }
    if status in {"semantic_candidate", "suffix_mismatch"}:
        return {
            "rule_field": rule_field,
            "dataset_field": dataset_field,
            "source_status": status,
            "status": "needs_confirmation",
            "rewrite_action": "wait_for_confirmation",
            "unit_status": "unchecked",
            "statistic_status": "unchecked",
            "requires_regression_validation": False,
            "issues": ["字段映射仍是候选，不能进入执行计划。"],
            "reason": mapping.get("reason") or "候选映射需要人工确认。",
        }
    if status == "missing" or not dataset_field:
        return {
            "rule_field": rule_field,
            "dataset_field": dataset_field or None,
            "source_status": status,
            "status": "blocked",
            "rewrite_action": "blocked",
            "unit_status": "unavailable",
            "statistic_status": "unavailable",
            "requires_regression_validation": False,
            "issues": ["缺少可映射的数据字段。"],
            "reason": mapping.get("reason") or "字段缺失，不能执行。",
        }

    rewrite_action = "rewrite_field"
    requires_regression = True
    unit_status = "compatible"
    statistic_status = "compatible"
    if rule_unit == "unknown" or candidate_unit == "unknown":
        unit_status = "unknown_allowed" if allow_unknown_unit else "unknown"
        if not allow_unknown_unit:
            issues.append("单位线索未知，需复核后再执行。")
    elif rule_unit != candidate_unit:
        unit_status = "mismatch"
        issues.append(f"单位不一致：规则字段 {rule_unit}，数据字段 {candidate_unit}。")

    if rule_statistic == "unknown" or candidate_statistic == "unknown":
        statistic_status = "unknown"
        issues.append("统计口径未知，需复核后再执行。")
    elif rule_statistic != candidate_statistic:
        statistic_status = "mismatch_allowed" if allow_statistic_mismatch else "mismatch"
        if not allow_statistic_mismatch:
            issues.append(f"统计口径不一致：规则字段 {rule_statistic}，数据字段 {candidate_statistic}。")

    if mapping.get("ambiguous"):
        issues.append("存在多个相近候选，需复核歧义。")
    mapped_status = "ready_for_regression" if not issues else "needs_review"
    return {
        "rule_field": rule_field,
        "dataset_field": dataset_field,
        "source_status": status,
        "status": mapped_status,
        "rewrite_action": rewrite_action,
        "unit": rule_unit,
        "candidate_unit": candidate_unit,
        "unit_status": unit_status,
        "statistic": rule_statistic,
        "candidate_statistic": candidate_statistic,
        "statistic_status": statistic_status,
        "requires_regression_validation": requires_regression,
        "issues": issues,
        "semantic_role": mapping.get("semantic_role", ""),
        "match_method": mapping.get("match_method", ""),
        "reason": mapping.get("reason") or "字段映射已确认，可生成重写预案但仍需验证。",
    }


def _validate_mapping_regression_rule(
    df: pd.DataFrame,
    rule: dict[str, Any],
    *,
    require_strict_unit: bool,
    require_strict_statistic: bool,
) -> tuple[str, dict[str, Any]]:
    """Validate one mapping execution plan rule without mutating the rule."""
    source_status = str(rule.get("status") or "unknown")
    condition = str(rule.get("condition") or "")
    validated_condition = str(rule.get("rewritten_condition_preview") or condition)
    rewrite_pairs = rule.get("rewrite_pairs") or []
    issues = [str(issue) for issue in rule.get("issues") or [] if str(issue)]
    field_plan_issues: list[str] = []

    for field_plan in rule.get("field_plans") or []:
        unit_status = str(field_plan.get("unit_status") or "unknown")
        statistic_status = str(field_plan.get("statistic_status") or "unknown")
        if require_strict_unit and unit_status not in {"compatible"}:
            field_plan_issues.append(
                f"{field_plan.get('rule_field')} -> {field_plan.get('dataset_field')} 单位状态为 {unit_status}。"
            )
        if require_strict_statistic and statistic_status not in {"compatible"}:
            field_plan_issues.append(
                f"{field_plan.get('rule_field')} -> {field_plan.get('dataset_field')} 统计口径状态为 {statistic_status}。"
            )
    issues.extend(field_plan_issues)

    if source_status == "executable_without_mapping":
        mask, error = _evaluate_condition_expression(df, condition)
        if error:
            status = "failed"
            issues.append(f"原始规则条件无法执行：{error}")
            recommendation = "规则本应无需映射但执行失败，请复核字段、条件表达式和数据类型。"
        else:
            status = "passed_no_mapping"
            recommendation = "规则无需字段映射，已验证原始条件可执行。"
        return status, _mapping_regression_rule_record(
            rule,
            status=status,
            condition=condition,
            validated_condition=condition,
            rewrite_pairs=[],
            mask=None if error else mask,
            issues=issues,
            recommendation=recommendation,
        )

    if source_status == "ready_for_regression":
        if not rewrite_pairs:
            issues.append("规则标记为待回归验证，但没有字段重写对。")
        if condition == validated_condition:
            issues.append("重写条件与原始条件相同，需确认映射预案是否生效。")
        mask, error = _evaluate_condition_expression(df, validated_condition)
        if error:
            status = "failed"
            issues.append(f"重写后规则条件无法执行：{error}")
            recommendation = "字段映射预案未通过技术执行验证，不能进入自动执行。"
        elif issues:
            status = "needs_review"
            recommendation = "重写后规则可以执行，但单位、统计口径或映射预案仍需人工复核。"
        else:
            status = "regression_passed"
            recommendation = "重写后规则可以执行；下一步应在批量数据上复验，确认该映射不是单数据集偶然成立。"
        return status, _mapping_regression_rule_record(
            rule,
            status=status,
            condition=condition,
            validated_condition=validated_condition,
            rewrite_pairs=rewrite_pairs,
            mask=None if error else mask,
            issues=issues,
            recommendation=recommendation,
        )

    if source_status == "needs_mapping_confirmation":
        status = "needs_confirmation"
        issues.append("仍有字段候选映射未确认，不能做回归验证。")
        recommendation = "先在 Schema 语义映射页确认或驳回候选映射。"
    elif source_status == "needs_review":
        status = "needs_review"
        issues.append("映射执行计划已要求人工复核。")
        recommendation = "先复核单位、统计口径或歧义风险，再重新验证。"
    else:
        status = "blocked"
        issues.append("映射执行计划处于阻断状态或缺少必要字段。")
        recommendation = "先修复字段适配或规则结构，不能强行验证映射。"
    return status, _mapping_regression_rule_record(
        rule,
        status=status,
        condition=condition,
        validated_condition=validated_condition,
        rewrite_pairs=rewrite_pairs,
        mask=None,
        issues=issues,
        recommendation=recommendation,
    )


def _mapping_regression_rule_record(
    rule: dict[str, Any],
    *,
    status: str,
    condition: str,
    validated_condition: str,
    rewrite_pairs: list[dict[str, Any]],
    mask: pd.Series | None,
    issues: list[str],
    recommendation: str,
) -> dict[str, Any]:
    matched_rows = int(mask.sum()) if mask is not None else 0
    total_rows = int(len(mask)) if mask is not None else 0
    return {
        "rule_id": rule.get("rule_id"),
        "name": rule.get("name"),
        "source_status": rule.get("status"),
        "status": status,
        "condition": condition,
        "validated_condition": validated_condition,
        "rewrite_pairs": rewrite_pairs,
        "matched_rows": matched_rows,
        "match_rate": round(matched_rows / total_rows, 6) if total_rows else 0.0,
        "sample_indices": [int(i) for i in mask[mask].index.tolist()[:20]] if mask is not None else [],
        "issues": list(dict.fromkeys(issues)),
        "recommendation": recommendation,
    }


def _rewrite_condition_fields(condition: str, rewrite_pairs: list[dict[str, Any]]) -> str:
    """Preview field-name replacement without mutating the real rule library."""
    rewritten = str(condition or "")
    pairs = sorted(rewrite_pairs, key=lambda item: len(str(item.get("rule_field") or "")), reverse=True)
    for item in pairs:
        rule_field = str(item.get("rule_field") or "")
        dataset_field = str(item.get("dataset_field") or "")
        if not rule_field or not dataset_field or rule_field == dataset_field:
            continue
        rewritten = re.sub(rf"(?<![\w]){re.escape(rule_field)}(?![\w])", dataset_field, rewritten)
    return rewritten


def _generalization_check(
    check_id: str,
    name: str,
    passed: bool,
    failed_message: str,
    *,
    evidence: str = "",
    status_if_failed: str = "blocked",
) -> dict[str, Any]:
    status = "passed" if passed else status_if_failed
    return {
        "id": check_id,
        "name": name,
        "status": status,
        "message": "检查通过。" if passed else failed_message,
        "evidence": evidence,
    }


def _generalization_recommendations(
    *,
    confirmed_mapping_count: int,
    needs_mapping_count: int,
    not_applicable_count: int,
    semantic_coverage: float,
    min_semantic_coverage: float,
    unresolved_rule_ratio: float,
    max_unresolved_rule_ratio: float,
) -> list[str]:
    recommendations: list[str] = [
        "所有规则、字段和阈值调整都必须基于多数据集证据，不能只因为当前数据集结果不好看而修改。",
    ]
    if semantic_coverage < min_semantic_coverage:
        recommendations.append("扩展通用语义角色或字段识别逻辑，并用多个客户数据变体回归验证。")
    if confirmed_mapping_count:
        recommendations.append("为已确认映射生成映射执行计划，逐项复核单位、统计窗口和阈值适用性。")
    if needs_mapping_count:
        recommendations.append("把候选映射交给人工或第二分析 Agent 复核，确认后再写入 schema_mapping_overrides.yaml。")
    if not_applicable_count:
        recommendations.append("对不适用规则补充通用转换配置或建立新规则，不要强行套用当前规则阈值。")
    if unresolved_rule_ratio > max_unresolved_rule_ratio:
        recommendations.append("未解决规则比例较高，建议先做批量泛化验收，再考虑规则库 v3。")
    return list(dict.fromkeys(recommendations))


def _generalization_status_label(status: str) -> str:
    return {
        "passed": "通过",
        "needs_review": "需复核",
        "blocked": "阻断",
    }.get(status, status)


def _mapping_execution_status_label(status: str) -> str:
    return {
        "passed": "通过",
        "needs_review": "需复核",
        "blocked": "阻断",
        "executable_without_mapping": "无需映射可执行",
        "ready_for_regression": "待回归验证",
        "needs_mapping_confirmation": "需映射确认",
        "needs_confirmation": "需确认",
    }.get(status, status)


def _mapping_regression_status_label(status: str) -> str:
    return {
        "passed": "通过",
        "needs_review": "需复核",
        "failed": "失败",
        "blocked": "阻断",
        "passed_no_mapping": "无需映射已通过",
        "regression_passed": "映射回归通过",
        "needs_confirmation": "需映射确认",
    }.get(status, status)


def _adaptation_status_label(status: str) -> str:
    return {
        "ready": "可直接执行",
        "mapping_confirmed": "映射已确认",
        "needs_mapping_confirmation": "需映射确认",
        "needs_review": "需复核",
        "not_applicable": "不适用",
        "disabled": "已停用",
    }.get(status, status)


def _schema_mapping_review_status_label(status: str) -> str:
    return {
        "ready_for_human_review": "待人工复核",
        "needs_human_review": "需人工复核",
        "has_confirm_suggestions": "有确认建议",
        "already_resolved": "已处理",
        "already_safe": "已安全处理",
        "suggest_confirm_candidate": "可考虑确认",
        "manual_review_required": "需专家复核",
        "not_applicable": "不适用",
        "not_run": "未运行",
    }.get(status, status)


def _field_mapping_status_label(status: str) -> str:
    return {
        "exact": "精确匹配",
        "confirmed_mapping": "已确认映射",
        "suffix_mismatch": "后缀口径不一致",
        "semantic_candidate": "语义候选",
        "missing": "缺失",
    }.get(status, status)


def evaluate_rule_condition(df: pd.DataFrame, rule: RuleCard) -> tuple[pd.Series, str | None]:
    """Evaluate a RuleCard against a DataFrame and return the matched-row mask."""
    return _evaluate_condition_expression(df, rule.condition or rule.description)


def _evaluate_condition_expression(df: pd.DataFrame, condition: str) -> tuple[pd.Series, str | None]:
    """Evaluate a supported rule condition string against a DataFrame."""
    # MVP 支持 AND/&&/且/并且，不支持 OR。多个子句都成立才算命中。
    clauses = re.split(r"\s+(?:and|AND|&&|且|并且)\s+", condition)
    mask = pd.Series(True, index=df.index)
    parsed_any = False
    errors = []
    for clause in clauses:
        clause = clause.strip()
        if not clause:
            continue
        clause_mask, error = evaluate_clause(df, clause)
        if error:
            errors.append(error)
            continue
        parsed_any = True
        mask &= clause_mask
    if not parsed_any:
        return mask, "; ".join(errors) or "No supported condition found."
    return mask.fillna(False), None


def evaluate_clause(df: pd.DataFrame, clause: str) -> tuple[pd.Series, str | None]:
    """Evaluate one simple clause like ``temperature > 80`` or ``status == WARN``."""
    normalized = (
        clause.replace("大于等于", ">=")
        .replace("不小于", ">=")
        .replace("小于等于", "<=")
        .replace("不大于", "<=")
        .replace("大于", ">")
        .replace("小于", "<")
        .replace("等于", "==")
        .replace("不等于", "!=")
    )
    # Longest field names first prevents matching "temp" before "temperature".
    for col in sorted([str(c) for c in df.columns], key=len, reverse=True):
        pattern = re.compile(rf"{re.escape(col)}\s*(>=|<=|==|!=|>|<|contains|not_contains)\s*(.+)$", re.IGNORECASE)
        match = pattern.search(normalized)
        if not match:
            continue
        op = match.group(1)
        raw_value = match.group(2).strip().strip("\"'")
        series = df[col]
        if op in {"contains", "not_contains"}:
            # contains/not_contains 统一按字符串包含处理，适合文本或类别字段。
            result = series.astype(str).str.contains(raw_value, case=False, na=False)
            return ((~result) if op == "not_contains" else result), None
        value, value_error = coerce_rule_value(raw_value, series)
        if value_error:
            return pd.Series(False, index=df.index), value_error
        try:
            # pandas 会返回布尔 Series；上层再与其他子句做 AND。
            if op == ">=":
                return series >= value, None
            if op == "<=":
                return series <= value, None
            if op == ">":
                return series > value, None
            if op == "<":
                return series < value, None
            if op == "==":
                return series.astype(str) == str(value) if isinstance(value, str) else series == value, None
            if op == "!=":
                return series.astype(str) != str(value) if isinstance(value, str) else series != value, None
        except TypeError as exc:
            return pd.Series(False, index=df.index), f"Invalid comparison for {col}: {exc}"
    return pd.Series(False, index=df.index), f"Could not parse clause: {clause}"


def coerce_rule_value(raw_value: str, series: pd.Series) -> tuple[Any, str | None]:
    """Coerce the right-hand side of a rule according to the target column type.

    Natural-language rules often include labels after the value, e.g.
    ``temperature > 80 异常``. For numeric columns we accept that form and extract
    the first numeric literal, while preserving strict behavior for non-numeric
    equality rules such as ``status == WARN``.
    """
    if pd.api.types.is_numeric_dtype(series):
        # 数值列允许右侧带中文后缀，例如 “80 异常”，只提取第一个数字。
        match = re.search(r"[-+]?\d+(?:\.\d+)?", raw_value)
        if not match:
            return None, f"Numeric rule value not found in: {raw_value}"
        number = float(match.group(0))
        return int(number) if number.is_integer() else number, None
    return coerce_value(raw_value), None


def coerce_value(value: str) -> Any:
    """Coerce rule literal values to int/float when possible."""
    stripped = value.strip()
    # 非数值字符串保持原样，便于 status == WARN 这类规则执行。
    if re.fullmatch(r"[-+]?\d+", stripped):
        return int(stripped)
    if re.fullmatch(r"[-+]?\d*\.\d+", stripped):
        as_float = float(stripped)
        if math.isfinite(as_float):
            return as_float
    return stripped
