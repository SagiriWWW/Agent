"""Agent orchestration for provider tool-calling and local fallback mode.

AgentRunner 只负责“模型如何决定下一步”，不直接实现数据分析逻辑。
所有真实数据处理都委托给 ToolExecutor，这样新增 tool 时通常不用改
AgentRunner，除非要改变整体编排策略或提示词。
"""

from __future__ import annotations

from typing import Any

from aircraft_agent.agent.model_providers import ChatCompletionsAdapter, ProviderToolResult, build_llm_provider
from aircraft_agent.agent.input_policy import VERSION, InputPolicyError, assess_request, build_tool_view
from aircraft_agent.common.config import Settings
from aircraft_agent.telemetry.demo_data import TELEMETRY_ANALYSIS_COLUMNS
from aircraft_agent.agent.runtime import RunContext
from aircraft_agent.evidence.chain import evidence_node_reference
from aircraft_agent.tools.executor import OPENAI_TOOL_SCHEMAS, ToolExecutor
from aircraft_agent.common.utils import stable_json, truncate_text


class AgentRunner:
    """Runs one end-to-end analysis task for a RunContext.

    AgentRunner 是整个项目的“编排中枢”。它不直接读 CSV、不直接算异常、
    不直接挖规则，而是决定：
    - 什么时候调用哪个工具；
    - 工具参数是什么；
    - 工具返回后如何继续；
    - 最后如何生成总结和报告。

    这样可以保持一个清晰边界：Agent 负责思考和路由，ToolExecutor 负责
    本地可信执行。
    """

    def __init__(self, context: RunContext, executor: ToolExecutor, settings: Settings) -> None:
        self.context = context
        self.executor = executor
        self.settings = settings

    def run(self) -> dict[str, Any]:
        """Choose model-tool mode when configured, otherwise use local demo mode.

        运行策略：
        - 有 API Key：优先让模型通过 function calling 自主选择工具；
        - API 失败且没有 force_openai：回退到本地固定计划，保证 demo 可运行；
        - 没有 API Key：直接走本地固定计划。
        """
        if self.settings.openai_api_key:
            try:
                return self._run_with_provider()
            except Exception as exc:
                if self.settings.force_openai or isinstance(exc, InputPolicyError):
                    raise
                reason = _friendly_model_error(exc)
                # Demo/开发阶段优先不中断流程：API 失败时仍能产出报告和证据链。
                self.context.evidence.add(
                    type="model_call",
                    tool_name=f"{self.settings.api_protocol}_provider",
                    arguments={
                        "provider": self.settings.provider,
                        "model": self.settings.openai_model,
                        "base_url": self.settings.base_url,
                        "api_protocol": self.settings.api_protocol,
                    },
                    output={"error": str(exc)},
                    summary=reason,
                )
                return self._run_offline(reason=reason)
        if self.settings.force_openai:
            raise RuntimeError("A provider API key is required when AGENT_MVP_FORCE_MODEL_API=1.")
        reason = (
            "内置 Demo 已禁用模型 API，使用本地确定性 Agent 计划。"
            if self.settings.provider == "local_demo"
            else "未设置模型 API Key，使用本地确定性 Agent 计划。"
        )
        return self._run_offline(reason=reason)

    def _run_with_provider(self) -> dict[str, Any]:
        """Run a protocol-neutral model tool-calling loop."""
        provider = build_llm_provider(self.settings)
        policy = self.settings.model_input_policy
        from aircraft_agent.agent import ollama_budget, ollama_native_budget
        render_policies = (ollama_budget.VERSION, ollama_budget.COMPARISON_VERSION, ollama_native_budget.VERSION)
        if policy:
            allowed_models = ("qwen3:4b",)
            if policy == ollama_budget.COMPARISON_VERSION:
                allowed_models = ("qwen3:4b", ollama_budget.CANDIDATE_MODEL)
            elif policy == ollama_native_budget.VERSION:
                allowed_models = (ollama_budget.CANDIDATE_MODEL,)
            if policy not in (VERSION, *render_policies) or not isinstance(provider, ChatCompletionsAdapter) or self.settings.openai_model not in allowed_models:
                raise InputPolicyError("input_policy_unsupported", {"policy": policy, "protocol": self.settings.api_protocol,
                                                                   "model": self.settings.openai_model})
            if policy in render_policies and self.settings.base_url != "http://127.0.0.1:11436/v1":
                raise InputPolicyError("input_count_identity_mismatch", {"reason": "provider endpoint outside frozen local gateway"})
            def guard(request: dict[str, Any]) -> None:
                try:
                    audit = (ollama_budget.assess_request(request, policy_version=policy) if policy in render_policies else assess_request(request))
                except InputPolicyError as exc:
                    self.context.emit_event("model_input_budget", payload={"status": exc.status, **exc.details})
                    raise
                self.context.emit_event("model_input_budget", payload=audit)
                if audit["status"] not in ("conservative_admit", "calibrated_admit"):
                    raise InputPolicyError(audit["status"], audit)
            provider.request_guard = guard
        response = provider.start(
            system=self._system_prompt(),
            user=self._initial_user_prompt(),
            tools=OPENAI_TOOL_SCHEMAS,
        )

        final_text = ""
        for step in range(self.settings.max_tool_steps):
            calls = response.function_calls
            # 记录模型每轮输出，便于事后追踪“模型为何调用这些工具”。
            self.context.evidence.add(
                type="model_call",
                tool_name=provider.tool_name,
                arguments={
                    "provider": self.settings.provider,
                    "model": self.settings.openai_model,
                    "base_url": self.settings.base_url,
                    "api_protocol": getattr(provider, "protocol", self.settings.api_protocol),
                    "step": step,
                },
                output={
                    "response_id": response.raw_response_id or response.state_ref,
                    "function_calls": [{"name": call.name, "call_id": call.call_id} for call in calls],
                    "output_text": truncate_text(response.output_text or "", 1200),
                },
                summary=f"模型响应已收到，包含 {len(calls)} 个工具调用。",
                model_response_id=response.raw_response_id or response.state_ref,
            )
            if not calls:
                # 模型不再要求工具调用时，说明它准备给最终回答。
                final_text = response.output_text or ""
                break

            tool_outputs: list[ProviderToolResult] = []
            for call in calls:
                step_number = self._start_model_tool_step(call.name, call.arguments)
                try:
                    result = self.executor.execute(call.name, call.arguments)
                    observation = _result_summary(result)
                    status = "complete"
                    tool_evidence_reference = _latest_tool_evidence_reference(self.context, call.name)
                except Exception as exc:
                    # 本地工具失败不等于模型 API 失败。把错误作为工具输出交还给模型，
                    # 模型可以改用其他工具或生成“该规则不可解析”的结论。
                    result = {"error": str(exc), "summary": f"工具 {call.name} 执行失败：{exc}"}
                    observation = str(result["summary"])
                    status = "failed"
                    node = self.context.evidence.add(
                        type="tool_call",
                        tool_name=call.name,
                        arguments=call.arguments,
                        output=result,
                        summary=result["summary"],
                    )
                    tool_evidence_reference = evidence_node_reference(node)
                self._finish_model_tool_step(
                    step=step_number,
                    name=call.name,
                    arguments=call.arguments,
                    observation=observation,
                    status=status,
                    tool_evidence_reference=tool_evidence_reference,
                )
                output = stable_json(result)
                if policy:
                    from aircraft_agent.common.utils import hash_json
                    source_node = next((node for node in reversed(self.context.evidence.nodes)
                                        if node.type == "tool_call" and node.tool_name == call.name), None)
                    if source_node is None or source_node.output_hash != hash_json(result):
                        raise InputPolicyError("tool_view_source_mismatch", {"call_id": call.call_id, "tool": call.name})
                    source = {"call_id": call.call_id, "provider_response_id": response.raw_response_id,
                              "tool_evidence_reference": evidence_node_reference(source_node),
                              "source_event": "tool_finished" if status == "complete" else "model_tool_view",
                              "full_error_result": result if status == "failed" else None}
                    try:
                        output, audit = build_tool_view(call.name, result, resource_version=ollama_budget.VIEW_RESOURCE_VERSION if policy in render_policies else None)
                    except InputPolicyError as exc:
                        self.context.emit_event("model_tool_view", payload={**source, **exc.details, "status": exc.status})
                        raise
                    self.context.emit_event("model_tool_view", payload={**source, **audit, "status": "prepared"})
                tool_outputs.append(ProviderToolResult(call_id=call.call_id, output=output))
            response = provider.continue_with_tool_outputs(
                previous=response,
                tool_outputs=tool_outputs,
                tools=OPENAI_TOOL_SCHEMAS,
            )
        else:
            final_text = "已达到最大工具调用步数；报告将基于已收集的本地工具输出生成。"

        self.context.final_text = final_text
        self.context.status = "complete"
        # The model may call write_report before the final response. Always rewrite
        # once after completion so status, final summary, and evidence count are fresh.
        if any(item.get("tool") == "write_report" for item in self.context.tool_results):
            self.executor.write_report(final_summary=final_text)
            self.context.emit_event(
                "report_refreshed",
                title="报告已刷新",
                message="模型已调用过 write_report，系统已直接刷新最终报告，避免重复显示一次报告工具步骤。",
                payload={"report_html_path": str(self.context.report_html_path) if self.context.report_html_path else None},
            )
        else:
            step_number = self._start_model_tool_step("write_report", {"final_summary": final_text})
            result = self.executor.execute("write_report", {"final_summary": final_text})
            self._finish_model_tool_step(
                step=step_number,
                name="write_report",
                arguments={"final_summary": final_text},
                observation=_result_summary(result),
                status="complete",
                tool_evidence_reference=_latest_tool_evidence_reference(self.context, "write_report"),
            )
        return {
            "mode": self.settings.provider,
            "api_protocol": self.settings.api_protocol,
            "final_text": final_text,
            "report": str(self.context.report_html_path),
        }

    def _start_model_tool_step(self, name: str, arguments: dict[str, Any]) -> int:
        """Add one visible dynamic plan item for a model-selected tool call."""
        step_number = len(self.context.agent_plan) + 1
        plan_item = {
            "step": step_number,
            "thought": "模型根据当前目标、上传文件元数据和工具输出，动态选择本地工具调用。",
            "action": name,
            "arguments": arguments,
            "expected": "工具返回结构化结果，并写入证据链与事件流。",
        }
        self.context.agent_plan.append(plan_item)
        self.context.save_manifest()
        self.context.emit_event(
            "step_started",
            title=f"Agent 步骤 {step_number} 开始",
            message=f"模型选择调用本地工具 {name}。",
            payload=plan_item,
        )
        return step_number

    def _finish_model_tool_step(
        self,
        *,
        step: int,
        name: str,
        arguments: dict[str, Any],
        observation: str,
        status: str,
        tool_evidence_reference: dict[str, Any] | None = None,
    ) -> None:
        """Record a model-selected tool call as a visible Agent trace item."""
        thought = "模型动态编排本地工具；页面展示的是实际工具调用顺序。"
        node = self.context.evidence.add(
            type="agent_trace",
            tool_name="agent_orchestrator",
            arguments={
                "step": step,
                "thought": thought,
                "action": name,
                "tool_arguments": arguments,
                "mode": self.settings.provider,
                "tool_evidence_reference": tool_evidence_reference,
            },
            output={"observation": observation, "status": status, "tool_evidence_reference": tool_evidence_reference},
            summary=f"Agent 第 {step} 步调用 {name}：{observation}",
        )
        related = [tool_evidence_reference] if tool_evidence_reference else None
        self.context.record_agent_trace(
            step=step,
            thought=thought,
            action=name,
            arguments=arguments,
            observation=observation,
            status=status,
            evidence_reference=evidence_node_reference(node),
            related_evidence=related,
        )

    def _run_offline(self, reason: str) -> dict[str, Any]:
        """Run a deterministic Agent plan so the MVP is demonstrable without an API key.

        这个模式是当前 demo 的核心：不依赖任何外部模型，也能展示项目书里的
        Agent 主结构。它会生成一套可见计划，然后逐步调用工具、记录观察、
        校验结果、生成报告。
        """
        plan = self._build_demo_plan()
        self.context.agent_plan = plan
        self.context.save_manifest()
        self.context.emit_event(
            "plan_created",
            title="Agent 计划已生成",
            message=f"Agent 已创建 {len(plan)} 步本地工具箱执行计划。",
            payload={"plan_steps": len(plan), "plan": plan},
        )
        self.context.evidence.add(
            type="agent_plan",
            tool_name="agent_orchestrator",
            arguments={"goal": self.context.user_goal, "mode": "local_demo"},
            output={"plan": plan, "toolbox": self.context.toolbox_manifest},
            summary=f"Agent 已创建 {len(plan)} 步本地工具箱执行计划。",
        )

        for step in plan:
            name = step["action"]
            args = dict(step.get("arguments") or {})
            self.context.emit_event(
                "step_started",
                title=f"Agent 步骤 {step['step']} 开始",
                message=f"开始调用工具 {name}。",
                payload={
                    "step": step["step"],
                    "thought": step["thought"],
                    "action": name,
                    "arguments": args,
                },
            )
            try:
                if name == "write_report":
                    # write_report 是计划的一部分。生成报告前先把 final_summary
                    # 填进去，确保报告里有完整 Agent 总结。
                    args["final_summary"] = self._compose_final_summary(reason)
                result = self.executor.execute(name, args)
                observation = _result_summary(result)
                status = "complete"
                tool_evidence_reference = _latest_tool_evidence_reference(self.context, name)
            except Exception as exc:
                # 单个工具失败时不直接让整个 demo 崩掉，而是把失败也作为证据写入。
                # 这能模拟真实 Agent 系统中的“工具失败 -> 记录 -> 报告/复核”链路。
                result = {"error": str(exc)}
                observation = f"{name} 执行失败：{exc}"
                status = "failed"
                node = self.context.evidence.add(
                    type="tool_call",
                    tool_name=name,
                    arguments=args,
                    output=result,
                    summary=observation,
                )
                tool_evidence_reference = evidence_node_reference(node)
            trace_node = self.context.evidence.add(
                type="agent_trace",
                tool_name="agent_orchestrator",
                arguments={
                    "step": step["step"],
                    "thought": step["thought"],
                    "action": name,
                    "tool_arguments": args,
                    "tool_evidence_reference": tool_evidence_reference,
                },
                output={"observation": observation, "status": status, "tool_evidence_reference": tool_evidence_reference},
                summary=f"Agent 第 {step['step']} 步调用 {name}：{observation}",
            )
            self.context.record_agent_trace(
                # agent_trace 是给人看的执行轨迹；evidence_chain 是给审计看的哈希链。
                step=int(step["step"]),
                thought=str(step["thought"]),
                action=name,
                arguments=args,
                observation=observation,
                status=status,
                evidence_reference=evidence_node_reference(trace_node),
                related_evidence=[tool_evidence_reference] if tool_evidence_reference else None,
            )

        final_summary = self._compose_final_summary(reason)
        self.context.final_text = final_summary
        self.context.status = "complete"
        # 计划中的 write_report 已经写过一次，但那时 context.status 还是 running。
        # 这里直接刷新报告，让最终报告显示 complete。
        self.executor.write_report(final_summary=final_summary)
        return {
            "mode": "offline",
            "final_text": final_summary,
            "report": str(self.context.report_html_path),
            "plan_steps": len(plan),
        }

    def _build_demo_plan(self) -> list[dict[str, Any]]:
        """Build the visible Agent plan used by the local demo mode.

        这是“半固定流程 + Agent 路由”的 MVP 版本。真实模型接入后，可以由
        模型动态生成工具调用；但为了演示稳定和可验收，内置 demo 使用固定
        计划。
        """
        return [
            {
                "step": 1,
                "thought": "先建立数据画像，确认字段、类型、缺失和样例，避免让模型直接处理原始大表。",
                "action": "profile_dataset",
                "arguments": {},
                "expected": "数据集规模、字段清单、缺失率和分布摘要。",
            },
            {
                "step": 2,
                "thought": "把规则库转成结构化 RuleCard，让后续规则命中可以被审计。",
                "action": "parse_rule_library",
                "arguments": {},
                "expected": "规则编号、目标字段、条件表达式和来源。",
            },
            {
                "step": 3,
                "thought": "校验结构化规则库的字段绑定、条件可执行性、严重程度和建议动作，形成复核清单。",
                "action": "validate_rule_library",
                "arguments": {},
                "expected": "可执行规则数、需复核规则数、规则问题清单和 rule_validation 产物。",
            },
            {
                "step": 4,
                "thought": "基于数据字段和规则绑定生成字段中文名、字段分组和忽略字段建议，作为可人工复核的 Schema 草案。",
                "action": "suggest_schema_metadata",
                "arguments": {"max_fields": 120},
                "expected": "字段中文名、分组、角色、忽略字段和 schema_suggestions 产物。",
            },
            {
                "step": 5,
                "thought": "把 Schema 草案整理成低置信度字段、忽略字段和规则绑定字段复核清单，避免字段语义直接黑箱固化。",
                "action": "review_schema_metadata",
                "arguments": {},
                "expected": "schema_review 复核清单和需人工确认的字段项。",
            },
            {
                "step": 6,
                "thought": "把数据字段映射到稳定语义角色，识别单位和统计口径线索，为未知客户数据生成可复核候选映射。",
                "action": "review_schema_mapping",
                "arguments": {},
                "expected": "字段语义角色、规则字段候选映射、低置信度和歧义映射清单。",
            },
            {
                "step": 7,
                "thought": "在规则执行前审查数据字段与规则字段的适配关系，避免把当前规则库强行套到不一致的新数据结构上。",
                "action": "review_dataset_adaptation",
                "arguments": {},
                "expected": "可直接执行规则、需映射确认规则、不适用规则和 dataset_adaptation 产物。",
            },
            {
                "step": 8,
                "thought": "把候选字段映射按可考虑确认、需专家复核、统计口径风险和不适用分组，辅助人工复核而不自动写入配置。",
                "action": "review_schema_mapping_candidates",
                "arguments": {},
                "expected": "映射候选复核建议、风险标签和 schema_mapping_review 产物。",
            },
            {
                "step": 9,
                "thought": "把已确认字段映射转成可审查的执行计划，标出单位、统计口径和回归验证风险，但不自动改写规则。",
                "action": "build_mapping_execution_plan",
                "arguments": {},
                "expected": "字段重写预案、风险项、回归验证要求和 mapping_execution_plan 产物。",
            },
            {
                "step": 10,
                "thought": "验证已确认字段映射生成的重写预案是否能透明执行，但不把单数据集命中率当作业务正确性证明。",
                "action": "validate_mapping_regression",
                "arguments": {},
                "expected": "映射回归验证状态、重写后条件执行证据、需确认或需复核的映射风险。",
            },
            {
                "step": 11,
                "thought": "在分析前检查数据质量，先暴露缺失字段和潜在处理建议。",
                "action": "check_missing_values",
                "arguments": {},
                "expected": "缺失字段、缺失率和处理建议。",
            },
            {
                "step": 12,
                "thought": "用透明统计规则定位数值离群点，给专家提供可复核边界。",
                "action": "detect_outliers",
                "arguments": {"method": "iqr", "columns": TELEMETRY_ANALYSIS_COLUMNS},
                "expected": "离群数量、比例、上下界和样例索引。",
            },
            {
                "step": 13,
                "thought": "执行规则库，区分可自动计算规则和需要人工复核的规则。",
                "action": "apply_rules",
                "arguments": {},
                "expected": "每条规则的命中行数、命中率和样例索引。",
            },
            {
                "step": 14,
                "thought": "在解释规则命中前执行泛化验收门禁，确认系统没有把候选映射、不适用规则或当前数据集特征误当作自动诊断依据。",
                "action": "review_generalization_readiness",
                "arguments": {},
                "expected": "泛化门禁状态、反过拟合风险、是否允许自动诊断解释和后续复核建议。",
            },
            {
                "step": 15,
                "thought": "在关联规则挖掘前确认 FP-growth 后端，明确是 Datamining、mlxtend 还是本地 fallback。",
                "action": "check_association_backend",
                "arguments": {},
                "expected": "关联挖掘后端状态和路径。",
            },
            {
                "step": 16,
                "thought": "用更保守的阈值挖掘字段组合关联，并过滤只命中少量样本的弱规则。",
                "action": "mine_association_rules",
                "arguments": {
                    "min_support": 0.2,
                    "min_confidence": 0.7,
                    "min_lift": 1.2,
                    "min_count": 10,
                    "max_rules": 15,
                    "max_rows": 2000,
                    "columns": TELEMETRY_ANALYSIS_COLUMNS,
                },
                "expected": "通过最小命中数过滤后的关联规则、支持度、置信度、lift 和命中行数。",
            },
            {
                "step": 17,
                "thought": "补充数值字段相关性，帮助判断异常指标是否同步变化。",
                "action": "compute_correlations",
                "arguments": {"top_n": 12, "columns": TELEMETRY_ANALYSIS_COLUMNS},
                "expected": "最强相关字段对。",
            },
            {
                "step": 18,
                "thought": "把工具结果压缩成面向报告的要点，保留来源而不编造结论。",
                "action": "summarize_findings",
                "arguments": {"focus": self.context.user_goal},
                "expected": "工具调用摘要。",
            },
            {
                "step": 19,
                "thought": "在生成报告前做结果校验，确认必需工具和证据链都存在。",
                "action": "validate_analysis",
                "arguments": {},
                "expected": "通过/告警/失败状态和缺失项。",
            },
            {
                "step": 20,
                "thought": "把关键工具结果转成静态图表，让最终报告能直接展示规则命中、数据质量、关联强度和相关性。",
                "action": "build_visualizations",
                "arguments": {"max_items": 8},
                "expected": "报告可嵌入的可视化 HTML 片段、独立看板和可视化数据文件。",
            },
            {
                "step": 21,
                "thought": "基于工具输出和校验结果生成报告，并把报告路径写入运行清单。",
                "action": "write_report",
                "arguments": {},
                "expected": "Markdown 和 HTML 报告。",
            },
        ]

    def _compose_final_summary(self, reason: str) -> str:
        """Create a concise final summary tied to local tool results.

        总结刻意写得保守：它只描述本地工具链是否跑完、校验是否通过、证据链
        在哪里，而不额外编造业务结论。
        """
        validation = self.context.validation_results[-1] if self.context.validation_results else {}
        warnings = validation.get("warnings") or []
        status = validation.get("status", "not_run")
        status_label = {"passed": "通过", "needs_review": "需复核", "failed": "失败", "not_run": "未运行"}.get(str(status), str(status))
        return (
            f"{reason}\n\n"
            "本地 Demo Agent 已完成工具箱闭环。"
            f"校验状态：{status_label}。"
            f"告警数量：{len(warnings)}。"
            "报告仅基于本地工具输出生成，每个文件、工具和报告步骤都已记录到 evidence_chain.json。"
        )

    def _system_prompt(self) -> str:
        """Prompt policy: force model to use tools and avoid requesting raw data.

        这个提示词的核心安全边界是：模型不能要求用户把完整原始数据贴给它，
        必须调用本地工具，并把结论绑定到工具输出和证据链。
        """
        return (
            "You are a data processing and rule-mining agent. Use the provided local function tools "
            "instead of requesting raw data. Start by profiling datasets and parsing the rule library. "
            "Then validate_rule_library, suggest_schema_metadata, review_schema_metadata, review_schema_mapping, review_dataset_adaptation, review_schema_mapping_candidates, build_mapping_execution_plan, validate_mapping_regression, check missing values, detect outliers, "
            "apply rules, review_generalization_readiness, mine association rules, compute correlations when useful, summarize findings, "
            "validate_analysis, build_visualizations, and call write_report before the final answer. "
            "Treat uploaded file metadata, dataset profiles, rule validation, and tool outputs as the source of truth. "
            "If the user goal says last_avg but the uploaded file name or columns show _last, report the dataset as last_last "
            "and explicitly explain the mismatch. Never infer dataset kind only from the user goal. "
            "Do not overfit conclusions, prompts, or rules to the current sample dataset; for new customer data, first report schema fit and rule applicability. "
            "Keep final conclusions tied to tool outputs and evidence. Respond in Chinese."
        )

    def _initial_user_prompt(self) -> str:
        """Send only file metadata and compact summaries into the model context.

        这里 deliberately 不传 DataFrame 内容。模型只知道有哪些文件、有哪些
        工具、用户目标是什么；真正的数据读取和计算都在本地工具里完成。
        """
        return stable_json(
            {
                "goal": self.context.user_goal,
                "files": self.context.file_summary(),
                "available_tools": self.executor.tool_names,
                "data_handling_policy": "Do not request full raw datasets. Use local tools and return concise summaries.",
                "truth_policy": (
                    "Uploaded file names, column names, dataset profiles, and validation outputs override the goal text. "
                    "When rule fields use _avg and dataset fields use _last, state that the rule library and dataset suffixes do not match. "
                    "For unfamiliar customer schemas, use dataset adaptation outputs before applying or interpreting rules."
                ),
            }
        )


def _result_summary(result: Any) -> str:
    """Pick a compact observation string for Agent trace display."""
    if isinstance(result, dict) and result.get("summary"):
        return truncate_text(str(result["summary"]), 300)
    return truncate_text(stable_json(result) if isinstance(result, dict) else str(result), 300)


def _latest_tool_evidence_reference(context: RunContext, name: str) -> dict[str, Any] | None:
    """Return the most recent evidence reference for a completed tool call."""
    for item in reversed(context.tool_results):
        if item.get("tool") == name and isinstance(item.get("evidence_reference"), dict):
            return item["evidence_reference"]
    return None


def _friendly_model_error(exc: Exception) -> str:
    """Return a customer-readable model failure reason for reports."""
    message = str(exc)
    lowered = message.lower()
    if "free quota" in lowered or "permission_denied" in lowered or "quota" in lowered:
        return "模型服务权限或额度不可用，已回退到本地确定性流程。"
    if "api key" in lowered or "unauthorized" in lowered or "authentication" in lowered:
        return "模型服务认证不可用，已回退到本地确定性流程。"
    return "模型 API 调用失败，已回退到本地确定性流程。"
