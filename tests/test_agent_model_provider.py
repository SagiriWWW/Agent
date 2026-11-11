from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from aircraft_agent.agent import runner as runner_module
from aircraft_agent.agent.model_providers import ProviderResponse, ProviderToolCall
from aircraft_agent.agent.runner import AgentRunner
from aircraft_agent.agent.runtime import RunStore
from aircraft_agent.common.config import LLM_API_PROTOCOL_CHAT_COMPLETIONS, Settings
from aircraft_agent.tools.executor import ToolExecutor


class FakeProvider:
    protocol = LLM_API_PROTOCOL_CHAT_COMPLETIONS
    tool_name = "fake_chat_completions"

    def __init__(self) -> None:
        self.step = 0
        self.tool_outputs = []

    def start(self, *, system: str, user: str, tools: list[dict]) -> ProviderResponse:
        self.step += 1
        assert "Use the provided local function tools" in system
        return ProviderResponse(
            function_calls=[ProviderToolCall(name="profile_dataset", call_id="call-profile", arguments={})],
            state_ref="fake-1",
        )

    def continue_with_tool_outputs(
        self,
        *,
        previous: ProviderResponse,
        tool_outputs: list,
        tools: list[dict],
    ) -> ProviderResponse:
        self.step += 1
        self.tool_outputs.extend(tool_outputs)
        if self.step == 2:
            return ProviderResponse(
                function_calls=[ProviderToolCall(name="parse_rule_library", call_id="call-rules", arguments={})],
                state_ref="fake-2",
            )
        return ProviderResponse(output_text="模型完成", state_ref="fake-3")


class FailingProvider:
    protocol = LLM_API_PROTOCOL_CHAT_COMPLETIONS
    tool_name = "fake_chat_completions"

    def start(self, *, system: str, user: str, tools: list[dict]) -> ProviderResponse:
        raise RuntimeError("local runtime unavailable")


def _context_and_settings(tmp_path: Path):
    store = RunStore(tmp_path / "runtime")
    context = store.create_run("model provider")
    store.save_bytes(context, data=b"R001: temperature > 80\n", original_name="rules.md", kind="rule_library")
    store.save_bytes(context, data=b"temperature\n82\n", original_name="data.csv", kind="dataset")
    settings = Settings(
        storage_dir=tmp_path / "runtime",
        provider="custom_openai",
        openai_api_key="local",
        openai_model="local-model",
        base_url="http://127.0.0.1:11434/v1",
        force_openai=False,
        max_tool_steps=4,
        api_protocol=LLM_API_PROTOCOL_CHAT_COMPLETIONS,
    )
    return context, settings


def test_agent_runner_uses_provider_and_records_evidence(tmp_path: Path, monkeypatch) -> None:
    context, settings = _context_and_settings(tmp_path)
    fake = FakeProvider()
    monkeypatch.setattr(runner_module, "build_llm_provider", lambda settings: fake)

    result = AgentRunner(context, ToolExecutor(context), settings).run()

    assert result["mode"] == "custom_openai"
    assert result["api_protocol"] == LLM_API_PROTOCOL_CHAT_COMPLETIONS
    assert context.tool_results[0]["tool"] == "profile_dataset"
    assert context.tool_results[0]["evidence_reference"]["node_hash"]
    assert any(item["action"] == "profile_dataset" for item in context.agent_trace)
    assert any(item.get("related_evidence") for item in context.agent_trace)
    model_nodes = [node for node in context.evidence.nodes if node.tool_name == "fake_chat_completions"]
    assert model_nodes
    assert all(node.arguments["api_protocol"] == LLM_API_PROTOCOL_CHAT_COMPLETIONS for node in model_nodes)
    assert fake.tool_outputs
    assert context.report_html_path and context.report_html_path.exists()


def test_agent_runner_provider_failure_falls_back_to_offline(tmp_path: Path, monkeypatch) -> None:
    context, settings = _context_and_settings(tmp_path)
    monkeypatch.setattr(runner_module, "build_llm_provider", lambda settings: FailingProvider())

    result = AgentRunner(context, ToolExecutor(context), settings).run()

    assert result["mode"] == "offline"
    assert context.report_html_path and context.report_html_path.exists()
    fallback_nodes = [node for node in context.evidence.nodes if node.type == "model_call" and "回退" in node.summary]
    assert fallback_nodes
    assert fallback_nodes[0].arguments["api_protocol"] == LLM_API_PROTOCOL_CHAT_COMPLETIONS


def test_agent_runner_provider_failure_is_fatal_when_model_required(tmp_path: Path, monkeypatch) -> None:
    context, settings = _context_and_settings(tmp_path)
    monkeypatch.setattr(runner_module, "build_llm_provider", lambda settings: FailingProvider())

    with pytest.raises(RuntimeError, match="local runtime unavailable"):
        AgentRunner(context, ToolExecutor(context), replace(settings, force_openai=True)).run()


def test_input_policy_maps_multicall_success_error_alias_and_repeat(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace
    from aircraft_agent.agent.model_providers import ChatCompletionsAdapter
    from aircraft_agent.agent.input_policy import VERSION
    from aircraft_agent.common.utils import hash_json
    from aircraft_agent.tools.catalog import TOOL_NAME_ALIASES
    context, settings = _context_and_settings(tmp_path)
    alias = next(iter(TOOL_NAME_ALIASES))
    captured = []
    def create(**request):
        captured.append(request)
        if len(captured) == 1:
            return {'id':'response-1','choices':[{'message':{'content':'', 'tool_calls':[
                {'id':str(i),'type':'function','function':{'name':name,'arguments':json.dumps(arguments)}}
                for i,(name,arguments) in enumerate([('profile_dataset',{}),('profile_dataset',{}),('review_schema_metadata',{'dataset_id':'bad'}),(alias,{})])
            ]}}]}
        return {'id':'response-2','choices':[{'message':{'content':'done'}}]}
    provider = ChatCompletionsAdapter(api_key='x',model='qwen3:4b',base_url=None,
        client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
    monkeypatch.setattr(runner_module,'build_llm_provider',lambda settings:provider)
    AgentRunner(context,ToolExecutor(context),replace(settings,model_input_policy=VERSION,openai_model='qwen3:4b')).run()
    events=[json.loads(line) for line in context.event_path.read_text().splitlines()]
    mappings=[e['payload'] for e in events if e['type']=='model_tool_view']
    assert len(mappings)==4 and len({m['call_id'] for m in mappings})==4
    tools=[m for m in captured[1]['messages'] if m['role']=='tool']
    assert len(tools)==4
    for tool,mapping in zip(tools,mappings):
        node=next(n for n in context.evidence.nodes if n.node_id==mapping['tool_evidence_reference']['node_id'])
        assert node.output_hash==mapping['original_output_hash']
        assert hash_json(json.loads(tool['content']))==mapping['view_hash']
    assert mappings[2]['result_is_error'] and mappings[2]['full_error_result']['error']
    assert mappings[3]['canonical_tool']==TOOL_NAME_ALIASES[alias]


def test_input_policy_refusal_never_falls_back(tmp_path, monkeypatch):
    from aircraft_agent.agent.input_policy import InputPolicyError, VERSION
    context,settings=_context_and_settings(tmp_path)
    monkeypatch.setattr(runner_module,'build_llm_provider',lambda settings:FailingProvider())
    with pytest.raises(InputPolicyError,match='input_policy_unsupported'):
        AgentRunner(context,ToolExecutor(context),replace(settings,model_input_policy=VERSION)).run()
    assert context.report_html_path is None
