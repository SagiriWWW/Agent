from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from aircraft_agent.agent import model_providers
from aircraft_agent.agent.model_providers import (
    ChatCompletionsAdapter,
    ProviderToolResult,
    ResponsesAdapter,
    build_llm_provider,
    to_chat_completions_tools,
)
from aircraft_agent.common.config import LLM_API_PROTOCOL_CHAT_COMPLETIONS, Settings
from aircraft_agent.tools.executor import OPENAI_TOOL_SCHEMAS


class FakeResponsesAPI:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        if "previous_response_id" not in kwargs:
            return {
                "id": "resp-1",
                "output": [
                    {
                        "type": "function_call",
                        "name": "profile_dataset",
                        "call_id": "call-1",
                        "arguments": json.dumps({"dataset_id": "d1"}),
                    }
                ],
                "output_text": "",
            }
        return {"id": "resp-2", "output": [], "output_text": "完成"}


class FakeChatCompletionsAPI:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        if len(kwargs["messages"]) == 2:
            return {
                "id": "chat-1",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "tool-1",
                                    "type": "function",
                                    "function": {
                                        "name": "parse_rule_library",
                                        "arguments": json.dumps({"rule_file_id": "r1"}),
                                    },
                                }
                            ],
                        }
                    }
                ],
            }
        return {"id": "chat-2", "choices": [{"message": {"role": "assistant", "content": "完成"}}]}


class FakeClient:
    def __init__(self) -> None:
        self.responses = FakeResponsesAPI()
        self.chat = type("Chat", (), {"completions": FakeChatCompletionsAPI()})()


def test_to_chat_completions_tools_wraps_responses_schema() -> None:
    converted = to_chat_completions_tools(OPENAI_TOOL_SCHEMAS[:1])

    assert converted == [
        {
            "type": "function",
            "function": {
                "name": "profile_dataset",
                "description": OPENAI_TOOL_SCHEMAS[0]["description"],
                "parameters": OPENAI_TOOL_SCHEMAS[0]["parameters"],
            },
        }
    ]


def test_responses_adapter_round_trip_with_fake_client() -> None:
    client = FakeClient()
    adapter = ResponsesAdapter(api_key="key", model="model", base_url=None, client=client)

    first = adapter.start(system="sys", user="user", tools=OPENAI_TOOL_SCHEMAS[:1])
    assert first.state_ref == "resp-1"
    assert first.function_calls[0].name == "profile_dataset"
    assert first.function_calls[0].arguments == {"dataset_id": "d1"}

    second = adapter.continue_with_tool_outputs(
        previous=first,
        tool_outputs=[ProviderToolResult(call_id="call-1", output='{"summary":"ok"}')],
        tools=OPENAI_TOOL_SCHEMAS[:1],
    )

    assert second.output_text == "完成"
    assert client.responses.calls[1]["previous_response_id"] == "resp-1"
    assert client.responses.calls[1]["input"][0]["type"] == "function_call_output"


def test_chat_completions_adapter_round_trip_with_fake_client() -> None:
    client = FakeClient()
    adapter = ChatCompletionsAdapter(
        api_key="local",
        model="local-model",
        base_url="http://127.0.0.1:11434/v1",
        client=client,
    )

    first = adapter.start(system="sys", user="user", tools=OPENAI_TOOL_SCHEMAS[:1])
    assert first.state_ref == "chat-1"
    assert first.function_calls[0].name == "parse_rule_library"
    assert first.function_calls[0].call_id == "tool-1"
    assert first.function_calls[0].arguments == {"rule_file_id": "r1"}
    assert first.messages[-1]["role"] == "assistant"
    assert first.messages[-1]["tool_calls"][0]["id"] == "tool-1"

    second = adapter.continue_with_tool_outputs(
        previous=first,
        tool_outputs=[ProviderToolResult(call_id="tool-1", output='{"summary":"ok"}')],
        tools=OPENAI_TOOL_SCHEMAS[:1],
    )

    assert second.output_text == "完成"
    assert client.chat.completions.calls[1]["messages"][-1] == {
        "role": "tool",
        "tool_call_id": "tool-1",
        "content": '{"summary":"ok"}',
    }


def test_build_llm_provider_selects_chat_completions(monkeypatch) -> None:
    monkeypatch.setattr(model_providers, "_build_openai_client", lambda **kwargs: FakeClient())
    settings = Settings(
        storage_dir=Path("runtime"),
        provider="custom_openai",
        openai_api_key="local",
        openai_model="local-model",
        base_url="http://127.0.0.1:11434/v1",
        force_openai=False,
        max_tool_steps=4,
        api_protocol=LLM_API_PROTOCOL_CHAT_COMPLETIONS,
    )

    provider = build_llm_provider(settings)

    assert isinstance(provider, ChatCompletionsAdapter)
