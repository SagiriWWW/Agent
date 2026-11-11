"""Protocol adapters for model-driven local tool routing."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Protocol

from aircraft_agent.common.config import (
    LLM_API_PROTOCOL_CHAT_COMPLETIONS,
    LLM_API_PROTOCOL_RESPONSES,
    Settings,
)


@dataclass(frozen=True)
class ProviderToolCall:
    name: str
    call_id: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProviderToolResult:
    call_id: str
    output: str


@dataclass
class ProviderResponse:
    function_calls: list[ProviderToolCall] = field(default_factory=list)
    output_text: str = ""
    state_ref: str | None = None
    messages: list[dict[str, Any]] = field(default_factory=list)
    raw_response_id: str | None = None


class LLMProvider(Protocol):
    protocol: str
    tool_name: str

    def start(self, *, system: str, user: str, tools: list[dict[str, Any]]) -> ProviderResponse:
        ...

    def continue_with_tool_outputs(
        self,
        *,
        previous: ProviderResponse,
        tool_outputs: list[ProviderToolResult],
        tools: list[dict[str, Any]],
    ) -> ProviderResponse:
        ...


class ResponsesAdapter:
    protocol = LLM_API_PROTOCOL_RESPONSES
    tool_name = "openai_responses"

    def __init__(self, *, api_key: str, model: str, base_url: str | None, client: Any | None = None) -> None:
        self.model = model
        self.client = client or _build_openai_client(api_key=api_key, base_url=base_url)

    def start(self, *, system: str, user: str, tools: list[dict[str, Any]]) -> ProviderResponse:
        response = self.client.responses.create(
            model=self.model,
            input=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            tools=tools,
        )
        return _responses_to_provider_response(response)

    def continue_with_tool_outputs(
        self,
        *,
        previous: ProviderResponse,
        tool_outputs: list[ProviderToolResult],
        tools: list[dict[str, Any]],
    ) -> ProviderResponse:
        response = self.client.responses.create(
            model=self.model,
            previous_response_id=previous.state_ref,
            input=[
                {"type": "function_call_output", "call_id": item.call_id, "output": item.output}
                for item in tool_outputs
            ],
            tools=tools,
        )
        return _responses_to_provider_response(response)


class ChatCompletionsAdapter:
    protocol = LLM_API_PROTOCOL_CHAT_COMPLETIONS
    tool_name = "chat_completions"

    def __init__(self, *, api_key: str, model: str, base_url: str | None, client: Any | None = None) -> None:
        self.request_guard = None
        self.model = model
        self.client = client or _build_openai_client(api_key=api_key, base_url=base_url)

    def start(self, *, system: str, user: str, tools: list[dict[str, Any]]) -> ProviderResponse:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        request = dict(model=self.model, messages=messages, tools=to_chat_completions_tools(tools))
        if self.request_guard is not None:
            from aircraft_agent.agent.input_policy import REQUEST_SETTINGS
            request.update(REQUEST_SETTINGS)
            self.request_guard(request)
        response = self.client.chat.completions.create(**request)
        return _chat_to_provider_response(response, messages)

    def continue_with_tool_outputs(
        self,
        *,
        previous: ProviderResponse,
        tool_outputs: list[ProviderToolResult],
        tools: list[dict[str, Any]],
    ) -> ProviderResponse:
        messages = list(previous.messages)
        messages.extend(
            {"role": "tool", "tool_call_id": item.call_id, "content": item.output}
            for item in tool_outputs
        )
        request = dict(model=self.model, messages=messages, tools=to_chat_completions_tools(tools))
        if self.request_guard is not None:
            from aircraft_agent.agent.input_policy import REQUEST_SETTINGS
            request.update(REQUEST_SETTINGS)
            self.request_guard(request)
        response = self.client.chat.completions.create(**request)
        return _chat_to_provider_response(response, messages)


def build_llm_provider(settings: Settings) -> LLMProvider:
    if not settings.openai_api_key:
        raise RuntimeError("A provider API key is required to build an LLM provider.")
    adapter = ChatCompletionsAdapter if settings.api_protocol == LLM_API_PROTOCOL_CHAT_COMPLETIONS else ResponsesAdapter
    return adapter(
        api_key=settings.openai_api_key,
        model=settings.openai_model,
        base_url=settings.base_url,
    )


def to_chat_completions_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for tool in tools:
        if "function" in tool:
            converted.append(tool)
            continue
        converted.append(
            {
                "type": "function",
                "function": {
                    "name": tool.get("name"),
                    "description": tool.get("description", ""),
                    "parameters": tool.get("parameters", {"type": "object", "properties": {}}),
                },
            }
        )
    return converted


def _build_openai_client(*, api_key: str, base_url: str | None) -> Any:
    try:
        from openai import OpenAI
    except Exception as exc:
        raise RuntimeError("The openai package is not installed. Run pip install -r requirements.txt.") from exc
    kwargs: dict[str, Any] = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


def _responses_to_provider_response(response: Any) -> ProviderResponse:
    response_id = _get(response, "id")
    return ProviderResponse(
        function_calls=_extract_responses_function_calls(response),
        output_text=str(_get(response, "output_text") or ""),
        state_ref=str(response_id) if response_id else None,
        raw_response_id=str(response_id) if response_id else None,
    )


def _extract_responses_function_calls(response: Any) -> list[ProviderToolCall]:
    calls: list[ProviderToolCall] = []
    for item in _get(response, "output") or []:
        if _get(item, "type") != "function_call":
            continue
        name = _get(item, "name")
        call_id = _get(item, "call_id")
        if name and call_id:
            calls.append(
                ProviderToolCall(
                    name=str(name),
                    call_id=str(call_id),
                    arguments=_parse_arguments(_get(item, "arguments")),
                )
            )
    return calls


def _chat_to_provider_response(response: Any, previous_messages: list[dict[str, Any]]) -> ProviderResponse:
    response_id = _get(response, "id")
    choices = _get(response, "choices") or []
    message = _get(choices[0], "message") if choices else {}
    content = _get(message, "content") or ""
    assistant_message: dict[str, Any] = {"role": "assistant", "content": content}
    tool_calls_payload: list[dict[str, Any]] = []
    calls: list[ProviderToolCall] = []

    for tool_call in _get(message, "tool_calls") or []:
        function = _get(tool_call, "function") or {}
        call_id = _get(tool_call, "id")
        name = _get(function, "name")
        if not call_id or not name:
            continue
        arguments = _parse_arguments(_get(function, "arguments"))
        tool_calls_payload.append(
            {
                "id": str(call_id),
                "type": _get(tool_call, "type") or "function",
                "function": {"name": str(name), "arguments": json.dumps(arguments, ensure_ascii=False)},
            }
        )
        calls.append(ProviderToolCall(name=str(name), call_id=str(call_id), arguments=arguments))

    if tool_calls_payload:
        assistant_message["tool_calls"] = tool_calls_payload
    return ProviderResponse(
        function_calls=calls,
        output_text=str(content),
        state_ref=str(response_id) if response_id else None,
        messages=[*previous_messages, assistant_message],
        raw_response_id=str(response_id) if response_id else None,
    )


def _parse_arguments(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(str(raw or "{}"))
    except (json.JSONDecodeError, TypeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _get(obj: Any, key: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)
