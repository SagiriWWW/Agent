"""Opt-in deterministic model views and conservative Chat request admission.

Business results and evidence hashes remain owned by the executor. This module
never edits history, source results, contracts, or report inputs.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any

from aircraft_agent.common.utils import hash_json, stable_json
from aircraft_agent.tools.catalog import OPENAI_TOOL_SCHEMAS, TOOL_NAME_ALIASES

VERSION = "tool-view-v1-byte-budget-v1"
VIEW_VERSION = "tool-view-v1"
CONTEXT = 32768
OUTPUT_RESERVE = 4096
REQUEST_SETTINGS = {"temperature": 0, "max_tokens": OUTPUT_RESERVE, "reasoning_effort": "none", "seed": 17}
MAX_VIEW_BYTES = 16384
KNOWN_TOOLS = {item["name"] for item in OPENAI_TOOL_SCHEMAS}


class InputPolicyError(RuntimeError):
    """Explicit admission failure; must never trigger deterministic fallback."""

    def __init__(self, status: str, details: dict[str, Any]):
        self.status = status
        self.details = details
        super().__init__(f"{status}: {stable_json(details)}")


def sizes(value: Any) -> dict[str, int]:
    text = stable_json(value)
    return {"bytes": len(text.encode("utf-8")), "characters": len(text)}


def build_tool_view(name: str, result: Any, *, resource_version: str | None = None) -> tuple[str, dict[str, Any]]:
    """Project only declared non-decision fields; unknown fields stay intact.

    All list rows are retained (projection, not sampling). Diagnostic statistics,
    actual IDs, mappings, status/error and artifact paths are not shortened.
    Unregistered tools are allowed only for complete execution errors.
    """
    canonical = TOOL_NAME_ALIASES.get(name, name)
    value = deepcopy(result)
    omissions: list[dict[str, Any]] = []
    is_error = isinstance(value, dict) and "error" in value
    if canonical not in KNOWN_TOOLS and not is_error:
        raise InputPolicyError("tool_view_unsupported", {"tool": name})

    def omit_field(obj: dict, key: str, path: str) -> None:
        if key in obj:
            old = obj.pop(key)
            omissions.append({"path": path, "omitted_fields": [key],
                              "original_count": len(old) if isinstance(old, (list, dict)) else None,
                              "returned_count": 0, "reason": "declared descriptive/sample projection"})

    def project_rows(rows: list, fields: set[str], path: str) -> None:
        omitted = set()
        for row in rows:
            if not isinstance(row, dict):
                raise InputPolicyError("tool_view_invalid_shape", {"path": path})
            for key in fields:
                if key in row:
                    omitted.add(key)
                    del row[key]
        omissions.append({"path": path, "original_count": len(rows), "returned_count": len(rows),
                          "omitted_fields": sorted(omitted), "reason": "all rows retained; field projection"})

    if not is_error and isinstance(value, dict):
        if canonical == "profile_dataset" and isinstance(value.get("profile"), dict):
            profile = value["profile"]
            omit_field(profile, "sample_rows", "$.profile.sample_rows")
            if isinstance(profile.get("column_profiles"), list):
                project_rows(profile["column_profiles"], {"sample_values", "top_values"}, "$.profile.column_profiles")
        elif canonical == "review_schema_mapping":
            if isinstance(value.get("fields"), list):
                project_rows(value["fields"], {"tokens", "source_hint", "aliases", "display_name"}, "$.fields")
            # Duplicate list is omitted only after value equality is established.
            if "rule_fields" in value and value.get("mappings") == value["rule_fields"]:
                old = value.pop("rule_fields")
                omissions.append({"path": "$.rule_fields", "original_count": len(old), "returned_count": 0,
                                  "omitted_fields": ["rule_fields"], "equivalent_to": "$.mappings",
                                  "reason": "identical list retained at mappings"})
        elif canonical == "suggest_schema_metadata" and isinstance(value.get("schema"), dict):
            fields = value["schema"].get("fields")
            if isinstance(fields, dict):
                project_rows(list(fields.values()), {"display_name"}, "$.schema.fields.*")
    view = {"view_version": VIEW_VERSION, "tool": canonical, "result": value,
            "omissions": omissions, "source_scope": "Projected tool result; artifact location does not mean its content was read."}
    original_size, view_size = sizes(result), sizes(view)
    audit = {"tool": name, "canonical_tool": canonical, "original_output_hash": hash_json(result),
             "view_version": VIEW_VERSION, "view_hash": hash_json(view), "original_size": original_size,
             "view_size": view_size, "byte_reduction": original_size["bytes"] - view_size["bytes"],
             "omissions": omissions, "result_is_error": is_error}
    limit = MAX_VIEW_BYTES
    status = "tool_view_budget_exceeded"
    if resource_version is not None:
        from aircraft_agent.agent.ollama_budget import VIEW_RESOURCE_VERSION, MAX_VIEW_RESOURCE_BYTES
        if resource_version != VIEW_RESOURCE_VERSION:
            raise InputPolicyError("tool_view_resource_unknown", {"version": resource_version})
        limit = MAX_VIEW_RESOURCE_BYTES
        status = "tool_view_resource_exceeded"
        audit.update(resource_version=resource_version, resource_limit_bytes=limit,
                     resource_status="admit" if view_size["bytes"] <= limit else "exceeded")
    if view_size["bytes"] > limit:
        raise InputPolicyError(status, audit)
    return stable_json(view), audit


def validate_pairs(messages: list[dict[str, Any]]) -> None:
    """Require each assistant batch to be immediately and completely answered."""
    pending: set[str] = set()
    seen: set[str] = set()
    if len(messages) < 2 or messages[0].get("role") != "system" or messages[1].get("role") != "user":
        raise InputPolicyError("input_message_pairing_invalid", {"reason": "system/task prefix missing"})
    for message in messages:
        role = message.get("role")
        if role == "tool":
            call_id = message.get("tool_call_id")
            if call_id not in pending:
                raise InputPolicyError("input_message_pairing_invalid", {"reason": "orphan or duplicate output", "call_id": call_id})
            pending.remove(call_id)
            continue
        if pending:
            raise InputPolicyError("input_message_pairing_invalid", {"reason": "unanswered batch", "call_ids": sorted(pending)})
        for call in message.get("tool_calls", []):
            call_id = call.get("id")
            if role != "assistant" or not isinstance(call_id, str) or not call_id or call_id in seen:
                raise InputPolicyError("input_message_pairing_invalid", {"reason": "invalid or reused call ID"})
            pending.add(call_id)
            seen.add(call_id)
    if pending:
        raise InputPolicyError("input_message_pairing_invalid", {"reason": "missing outputs", "call_ids": sorted(pending)})


def assess_request(request: dict[str, Any], *, context: int = CONTEXT) -> dict[str, Any]:
    """Bytes-as-units, NOT exact tokens or a claimed tokenizer upper bound.

    UTF-8 bytes cover literal payload conservatively for byte-level BPE. The
    unmeasured service template gets fixed + per-message/tool headroom. Error
    relative to actual rendered tokens is unknown; admit is not completeness.
    This opt-in profile is restricted to the frozen local qwen3 experiment.
    """
    validate_pairs(request["messages"])
    if request.get("max_tokens") != OUTPUT_RESERVE:
        raise InputPolicyError("input_budget_configuration_invalid", {"reason": "output reserve mismatch"})
    size = sizes(request)
    template = 2048 + 128 * len(request["messages"]) + 128 * len(request.get("tools", []))
    total = size["bytes"] + template + OUTPUT_RESERVE
    return {"policy_version": VERSION, "status": "conservative_admit" if total <= context else "input_budget_exceeded",
            "request_hash": hash_json(request), "request_size": size, "context_tokens": context,
            "output_reserve_tokens": OUTPUT_RESERVE, "template_allowance_units": template,
            "input_budget_units": size["bytes"] + template, "total_budget_units": total,
            "exact_input_tokens": None, "estimation_error_tokens": None,
            "count_method": "UTF-8 bytes of canonical full request + 2048 + 128/message + 128/tool; no character/token ratio",
            "limitation": "Conservative admission heuristic, not exact service template/token count or proof of complete input; verify full server log interval.",
            "history_removed": False}
