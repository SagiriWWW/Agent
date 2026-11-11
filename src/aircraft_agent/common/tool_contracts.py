"""Version 1 contracts for the five stable core tool boundaries.

Business payloads remain flat dictionaries. Nested domain data is deliberately
open: this module does not reinterpret rules, datasets or evidence schemas.
"""
from __future__ import annotations

import traceback
from functools import wraps
from inspect import signature
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator, model_validator

TOOL_SCHEMA_VERSION = 1


class VersionedContract(BaseModel):
    schema_version: Literal[1] = TOOL_SCHEMA_VERSION

    @field_validator("schema_version", mode="before")
    @classmethod
    def validate_version(cls, value):
        if type(value) is not int or value != TOOL_SCHEMA_VERSION:
            raise ValueError("schema_version must be integer 1")
        return value


class ToolInput(VersionedContract):
    model_config = ConfigDict(extra="forbid", strict=True)


class ProfileDatasetInput(ToolInput):
    dataset_id: str | None = None


class ParseRuleLibraryInput(ToolInput):
    rule_file_id: str | None = None


class ApplyRulesInput(ProfileDatasetInput):
    rule_file_id: str | None = None


class ValidateAnalysisInput(ToolInput):
    required_tools: list[str] | None = None
    strict: bool = False


class WriteReportInput(ToolInput):
    final_summary: str = ""


class ToolFailure(VersionedContract):
    model_config = ConfigDict(extra="forbid", strict=True)
    code: str
    message: str
    stage: str
    exception_type: str | None = None
    traceback: str | None = None

    @classmethod
    def from_exception(cls, exc: Exception, *, stage: str = "execution", code: str = "TOOL_EXECUTION_FAILED") -> ToolFailure:
        if isinstance(exc, (ToolContractError, ToolArgumentError)):
            stage, code = exc.stage, exc.code
        return cls(
            code=code, message=str(exc), stage=stage,
            exception_type=f"{type(exc).__module__}.{type(exc).__qualname__}",
            traceback="".join(traceback.format_exception(exc)),
        )


class ToolOutput(VersionedContract):
    model_config = ConfigDict(extra="allow", strict=True)
    summary: str


class ProfileDatasetOutput(ToolOutput):
    dataset_id: str
    source_file: str
    dataset_kind_hint: str
    profile: dict[str, Any]


class ParseRuleLibraryOutput(ToolOutput):
    rule_file_id: str
    rules: list[dict[str, Any]]


class RuleExecutionResult(BaseModel):
    model_config = ConfigDict(extra="allow", strict=True)
    rule_id: str
    name: str
    status: Literal["evaluated", "disabled", "unparseable"]
    failure: ToolFailure | None = None

    @model_validator(mode="after")
    def require_failure(self):
        if self.status == "unparseable" and self.failure is None:
            raise ValueError("unparseable rules require a structured failure")
        return self


class ApplyRulesOutput(ToolOutput):
    dataset_id: str
    rule_results: list[RuleExecutionResult]


class ValidateAnalysisOutput(ToolOutput):
    status: Literal["passed", "needs_review", "failed"]
    required_tools: list[str]
    completed_tools: list[str]
    missing_tools: list[str]
    warnings: list[str]
    errors: list[str]
    evidence_nodes: int
    evidence_schema_version: int | None
    chain_hash: str | None
    evidence_verification: dict[str, Any]
    failure: ToolFailure | None = None

    @model_validator(mode="after")
    def require_failure(self):
        if self.status == "failed" and self.failure is None:
            raise ValueError("failed validation requires a structured failure")
        return self


class WriteReportOutput(ToolOutput):
    report_markdown: str
    report_html: str


CORE_TOOL_CONTRACTS = {
    "profile_dataset": (ProfileDatasetInput, ProfileDatasetOutput),
    "parse_rule_library": (ParseRuleLibraryInput, ParseRuleLibraryOutput),
    "apply_rules": (ApplyRulesInput, ApplyRulesOutput),
    "validate_analysis": (ValidateAnalysisInput, ValidateAnalysisOutput),
    "write_report": (WriteReportInput, WriteReportOutput),
}


class ToolContractError(ValueError):
    """Boundary validation failure; the original ValidationError is __cause__."""

    def __init__(self, message: str, *, stage: str, code: str) -> None:
        super().__init__(message)
        self.stage = stage
        self.code = code


class ToolArgumentError(TypeError):
    """Signature mismatch, retaining TypeError compatibility and its cause."""
    stage = "input"
    code = "INVALID_TOOL_INPUT"


def versioned_tool(function):
    """Validate inputs/outputs while preserving positional calls and flat results."""
    input_model, output_model = CORE_TOOL_CONTRACTS[function.__name__]
    call_signature = signature(function)

    @wraps(function)
    def wrapped(self, *args, **kwargs):
        version = kwargs.pop("schema_version", TOOL_SCHEMA_VERSION)
        try:
            bound = call_signature.bind(self, *args, **kwargs)
        except TypeError as exc:
            raise ToolArgumentError(str(exc)) from exc
        arguments = {key: value for key, value in bound.arguments.items() if key != "self"}
        try:
            # bool compares equal to 1 in Python; explicitly reject it as a version.
            if type(version) is not int or version != TOOL_SCHEMA_VERSION:
                raise ToolContractError(f"Unsupported schema_version: {version!r}", stage="input", code="UNSUPPORTED_SCHEMA_VERSION")
            input_model.model_validate({**arguments, "schema_version": version})
        except ValidationError as exc:
            raise ToolContractError(str(exc), stage="input", code="INVALID_TOOL_INPUT") from exc
        result = function(self, *args, **kwargs)
        try:
            output_model.model_validate(result)
        except ValidationError as exc:
            raise ToolContractError(str(exc), stage="output", code="INVALID_TOOL_OUTPUT") from exc
        # Preserve original optional fields and nested payloads, adding only version.
        return {"schema_version": TOOL_SCHEMA_VERSION, **result}

    return wrapped
