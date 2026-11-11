"""Contract compatibility, failure evidence and deterministic presentation gates."""
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from aircraft_agent.agent.runtime import RunStore
from aircraft_agent.common.tool_contracts import (
    CORE_TOOL_CONTRACTS, ProfileDatasetInput, ToolContractError, ToolFailure,
    versioned_tool,
)
from aircraft_agent.evidence.chain import verify_chain
from aircraft_agent.tools.catalog import OPENAI_TOOL_SCHEMAS
from aircraft_agent.tools.executor import ToolExecutor


@pytest.fixture
def executor(tmp_path):
    store = RunStore(tmp_path)
    context = store.create_run("contract compatibility")
    store.save_bytes(context, data=b"temperature,status\n82,WARN\n70,OK\n", original_name="data.csv", kind="dataset")
    store.save_bytes(context, data=b"R001: temperature > 80\n", original_name="rules.md", kind="rule_library")
    return ToolExecutor(context)


@pytest.mark.parametrize("name", CORE_TOOL_CONTRACTS)
def test_core_legacy_and_explicit_version_outputs(executor, name):
    arguments = {"required_tools": ["profile_dataset"]} if name == "validate_analysis" else {}
    legacy = executor.execute(name, arguments, record=False)
    explicit = executor.execute(name, {**arguments, "schema_version": 1}, record=False)
    assert legacy == explicit
    assert legacy["schema_version"] == 1
    CORE_TOOL_CONTRACTS[name][1].model_validate(legacy)
    schema = next(item for item in OPENAI_TOOL_SCHEMAS if item["name"] == name)
    assert schema["parameters"]["properties"]["schema_version"]["enum"] == [1]
    assert "schema_version" not in schema["parameters"].get("required", [])


def test_direct_positional_alias_and_old_import(executor):
    from agent_mvp.tools import ToolExecutor as OldExecutor
    assert OldExecutor is ToolExecutor
    dataset_id = executor.context.dataset_file_ids[0]
    assert executor.profile_dataset(dataset_id) == executor.profile_dataset(dataset_id, schema_version=1)
    assert executor.execute("apply_rule_library", {"schema_version": 1}, record=False) == executor.apply_rules(dataset_id)
    # Existing fallback selection of unknown file IDs is intentionally unchanged.
    assert executor.profile_dataset("old-unknown-id") == executor.profile_dataset(dataset_id)


@pytest.mark.parametrize("arguments,code", [
    ({"schema_version": 2}, "UNSUPPORTED_SCHEMA_VERSION"),
    ({"schema_version": True}, "UNSUPPORTED_SCHEMA_VERSION"),
    ({"schema_version": "1"}, "UNSUPPORTED_SCHEMA_VERSION"),
    ({"dataset_id": 12}, "INVALID_TOOL_INPUT"),
    ({"unexpected": "x"}, "INVALID_TOOL_INPUT"),
])
def test_input_failures_do_not_record_success(executor, arguments, code):
    before = len(executor.context.evidence.nodes)
    with pytest.raises((ValueError, TypeError)):
        executor.execute("profile_dataset", arguments)
    events = [json.loads(line) for line in executor.context.event_path.read_text().splitlines()]
    failure = events[-1]["payload"]["failure"]
    assert failure["code"] == code
    assert failure["stage"] == "input"
    assert failure["schema_version"] == 1
    assert failure["exception_type"] and failure["traceback"]
    assert not executor.context.tool_results
    assert len(executor.context.evidence.nodes) == before
    assert events[-1]["type"] == "tool_failed"


@pytest.mark.parametrize("version", [2, True, "1", 1.0])
def test_input_model_rejects_invalid_versions(version):
    with pytest.raises(ValidationError):
        ProfileDatasetInput(schema_version=version)


def test_execution_failure_preserves_original_exception_and_cause(executor, monkeypatch):
    cause = OSError("original disk error")
    problem = RuntimeError("dataset read failed")
    problem.__cause__ = cause
    def fail(**kwargs):
        raise problem
    monkeypatch.setitem(executor.registry, "profile_dataset", fail)
    with pytest.raises(RuntimeError) as caught:
        executor.execute("profile_dataset")
    assert caught.value is problem
    assert caught.value.__cause__ is cause
    event = json.loads(executor.context.event_path.read_text().splitlines()[-1])
    failure = ToolFailure.model_validate(event["payload"]["failure"])
    assert failure.code == "TOOL_EXECUTION_FAILED"
    assert failure.stage == "execution"
    assert "original disk error" in failure.traceback
    assert "dataset read failed" in failure.traceback
    assert not executor.context.tool_results


def test_unknown_tool_and_record_false(executor):
    with pytest.raises(ValueError, match="Unknown tool"):
        executor.execute("not_registered")
    event = json.loads(executor.context.event_path.read_text().splitlines()[-1])
    assert event["payload"]["failure"]["code"] == "UNKNOWN_TOOL"
    assert event["payload"]["failure"]["stage"] == "dispatch"
    original = executor.context.event_path.read_bytes()
    with pytest.raises(ValueError):
        executor.execute("not_registered", record=False)
    assert executor.context.event_path.read_bytes() == original


def test_malformed_output_fails_before_success_evidence(executor, monkeypatch):
    def profile_dataset(self, dataset_id=None):
        return {"summary": "looks successful but lacks profile"}
    monkeypatch.setitem(executor.registry, "profile_dataset", versioned_tool(profile_dataset).__get__(executor))
    with pytest.raises(ToolContractError) as caught:
        executor.execute("profile_dataset")
    assert isinstance(caught.value.__cause__, ValidationError)
    event = json.loads(executor.context.event_path.read_text().splitlines()[-1])
    assert event["payload"]["failure"]["stage"] == "output"
    assert event["payload"]["failure"]["code"] == "INVALID_TOOL_OUTPUT"
    assert not executor.context.tool_results


def test_domain_failure_remains_failed_and_rules_remain_unparseable(executor):
    result = executor.execute("validate_analysis", {"required_tools": ["profile_dataset"], "strict": True})
    assert result["status"] == "failed"
    assert result["failure"]["code"] == "ANALYSIS_VALIDATION_FAILED"
    assert result["errors"]
    executor.parse_rule_library()
    executor.context.rules[0].condition = "absent_field > 80"
    rules = executor.execute("apply_rules")
    entry = rules["rule_results"][0]
    assert entry["status"] == "unparseable"
    assert entry["failure"]["code"] == "RULE_UNPARSEABLE"
    assert entry["failure"]["stage"] == "rule_evaluation"
    assert entry["message"] == entry["failure"]["message"]
    assert verify_chain(executor.context.evidence.path).valid


def test_versioned_evidence_and_report_rewrite(executor):
    executor.execute("profile_dataset", ProfileDatasetInput().model_dump())
    result = executor.execute("write_report", {"schema_version": 1})
    recorded = executor.context.tool_results[-1]["result"]
    assert result == recorded
    assert all(item["result"]["schema_version"] == 1 for item in executor.context.tool_results)
    assert Path(result["report_html"]).exists()
    assert verify_chain(executor.context.evidence.path).valid


def test_verification_exception_retains_context(executor, monkeypatch):
    import aircraft_agent.tools.executor as module
    def fail(source):
        raise OSError("cannot read chain") from ValueError("underlying verification cause")
    monkeypatch.setattr(module, "verify_chain", fail)
    result = executor.validate_analysis(required_tools=["profile_dataset"])
    assert result["status"] == "failed"
    failure = result["evidence_verification"]["errors"][0]["failure"]
    assert failure["code"] == "EVIDENCE_VERIFICATION_FAILED"
    assert failure["stage"] == "evidence_verification"
    assert "underlying verification cause" in failure["traceback"]


@pytest.mark.parametrize("name", CORE_TOOL_CONTRACTS)
def test_advertised_input_fields_match_models(name):
    parameters = next(item["parameters"] for item in OPENAI_TOOL_SCHEMAS if item["name"] == name)
    assert set(parameters["properties"]) == set(CORE_TOOL_CONTRACTS[name][0].model_fields)
