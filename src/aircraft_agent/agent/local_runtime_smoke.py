"""Smoke and benchmark helpers for real local model runtimes."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aircraft_agent.common.utils import read_json, utc_now_iso, write_json, write_text_atomic
from aircraft_agent.telemetry.demo_data import DEMO_DATASET_PATH, DEMO_RULE_PATH


TIMEOUT_EXIT_CODE = 124
REQUESTED_PROVIDER = "custom_openai"
REQUIRED_CRITICAL_TOOLS = (
    "profile_dataset",
    "parse_rule_library",
    "apply_rules",
    "validate_analysis",
)


@dataclass(frozen=True)
class RuntimeCandidate:
    label: str
    base_url: str


DEFAULT_RUNTIME_CANDIDATES = [
    RuntimeCandidate(label="ollama", base_url="http://127.0.0.1:11434/v1"),
    RuntimeCandidate(label="vllm", base_url="http://127.0.0.1:8000/v1"),
    RuntimeCandidate(label="lm_studio", base_url="http://127.0.0.1:1234/v1"),
]


@dataclass(frozen=True)
class SmokeConfig:
    base_url: str
    model: str
    api_key: str = "local"
    protocol: str = "chat_completions"
    runtime_label: str = "local_runtime"
    output_dir: Path = Path("docs/local_runtime_benchmarks")
    timeout: int = 180
    force_model_api: bool = False


def probe_models_endpoint(base_url: str, timeout: float = 2.0) -> bool:
    url = base_url.rstrip("/") + "/models"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return 200 <= int(response.status) < 500
    except Exception:
        return False


def discover_runtime_candidates(
    *,
    candidates: list[RuntimeCandidate] | None = None,
    probe=probe_models_endpoint,
) -> list[RuntimeCandidate]:
    selected_candidates = DEFAULT_RUNTIME_CANDIDATES if candidates is None else candidates
    return [candidate for candidate in selected_candidates if probe(candidate.base_url)]


def write_blocked_result(output_dir: Path, *, checked: list[str]) -> dict[str, str]:
    summary = {
        "record_id": str(uuid.uuid4()),
        "recorded_at": utc_now_iso(),
        "run_id": None,
        "runtime_label": None,
        "model": None,
        "base_url": None,
        "elapsed_seconds": 0.0,
        "chain_hash": None,
        "verify_chain_valid": False,
        "verify_chain_verified": False,
        "validation_status": "not_run",
        "fallback_detected": False,
        "status": "blocked",
        "benchmark_success": False,
        "benchmark_verdict": "BLOCKED",
        "reason": "No reachable local runtime was detected.",
        "checked": checked,
        "notes": ["No reachable local runtime; harness and fake tests can still be completed."],
    }
    return write_smoke_outputs(summary, output_dir)


def extract_run_summary(
    run_dir: Path,
    *,
    elapsed_seconds: float,
    runtime_label: str,
    model: str,
    base_url: str,
    requested_api_protocol: str | None = None,
) -> dict[str, Any]:
    manifest = _read_json_object(run_dir / "manifest.json")
    metadata_path = run_dir / "run_metadata.json"
    evidence_path = run_dir / "evidence_chain.json"
    metadata = _read_json_object(metadata_path) if metadata_path.exists() else {}
    evidence = _read_json_object(evidence_path) if evidence_path.exists() else {}
    validation_results = manifest.get("validation_results") or []
    if not isinstance(validation_results, list):
        raise ValueError("manifest.json validation_results must be a list")
    validation = validation_results[-1] if validation_results else {}
    if not isinstance(validation, dict):
        raise ValueError("manifest.json validation result must be an object")
    evidence_verification = manifest.get("evidence_verification") or validation.get("evidence_verification") or {}
    if not isinstance(evidence_verification, dict):
        raise ValueError("manifest.json evidence_verification must be an object")
    nodes = evidence.get("nodes") or []
    if not isinstance(nodes, list) or any(not isinstance(node, dict) for node in nodes):
        raise ValueError("evidence_chain.json nodes must be a list of objects")
    metadata_evidence = metadata.get("evidence") or {}
    if not isinstance(metadata_evidence, dict):
        raise ValueError("run_metadata.json evidence must be an object")
    agent_trace = manifest.get("agent_trace") or []
    if not isinstance(agent_trace, list):
        raise ValueError("manifest.json agent_trace must be a list")
    missing_tools = validation.get("missing_tools") or []
    if not isinstance(missing_tools, list):
        raise ValueError("manifest.json missing_tools must be a list")
    run_status = manifest.get("status")
    run_id = manifest.get("run_id") or run_dir.name
    recorded_at = manifest.get("updated_at") or manifest.get("created_at")
    if run_status == "complete" and not recorded_at:
        raise ValueError("completed manifest.json must include updated_at or created_at")
    report_html_path = manifest.get("report_html_path")
    report_markdown_path = manifest.get("report_markdown_path")
    report_html_present = _path_present(report_html_path, run_dir)
    report_markdown_present = _path_present(report_markdown_path, run_dir)
    fallback_detected = any(_is_model_fallback(node) for node in nodes)
    provider_response_nodes = [
        node
        for node in nodes
        if node.get("type") == "model_call" and node.get("model_response_id") and not _is_model_fallback(node)
    ]
    provider_model_response_count = len(provider_response_nodes)
    observed_arguments = provider_response_nodes[0].get("arguments") if provider_response_nodes else {}
    if not isinstance(observed_arguments, dict):
        observed_arguments = {}
    observed_config = {
        "provider": observed_arguments.get("provider"),
        "model": observed_arguments.get("model"),
        "base_url": observed_arguments.get("base_url"),
        "api_protocol": observed_arguments.get("api_protocol"),
    }
    requested_config = {
        "provider": REQUESTED_PROVIDER,
        "model": model,
        "base_url": base_url,
        "api_protocol": requested_api_protocol,
    }
    runtime_config_mismatch_reasons = _runtime_config_mismatch_reasons(
        requested_config,
        observed_config,
    )
    called_tool_names = list(
        dict.fromkeys(
            str(node.get("tool_name"))
            for node in nodes
            if node.get("type") == "tool_call" and node.get("tool_name")
        )
    )
    required_critical_tools = list(REQUIRED_CRITICAL_TOOLS)
    missing_critical_tools = [
        tool_name for tool_name in required_critical_tools if tool_name not in called_tool_names
    ]
    chain_hash = (
        manifest.get("chain_hash")
        or metadata_evidence.get("chain_hash")
        or evidence.get("chain_hash")
    )
    notes: list[str] = []
    if missing_tools:
        notes.append(f"missing_tools={missing_tools}")
    if missing_critical_tools:
        notes.append(f"missing_critical_tools={missing_critical_tools}")
    if not evidence_verification.get("valid"):
        notes.append("evidence verification did not pass")
    if not evidence_verification.get("verified"):
        notes.append("evidence verification was not verified")
    if not chain_hash:
        notes.append("chain_hash missing")
    notes.extend(runtime_config_mismatch_reasons)

    summary = {
        "record_id": run_id,
        "recorded_at": recorded_at,
        "run_id": run_id,
        "runtime_label": runtime_label,
        "model": model,
        "base_url": base_url,
        "api_protocol": requested_api_protocol,
        "requested_provider": REQUESTED_PROVIDER,
        "requested_model": model,
        "requested_base_url": base_url,
        "requested_api_protocol": requested_api_protocol,
        "observed_provider": observed_config["provider"],
        "observed_model": observed_config["model"],
        "observed_base_url": observed_config["base_url"],
        "observed_api_protocol": observed_config["api_protocol"],
        "runtime_config_matches": not runtime_config_mismatch_reasons,
        "runtime_config_mismatch_reasons": runtime_config_mismatch_reasons,
        "status": run_status,
        "run_status": run_status,
        "validation_status": validation.get("status"),
        "missing_tools": missing_tools,
        "verify_chain_valid": bool(evidence_verification.get("valid")),
        "verify_chain_verified": bool(evidence_verification.get("verified")),
        "chain_hash": chain_hash,
        "evidence_node_count": metadata_evidence.get("node_count") or len(nodes),
        "model_call_count": sum(1 for node in nodes if node.get("type") == "model_call"),
        "provider_model_response_count": provider_model_response_count,
        "tool_call_count": sum(1 for node in nodes if node.get("type") == "tool_call"),
        "called_tool_names": called_tool_names,
        "required_critical_tools": required_critical_tools,
        "missing_critical_tools": missing_critical_tools,
        "agent_step_count": len(agent_trace),
        "report_html_path": report_html_path,
        "report_markdown_path": report_markdown_path,
        "report_html_present": report_html_present,
        "report_markdown_present": report_markdown_present,
        "reports_present": report_html_present and report_markdown_present,
        "elapsed_seconds": round(float(elapsed_seconds), 3),
        "fallback_detected": fallback_detected,
        "notes": notes,
    }
    return _apply_benchmark_verdict(summary)


def write_smoke_outputs(summary: dict[str, Any], output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = summary.get("run_id")
    if run_id:
        stem = f"{summary.get('runtime_label', 'runtime')}_{summary.get('model', 'model')}_{run_id}"
    else:
        stem = summary.get("record_id") or (
            f"{summary.get('runtime_label', 'runtime')}_{summary.get('model', 'model')}_run"
        )
    safe_stem = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in stem)
    json_path = output_dir / f"{safe_stem}.json"
    markdown_path = output_dir / f"{safe_stem}.md"
    write_json(json_path, summary)
    write_text_atomic(markdown_path, _render_markdown(summary), encoding="utf-8")
    return {"json": str(json_path), "markdown": str(markdown_path)}


def run_cli_demo(config: SmokeConfig) -> tuple[int, str, str, float]:
    env = os.environ.copy()
    env.update(
        {
            "LLM_PROVIDER": "custom_openai",
            "LLM_API_PROTOCOL": config.protocol,
            "LLM_BASE_URL": config.base_url,
            "LLM_API_KEY": config.api_key,
            "LLM_MODEL": config.model,
            "AGENT_MVP_FORCE_MODEL_API": "1" if config.force_model_api else "0",
        }
    )
    start = time.perf_counter()
    command = [
        sys.executable,
        "-m",
        "aircraft_agent.interface.cli",
        "--rule",
        str(DEMO_RULE_PATH),
        "--dataset",
        str(DEMO_DATASET_PATH),
    ]
    try:
        completed = subprocess.run(
            command,
            cwd=Path.cwd(),
            env=env,
            text=True,
            capture_output=True,
            timeout=config.timeout,
        )
    except subprocess.TimeoutExpired as exc:
        elapsed = time.perf_counter() - start
        stdout = _bounded_text(exc.stdout or exc.output)
        partial_stderr = _bounded_text(exc.stderr)
        timeout_message = f"CLI smoke timed out after {config.timeout} seconds"
        stderr = "\n".join(part for part in (partial_stderr, timeout_message) if part)
        return TIMEOUT_EXIT_CODE, stdout, _bounded_text(stderr), elapsed
    elapsed = time.perf_counter() - start
    return completed.returncode, completed.stdout, completed.stderr, elapsed


def find_run_id(stdout: str) -> str | None:
    for line in stdout.splitlines():
        if line.startswith("运行 ID:"):
            return line.split(":", 1)[1].strip()
    return None


def run_smoke(config: SmokeConfig) -> dict[str, Any]:
    code, stdout, stderr, elapsed = run_cli_demo(config)
    run_id = find_run_id(stdout)
    if code != 0 or not run_id:
        reason = (
            f"CLI smoke timed out after {config.timeout} seconds"
            if code == TIMEOUT_EXIT_CODE
            else "CLI smoke failed or did not print a run id"
        )
        summary = _failed_attempt_summary(
            config,
            run_id=None,
            elapsed_seconds=elapsed,
            stdout=stdout,
            stderr=stderr,
            reason=reason,
            notes=[reason],
        )
        write_smoke_outputs(summary, config.output_dir)
        return summary
    run_dir = Path(os.environ.get("AGENT_MVP_STORAGE_DIR", "outputs/runs")).resolve() / run_id
    try:
        summary = extract_run_summary(
            run_dir,
            elapsed_seconds=elapsed,
            runtime_label=config.runtime_label,
            model=config.model,
            base_url=config.base_url,
            requested_api_protocol=config.protocol,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        extraction_error = f"{type(exc).__name__}: {exc}"
        reason = f"Artifact extraction failed for run {run_id}."
        summary = _failed_attempt_summary(
            config,
            run_id=run_id,
            elapsed_seconds=elapsed,
            stdout=stdout,
            stderr=stderr,
            reason=reason,
            notes=[f"Artifact extraction failed: {extraction_error}"],
            artifact_extraction_error=extraction_error,
        )
        write_smoke_outputs(summary, config.output_dir)
        return summary
    summary["stdout"] = _bounded_text(stdout)
    summary["stderr"] = _bounded_text(stderr)
    write_smoke_outputs(summary, config.output_dir)
    return summary


def _failed_attempt_summary(
    config: SmokeConfig,
    *,
    run_id: str | None,
    elapsed_seconds: float,
    stdout: Any,
    stderr: Any,
    reason: str,
    notes: list[str],
    artifact_extraction_error: str | None = None,
) -> dict[str, Any]:
    requested_config = {
        "provider": REQUESTED_PROVIDER,
        "model": config.model,
        "base_url": config.base_url,
        "api_protocol": config.protocol,
    }
    observed_config = {field: None for field in requested_config}
    runtime_config_mismatch_reasons = _runtime_config_mismatch_reasons(
        requested_config,
        observed_config,
    )
    return _apply_benchmark_verdict(
        {
            "record_id": str(uuid.uuid4()),
            "recorded_at": utc_now_iso(),
            "run_id": run_id,
            "runtime_label": config.runtime_label,
            "model": config.model,
            "base_url": config.base_url,
            "api_protocol": config.protocol,
            "requested_provider": REQUESTED_PROVIDER,
            "requested_model": config.model,
            "requested_base_url": config.base_url,
            "requested_api_protocol": config.protocol,
            "observed_provider": None,
            "observed_model": None,
            "observed_base_url": None,
            "observed_api_protocol": None,
            "runtime_config_matches": False,
            "runtime_config_mismatch_reasons": runtime_config_mismatch_reasons,
            "status": "failed",
            "run_status": "failed",
            "validation_status": None,
            "missing_tools": [],
            "verify_chain_valid": False,
            "verify_chain_verified": False,
            "chain_hash": None,
            "evidence_node_count": 0,
            "model_call_count": 0,
            "provider_model_response_count": 0,
            "tool_call_count": 0,
            "called_tool_names": [],
            "required_critical_tools": list(REQUIRED_CRITICAL_TOOLS),
            "missing_critical_tools": list(REQUIRED_CRITICAL_TOOLS),
            "agent_step_count": 0,
            "report_html_path": None,
            "report_markdown_path": None,
            "report_html_present": False,
            "report_markdown_present": False,
            "reports_present": False,
            "elapsed_seconds": round(float(elapsed_seconds), 3),
            "fallback_detected": False,
            "stdout": _bounded_text(stdout),
            "stderr": _bounded_text(stderr),
            "reason": reason,
            "artifact_extraction_error": artifact_extraction_error,
            "notes": notes,
        }
    )


def _render_markdown(summary: dict[str, Any]) -> str:
    if summary.get("status") == "blocked":
        lines = [
            "# Local Runtime Smoke Result",
            "",
            "- Verdict: **BLOCKED**",
            f"- Record ID: `{summary.get('record_id')}`",
            f"- Recorded at: `{summary.get('recorded_at')}`",
            f"- Reason: {summary.get('reason')}",
            f"- Checked: `{', '.join(summary.get('checked') or [])}`",
            f"- Run ID: `{summary.get('run_id')}`",
            f"- Runtime: `{summary.get('runtime_label')}`",
            f"- Model: `{summary.get('model')}`",
            f"- Base URL: `{summary.get('base_url')}`",
            f"- Status: `{summary.get('status')}`",
            f"- Validation status: `{summary.get('validation_status')}`",
            f"- verify_chain valid: `{summary.get('verify_chain_valid')}`",
            f"- verify_chain verified: `{summary.get('verify_chain_verified')}`",
            f"- chain_hash: `{summary.get('chain_hash')}`",
            f"- Elapsed seconds: `{summary.get('elapsed_seconds')}`",
            f"- Fallback detected: `{summary.get('fallback_detected')}`",
            "",
            "## Notes",
        ]
        notes = summary.get("notes") or []
        lines.extend([f"- {note}" for note in notes] or ["- None"])
        return "\n".join(lines) + "\n"

    verdict = summary.get("benchmark_verdict") or "FAIL"
    lines = [
        "# Local Runtime Smoke Result",
        "",
        f"- Verdict: **{verdict}**",
        f"- Benchmark success: `{summary.get('benchmark_success')}`",
        f"- Record ID: `{summary.get('record_id')}`",
        f"- Recorded at: `{summary.get('recorded_at')}`",
        f"- Runtime: `{summary.get('runtime_label')}`",
        f"- Model: `{summary.get('model')}`",
        f"- Base URL: `{summary.get('base_url')}`",
        f"- API protocol: `{summary.get('api_protocol')}`",
        f"- Requested provider: `{summary.get('requested_provider')}`",
        f"- Requested model: `{summary.get('requested_model')}`",
        f"- Requested base URL: `{summary.get('requested_base_url')}`",
        f"- Requested API protocol: `{summary.get('requested_api_protocol')}`",
        f"- Observed provider: `{summary.get('observed_provider')}`",
        f"- Observed model: `{summary.get('observed_model')}`",
        f"- Observed base URL: `{summary.get('observed_base_url')}`",
        f"- Observed API protocol: `{summary.get('observed_api_protocol')}`",
        f"- Runtime config matches: `{summary.get('runtime_config_matches')}`",
        f"- Runtime config mismatch reasons: `{_display_list(summary.get('runtime_config_mismatch_reasons'))}`",
        f"- Run ID: `{summary.get('run_id', 'unknown')}`",
        f"- Status: `{summary.get('status')}`",
        f"- Validation status: `{summary.get('validation_status')}`",
        f"- Validation missing tools: `{_display_list(summary.get('missing_tools'))}`",
        f"- verify_chain valid: `{summary.get('verify_chain_valid')}`",
        f"- verify_chain verified: `{summary.get('verify_chain_verified')}`",
        f"- chain_hash: `{summary.get('chain_hash')}`",
        f"- Elapsed seconds: `{summary.get('elapsed_seconds')}`",
        f"- Model calls: `{summary.get('model_call_count')}`",
        f"- Provider model responses: `{summary.get('provider_model_response_count')}`",
        f"- Tool calls: `{summary.get('tool_call_count')}`",
        f"- Called tools: `{_display_list(summary.get('called_tool_names'))}`",
        f"- Required critical tools: `{_display_list(summary.get('required_critical_tools'))}`",
        f"- Missing critical tools: `{_display_list(summary.get('missing_critical_tools'))}`",
        f"- Agent steps: `{summary.get('agent_step_count')}`",
        f"- Reports present: `{summary.get('reports_present')}`",
        f"- Fallback detected: `{summary.get('fallback_detected')}`",
        f"- Reason: {summary.get('reason') or 'None'}",
        f"- Artifact extraction error: {summary.get('artifact_extraction_error') or 'None'}",
        "",
        "## Benchmark failure reasons",
    ]
    reasons = summary.get("benchmark_failure_reasons") or []
    lines.extend([f"- {reason}" for reason in reasons] or ["- None"])
    lines.extend([
        "",
        "## Notes",
    ])
    notes = summary.get("notes") or []
    lines.extend([f"- {note}" for note in notes] or ["- None"])
    return "\n".join(lines) + "\n"


def _display_list(values: Any) -> str:
    return ", ".join(str(value) for value in (values or [])) or "None"


def _read_json_object(path: Path) -> dict[str, Any]:
    value = read_json(path)
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return value


def _normalized_runtime_value(field: str, value: Any) -> Any:
    if field == "base_url" and isinstance(value, str):
        return value.rstrip("/")
    return value


def _runtime_config_mismatch_reasons(
    requested: dict[str, Any],
    observed: dict[str, Any],
) -> list[str]:
    reasons: list[str] = []
    for field in ("provider", "model", "base_url", "api_protocol"):
        requested_value = _normalized_runtime_value(field, requested.get(field))
        observed_value = _normalized_runtime_value(field, observed.get(field))
        if observed_value is None or observed_value == "":
            reasons.append(f"{field} missing from provider response evidence")
        elif observed_value != requested_value:
            reasons.append(
                f"{field} mismatch: requested {requested_value!r}, observed {observed_value!r}"
            )
    return reasons


def _path_present(value: Any, run_dir: Path) -> bool:
    if not value:
        return False
    path = Path(str(value))
    if path.is_file():
        return True
    return not path.is_absolute() and (run_dir / path).is_file()


def _is_model_fallback(node: dict[str, Any]) -> bool:
    if node.get("type") != "model_call":
        return False
    summary = str(node.get("summary", ""))
    arguments = node.get("arguments") or {}
    return "回退" in summary or "本地确定性" in summary or arguments.get("mode") == "local_demo"


def _apply_benchmark_verdict(summary: dict[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    if summary.get("status") != "complete":
        reasons.append("run did not complete")
    if summary.get("validation_status") != "passed":
        reasons.append("validation did not pass")
    missing_tools = summary.get("missing_tools") or []
    if missing_tools:
        reasons.append(f"validation reports missing tools: {_display_list(missing_tools)}")
    missing_critical_tools = summary.get("missing_critical_tools") or []
    if missing_critical_tools:
        reasons.append(f"critical tools missing: {_display_list(missing_critical_tools)}")
    if not summary.get("verify_chain_valid"):
        reasons.append("evidence verification did not pass")
    if not summary.get("verify_chain_verified"):
        reasons.append("evidence verification was not verified")
    if not str(summary.get("chain_hash") or "").strip():
        reasons.append("chain hash missing")
    if not summary.get("reports_present"):
        reasons.append("reports missing")
    if int(summary.get("provider_model_response_count") or 0) < 1:
        reasons.append("no provider model response")
    if not summary.get("runtime_config_matches"):
        reasons.append("runtime configuration mismatch")
    if summary.get("fallback_detected"):
        reasons.append("deterministic fallback detected")
    summary["benchmark_failure_reasons"] = reasons
    summary["benchmark_success"] = not reasons
    summary["benchmark_verdict"] = "PASS" if not reasons else "FAIL"
    return summary


def _bounded_text(value: Any, limit: int = 4000) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        text = value.decode("utf-8", errors="replace")
    else:
        text = str(value)
    return text[-limit:]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run local runtime smoke benchmark for aircraft-agent.")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key", default="local")
    parser.add_argument("--protocol", default="chat_completions")
    parser.add_argument("--runtime-label", default="local_runtime")
    parser.add_argument("--output-dir", type=Path, default=Path("docs/local_runtime_benchmarks"))
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--force-model-api", action="store_true")
    args = parser.parse_args(argv)
    summary = run_smoke(
        SmokeConfig(
            base_url=args.base_url,
            model=args.model,
            api_key=args.api_key,
            protocol=args.protocol,
            runtime_label=args.runtime_label,
            output_dir=args.output_dir,
            timeout=args.timeout,
            force_model_api=args.force_model_api,
        )
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary.get("benchmark_success") else 1


if __name__ == "__main__":
    raise SystemExit(main())
