"""Minimal structure tests for the aircraft_agent package layout (Stage B).

These assert the package/import surface only — no feature behavior.
"""

import aircraft_agent


def test_package_imports_and_has_version() -> None:
    assert hasattr(aircraft_agent, "__version__")


def test_agent_runner_importable() -> None:
    from aircraft_agent.agent.runner import AgentRunner

    assert AgentRunner is not None


def test_tool_executor_importable() -> None:
    from aircraft_agent.tools.executor import ToolExecutor

    assert ToolExecutor is not None


def test_project_root_and_configs_resolve() -> None:
    from aircraft_agent.common.config import PROJECT_ROOT

    assert (PROJECT_ROOT / "configs" / "telemetry_schema.yaml").exists()


def test_agent_mvp_shim_bridges_to_canonical_package() -> None:
    from agent_mvp.tools import ToolExecutor as ShimToolExecutor
    from aircraft_agent.tools.executor import ToolExecutor as RealToolExecutor

    assert ShimToolExecutor is RealToolExecutor
