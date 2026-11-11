"""Environment-driven settings for the MVP.

配置来源保持简单：启动时先尝试加载工作目录下的 .env，再从环境变量读取。
显式 export 的系统环境变量优先级更高，便于后续容器化/生产部署覆盖本地配置。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path


def _find_project_root() -> Path:
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if (parent / "configs").is_dir() and (parent / "pyproject.toml").is_file():
            return parent
    return here.parents[3]


PROJECT_ROOT = _find_project_root()


_ENV_LOADED = False

LLM_API_PROTOCOL_RESPONSES = "responses"
LLM_API_PROTOCOL_CHAT_COMPLETIONS = "chat_completions"


@dataclass(frozen=True)
class Settings:
    """Runtime configuration shared by app, CLI, and AgentRunner.

    这个对象是整个系统的“运行配置快照”。FastAPI、CLI 和 AgentRunner 都只
    读取这个对象，而不是到处直接读环境变量，这样后续改成本地模型、百炼、
    OpenAI 或离线 demo 时，只需要集中改这里的解析逻辑。
    """

    storage_dir: Path
    provider: str
    openai_api_key: str | None
    openai_model: str
    base_url: str | None
    force_openai: bool
    max_tool_steps: int
    api_protocol: str = LLM_API_PROTOCOL_RESPONSES
    model_input_policy: str | None = None


MODEL_RUN_MODE_LOCAL = "local"
MODEL_RUN_MODE_AUTO = "model_auto"
MODEL_RUN_MODE_REQUIRED = "model_required"


def build_run_settings(
    base: Settings,
    *,
    mode: str,
    provider: str | None = None,
    model: str | None = None,
) -> tuple[Settings, dict[str, str | bool | None]]:
    """Build per-run model settings from a front-end task selection.

    `.env` 仍然是全局默认配置；前端提交的 provider/model 只影响本次运行。
    这样客户演示时可以一单一选，而不会改动服务器进程的全局配置。
    """
    run_mode = _normalize_run_mode(mode)
    selected_model = (model or "").strip()

    if run_mode == MODEL_RUN_MODE_LOCAL:
        local_model = selected_model or base.openai_model
        run_settings = replace(
            base,
            provider="local_demo",
            openai_api_key=None,
            openai_model=local_model,
            base_url=None,
            force_openai=False,
        )
        return run_settings, {
            "mode": MODEL_RUN_MODE_LOCAL,
            "mode_label": "本地稳定模式",
            "provider": "local_demo",
            "model": local_model,
            "base_url": None,
            "api_protocol": base.api_protocol,
            "model_api_available": False,
            "force_model_api": False,
        }

    selected_provider = _normalize_provider(provider or base.provider)
    api_key, default_model, base_url = _resolve_model_config(selected_provider)
    selected_model = selected_model or default_model or base.openai_model
    force_model_api = run_mode == MODEL_RUN_MODE_REQUIRED
    run_settings = replace(
        base,
        provider=selected_provider,
        openai_api_key=api_key,
        openai_model=selected_model,
        base_url=base_url,
        force_openai=force_model_api,
    )
    return run_settings, {
        "mode": run_mode,
        "mode_label": "强制外部模型" if force_model_api else "外部模型优先",
        "provider": selected_provider,
        "model": selected_model,
        "base_url": base_url,
        "api_protocol": base.api_protocol,
        "model_api_available": bool(api_key),
        "force_model_api": force_model_api,
    }


def get_settings() -> Settings:
    """Read settings from .env/environment variables with MVP-safe defaults.

    读取顺序是：
    1. 尝试加载当前目录下的 `.env`；
    2. 读取已经 export 的系统环境变量；
    3. 没有显式配置时使用 MVP 默认值。

    注意：python-dotenv 的 `override=False` 会让 shell 中 export 的值优先，
    这对服务器部署很重要，因为生产环境通常由平台注入环境变量。
    """
    load_local_env()
    # storage_dir 是所有运行结果、上传文件、报告和证据链的根目录。
    # 默认写入 outputs/runs/<run_id>/；仍可用 AGENT_MVP_STORAGE_DIR 覆盖（如旧的 runtime/）。
    storage_dir = Path(os.getenv("AGENT_MVP_STORAGE_DIR", "outputs/runs")).resolve()
    # provider 只决定“如何解析模型配置”；真实工具执行仍然在本地。
    provider = os.getenv("LLM_PROVIDER", os.getenv("MODEL_PROVIDER", "openai")).lower()
    # force_openai=True 时，如果 API 不可用就直接报错；默认 False 更适合 demo，
    # 因为 API 出问题时仍可回退到本地 deterministic Agent 流程。
    force_openai = (
        os.getenv("AGENT_MVP_FORCE_MODEL_API", os.getenv("AGENT_MVP_FORCE_OPENAI", "0")).lower()
        in {"1", "true", "yes"}
    )
    max_tool_steps = int(os.getenv("AGENT_MVP_MAX_TOOL_STEPS", "10"))
    api_protocol = _normalize_api_protocol(os.getenv("LLM_API_PROTOCOL"))
    api_key, model, base_url = _resolve_model_config(provider)
    return Settings(
        storage_dir=storage_dir,
        provider=provider,
        openai_api_key=api_key,
        openai_model=model,
        base_url=base_url,
        force_openai=force_openai,
        max_tool_steps=max_tool_steps,
        api_protocol=api_protocol,
        model_input_policy=os.getenv("AGENT_MVP_MODEL_INPUT_POLICY") or None,
    )


def load_local_env(env_path: str | Path = ".env") -> None:
    """Load local .env once, without overriding already-exported variables.

    This lets ``uvicorn agent_mvp.app:app`` pick up .env automatically. If a
    deployment platform or shell already provides a variable, that value wins.
    """
    global _ENV_LOADED
    if _ENV_LOADED:
        # 配置只在进程启动/首次读取时加载一次。修改 .env 后需要重启服务，
        # 否则运行中的 FastAPI 进程不会自动刷新配置。
        return
    _ENV_LOADED = True

    path = Path(env_path)
    if not path.exists():
        return

    try:
        from dotenv import load_dotenv
    except ImportError:
        _load_env_file_manually(path)
        return

    load_dotenv(dotenv_path=path, override=False)


def _load_env_file_manually(path: Path) -> None:
    """Tiny fallback parser used if python-dotenv is unavailable.

    这里只支持最简单的 KEY=VALUE 形式，是为了让项目在缺少 python-dotenv
    时仍然可运行；复杂 `.env` 语法仍建议安装 python-dotenv。
    """
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def _normalize_run_mode(mode: str | None) -> str:
    value = (mode or MODEL_RUN_MODE_AUTO).strip().lower()
    aliases = {
        "offline": MODEL_RUN_MODE_LOCAL,
        "local_demo": MODEL_RUN_MODE_LOCAL,
        "demo": MODEL_RUN_MODE_LOCAL,
        "auto": MODEL_RUN_MODE_AUTO,
        "model": MODEL_RUN_MODE_AUTO,
        "external": MODEL_RUN_MODE_AUTO,
        "required": MODEL_RUN_MODE_REQUIRED,
        "force": MODEL_RUN_MODE_REQUIRED,
        "strict": MODEL_RUN_MODE_REQUIRED,
    }
    value = aliases.get(value, value)
    if value not in {MODEL_RUN_MODE_LOCAL, MODEL_RUN_MODE_AUTO, MODEL_RUN_MODE_REQUIRED}:
        return MODEL_RUN_MODE_AUTO
    return value


def _normalize_api_protocol(value: str | None) -> str:
    normalized = (value or LLM_API_PROTOCOL_RESPONSES).strip().lower().replace("-", "_")
    aliases = {
        "response": LLM_API_PROTOCOL_RESPONSES,
        "responses_api": LLM_API_PROTOCOL_RESPONSES,
        "chat": LLM_API_PROTOCOL_CHAT_COMPLETIONS,
        "chat_completion": LLM_API_PROTOCOL_CHAT_COMPLETIONS,
        "chat_completions_api": LLM_API_PROTOCOL_CHAT_COMPLETIONS,
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in {LLM_API_PROTOCOL_RESPONSES, LLM_API_PROTOCOL_CHAT_COMPLETIONS}:
        return LLM_API_PROTOCOL_RESPONSES
    return normalized


def _normalize_provider(provider: str | None) -> str:
    value = (provider or "openai").strip().lower()
    aliases = {
        "model-studio": "bailian",
        "model_studio": "bailian",
        "dashscope": "bailian",
        "aliyun": "bailian",
        "custom": "custom_openai",
        "openai_compatible": "custom_openai",
    }
    return aliases.get(value, value)


def _resolve_model_config(provider: str) -> tuple[str | None, str, str | None]:
    """Resolve provider-specific API key, model, and OpenAI-compatible base_url.

    百炼（Model Studio/DashScope）提供 OpenAI 兼容接口，所以项目仍然使用
    openai Python SDK；区别只在 api_key、base_url 和 model。
    """
    # 通用变量用于兼容“非 OpenAI 但兼容 OpenAI API 格式”的模型服务。
    generic_api_key = os.getenv("LLM_API_KEY")
    generic_model = os.getenv("LLM_MODEL")
    generic_base_url = os.getenv("LLM_BASE_URL")

    if provider in {"bailian", "dashscope", "modelstudio", "model_studio"}:
        # 百炼/通义走 OpenAI-compatible 接口，因此后续 Agent 仍然可以复用
        # OpenAI Python SDK，只是 base_url、api_key 和 model 名不同。
        return (
            os.getenv("DASHSCOPE_API_KEY") or os.getenv("BAILIAN_API_KEY") or generic_api_key,
            os.getenv("BAILIAN_MODEL") or os.getenv("DASHSCOPE_MODEL") or generic_model or "qwen-plus",
            os.getenv("BAILIAN_BASE_URL")
            or os.getenv("DASHSCOPE_BASE_URL")
            or generic_base_url
            or "https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

    # 默认 provider=openai；也允许通过 LLM_* 变量接入本地 OpenAI-compatible
    # 服务，例如 vLLM 或其他本地推理网关。
    return (
        os.getenv("OPENAI_API_KEY") or generic_api_key,
        os.getenv("OPENAI_MODEL") or generic_model or "gpt-5.5",
        os.getenv("OPENAI_BASE_URL") or generic_base_url,
    )
