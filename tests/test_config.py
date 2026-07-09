from agent_mvp import config
from agent_mvp.config import MODEL_RUN_MODE_LOCAL, MODEL_RUN_MODE_REQUIRED, build_run_settings, get_settings, load_local_env


def test_bailian_config(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "bailian")
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
    monkeypatch.setenv("BAILIAN_MODEL", "qwen-plus")
    monkeypatch.delenv("BAILIAN_BASE_URL", raising=False)

    settings = get_settings()

    assert settings.provider == "bailian"
    assert settings.openai_api_key == "test-key"
    assert settings.openai_model == "qwen-plus"
    assert settings.base_url == "https://dashscope.aliyuncs.com/compatible-mode/v1"


def test_load_local_env_file(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.delenv("BAILIAN_MODEL", raising=False)
    monkeypatch.setattr(config, "_ENV_LOADED", False)
    env_path = tmp_path / ".env"
    env_path.write_text(
        "LLM_PROVIDER=bailian\n"
        "DASHSCOPE_API_KEY=test-key-from-env-file\n"
        "BAILIAN_MODEL=qwen-turbo\n",
        encoding="utf-8",
    )

    load_local_env(env_path)
    settings = get_settings()

    assert settings.provider == "bailian"
    assert settings.openai_api_key == "test-key-from-env-file"
    assert settings.openai_model == "qwen-turbo"


def test_build_run_settings_from_frontend_selection(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-default")
    monkeypatch.setenv("DASHSCOPE_API_KEY", "bailian-key")
    monkeypatch.setenv("BAILIAN_MODEL", "qwen-default")
    base = get_settings()

    local_settings, local_config = build_run_settings(base, mode=MODEL_RUN_MODE_LOCAL, provider="openai", model="gpt-test")
    required_settings, required_config = build_run_settings(
        base,
        mode=MODEL_RUN_MODE_REQUIRED,
        provider="bailian",
        model="qwen-plus",
    )

    assert local_settings.provider == "local_demo"
    assert local_settings.openai_api_key is None
    assert local_settings.force_openai is False
    assert local_config["mode"] == MODEL_RUN_MODE_LOCAL
    assert local_config["model_api_available"] is False
    assert required_settings.provider == "bailian"
    assert required_settings.openai_api_key == "bailian-key"
    assert required_settings.openai_model == "qwen-plus"
    assert required_settings.force_openai is True
    assert required_config["mode"] == MODEL_RUN_MODE_REQUIRED
    assert required_config["model_api_available"] is True
