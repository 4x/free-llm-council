import inspect
from pathlib import Path

import pytest

from llm_council import ConfigurationError, CouncilError, Settings, query, query_result
from llm_council.council import JUDGE_SYSTEM


def test_judge_prompt_env_and_override(tmp_path, monkeypatch):
    """Environment and per-call settings override the environment file."""
    env_file = tmp_path / "settings.env"
    env_file.write_text("JUDGE_PROMPT=prompts/judge.md\n", encoding="utf-8")
    monkeypatch.delenv("JUDGE_PROMPT", raising=False)
    settings = Settings.from_env(env_file)
    assert settings.judge_prompt == Path("prompts/judge.md")
    monkeypatch.setenv("JUDGE_PROMPT", "custom.txt")
    assert Settings.from_env(env_file).judge_prompt == Path("custom.txt")
    assert settings.with_overrides(judge_prompt="override.md").judge_prompt == Path("override.md")
    assert settings.with_overrides(judge_prompt="").judge_prompt is None


def test_external_prompt_is_used_by_every_fallback(config, router, tmp_path):
    """Fallback judges receive the same exact UTF-8 system prompt."""
    prompt = tmp_path / "judge.txt"
    text = "Custom judge instructions π\nPreserve this newline.\n"
    prompt.write_text(text, encoding="utf-8-sig")
    router.judge_failure = True
    with pytest.raises(CouncilError, match="fallbacks failed"):
        query_result("Question", settings=config, judge_prompt=prompt, no_save=True)
    judge_calls = [call for call in router.calls if call["messages"][0]["role"] == "system"]
    assert len(judge_calls) == 2
    assert all(call["messages"][0]["content"] == text for call in judge_calls)


@pytest.mark.parametrize("content", [None, b" \n", b"\xff"])
def test_bad_prompt_fails_before_network_or_storage(config, router, tmp_path, content):
    """Configured invalid files never silently select the built-in backup."""
    path = tmp_path / "judge.md"
    if content is not None:
        path.write_bytes(content)
    with pytest.raises(ConfigurationError, match="JUDGE_PROMPT"):
        query_result("Question", settings=config, judge_prompt=path)
    assert not router.gets and not router.calls
    assert not config.duckdb_path.exists()


def test_unset_prompt_uses_backup(config, router):
    """The built-in prompt remains available without external configuration."""
    query_result("Question", settings=config, no_save=True)
    assert router.calls[-1]["messages"][0]["content"] == JUDGE_SYSTEM


@pytest.mark.parametrize("api", [query, query_result])
def test_judge_prompt_is_explicit_optional_keyword_parameter(api):
    """Public signatures expose the override for IDEs and introspection."""
    parameter = inspect.signature(api).parameters["judge_prompt"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is None
    assert parameter.annotation == "str | Path | None"


@pytest.mark.parametrize("api", [query, query_result])
@pytest.mark.parametrize("path_type", [str, Path])
def test_public_api_judge_prompt_overrides_env_file(
    api, path_type, config, router, tmp_path, monkeypatch
):
    """A direct API path overrides .env and process environment at inference time."""
    env_file = tmp_path / "settings.env"
    env_file.write_text("JUDGE_PROMPT=missing-env-prompt.md\n", encoding="utf-8")
    prompt_path = tmp_path / "custom.md"
    prompt_path.write_text("Per-call judge instructions π", encoding="utf-8")
    monkeypatch.delenv("JUDGE_PROMPT", raising=False)
    options = dict(
        env_file=env_file,
        api_key=config.api_key,
        base_url=config.base_url,
        judge=config.judge,
        router_models=config.configured_routers,
        request_interval=0,
        max_retries=0,
        no_save=True,
    )
    for environment_prompt in (None, "missing-process-prompt.md"):
        if environment_prompt is not None:
            monkeypatch.setenv("JUDGE_PROMPT", environment_prompt)
        result = api("Question", judge_prompt=path_type(prompt_path), **options)
        synthesis = result if isinstance(result, str) else result.synthesis
        assert synthesis == "Synthesized fixture answer"
        assert router.calls[-1]["messages"][0]["content"] == "Per-call judge instructions π"


@pytest.mark.parametrize("api", [query, query_result])
def test_public_api_none_preserves_settings_and_blank_selects_backup(api, config, router, tmp_path):
    """Defaults do not erase configured prompts; blank explicitly selects the backup."""
    prompt_path = tmp_path / "custom.txt"
    prompt_path.write_text("Configured instructions", encoding="utf-8")
    settings = config.with_overrides(judge_prompt=prompt_path)
    api("Question", settings=settings, no_save=True)
    assert router.calls[-1]["messages"][0]["content"] == "Configured instructions"
    api("Question", settings=settings, judge_prompt=None, no_save=True)
    assert router.calls[-1]["messages"][0]["content"] == "Configured instructions"
    api("Question", settings=settings, judge_prompt="", no_save=True)
    assert router.calls[-1]["messages"][0]["content"] == JUDGE_SYSTEM


def test_invalid_path_is_a_configuration_error(config, router):
    """Invalid filesystem paths must not escape the public error contract."""
    with pytest.raises(ConfigurationError, match="JUDGE_PROMPT"):
        query("Question", settings=config, judge_prompt="invalid\x00prompt.md", no_save=True)
    assert not router.gets and not router.calls


def test_example_prompt_is_sent_unchanged(config, router):
    """The editable external prompt need not match the built-in backup."""
    path = Path(__file__).resolve().parents[1] / "prompts" / "judge.md"
    text = path.read_text(encoding="utf-8-sig")
    assert text.strip()
    query("Question", settings=config, judge_prompt=path, no_save=True)
    assert router.calls[-1]["messages"][0]["content"] == text
