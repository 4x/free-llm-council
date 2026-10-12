import json
import os
import subprocess
import sys
from pathlib import Path

import duckdb
import pytest
from conftest import model

from llm_council import cli
from llm_council.errors import ConfigurationError


@pytest.fixture
def cli_env(router):
    env = os.environ.copy()
    env.update(
        OPENROUTER_API_KEY="fixture-key",
        OPENROUTER_BASE_URL=router.url,
        REQUEST_INTERVAL="0",
        MAX_RETRIES="0",
        JUDGE="test/beta:free",
        SAVE_PROMPT="DUCKDB",
        SAVE_INITIAL_ANSWERS="NO",
        SAVE_SYNTHESIS="NO",
        LOG_LEVEL="INFO",
    )
    return env


def run_cli(args, cwd, env, module=False):
    command = (
        [sys.executable, "-m", "llm_council"]
        if module
        else [
            str(
                Path(sys.executable).parent
                / ("llm-council.exe" if os.name == "nt" else "llm-council")
            )
        ]
    )
    return subprocess.run(
        command + args,
        cwd=cwd,
        env=env,
        text=True,
        encoding="utf-8",
        capture_output=True,
        timeout=30,
    )


def test_installed_cli_question_markdown(cli_env, router, tmp_path):
    result = run_cli(
        ["--question", "Explain Riemann", "--save-initial-answers", "markdown"], tmp_path, cli_env
    )
    assert result.returncode == 0, result.stderr
    assert "Synthesized fixture answer" in result.stdout
    assert "INFO" in result.stderr
    assert "Starting council run" in result.stderr
    assert "beta synthesized 2 responses in" in result.stderr
    assert "Explain Riemann" not in result.stderr
    assert "fixture-key" not in result.stderr
    assert len(router.calls) == 3
    folders = list((tmp_path / "council-output").iterdir())
    assert len(folders) == 1 and len(list(folders[0].glob("initial-*.md"))) == 2
    assert folders[0].name.startswith("explain-riemann-")
    assert len(folders[0].name.rsplit("-", 1)[-1]) == 13
    assert folders[0].name.rsplit("-", 1)[-1].startswith("20")
    with duckdb.connect(str(tmp_path / "council.duckdb")) as db:
        assert db.execute("SELECT prompt FROM questions").fetchone() == ("Explain Riemann",)


def test_cli_followup_uses_cached_synthesis_in_messages(cli_env, router, tmp_path):
    first = run_cli(["-q", "Original question"], tmp_path, cli_env)
    assert first.returncode == 0, first.stderr
    cache = tmp_path / ".llm-council" / "latest-synthesis.md"
    previous = cache.read_text(encoding="utf-8")

    router.calls.clear()
    followup = run_cli(
        ["--followup", "Clarify the previous answer", "--no-save"], tmp_path, cli_env
    )
    assert followup.returncode == 0, followup.stderr
    assert router.calls[0]["messages"] == [
        {"role": "assistant", "content": previous},
        {"role": "user", "content": "Clarify the previous answer"},
    ]
    judge_call = next(call for call in router.calls if call["messages"][0]["role"] == "system")
    assert judge_call["messages"][1] == {"role": "assistant", "content": previous}
    assert judge_call["messages"][2]["role"] == "user"
    assert cache.read_text(encoding="utf-8") == previous


def test_followup_without_cache_fails_before_model_requests(cli_env, router, tmp_path):
    result = run_cli(["--followup", "Ask for details"], tmp_path, cli_env)
    assert result.returncode == 1
    assert "No previous synthesis is cached" in result.stderr
    assert router.calls == []


def test_copy_option_passes_exact_synthesis_to_clipboard(monkeypatch):
    copied: list[str] = []
    monkeypatch.setattr(cli, "query", lambda **_: "Exact **Markdown** synthesis")
    monkeypatch.setattr(cli, "_copy_to_clipboard", copied.append)
    monkeypatch.setattr(cli, "_cache_latest_synthesis", lambda _: None)

    assert cli.main(["-q", "Question", "--copy", "--no-save"]) == 0
    assert copied == ["Exact **Markdown** synthesis"]


def test_copy_to_clipboard_uses_windows_clipboard(monkeypatch):
    invocations = []
    monkeypatch.setattr(cli.sys, "platform", "win32")
    monkeypatch.setattr(cli.shutil, "which", lambda executable: "clip.exe")
    monkeypatch.setattr(
        cli.subprocess, "run", lambda command, **kwargs: invocations.append((command, kwargs))
    )

    cli._copy_to_clipboard("Exact synthesis")
    assert invocations == [
        (("clip",), {"input": "Exact synthesis", "text": True, "check": True, "timeout": 15})
    ]


def test_copy_without_clipboard_utility_is_actionable(monkeypatch):
    monkeypatch.setattr(cli.sys, "platform", "linux")
    monkeypatch.setattr(cli.shutil, "which", lambda _: None)
    with pytest.raises(ConfigurationError, match="No clipboard utility found"):
        cli._copy_to_clipboard("Synthesis")


def test_cli_renders_synthesis_as_markdown(cli_env, router, tmp_path):
    router.reply = {
        "model": "test/beta:free",
        "choices": [
            {
                "message": {"content": "# Rendered heading\n\n**Bold answer**"},
                "finish_reason": "stop",
            }
        ],
    }
    result = run_cli(["-q", "Question", "--no-save"], tmp_path, cli_env)
    assert result.returncode == 0, result.stderr
    assert "Rendered heading" in result.stdout
    assert "Bold answer" in result.stdout
    assert "# Rendered heading" not in result.stdout
    assert "**Bold answer**" not in result.stdout


def test_cli_prints_synthesis_despite_partial_failure_and_warnings_as_errors(
    cli_env, router, tmp_path
):
    router.fail_models.add("test/alpha:free")
    cli_env["PYTHONWARNINGS"] = "error"
    result = run_cli(["-q", "Question", "--no-save"], tmp_path, cli_env)
    assert result.returncode == 0, result.stderr
    assert "Synthesized fixture answer" in result.stdout
    assert "WARNING" in result.stderr
    assert "Model test/alpha:free failed" in result.stderr
    assert "1/2 council members failed; 1 valid answers remain" in result.stderr
    assert "Synthesizing 1 valid answers" in result.stderr
    assert len(router.calls) == 3


def test_cli_env_log_level_and_flag_override(cli_env, tmp_path):
    cli_env.pop("LOG_LEVEL")
    (tmp_path / ".env").write_text("LOG_LEVEL=ERROR\n", encoding="utf-8")
    result = run_cli(["-q", "Question", "--no-save"], tmp_path, cli_env)
    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    assert "Synthesized fixture answer" in result.stdout
    result = run_cli(["-q", "Question", "--no-save", "--log-level", "debug"], tmp_path, cli_env)
    assert result.returncode == 0, result.stderr
    assert "DEBUG" in result.stderr
    assert "OpenRouter POST chat/completions" in result.stderr
    assert "Synthesized fixture answer" in result.stdout


def test_cli_partial_failure_suppresses_all_markdown_artifacts(cli_env, router, tmp_path):
    router.fail_models.add("test/alpha:free")
    cli_env["SAVE_PROMPT"] = "MARKDOWN"
    cli_env["SAVE_INITIAL_ANSWERS"] = "MARKDOWN"
    cli_env["SAVE_SYNTHESIS"] = "MARKDOWN"
    result = run_cli(["-q", "Question"], tmp_path, cli_env)
    assert result.returncode == 0, result.stderr
    assert "Synthesized fixture answer" in result.stdout
    assert not (tmp_path / "council-output").exists()


def test_cli_all_models_fail_without_markdown_artifacts(cli_env, router, tmp_path):
    router.fail_models.update({"test/alpha:free", "test/beta:free"})
    cli_env["SAVE_INITIAL_ANSWERS"] = "markdown"
    result = run_cli(["-q", "Question"], tmp_path, cli_env)
    assert result.returncode == 1
    assert "All free model requests failed" not in result.stdout
    assert "All free model requests failed" in result.stderr
    assert len(router.calls) == 2
    assert not (tmp_path / "council-output").exists()


def test_cli_excludes_harness_nontext_and_configured_models(cli_env, router, tmp_path):
    image = model("test/image:free")
    image["architecture"]["output_modalities"] = ["image"]
    router.catalog.extend([model("thinkingmachines/inkling-small:free"), image])
    cli_env["EXCLUDED_MODELS"] = "test/alpha:free"
    cli_env["JUDGE"] = "random"
    result = run_cli(["-q", "Question", "--no-save"], tmp_path, cli_env)
    assert result.returncode == 0, result.stderr
    assert "Synthesized fixture answer" in result.stdout
    assert {call["model"] for call in router.calls} == {"test/beta:free"}
    assert len(router.calls) == 2
    router.calls.clear()
    result = run_cli(
        ["-q", "Question", "--no-save", "--excluded-models", "test/beta:free"], tmp_path, cli_env
    )
    assert result.returncode == 0, result.stderr
    assert "Synthesized fixture answer" in result.stdout
    assert {call["model"] for call in router.calls} == {"test/alpha:free"}
    assert len(router.calls) == 2


def test_cli_file_and_question_are_combined_file_first(cli_env, router, tmp_path):
    context_file = tmp_path / "context.md"
    context_file.write_text("Prior file context", encoding="utf-8")
    result = run_cli(
        ["--file", str(context_file), "--question", "Current question", "--no-save"],
        tmp_path,
        cli_env,
    )
    assert result.returncode == 0, result.stderr
    combined = "Prior file context\n\nCurrent question"
    assert router.calls[0]["messages"][0]["content"] == combined
    judge_call = next(call for call in router.calls if call["messages"][0]["role"] == "system")
    assert json.loads(judge_call["messages"][1]["content"])["original_question"] == combined


def test_cli_question_is_literal_even_if_file_exists(cli_env, router, tmp_path):
    (tmp_path / "prompt.md").write_text("File contents", encoding="utf-8")
    result = run_cli(["--question", "prompt.md", "--no-save"], tmp_path, cli_env)
    assert result.returncode == 0, result.stderr
    assert router.calls[0]["messages"][0]["content"] == "prompt.md"


def test_cli_file_synthesis_duckdb(cli_env, tmp_path):
    (tmp_path / "prompt.md").write_text("Explain beta distributions", encoding="utf-8")
    result = run_cli(["--file", "prompt.md", "--save-synthesis", "duckdb"], tmp_path, cli_env)
    assert result.returncode == 0, result.stderr
    with duckdb.connect(str(tmp_path / "council.duckdb")) as db:
        assert db.execute("SELECT prompt FROM questions").fetchone() == (
            "Explain beta distributions",
        )
        assert db.execute("SELECT stage, content FROM responses").fetchone() == (
            "synthesis",
            "Synthesized fixture answer",
        )


def test_module_no_save_overrides_env_and_flags(cli_env, tmp_path):
    cli_env.update(SAVE_INITIAL_ANSWERS="BOTH", SAVE_SYNTHESIS="BOTH")
    result = run_cli(
        ["-q", "Question", "--no-save", "--save-prompt", "both"], tmp_path, cli_env, module=True
    )
    assert result.returncode == 0, result.stderr
    assert list(tmp_path.iterdir()) == []


def test_cli_env_file(cli_env, tmp_path):
    cli_env.pop("SAVE_PROMPT")
    (tmp_path / "settings.env").write_text("SAVE_PROMPT=NO\n", encoding="utf-8")
    result = run_cli(["-q", "Question", "--env-file", "settings.env"], tmp_path, cli_env)
    assert result.returncode == 0, result.stderr
    assert not (tmp_path / "council.duckdb").exists()


@pytest.mark.parametrize(
    "args",
    [
        [],
        ["--question", "x", "--followup", "y"],
        ["-q", "x", "--save-synthesis", "invalid"],
    ],
)
def test_cli_invalid_usage(cli_env, router, tmp_path, args):
    result = run_cli(args, tmp_path, cli_env)
    assert result.returncode == 2
    assert not router.calls


def test_cli_missing_question_and_file_is_usage_error(cli_env, router, tmp_path):
    result = run_cli([], tmp_path, cli_env)
    assert result.returncode == 2
    assert not router.calls


def test_cli_missing_key_actionable_error(cli_env, tmp_path):
    cli_env["OPENROUTER_API_KEY"] = ""
    result = run_cli(["-q", "Question"], tmp_path, cli_env)
    assert result.returncode == 1
    assert "OPENROUTER_API_KEY" in result.stderr
    assert not result.stdout


def test_help_and_version_do_not_need_key(cli_env, tmp_path):
    cli_env["OPENROUTER_API_KEY"] = ""
    for arg in ("--help", "--version"):
        result = run_cli([arg], tmp_path, cli_env)
        assert result.returncode == 0, result.stderr
        assert "llm-council" in result.stdout
