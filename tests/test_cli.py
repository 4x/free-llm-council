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
        ROUTER_MODELS="openrouter/free",
        FREE_ROUTER_MODEL="",
        AUTO_ROUTER_MODEL="",
        JUDGE_PROMPT="",
        EXCLUDED_MODELS="",
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
        {"role": "user", "content": "Original question"},
        {"role": "assistant", "content": previous},
        {"role": "user", "content": "Clarify the previous answer"},
    ]
    judge_call = next(call for call in router.calls if call["messages"][0]["role"] == "system")
    assert judge_call["messages"][1:3] == [
        {"role": "user", "content": "Original question"},
        {"role": "assistant", "content": previous},
    ]
    assert judge_call["messages"][3]["role"] == "user"
    assert cache.read_text(encoding="utf-8") == previous


def test_cli_subsequent_followups_preserve_alternating_roles(cli_env, router, tmp_path):
    """Chained --followup calls keep user/assistant/user/assistant/... role alternation."""
    assert run_cli(["-q", "First question"], tmp_path, cli_env).returncode == 0
    router.reply = {
        "model": "test/beta:free",
        "choices": [{"message": {"content": "Second synthesis"}, "finish_reason": "stop"}],
    }
    assert run_cli(["--followup", "First followup"], tmp_path, cli_env).returncode == 0

    router.calls.clear()
    router.reply = {
        "model": "test/beta:free",
        "choices": [{"message": {"content": "Third synthesis"}, "finish_reason": "stop"}],
    }
    second_followup = run_cli(["--followup", "Second followup"], tmp_path, cli_env)
    assert second_followup.returncode == 0, second_followup.stderr
    assert router.calls[0]["messages"] == [
        {"role": "user", "content": "First question"},
        {"role": "assistant", "content": "Synthesized fixture answer"},
        {"role": "user", "content": "First followup"},
        {"role": "assistant", "content": "Second synthesis"},
        {"role": "user", "content": "Second followup"},
    ]
    judge_call = next(call for call in router.calls if call["messages"][0]["role"] == "system")
    assert [msg["role"] for msg in judge_call["messages"]] == [
        "system",
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
    ]


def test_cli_followup_with_legacy_synthesis_cache_only_alternates_roles(cli_env, router, tmp_path):
    """A workspace with only latest-synthesis.md still sends alternating user/assistant roles."""
    cache_dir = tmp_path / ".llm-council"
    cache_dir.mkdir()
    (cache_dir / "latest-synthesis.md").write_text("Legacy synthesis", encoding="utf-8")
    result = run_cli(["--followup", "Followup question", "--no-save"], tmp_path, cli_env)
    assert result.returncode == 0, result.stderr
    assert [msg["role"] for msg in router.calls[0]["messages"]] == ["user", "assistant", "user"]
    assert router.calls[0]["messages"][1] == {"role": "assistant", "content": "Legacy synthesis"}
    judge_call = next(call for call in router.calls if call["messages"][0]["role"] == "system")
    assert [msg["role"] for msg in judge_call["messages"]] == [
        "system",
        "user",
        "assistant",
        "user",
    ]


def test_cli_followup_prefers_edited_latest_synthesis_over_stale_history(cli_env, router, tmp_path):
    """Editing latest-synthesis.md invalidates stale latest-conversation.json history."""
    assert run_cli(["-q", "First question"], tmp_path, cli_env).returncode == 0
    cache = tmp_path / ".llm-council" / "latest-synthesis.md"
    cache.write_text("Manually edited synthesis", encoding="utf-8")
    router.calls.clear()
    followup = run_cli(["--followup", "Clarify edited text", "--no-save"], tmp_path, cli_env)
    assert followup.returncode == 0, followup.stderr
    assert router.calls[0]["messages"] == [
        {"role": "user", "content": "Previous question"},
        {"role": "assistant", "content": "Manually edited synthesis"},
        {"role": "user", "content": "Clarify edited text"},
    ]


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
    assert "fixture unavailable" in " ".join(result.stderr.split())
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


def test_cli_partial_failure_still_saves_requested_markdown_artifacts(cli_env, router, tmp_path):
    router.fail_models.add("test/alpha:free")
    cli_env["SAVE_PROMPT"] = "MARKDOWN"
    cli_env["SAVE_INITIAL_ANSWERS"] = "MARKDOWN"
    cli_env["SAVE_SYNTHESIS"] = "BOTH"
    result = run_cli(["-q", "Question"], tmp_path, cli_env)
    assert result.returncode == 0, result.stderr
    assert "Synthesized fixture answer" in result.stdout
    folders = list((tmp_path / "council-output").iterdir())
    assert len(folders) == 1
    files = {path.name for path in folders[0].iterdir()}
    assert files == {"prompt.md", "initial-002-test-beta-free.md", "synthesis.md"}
    assert "Synthesized fixture answer" in (folders[0] / "synthesis.md").read_text(encoding="utf-8")


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
    router.catalog.extend([model("test/harness-only:free"), image])
    cli_env["EXCLUDED_MODELS"] = "test/harness-only:free,test/alpha:free"
    cli_env["JUDGE"] = "random"
    result = run_cli(["-q", "Question", "--no-save"], tmp_path, cli_env)
    assert result.returncode == 0, result.stderr
    assert "Synthesized fixture answer" in result.stdout
    assert {call["model"] for call in router.calls} == {"test/beta:free"}
    assert len(router.calls) == 2
    router.calls.clear()
    result = run_cli(
        [
            "-q",
            "Question",
            "--no-save",
            "--excluded-models",
            "test/harness-only:free,test/beta:free",
        ],
        tmp_path,
        cli_env,
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


def test_cli_external_judge_prompt_env_and_flag_override(cli_env, router, tmp_path):
    """Installed CLI resolves prompt files against cwd and honors its override flag."""
    (tmp_path / "judge.md").write_text("Custom π judge\n", encoding="utf-8-sig")
    (tmp_path / "override.txt").write_text("Override judge", encoding="utf-8")
    (tmp_path / ".env").write_text("JUDGE_PROMPT=judge.md\n", encoding="utf-8")
    cli_env.pop("JUDGE_PROMPT")
    result = run_cli(["-q", "Question", "--no-save"], tmp_path, cli_env)
    assert result.returncode == 0, result.stderr
    assert router.calls[-1]["messages"][0]["content"] == "Custom π judge\n"
    result = run_cli(
        ["-q", "Question", "--no-save", "--judge-prompt", "override.txt"], tmp_path, cli_env
    )
    assert result.returncode == 0, result.stderr
    assert router.calls[-1]["messages"][0]["content"] == "Override judge"


def test_cli_generation_error_keeps_code_metadata_and_redacts_saved_errors(
    cli_env, router, tmp_path
):
    """Provider diagnostics reach stderr and DuckDB without leaking the API key."""
    router.model_replies["test/alpha:free"] = {
        "choices": [
            {
                "error": {
                    "code": "provider_quota",
                    "message": "Quota reached for fixture-key",
                    "metadata": {"raw": "Reset tomorrow; fixture-key", "provider_name": "Fixture"},
                }
            }
        ]
    }
    cli_env["SAVE_INITIAL_ANSWERS"] = "DUCKDB"
    result = run_cli(["-q", "Question"], tmp_path, cli_env)
    assert result.returncode == 0, result.stderr
    output = " ".join(result.stderr.split())
    assert "provider_quota" in output and "Reset tomorrow" in output
    assert "fixture-key" not in output and "[redacted]" in output
    assert "Synthesized fixture answer" in result.stdout
    with duckdb.connect(str(tmp_path / "council.duckdb")) as db:
        error = db.execute("SELECT error FROM responses WHERE error IS NOT NULL").fetchone()[0]
    assert "provider_quota" in error and "Reset tomorrow" in error
    assert "fixture-key" not in error and error.count("[redacted]") == 2


@pytest.mark.parametrize("path_kind", ["missing", "directory", "invalid-utf8", "blank"])
def test_cli_invalid_judge_prompt_is_actionable_without_requests(
    cli_env, router, tmp_path, path_kind
):
    """Prompt-file validation produces normal CLI errors before network or storage."""
    path = tmp_path / "judge.md"
    if path_kind == "directory":
        path.mkdir()
    elif path_kind == "invalid-utf8":
        path.write_bytes(b"\xff")
    elif path_kind == "blank":
        path.write_text(" \n", encoding="utf-8")
    result = run_cli(["-q", "Question", "--judge-prompt", str(path)], tmp_path, cli_env)
    assert result.returncode == 1
    assert "JUDGE_PROMPT" in result.stderr and "Traceback" not in result.stderr
    assert not router.calls and not router.gets
    assert not (tmp_path / "council.duckdb").exists()


def test_cli_all_failures_are_visible_with_logs_disabled(cli_env, router, tmp_path):
    """The terminal error includes every provider failure even at CRITICAL verbosity."""
    router.fail_models.update({"test/alpha:free", "test/beta:free"})
    cli_env["LOG_LEVEL"] = "CRITICAL"
    result = run_cli(["-q", "Question", "--no-save"], tmp_path, cli_env)
    assert result.returncode == 1
    assert "test/alpha:free: OpenRouter error 503: fixture unavailable" in result.stderr
    assert "test/beta:free: OpenRouter error 503: fixture unavailable" in result.stderr
    assert "fixture-key" not in result.stderr


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
        ["--question", "x", "--export-markdown"],
        ["--export-markdown", "--no-save"],
        ["-q", "x", "--save-synthesis", "invalid"],
        ["--improve"],
        ["--improve", "x", "--question", "y"],
        ["--improve", "x", "--followup", "y"],
        ["--improve", "x", "--prompt", "1"],
        ["--prompt", "1", "--question", "x"],
        ["--prompt", "1", "--followup", "x"],
        ["--export-markdown", "--improve", "x"],
        ["--export-markdown", "--prompt", "1"],
    ],
)
def test_cli_invalid_usage(cli_env, router, tmp_path, args):
    result = run_cli(args, tmp_path, cli_env)
    assert result.returncode == 2
    assert not router.calls


@pytest.mark.parametrize("flag", ["--improve", "--improve-prompt"])
def test_cli_improve_prompt_and_subsequent_prompt_index(cli_env, router, tmp_path, flag):
    router.reply = {
        "model": "test/beta:free",
        "choices": [
            {
                "message": {
                    "content": json.dumps(
                        [
                            "Improved alpha prompt",
                            "Improved beta prompt",
                            "Improved gamma prompt",
                        ]
                    )
                },
                "finish_reason": "stop",
            }
        ],
    }
    improve_res = run_cli([flag, "Rough question"], tmp_path, cli_env)
    assert improve_res.returncode == 0, improve_res.stderr
    assert "1. Improved alpha prompt" in improve_res.stdout
    assert "2. Improved beta prompt" in improve_res.stdout
    assert "3. Improved gamma prompt" in improve_res.stdout
    assert len(router.calls) == 1
    assert router.calls[0]["messages"][1]["content"] == "Rough question"
    cache_file = tmp_path / ".llm-council" / "improved-prompts.json"
    assert json.loads(cache_file.read_text(encoding="utf-8")) == [
        "Improved alpha prompt",
        "Improved beta prompt",
        "Improved gamma prompt",
    ]

    router.calls.clear()
    router.reply = None
    run_res = run_cli(["--prompt", "2", "--no-save"], tmp_path, cli_env)
    assert run_res.returncode == 0, run_res.stderr
    assert "Synthesized fixture answer" in run_res.stdout
    assert len(router.calls) == 3
    assert router.calls[0]["messages"] == [{"role": "user", "content": "Improved beta prompt"}]
    judge_call = next(call for call in router.calls if call["messages"][0]["role"] == "system")
    assert json.loads(judge_call["messages"][1]["content"])["original_question"] == (
        "Improved beta prompt"
    )


def test_cli_improve_with_file_only(cli_env, router, tmp_path):
    (tmp_path / "draft.txt").write_text("File-based draft prompt", encoding="utf-8")
    router.reply = {
        "model": "test/beta:free",
        "choices": [
            {
                "message": {"content": '["Refined file prompt 1", "Refined file prompt 2"]'},
                "finish_reason": "stop",
            }
        ],
    }
    res = run_cli(["--file", "draft.txt", "--improve"], tmp_path, cli_env)
    assert res.returncode == 0, res.stderr
    assert "1. Refined file prompt 1" in res.stdout
    assert "2. Refined file prompt 2" in res.stdout
    assert len(router.calls) == 1
    assert router.calls[0]["messages"][1]["content"] == "File-based draft prompt"


def test_cli_prompt_index_errors_without_cache_or_out_of_range(cli_env, router, tmp_path):
    no_cache = run_cli(["--prompt", "1"], tmp_path, cli_env)
    assert no_cache.returncode == 1
    assert "No improved prompts are cached" in no_cache.stderr
    assert not router.calls

    cache_dir = tmp_path / ".llm-council"
    cache_dir.mkdir()
    (cache_dir / "improved-prompts.json").write_text(
        json.dumps(["Only one prompt"]), encoding="utf-8"
    )
    zero_idx = run_cli(["--prompt", "0"], tmp_path, cli_env)
    assert zero_idx.returncode == 1
    assert "at least 1" in zero_idx.stderr

    out_of_range = run_cli(["--prompt", "2"], tmp_path, cli_env)
    assert out_of_range.returncode == 1
    assert "out of range" in out_of_range.stderr
    assert not router.calls


def test_cli_export_markdown_backfills_stored_duckdb_runs(cli_env, router, tmp_path):
    """--export-markdown exports stored DuckDB syntheses without making API requests."""
    run_result = run_cli(
        ["-q", "Exported question", "--save-synthesis", "duckdb"],
        tmp_path,
        cli_env,
    )
    assert run_result.returncode == 0, run_result.stderr
    assert not (tmp_path / "council-output").exists()
    router.calls.clear()
    router.gets.clear()

    export_result = run_cli(["--export-markdown"], tmp_path, cli_env)
    assert export_result.returncode == 0, export_result.stderr
    assert not router.calls and not router.gets
    folders = list((tmp_path / "council-output").iterdir())
    assert len(folders) == 1
    assert (folders[0] / "synthesis.md").is_file()
    assert str(folders[0].name) in export_result.stdout


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
