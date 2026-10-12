from __future__ import annotations

import json
import warnings

import duckdb
import pytest
from conftest import model

from llm_council import ConfigurationError, CouncilError, SaveMode, Settings, query, query_result
from llm_council.models import council_models


def test_default_saves_only_prompt_and_calls_each_free_model(config, router):
    result = query_result("A detailed question", settings=config)
    assert result.synthesis == "Synthesized fixture answer"
    assert [answer.requested_model for answer in result.answers] == [
        "test/alpha:free",
        "test/beta:free",
    ]
    assert len(router.calls) == 3
    assert all(call["authorization"] == "Bearer fixture-key" for call in router.calls)
    assert all(
        call["provider"]["max_price"] == {"prompt": 0, "completion": 0} for call in router.calls
    )
    judge_input = json.loads(router.calls[-1]["messages"][1]["content"])
    assert judge_input["original_question"] == "A detailed question"
    assert [item["answer"] for item in judge_input["candidate_answers"]] == [
        "Answer: test/alpha:free",
        "Answer: test/beta:free",
    ]
    with duckdb.connect(str(config.duckdb_path)) as db:
        assert db.execute("SHOW TABLES").fetchall() == [("questions",), ("responses",)]
        assert db.execute("SELECT prompt FROM questions").fetchall() == [("A detailed question",)]
        assert db.execute("SELECT count(*) FROM responses").fetchone() == (0,)
    assert not config.markdown_dir.exists()


def test_both_storage_repeated_prompt_single_tables_and_one_folder_per_run(config):
    options = dict(
        settings=config, save_prompt="BOTH", save_initial_answers="both", save_synthesis="both"
    )
    first = query_result("Same prompt", **options)
    second = query_result("Same prompt", **options)
    assert first.question_id != second.question_id
    assert len(list(config.markdown_dir.iterdir())) == 2
    for result in (first, second):
        assert result.markdown_path is not None
        files = sorted(path.name for path in result.markdown_path.iterdir())
        assert len(files) == 4
        assert "prompt.md" in files and "synthesis.md" in files
        assert (result.markdown_path / "prompt.md").read_text(encoding="utf-8") == "Same prompt"
        assert "Synthesized fixture answer" in (result.markdown_path / "synthesis.md").read_text()
    with duckdb.connect(str(config.duckdb_path)) as db:
        assert db.execute("SHOW TABLES").fetchall() == [("questions",), ("responses",)]
        assert db.execute("SELECT count(*) FROM questions").fetchone() == (2,)
        assert db.execute(
            "SELECT stage, count(*) FROM responses GROUP BY stage ORDER BY stage"
        ).fetchall() == [("initial", 4), ("synthesis", 2)]
        assert db.execute(
            "SELECT count(*) FROM responses r JOIN questions q ON q.id=r.question_id"
        ).fetchone() == (6,)


def test_responses_only_do_not_persist_prompt(config):
    query("Private prompt", settings=config, save_prompt="none", save_synthesis="duckdb")
    with duckdb.connect(str(config.duckdb_path)) as db:
        assert db.execute("SELECT prompt, source_path FROM questions").fetchall() == [(None, None)]
        assert db.execute("SELECT stage FROM responses").fetchall() == [("synthesis",)]


def test_markdown_only_and_no_save_precedence(config):
    result = query_result(
        "Question",
        settings=config,
        save_prompt="none",
        save_initial_answers="markdown",
        save_synthesis="markdown",
    )
    assert not config.duckdb_path.exists()
    assert result.markdown_path is not None
    assert not (result.markdown_path / "prompt.md").exists()
    assert len(list(result.markdown_path.iterdir())) == 3
    before = list(config.markdown_dir.iterdir())
    query(
        "No saves",
        settings=config,
        save_prompt="both",
        save_initial_answers="both",
        save_synthesis="both",
        no_save=True,
    )
    assert not config.duckdb_path.exists()
    assert list(config.markdown_dir.iterdir()) == before


@pytest.mark.parametrize("mode", ["NONE", "NO", "neither", "none"])
def test_no_save_aliases(config, mode):
    assert query("Question", settings=config, save_prompt=mode) == "Synthesized fixture answer"
    assert not config.duckdb_path.exists()
    assert not config.markdown_dir.exists()


@pytest.mark.parametrize("method", ["string", "path", "file"])
def test_file_inputs_utf8(config, tmp_path, method):
    path = tmp_path / "prompt.md"
    path.write_text("Question with π and β", encoding="utf-8-sig")
    kwargs = (
        {"file": path}
        if method == "file"
        else {"prompt": str(path) if method == "string" else path}
    )
    result = query_result(**kwargs, settings=config)
    assert result.prompt == "Question with π and β"
    with duckdb.connect(str(config.duckdb_path)) as db:
        assert db.execute("SELECT source_path FROM questions").fetchone() == (str(path.resolve()),)


def test_literal_prompt_override(config, tmp_path):
    path = tmp_path / "prompt.md"
    path.write_text("File contents", encoding="utf-8")
    result = query_result(str(path), settings=config, literal_prompt=True, no_save=True)
    assert result.prompt == str(path)


def test_long_text_is_not_path(config):
    text = "A question? " * 1000
    assert query_result(text, settings=config, no_save=True).prompt == text


@pytest.mark.parametrize(
    "kwargs", [{}, {"prompt": ""}, {"prompt": "a", "file": "b"}, {"file": "missing-prompt.txt"}]
)
def test_invalid_input(config, router, kwargs):
    with pytest.raises(ConfigurationError):
        query(settings=config, **kwargs)
    assert not router.calls


def test_failed_judge_falls_back_to_a_successful_member(config, router, caplog):
    router.fail_models.add("test/beta:free")
    result = query_result("Question", settings=config, judge="test/beta:free", no_save=True)
    assert result.judge_requested == "test/alpha:free"
    assert result.synthesis == "Synthesized fixture answer"
    assert [call["model"] for call in router.calls] == [
        "test/alpha:free",
        "test/beta:free",
        "test/alpha:free",
    ]
    assert "Synthesizing 1 valid answers with test/alpha:free" in caplog.text


def test_partial_failure_saves_errors_and_synthesizes_successes(config, router, caplog):
    router.fail_models.add("test/alpha:free")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = query_result("Question", settings=config, save_initial_answers="both")
    assert result.synthesis == "Synthesized fixture answer"
    assert "1/2 council members failed; 1 valid answers remain" in caplog.text
    assert "Synthesizing 1 valid answers" in caplog.text
    assert len(result.failures) == 1
    judge_input = json.loads(router.calls[-1]["messages"][1]["content"])
    assert len(judge_input["candidate_answers"]) == 1
    with duckdb.connect(str(config.duckdb_path)) as db:
        assert db.execute("SELECT count(*) FROM responses WHERE error IS NOT NULL").fetchone() == (
            1,
        )
    assert result.markdown_path is None
    assert not config.markdown_dir.exists()


def test_markdown_folder_name_uses_prompt_words_and_minute_timestamp(config, router):
    result = query_result(
        "The quick brown fox jumps over lazy dogs in a field",
        settings=config,
        save_prompt="markdown",
        no_save=False,
    )
    assert result.markdown_path is not None
    assert result.markdown_path.parent == config.markdown_dir
    assert result.markdown_path.name.startswith("quick-brown-fox-jumps-over-lazy-")
    assert len(result.markdown_path.name.rsplit("-", 1)[-1]) == 13
    assert result.markdown_path.name.rsplit("-", 1)[-1].startswith("20")


def test_all_fail_does_not_call_judge_or_create_markdown(config, router, caplog):
    router.fail_models.update({"test/alpha:free", "test/beta:free"})
    with pytest.raises(CouncilError, match="All free"):
        query("Question", settings=config, save_initial_answers="markdown")
    assert "No valid answers; synthesis will not be attempted" in caplog.text
    assert len(router.calls) == 2
    assert not config.markdown_dir.exists()


def test_judge_failure_preserves_initial_answers_without_markdown_error(config, router):
    router.judge_failure = True
    with pytest.raises(CouncilError, match="fallbacks failed"):
        query("Question", settings=config, save_initial_answers="both", save_synthesis="both")
    with duckdb.connect(str(config.duckdb_path)) as db:
        assert db.execute("SELECT count(*) FROM responses").fetchone() == (4,)
        synthesis_count = db.execute(
            "SELECT count(*) FROM responses WHERE stage='synthesis'"
        ).fetchone()
        assert synthesis_count == (2,)
        assert db.execute("SELECT error FROM responses WHERE stage='synthesis'").fetchall() == [
            ("OpenRouter error 503: fixture unavailable",),
            ("OpenRouter error 503: fixture unavailable",),
        ]
    assert not config.markdown_dir.exists()


def test_catalog_pagination(config, router):
    router.page_size = 1
    result = query_result("Question", settings=config, no_save=True)
    assert len(result.answers) == 2
    assert [params["offset"] for params in router.gets] == [["0"], ["1"], ["2"], ["3"]]


def test_no_free_models(config, router):
    router.catalog = [model("paid", "0.1")]
    with pytest.raises(CouncilError, match="no free"):
        query("Question", settings=config)
    assert not router.calls and not config.duckdb_path.exists()


def test_free_detection_is_conservative():
    valid = model("free-without-suffix")
    paid_output = model("paid-output:free")
    paid_output["pricing"]["completion"] = "0.01"
    request_fee = model("request-fee")
    request_fee["pricing"]["request"] = "0.01"
    missing = model("missing")
    del missing["pricing"]["prompt"]
    bad = model("bad", "nonsense")
    conditional = model("conditional")
    conditional["pricing"]["overrides"] = [{"prompt": "1"}]
    reasoning = model("reasoning")
    reasoning["pricing"]["internal_reasoning"] = "0.1"
    image = model("image")
    image["architecture"]["output_modalities"] = ["image"]
    catalog = [
        valid,
        valid,
        paid_output,
        request_fee,
        missing,
        bad,
        conditional,
        reasoning,
        image,
        model("openrouter/free"),
    ]
    assert [entry["id"] for entry in council_models(catalog)] == ["free-without-suffix"]


def test_env_precedence_and_case_insensitivity(tmp_path, monkeypatch):
    path = tmp_path / ".env"
    path.write_text("SAVE_PROMPT=NO\nSAVE_INITIAL_ANSWERS=BOTH\nJUDGE=random\n", encoding="utf-8")
    monkeypatch.delenv("SAVE_INITIAL_ANSWERS", raising=False)
    monkeypatch.setenv("SAVE_PROMPT", "MARKDOWN")
    settings = Settings.from_env(path)
    assert settings.save_prompt == SaveMode.MARKDOWN
    assert settings.save_initial_answers == SaveMode.BOTH
    assert settings.with_overrides(save_prompt="duckdb").save_prompt == SaveMode.DUCKDB
    assert "SAVE_INITIAL_ANSWERS" not in __import__("os").environ


@pytest.mark.parametrize(
    "override",
    [
        {"max_concurrency": 0},
        {"max_retries": -1},
        {"timeout": 0},
        {"request_interval": float("nan")},
        {"max_tokens": 1.5},
        {"max_tokens": 0},
        {"save_prompt": "yes"},
        {"judge_scope": "paid"},
        {"unknown": 1},
        {"base_url": "ftp://invalid"},
    ],
)
def test_invalid_settings(override):
    with pytest.raises(ConfigurationError):
        Settings().with_overrides(**override)


def test_missing_api_key(config, router):
    with pytest.raises(ConfigurationError, match="OPENROUTER_API_KEY"):
        query("Question", settings=config.with_overrides(api_key=""))
    assert not router.calls
