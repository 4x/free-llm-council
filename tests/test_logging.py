import json
import logging

import pytest
from rich.logging import RichHandler

from llm_council import ConfigurationError, Settings, query_result
from llm_council.logging_config import configure_logging


def test_logging_defaults_to_info():
    assert Settings().log_level == "INFO"


def test_log_level_env_precedence(tmp_path, monkeypatch):
    path = tmp_path / ".env"
    path.write_text("LOG_LEVEL=warning\n", encoding="utf-8")
    monkeypatch.delenv("LOG_LEVEL", raising=False)
    assert Settings.from_env(path).log_level == "WARNING"
    monkeypatch.setenv("LOG_LEVEL", "error")
    settings = Settings.from_env(path)
    assert settings.log_level == "ERROR"
    assert settings.with_overrides(log_level=" debug ").log_level == "DEBUG"


@pytest.mark.parametrize("level", ["verbose", "", "60"])
def test_invalid_log_level_is_actionable(level):
    with pytest.raises(ConfigurationError, match="LOG_LEVEL"):
        Settings().with_overrides(log_level=level)


def test_package_logging_stderr_and_no_duplicate_handlers(monkeypatch, capsys):
    package_logger = logging.getLogger("llm_council")
    root = logging.getLogger()
    monkeypatch.setattr(root, "handlers", [])
    monkeypatch.setattr(package_logger, "handlers", [])
    monkeypatch.setattr(package_logger, "level", logging.NOTSET)
    monkeypatch.setattr(package_logger, "propagate", True)
    configure_logging("INFO")
    configure_logging("INFO")
    assert len(package_logger.handlers) == 1
    assert isinstance(package_logger.handlers[0], RichHandler)
    package_logger.info("Fixture progress")
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Fixture progress" in captured.err
    assert root.handlers == []


def test_package_logging_respects_application_handlers(monkeypatch):
    package_logger = logging.getLogger("llm_council")
    root = logging.getLogger()
    application_handler = logging.NullHandler()
    monkeypatch.setattr(root, "handlers", [application_handler])
    monkeypatch.setattr(package_logger, "handlers", [])
    monkeypatch.setattr(package_logger, "level", logging.NOTSET)
    monkeypatch.setattr(package_logger, "propagate", True)
    configure_logging("DEBUG")
    assert package_logger.level == logging.DEBUG
    assert package_logger.handlers == []
    assert root.handlers == [application_handler]


def test_repeated_queries_log_progress_without_prompt_answers_or_key(config, caplog):
    for _ in range(2):
        query_result("Sensitive prompt π", settings=config, no_save=True)
    assert caplog.text.count("Starting council run") == 2
    assert caplog.text.count(" synthesized 2 responses in ") == 2
    assert "Sensitive prompt" not in caplog.text
    assert "Synthesized fixture answer" not in caplog.text
    assert "fixture-key" not in caplog.text
    assert "httpx" not in {record.name for record in caplog.records}


@pytest.mark.parametrize(
    "reply",
    [
        {},
        {"choices": [{"message": {"content": "   "}}]},
        {"choices": [{"message": {"content": "partial"}, "finish_reason": "length"}]},
        {"choices": [{"error": {"message": "failed generation"}}]},
    ],
)
def test_invalid_member_answers_are_excluded_but_synthesis_continues(config, router, reply):
    router.model_replies["test/alpha:free"] = reply
    result = query_result("Question", settings=config, no_save=True)
    assert result.synthesis == "Synthesized fixture answer"
    assert len(result.failures) == 1
    candidates = json.loads(router.calls[-1]["messages"][1]["content"])["candidate_answers"]
    assert candidates == [{"model": "test/beta:free", "answer": "Answer: test/beta:free"}]
