from itertools import product
from pathlib import Path

import duckdb
import pytest

from llm_council import ModelAnswer, SaveMode, Settings, StorageError
from llm_council.storage import RunStorage


@pytest.mark.parametrize(
    "prompt_mode,initial_mode,synthesis_mode", list(product(SaveMode, repeat=3))
)
def test_every_save_mode_combination(tmp_path, prompt_mode, initial_mode, synthesis_mode):
    settings = Settings().with_overrides(
        save_prompt=prompt_mode,
        save_initial_answers=initial_mode,
        save_synthesis=synthesis_mode,
        duckdb_path=tmp_path / "data.duckdb",
        markdown_dir=tmp_path / "markdown",
    )
    answer = ModelAnswer("test/model", "actual/model", content="Answer", usage={"total_tokens": 5})
    with RunStorage(settings, "question-id", "The Question π") as storage:
        storage.save_question("Question π", "/source.txt")
        storage.save_answer(answer, "initial", 1)
        storage.save_answer(answer, "synthesis")
        folder = storage.folder
    modes = (prompt_mode, initial_mode, synthesis_mode)
    assert settings.duckdb_path.exists() == any(mode.duckdb for mode in modes)
    if settings.duckdb_path.exists():
        with duckdb.connect(str(settings.duckdb_path)) as db:
            assert db.execute("SHOW TABLES").fetchall() == [("questions",), ("responses",)]
            assert db.execute("SELECT prompt, source_path FROM questions").fetchall() == [
                ("Question π", "/source.txt") if prompt_mode.duckdb else (None, None)
            ]
            stages = db.execute("SELECT stage FROM responses ORDER BY stage").fetchall()
            assert stages == ([("initial",)] if initial_mode.duckdb else []) + (
                [("synthesis",)] if synthesis_mode.duckdb else []
            )
    expected = set()
    if prompt_mode.markdown:
        expected.add("prompt.md")
    if initial_mode.markdown:
        expected.add("initial-001-test-model.md")
    if synthesis_mode.markdown:
        expected.add("synthesis.md")
    if expected:
        storage = RunStorage(settings, "successful-question", "The Question π")
        storage.save_markdown_artifacts("Question π", (answer,), answer)
        assert storage.folder is not None
        assert settings.markdown_dir.exists()
        assert {path.name for path in storage.folder.iterdir()} == expected
        assert storage.folder.name.startswith("question-")
    else:
        assert folder is None
        assert not settings.markdown_dir.exists()


def test_storage_initialization_failure_is_actionable(tmp_path):
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("leave untouched", encoding="utf-8")
    settings = Settings().with_overrides(duckdb_path=blocker / "data.duckdb")
    with pytest.raises(StorageError, match="initialize council storage"):
        with RunStorage(settings, "question-id"):
            pass
    assert blocker.read_text() == "leave untouched"


def test_markdown_write_failure_is_not_swallowed(tmp_path, monkeypatch):
    settings = Settings().with_overrides(save_initial_answers="markdown", markdown_dir=tmp_path)
    with RunStorage(settings, "question-id", "Question") as storage:

        def fail_write(*_args, **_kwargs):
            raise OSError("fixture write failure")

        monkeypatch.setattr(Path, "write_text", fail_write)
        answer = ModelAnswer("test/model", "test/model", content="Actual answer")
        with pytest.raises(StorageError, match="Cannot save Markdown artifacts"):
            storage.save_markdown_artifacts("Question", (answer,), answer)
