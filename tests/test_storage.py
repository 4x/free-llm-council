from itertools import product
from pathlib import Path

import duckdb
import pytest

from llm_council import ModelAnswer, SaveMode, Settings, StorageError, export_markdown
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
        if expected == {"synthesis.md"}:
            assert storage.folder.is_file()
            assert storage.folder.parent == settings.markdown_dir
            assert storage.folder.name.startswith("question-synthesis-")
            assert storage.folder.suffix == ".md"
            assert "Answer" in storage.folder.read_text(encoding="utf-8")
        else:
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


def test_export_markdown_writes_stored_duckdb_runs_idempotently(tmp_path):
    """export_markdown backfills DuckDB-only runs into Markdown folders without duplicates."""
    settings = Settings().with_overrides(
        save_prompt="duckdb",
        save_initial_answers="duckdb",
        save_synthesis="duckdb",
        duckdb_path=tmp_path / "data.duckdb",
        markdown_dir=tmp_path / "markdown",
    )
    initial = ModelAnswer("test/alpha:free", "test/alpha:free", content="Initial answer")
    failed = ModelAnswer("test/beta:free", "test/beta:free", error="429 rate limited")
    synthesis = ModelAnswer("test/alpha:free", "test/alpha:free", content="Synthesized answer")
    with RunStorage(settings, "q1", "Explain normal distribution") as storage:
        storage.save_question("Explain normal distribution", None)
        storage.save_answer(initial, "initial", 1)
        storage.save_answer(failed, "initial", 2)
        storage.save_answer(synthesis, "synthesis")

    exported = export_markdown(settings=settings)
    assert len(exported) == 1
    assert exported[0].parent == settings.markdown_dir
    assert {path.name for path in exported[0].iterdir()} == {
        "prompt.md",
        "initial-001-test-alpha-free.md",
        "synthesis.md",
    }
    assert "Synthesized answer" in (exported[0] / "synthesis.md").read_text(encoding="utf-8")
    # Re-running export_markdown skips already exported runs.
    assert export_markdown(settings=settings) == ()


def test_export_markdown_respects_null_prompt_privacy_and_stage_modes(tmp_path):
    """Runs saved without a prompt do not create an empty prompt.md on export."""
    settings = Settings().with_overrides(
        save_prompt="none",
        save_initial_answers="duckdb",
        save_synthesis="duckdb",
        duckdb_path=tmp_path / "data.duckdb",
        markdown_dir=tmp_path / "markdown",
    )
    synthesis = ModelAnswer("test/alpha:free", "test/alpha:free", content="Private synthesis")
    with RunStorage(settings, "q-private", "") as storage:
        storage.save_question("Should not be saved", None)
        storage.save_answer(synthesis, "synthesis")

    exported = export_markdown(settings=settings, save_prompt="markdown", save_synthesis="markdown")
    assert len(exported) == 1
    assert exported[0].is_file()
    assert exported[0].parent == settings.markdown_dir
    assert exported[0].name.startswith("council-run-synthesis-")
    assert exported[0].suffix == ".md"
    assert "Private synthesis" in exported[0].read_text(encoding="utf-8")
