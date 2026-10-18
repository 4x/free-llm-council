"""Opt-in storage, independent for prompts, initial answers, and synthesis."""

from __future__ import annotations

import json
import logging
import re
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import duckdb

from .config import SaveMode, Settings
from .errors import StorageError
from .models import ModelAnswer
from .report import build_report_html

logger = logging.getLogger("llm_council")

# Shared by every exported artifact name so Markdown folders and HTML reports for
# the same run sort together and collide predictably.
ARTIFACT_STAMP_FORMAT = "%Y%m%dT%H%M"

# Bound on how many stored runs a single HTML export considers; effectively "all of
# them" for this tool while keeping the underlying history query finite.
EXPORT_RUN_LIMIT = 5000

HTML_REPORT_SUFFIX = ".html"

_PROMPT_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "with",
}


class RunStorage:
    def __init__(self, settings: Settings, question_id: str, prompt: str = "") -> None:
        self.settings = settings
        self.question_id = question_id
        self.prompt = prompt
        self.connection: Any = None
        self.folder: Path | None = None
        self._markdown_stamp = datetime.now(UTC).strftime(ARTIFACT_STAMP_FORMAT)
        self._prompt_written = False

    def __enter__(self) -> RunStorage:
        try:
            modes = (
                self.settings.save_prompt,
                self.settings.save_initial_answers,
                self.settings.save_synthesis,
            )
            if any(mode.duckdb for mode in modes):
                if str(self.settings.duckdb_path) != ":memory:":
                    self.settings.duckdb_path.parent.mkdir(parents=True, exist_ok=True)
                self.connection = duckdb.connect(str(self.settings.duckdb_path))
                self.connection.execute("""
                    CREATE TABLE IF NOT EXISTS questions (
                        id VARCHAR PRIMARY KEY,
                        created_at TIMESTAMPTZ NOT NULL,
                        prompt TEXT,
                        source_path VARCHAR
                    )
                """)
                self.connection.execute("""
                    CREATE TABLE IF NOT EXISTS responses (
                        id VARCHAR PRIMARY KEY,
                        question_id VARCHAR NOT NULL REFERENCES questions(id),
                        created_at TIMESTAMPTZ NOT NULL,
                        stage VARCHAR NOT NULL CHECK (stage IN ('initial', 'synthesis')),
                        requested_model VARCHAR NOT NULL,
                        model VARCHAR NOT NULL,
                        content TEXT,
                        error TEXT,
                        usage JSON NOT NULL
                    )
                """)
        except (duckdb.Error, OSError) as exc:
            if self.connection is not None:
                self.connection.close()
            raise StorageError(f"Cannot initialize council storage: {exc}") from exc
        return self

    def __exit__(self, *_: object) -> None:
        if self.connection is not None:
            self.connection.close()

    def save_question(self, prompt: str, source_path: str | None) -> None:
        try:
            if self.connection is not None:
                # Responses-only runs need a parent identity, not a saved copy of the prompt.
                persist = self.settings.save_prompt.duckdb
                self.connection.execute(
                    "INSERT INTO questions VALUES (?, ?, ?, ?)",
                    [
                        self.question_id,
                        datetime.now(UTC),
                        prompt if persist else None,
                        source_path if persist else None,
                    ],
                )
        except (duckdb.Error, OSError) as exc:
            raise StorageError(f"Cannot save prompt: {exc}") from exc

    def save_prompt_markdown(self) -> None:
        if self.settings.save_prompt.markdown:
            self._write("prompt.md", self.prompt)

    def save_markdown_answer(self, answer: ModelAnswer, stage: str, index: int = 0) -> None:
        mode = (
            self.settings.save_initial_answers
            if stage == "initial"
            else self.settings.save_synthesis
        )
        if mode.markdown and answer.content is not None and answer.content.strip():
            self._write(
                self._answer_filename(answer, stage, index), self._answer_markdown(answer, stage)
            )

    def save_answer(self, answer: ModelAnswer, stage: str, index: int = 0) -> None:
        mode: SaveMode = (
            self.settings.save_initial_answers
            if stage == "initial"
            else self.settings.save_synthesis
        )
        try:
            if mode.duckdb:
                self.connection.execute(
                    "INSERT INTO responses VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    [
                        str(uuid4()),
                        self.question_id,
                        datetime.now(UTC),
                        stage,
                        answer.requested_model,
                        answer.model,
                        answer.content,
                        answer.error,
                        json.dumps(answer.usage),
                    ],
                )
        except (duckdb.Error, OSError, StorageError) as exc:
            raise StorageError(f"Cannot save {stage} response: {exc}") from exc

    def _expected_markdown_files(
        self, prompt: str, answers: tuple[ModelAnswer, ...], synthesis: ModelAnswer
    ) -> dict[str, str]:
        files: dict[str, str] = {}
        if (
            self.settings.save_prompt.markdown
            and prompt.strip()
            and synthesis.content
            and synthesis.content.strip()
        ):
            files["prompt.md"] = prompt
        for index, answer in enumerate(answers, 1):
            mode = self.settings.save_initial_answers
            if mode.markdown and answer.content and answer.content.strip():
                files[self._answer_filename(answer, "initial", index)] = self._answer_markdown(
                    answer, "initial"
                )
        if (
            self.settings.save_synthesis.markdown
            and synthesis.content
            and synthesis.content.strip()
        ):
            files[self._answer_filename(synthesis, "synthesis", 0)] = self._answer_markdown(
                synthesis, "synthesis"
            )
        return files

    @staticmethod
    def _is_synthesis_only(files: dict[str, str]) -> bool:
        """Return True when synthesis.md is the only Markdown file requested."""
        return set(files.keys()) == {"synthesis.md"}

    def save_markdown_artifacts(
        self, prompt: str, answers: tuple[ModelAnswer, ...], synthesis: ModelAnswer
    ) -> None:
        """Atomically write requested Markdown outputs after synthesis succeeds.

        When synthesis.md is the only requested file, write it directly under
        markdown_dir as <slug>-synthesis-<stamp>.md instead of creating a
        subfolder just to hold synthesis.md.
        """
        files = self._expected_markdown_files(prompt, answers, synthesis)
        if not files:
            return
        try:
            self.settings.markdown_dir.mkdir(parents=True, exist_ok=True)
            if self._is_synthesis_only(files):
                self._save_single_synthesis_file(files["synthesis.md"])
                return
            base_name = self._folder_base_name()
            staging = self.settings.markdown_dir / f".{base_name}-{uuid4().hex}.tmp"
            staging.mkdir(parents=True, exist_ok=False)
            try:
                for filename, content in files.items():
                    (staging / filename).write_text(content, encoding="utf-8")
                candidate = self.settings.markdown_dir / base_name
                suffix = 2
                while True:
                    try:
                        staging.rename(candidate)
                        self.folder = candidate
                        return
                    except FileExistsError:
                        candidate = self.settings.markdown_dir / f"{base_name}-{suffix}"
                        suffix += 1
            except OSError:
                shutil.rmtree(staging, ignore_errors=True)
                raise
        except OSError as exc:
            raise StorageError(f"Cannot save Markdown artifacts: {exc}") from exc

    def _save_single_synthesis_file(self, content: str) -> None:
        """Atomically write a lone synthesis directly under markdown_dir."""
        base_name = self._single_file_base_name()
        stem = base_name[: -len(".md")]
        candidate = self.settings.markdown_dir / base_name
        suffix = 2
        while candidate.exists():
            candidate = self.settings.markdown_dir / f"{stem}-{suffix}.md"
            suffix += 1
        staging = self.settings.markdown_dir / f".{stem}-{uuid4().hex}.tmp"
        try:
            staging.write_text(content, encoding="utf-8")
            try:
                staging.rename(candidate)
            except FileExistsError:
                # A concurrent run claimed the name; pick the next free suffix.
                suffix = 2
                while True:
                    candidate = self.settings.markdown_dir / f"{stem}-{suffix}.md"
                    suffix += 1
                    if not candidate.exists():
                        break
                staging.rename(candidate)
            self.folder = candidate
        except OSError:
            staging.unlink(missing_ok=True)
            raise

    def _answer_filename(self, answer: ModelAnswer, stage: str, index: int) -> str:
        slug = re.sub(r"[^a-zA-Z0-9_.-]+", "-", answer.requested_model)[:100]
        return f"initial-{index:03d}-{slug}.md" if stage == "initial" else "synthesis.md"

    def _answer_markdown(self, answer: ModelAnswer, stage: str) -> str:
        assert answer.content is not None
        return (
            f"# {'Synthesis' if stage == 'synthesis' else 'Initial answer'}\n\n"
            f"- Requested model: `{answer.requested_model}`\n"
            f"- Answering model: `{answer.model}`\n\n"
            f"{answer.content}"
        )

    def _slug(self) -> str:
        words = [
            word
            for word in re.findall(r"[^\W_]+", self.prompt.lower())
            if word not in _PROMPT_STOP_WORDS
        ][:6]
        slug = "-".join(words) or "council-run"
        slug = re.sub(r"[^a-z0-9-]+", "-", slug).strip("-")[:72] or "council-run"
        return slug

    def report_base_name(self, stamp: str) -> str:
        """Return the file name for the self-contained HTML report of this run."""
        return f"{self._slug()}-report-{stamp}{HTML_REPORT_SUFFIX}"

    def _folder_base_name(self) -> str:
        return f"{self._slug()}-{self._markdown_stamp}"

    def _single_file_base_name(self) -> str:
        return f"{self._slug()}-synthesis-{self._markdown_stamp}.md"

    def _ensure_folder(self) -> None:
        if self.folder is None:
            base_name = self._folder_base_name()
            candidate = self.settings.markdown_dir / base_name
            suffix = 2
            while True:
                try:
                    candidate.mkdir(parents=True, exist_ok=False)
                    self.folder = candidate
                    break
                except FileExistsError:
                    candidate = self.settings.markdown_dir / f"{base_name}-{suffix}"
                    suffix += 1

    def _write(self, filename: str, text: str) -> None:
        self._ensure_folder()
        assert self.folder is not None
        (self.folder / filename).write_text(text, encoding="utf-8")


def export_markdown(
    *,
    env_file: str | Path | None = None,
    settings: Settings | None = None,
    **overrides: Any,
) -> tuple[Path, ...]:
    """Export stored DuckDB runs with non-empty syntheses into Markdown.

    Multi-file runs become folders; a lone synthesis becomes a single
    <slug>-synthesis-<stamp>.md file directly under markdown_dir.
    Respects configured save modes ( defaulting to exporting prompt, initial answers,
    and synthesis when none of the three are configured for Markdown) and skips runs
    whose target folder/file and contents already exist in markdown_dir.
    """
    config = settings if settings is not None else Settings.from_env(env_file)
    config = config.with_overrides(**overrides)
    if not any(
        mode.markdown
        for mode in (config.save_prompt, config.save_initial_answers, config.save_synthesis)
    ):
        config = config.with_overrides(
            save_prompt="markdown",
            save_initial_answers="markdown",
            save_synthesis="markdown",
        )
    if not config.duckdb_path.is_file():
        raise StorageError(f"DuckDB file {config.duckdb_path} does not exist.")
    exported: list[Path] = []
    try:
        with duckdb.connect(str(config.duckdb_path), read_only=True) as connection:
            tables = {row[0] for row in connection.execute("SHOW TABLES").fetchall()}
            if not {"questions", "responses"}.issubset(tables):
                return ()
            question_rows = connection.execute(
                """
                SELECT id, strftime(created_at AT TIME ZONE 'UTC', '%Y%m%dT%H%M'), prompt
                FROM questions
                ORDER BY created_at
                """
            ).fetchall()
            for question_id, stamp, prompt in question_rows:
                response_rows = connection.execute(
                    """
                    SELECT stage, requested_model, model, content
                    FROM responses
                    WHERE question_id = ?
                    ORDER BY created_at
                    """,
                    [question_id],
                ).fetchall()
                initial_answers = tuple(
                    ModelAnswer(
                        requested_model=req_model,
                        model=model_id,
                        content=content,
                    )
                    for stage, req_model, model_id, content in response_rows
                    if stage == "initial" and content and content.strip()
                )
                syntheses = [
                    ModelAnswer(
                        requested_model=req_model,
                        model=model_id,
                        content=content,
                    )
                    for stage, req_model, model_id, content in response_rows
                    if stage == "synthesis" and content and content.strip()
                ]
                if not syntheses:
                    continue
                synthesis = syntheses[-1]
                prompt_text = prompt or ""
                storage = RunStorage(config, str(question_id), prompt_text)
                if isinstance(stamp, str) and stamp:
                    storage._markdown_stamp = stamp
                expected_files = storage._expected_markdown_files(
                    prompt_text, initial_answers, synthesis
                )
                if not expected_files:
                    continue
                if RunStorage._is_synthesis_only(expected_files):
                    single_base = storage._single_file_base_name()
                    stem = single_base[: -len(".md")]
                    already_exported = False
                    if config.markdown_dir.is_dir():
                        for candidate in config.markdown_dir.glob(f"{stem}*.md"):
                            if (
                                candidate.is_file()
                                and candidate.read_text(encoding="utf-8")
                                == expected_files["synthesis.md"]
                            ):
                                already_exported = True
                                break
                    if already_exported:
                        continue
                    storage.save_markdown_artifacts(prompt_text, initial_answers, synthesis)
                    if storage.folder is not None:
                        exported.append(storage.folder)
                    continue
                base_name = storage._folder_base_name()
                already_exported = False
                if config.markdown_dir.is_dir():
                    for candidate in config.markdown_dir.glob(f"{base_name}*"):
                        if candidate.is_dir() and all(
                            (candidate / filename).is_file()
                            and (candidate / filename).read_text(encoding="utf-8") == content
                            for filename, content in expected_files.items()
                        ):
                            already_exported = True
                            break
                if already_exported:
                    continue
                storage.save_markdown_artifacts(prompt_text, initial_answers, synthesis)
                if storage.folder is not None:
                    exported.append(storage.folder)
    except (duckdb.Error, OSError) as exc:
        raise StorageError(f"Cannot export Markdown artifacts from DuckDB: {exc}") from exc
    return tuple(exported)


def _write_report(directory: Path, base_name: str, content: str) -> Path | None:
    """Write one report atomically, or return None when an identical one exists.

    Re-exporting is idempotent: an existing file with byte-identical content is left
    alone, and any other collision takes the next free ``-N`` suffix so concurrent
    exports never clobber each other.
    """
    stem = base_name[: -len(HTML_REPORT_SUFFIX)]
    if directory.is_dir():
        for existing in sorted(directory.glob(f"{stem}*{HTML_REPORT_SUFFIX}")):
            try:
                if existing.is_file() and existing.read_text(encoding="utf-8") == content:
                    logger.debug("Identical report already present, skipping: %s", existing)
                    return None
            except (OSError, UnicodeDecodeError) as exc:
                logger.warning("Cannot compare existing report %s: %s", existing, exc)
    try:
        directory.mkdir(parents=True, exist_ok=True)
        candidate = directory / base_name
        suffix = 2
        while candidate.exists():
            candidate = directory / f"{stem}-{suffix}{HTML_REPORT_SUFFIX}"
            suffix += 1
        staging = directory / f".{stem}-{uuid4().hex}.tmp"
        try:
            staging.write_text(content, encoding="utf-8")
            try:
                staging.rename(candidate)
            except FileExistsError:
                # A concurrent export claimed the name; take the next free suffix.
                suffix = 2
                while True:
                    candidate = directory / f"{stem}-{suffix}{HTML_REPORT_SUFFIX}"
                    suffix += 1
                    if not candidate.exists():
                        break
                staging.rename(candidate)
        except OSError:
            staging.unlink(missing_ok=True)
            raise
    except OSError as exc:
        raise StorageError(f"Cannot write HTML report into {directory}: {exc}") from exc
    return candidate


def export_html(
    *,
    env_file: str | Path | None = None,
    settings: Settings | None = None,
    run_id: str | None = None,
    **overrides: Any,
) -> tuple[Path, ...]:
    """Export stored DuckDB runs with non-empty syntheses as self-contained HTML reports.

    Each report is one ``<slug>-report-<stamp>.html`` file under ``html_dir`` that
    inlines its own styling, fonts, and scripts, so it can be read and printed with
    no network access. Pass ``run_id`` to export a single stored run. Runs whose
    report is already present with identical content are skipped.
    """
    config = settings if settings is not None else Settings.from_env(env_file)
    config = config.with_overrides(**overrides)
    if not config.duckdb_path.is_file():
        raise StorageError(f"DuckDB file {config.duckdb_path} does not exist.")
    runs = load_history(settings=config, limit=EXPORT_RUN_LIMIT)
    exported: list[Path] = []
    for run in runs:
        if run_id is not None and str(run.get("id")) != run_id:
            continue
        synthesis = run.get("synthesis")
        if not isinstance(synthesis, str) or not synthesis.strip():
            continue
        storage = RunStorage(config, str(run.get("id")), run.get("prompt") or "")
        stamp = datetime.fromtimestamp(float(run.get("created_at") or 0.0), tz=UTC).strftime(
            ARTIFACT_STAMP_FORMAT
        )
        base_name = storage.report_base_name(stamp)
        try:
            report = build_report_html(run)
        except StorageError as exc:
            logger.error("Cannot render HTML report for run %s: %s", run.get("id"), exc)
            raise
        path = _write_report(config.html_dir, base_name, report)
        if path is not None:
            exported.append(path)
    return tuple(exported)


def load_history(
    *,
    env_file: str | Path | None = None,
    settings: Settings | None = None,
    limit: int = 100,
    **overrides: Any,
) -> list[dict[str, Any]]:
    """Load persisted council runs from DuckDB ordered newest first."""
    base = settings or Settings.from_env(env_file)
    config = base.with_overrides(**overrides)
    if not config.duckdb_path.is_file():
        return []
    try:
        with duckdb.connect(str(config.duckdb_path), read_only=True) as connection:
            tables = {row[0] for row in connection.execute("SHOW TABLES").fetchall()}
            if "questions" not in tables:
                return []
            question_rows = connection.execute(
                """
                SELECT id, epoch(created_at), prompt, source_path
                FROM questions
                ORDER BY created_at DESC
                LIMIT ?
                """,
                [limit],
            ).fetchall()
            has_responses = "responses" in tables
            history: list[dict[str, Any]] = []
            for question_id, created_ts, prompt, source_path in question_rows:
                answers: list[dict[str, Any]] = []
                synthesis: str | None = None
                judge: str | None = None
                judge_requested: str | None = None
                synthesis_usage: dict[str, Any] = {}
                if has_responses:
                    response_rows = connection.execute(
                        """
                        SELECT stage, requested_model, model, content, error, usage
                        FROM responses
                        WHERE question_id = ?
                        ORDER BY created_at ASC
                        """,
                        [question_id],
                    ).fetchall()
                    for stage, req_model, model_id, content, error, raw_usage in response_rows:
                        parsed_usage: dict[str, Any] = {}
                        if isinstance(raw_usage, str) and raw_usage:
                            try:
                                loaded = json.loads(raw_usage)
                                if isinstance(loaded, dict):
                                    parsed_usage = loaded
                            except json.JSONDecodeError:
                                parsed_usage = {}
                        elif isinstance(raw_usage, dict):
                            parsed_usage = raw_usage
                        if stage == "initial":
                            answers.append(
                                {
                                    "requested_model": req_model,
                                    "model": model_id,
                                    "content": content,
                                    "error": error,
                                    "usage": parsed_usage,
                                }
                            )
                        elif stage == "synthesis" and content and content.strip():
                            synthesis = content
                            judge = model_id
                            judge_requested = req_model
                            synthesis_usage = parsed_usage
                history.append(
                    {
                        "id": str(question_id),
                        "created_at": float(created_ts) if created_ts is not None else 0.0,
                        "prompt": prompt,
                        "source_path": source_path,
                        "answers": answers,
                        "synthesis": synthesis,
                        "judge": judge,
                        "judge_requested": judge_requested,
                        "synthesis_usage": synthesis_usage,
                        "source": "db",
                    }
                )
            return history
    except (duckdb.Error, OSError) as exc:
        raise StorageError(f"Cannot load history from DuckDB: {exc}") from exc
