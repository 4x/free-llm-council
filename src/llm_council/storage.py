"""Opt-in storage, independent for prompts, initial answers, and synthesis."""

from __future__ import annotations

import json
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
        self._markdown_stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M")
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

    def save_markdown_artifacts(
        self, prompt: str, answers: tuple[ModelAnswer, ...], synthesis: ModelAnswer
    ) -> None:
        """Atomically write Markdown outputs after all model calls succeeded."""
        files: dict[str, str] = {}
        if self.settings.save_prompt.markdown and synthesis.content and synthesis.content.strip():
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
        if not files:
            return
        try:
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

    def _folder_base_name(self) -> str:
        words = [
            word
            for word in re.findall(r"[^\W_]+", self.prompt.lower())
            if word not in _PROMPT_STOP_WORDS
        ][:6]
        slug = "-".join(words) or "council-run"
        slug = re.sub(r"[^a-z0-9-]+", "-", slug).strip("-")[:72] or "council-run"
        return f"{slug}-{self._markdown_stamp}"

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
