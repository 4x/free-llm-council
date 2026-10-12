"""Configuration precedence: call arguments > environment > .env > defaults."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, fields, replace
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

from dotenv import dotenv_values

from .errors import ConfigurationError

LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")


class SaveMode(StrEnum):
    NONE = "none"
    DUCKDB = "duckdb"
    MARKDOWN = "markdown"
    BOTH = "both"

    @classmethod
    def parse(cls, value: str | SaveMode) -> SaveMode:
        if isinstance(value, cls):
            return value
        normalized = value.strip().lower()
        if normalized in {"no", "neither", "false", "off"}:
            normalized = "none"
        try:
            return cls(normalized)
        except ValueError as exc:
            raise ConfigurationError(
                f"Invalid save mode {value!r}; use none, duckdb, markdown, or both."
            ) from exc

    @property
    def duckdb(self) -> bool:
        return self in {self.DUCKDB, self.BOTH}

    @property
    def markdown(self) -> bool:
        return self in {self.MARKDOWN, self.BOTH}


@dataclass(frozen=True)
class Settings:
    api_key: str = ""
    base_url: str = "https://openrouter.ai/api/v1"
    save_prompt: SaveMode = SaveMode.DUCKDB
    save_initial_answers: SaveMode = SaveMode.NONE
    save_synthesis: SaveMode = SaveMode.NONE
    duckdb_path: Path = Path("council.duckdb")
    markdown_dir: Path = Path("council-output")
    judge: str = "random"
    judge_scope: Literal["free", "all"] = "free"
    max_concurrency: int = 3
    request_interval: float = 3.1
    timeout: float = 120.0
    max_retries: int = 2
    max_retry_wait: float = 60.0
    max_tokens: int = 2048
    log_level: str = "INFO"
    excluded_models: tuple[str, ...] = ()

    @classmethod
    def from_env(cls, env_file: str | Path | None = None) -> Settings:
        path = Path(env_file) if env_file is not None else Path.cwd() / ".env"
        values = {**dotenv_values(path), **os.environ}
        mapping = {
            "api_key": "OPENROUTER_API_KEY",
            "base_url": "OPENROUTER_BASE_URL",
            "save_prompt": "SAVE_PROMPT",
            "save_initial_answers": "SAVE_INITIAL_ANSWERS",
            "save_synthesis": "SAVE_SYNTHESIS",
            "duckdb_path": "DUCKDB_PATH",
            "markdown_dir": "MARKDOWN_DIR",
            "judge": "JUDGE",
            "judge_scope": "JUDGE_SCOPE",
            "max_concurrency": "MAX_CONCURRENCY",
            "request_interval": "REQUEST_INTERVAL",
            "timeout": "REQUEST_TIMEOUT",
            "max_retries": "MAX_RETRIES",
            "max_retry_wait": "MAX_RETRY_WAIT",
            "max_tokens": "MAX_TOKENS",
            "log_level": "LOG_LEVEL",
            "excluded_models": "EXCLUDED_MODELS",
        }
        overrides = {
            name: values[key] for name, key in mapping.items() if values.get(key) is not None
        }
        return cls().with_overrides(**overrides)

    def with_overrides(self, **overrides: Any) -> Settings:
        valid = {field.name for field in fields(self)}
        unknown = overrides.keys() - valid
        if unknown:
            raise ConfigurationError(f"Unknown settings: {', '.join(sorted(unknown))}")
        converted = {key: value for key, value in overrides.items() if value is not None}
        try:
            if "excluded_models" in converted:
                value = converted["excluded_models"]
                if isinstance(value, str):
                    value = value.split(",")
                if not isinstance(value, (tuple, list)) or any(
                    not isinstance(item, str) for item in value
                ):
                    raise ValueError(
                        "excluded_models must be model IDs or a comma-separated string"
                    )
                converted["excluded_models"] = tuple(
                    dict.fromkeys(item.strip() for item in value if item.strip())
                )
            if "log_level" in converted:
                converted["log_level"] = str(converted["log_level"]).strip().upper()
            for key in ("save_prompt", "save_initial_answers", "save_synthesis"):
                if key in converted:
                    converted[key] = SaveMode.parse(converted[key])
            for key in ("duckdb_path", "markdown_dir"):
                if key in converted:
                    converted[key] = Path(converted[key])
            for key in ("max_concurrency", "max_retries", "max_tokens"):
                if key in converted:
                    value = converted[key]
                    if isinstance(value, float) and not value.is_integer():
                        raise ValueError(f"{key} must be an integer")
                    converted[key] = int(value)
            for key in ("request_interval", "timeout", "max_retry_wait"):
                if key in converted:
                    converted[key] = float(converted[key])
            result = replace(self, **converted)
        except (ValueError, TypeError) as exc:
            raise ConfigurationError(f"Invalid configuration: {exc}") from exc
        result.validate()
        return result

    def validate(self) -> None:
        if self.log_level not in LOG_LEVELS:
            raise ConfigurationError(f"LOG_LEVEL must be one of {', '.join(LOG_LEVELS)}.")
        if self.judge_scope not in {"free", "all"}:
            raise ConfigurationError("JUDGE_SCOPE must be free or all.")
        if not self.judge.strip():
            raise ConfigurationError("JUDGE must be a strategy or an explicit model ID.")
        if self.max_concurrency < 1 or self.max_tokens == 0 or self.max_retries < 0:
            raise ConfigurationError(
                "Concurrency must be positive, MAX_TOKENS cannot be zero, "
                "and retries cannot be negative."
            )
        for name in ("request_interval", "timeout", "max_retry_wait"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ConfigurationError(f"{name} must be a finite non-negative number.")
        if self.timeout == 0:
            raise ConfigurationError("Timeout must be positive.")
        parsed = urlsplit(self.base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ConfigurationError("OPENROUTER_BASE_URL must be an HTTP(S) URL.")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ConfigurationError(
                "OPENROUTER_BASE_URL cannot include credentials, query, or fragment."
            )
