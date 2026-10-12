"""Command line entrypoint; stdout renders the synthesized answer as Markdown."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown

from .config import LOG_LEVELS
from .council import query
from .errors import ConfigurationError, CouncilError
from .judges import STRATEGIES


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        prog="llm-council",
        description="Ask every currently free OpenRouter text model, then synthesize with a judge.",
        epilog="Judge strategies: " + ", ".join(STRATEGIES) + "; or an explicit model ID.",
    )
    inputs = result.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--question", "-q", dest="prompt", help="Literal prompt text")
    inputs.add_argument("--file", "-f", help="UTF-8 prompt file")
    inputs.add_argument(
        "--followup", dest="followup_prompt", help="Ask a follow-up using the latest synthesis"
    )
    result.add_argument("--copy", action="store_true", help="Copy the synthesis to the clipboard")
    result.add_argument("--env-file", help="Settings file (default: .env in the current directory)")
    modes = ("none", "no", "neither", "duckdb", "markdown", "both")
    for option in ("prompt", "initial-answers", "synthesis"):
        result.add_argument(f"--save-{option}", type=str.lower, choices=modes)
    result.add_argument(
        "--no-save", action="store_true", help="Disable ALL database/Markdown saves"
    )
    result.add_argument("--judge", help="Judge strategy or OpenRouter model ID (default: random)")
    result.add_argument(
        "--judge-scope", choices=("free", "all"), help="Allow paid judges with 'all'"
    )
    result.add_argument("--duckdb-path", help="DuckDB file (default: council.duckdb)")
    result.add_argument("--markdown-dir", help="Markdown root directory (default: council-output)")
    result.add_argument(
        "--log-level",
        type=str.upper,
        choices=LOG_LEVELS,
        help="Log verbosity (default: INFO; overrides LOG_LEVEL from .env)",
    )
    result.add_argument(
        "--excluded-models", help="Comma-separated model IDs to skip (overrides EXCLUDED_MODELS)"
    )
    result.add_argument("--max-concurrency", type=int)
    result.add_argument("--request-interval", type=float, help="Seconds between request starts")
    result.add_argument("--timeout", type=float)
    result.add_argument("--max-retries", type=int)
    result.add_argument("--max-retry-wait", type=float)
    result.add_argument("--max-tokens", type=int, help="Maximum output tokens per model and judge")
    result.add_argument("--version", action="version", version="%(prog)s 0.1.0")
    return result


def _latest_synthesis_path() -> Path:
    return Path.cwd() / ".llm-council" / "latest-synthesis.md"


def _load_latest_synthesis() -> str:
    path = _latest_synthesis_path()
    try:
        synthesis = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ConfigurationError(
            "No previous synthesis is cached. Run a query first without --no-save."
        ) from exc
    except (OSError, UnicodeError) as exc:
        raise ConfigurationError(f"Cannot read cached synthesis {path}: {exc}") from exc
    if not synthesis.strip():
        raise ConfigurationError(f"Cached synthesis {path} is empty; run a new query first.")
    return synthesis


def _cache_latest_synthesis(synthesis: str) -> None:
    path = _latest_synthesis_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=".latest-", delete=False
        ) as temporary:
            temporary.write(synthesis)
            temporary_path = Path(temporary.name)
        temporary_path.replace(path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _copy_to_clipboard(synthesis: str) -> None:
    if sys.platform == "win32":
        commands = (("clip",),)
    elif sys.platform == "darwin":
        commands = (("pbcopy",),)
    else:
        commands = (
            ("wl-copy",),
            ("xclip", "-selection", "clipboard"),
            ("xsel", "--clipboard", "--input"),
        )
    command = next((candidate for candidate in commands if shutil.which(candidate[0])), None)
    if command is None:
        choices = ", ".join(candidate[0] for candidate in commands)
        raise ConfigurationError(
            f"No clipboard utility found ({choices}); install one or omit --copy."
        )
    try:
        subprocess.run(command, input=synthesis, text=True, check=True, timeout=15)
    except (OSError, subprocess.SubprocessError) as exc:
        raise ConfigurationError(f"Could not copy synthesis to the clipboard: {exc}") from exc


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        options = vars(args)
        followup_prompt = options.pop("followup_prompt")
        should_copy = options.pop("copy")
        previous_synthesis = _load_latest_synthesis() if followup_prompt is not None else None
        if followup_prompt is not None:
            options["prompt"] = followup_prompt
        answer = query(
            **options,
            literal_prompt=args.file is None,
            previous_synthesis=previous_synthesis,
        )
    except CouncilError as exc:
        print(f"llm-council: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("llm-council: interrupted", file=sys.stderr)
        return 130

    Console().print(Markdown(answer), markup=False)
    try:
        if not args.no_save:
            _cache_latest_synthesis(answer)
        if should_copy:
            _copy_to_clipboard(answer)
    except (OSError, CouncilError) as exc:
        print(f"llm-council: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("llm-council: interrupted", file=sys.stderr)
        return 130
    return 0
