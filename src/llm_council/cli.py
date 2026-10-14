"""Command line entrypoint; stdout renders the synthesized answer as Markdown."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown

from .config import LOG_LEVELS
from .council import _read_prompt, query
from .errors import ConfigurationError, CouncilError
from .judges import STRATEGIES
from .storage import export_markdown


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        prog="llm-council",
        description="Ask every currently free OpenRouter text model, then synthesize with a judge.",
        epilog="Judge strategies: " + ", ".join(STRATEGIES) + "; or an explicit model ID.",
    )
    result.add_argument("--question", "-q", dest="prompt", help="Literal question text")
    result.add_argument("--file", "-f", help="UTF-8 context file; combines before --question")
    result.set_defaults(followup_prompt=None)
    result.add_argument(
        "--followup", dest="followup_prompt", help="Ask a follow-up using the latest synthesis"
    )
    result.add_argument(
        "--export-markdown",
        action="store_true",
        help="Export stored DuckDB runs with syntheses into Markdown folders and exit",
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
        "--judge-prompt", help="UTF-8 judge system prompt file (overrides JUDGE_PROMPT)"
    )
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


def _latest_history_path() -> Path:
    return Path.cwd() / ".llm-council" / "latest-conversation.json"


def _load_latest_conversation() -> tuple[tuple[str, str], ...]:
    synthesis = _load_latest_synthesis()
    history_path = _latest_history_path()
    if history_path.is_file():
        try:
            payload = json.loads(history_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, ValueError) as exc:
            raise ConfigurationError(
                f"Cannot read cached conversation {history_path}: {exc}"
            ) from exc
        if (
            isinstance(payload, list)
            and payload
            and all(
                isinstance(turn, list)
                and len(turn) == 2
                and all(isinstance(part, str) and part.strip() for part in turn)
                for turn in payload
            )
        ):
            if payload[-1][1] == synthesis:
                return tuple((turn[0], turn[1]) for turn in payload)
            return (("Previous question", synthesis),)
        raise ConfigurationError(
            f"Cached conversation {history_path} is invalid; run a new query first."
        )
    return (("Previous question", synthesis),)


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


def _atomic_write_text(path: Path, content: str, prefix: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=prefix, delete=False
        ) as temporary:
            temporary.write(content)
            temporary_path = Path(temporary.name)
        temporary_path.replace(path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _cache_latest_synthesis(
    synthesis: str, conversation: Sequence[tuple[str, str]] | None = None
) -> None:
    _atomic_write_text(_latest_synthesis_path(), synthesis, ".latest-")
    history_path = _latest_history_path()
    if conversation is not None:
        _atomic_write_text(
            history_path,
            json.dumps([[user, assistant] for user, assistant in conversation], ensure_ascii=False),
            ".history-",
        )
    else:
        history_path.unlink(missing_ok=True)


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
    arg_parser = parser()
    args = arg_parser.parse_args(argv)
    followup_prompt = args.followup_prompt
    should_export = args.export_markdown
    if should_export and (
        followup_prompt is not None
        or args.file is not None
        or args.prompt is not None
        or args.no_save
    ):
        arg_parser.error(
            "--export-markdown cannot be combined with --question, --file, --followup, or --no-save"
        )
    if followup_prompt is not None and (args.file is not None or args.prompt is not None):
        arg_parser.error("--followup cannot be combined with --question or --file")
    if not should_export and followup_prompt is None and args.file is None and args.prompt is None:
        arg_parser.error("provide --question, --file, or both")
    try:
        options = vars(args)
        options.pop("followup_prompt")
        options.pop("export_markdown")
        should_copy = options.pop("copy")
        if should_export:
            options.pop("prompt")
            options.pop("file")
            options.pop("no_save")
            folders = export_markdown(**options)
            for folder in folders:
                print(folder)
            return 0
        prior_turns = _load_latest_conversation() if followup_prompt is not None else ()
        previous_synthesis = prior_turns[-1][1] if prior_turns else None
        if followup_prompt is not None:
            options["prompt"] = followup_prompt
        answer = query(
            **options,
            literal_prompt=args.file is None and followup_prompt is None,
            previous_synthesis=previous_synthesis,
            prior_turns=prior_turns or None,
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
            current_prompt = (
                followup_prompt
                if followup_prompt is not None
                else _read_prompt(args.prompt, args.file, literal_prompt=args.file is None)[0]
            )
            _cache_latest_synthesis(answer, (*prior_turns, (current_prompt, answer)))
        if should_copy:
            _copy_to_clipboard(answer)
    except (OSError, CouncilError) as exc:
        print(f"llm-council: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("llm-council: interrupted", file=sys.stderr)
        return 130
    return 0
