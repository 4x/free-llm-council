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
from .council import _read_prompt, improve_prompt, query
from .errors import ConfigurationError, CouncilError
from .judges import STRATEGIES
from .storage import export_html, export_markdown
from .web import DEFAULT_WEB_HOST, DEFAULT_WEB_PORT, serve_web


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        prog="llm-council",
        description="Ask every currently free OpenRouter text model, then synthesize with a judge.",
        epilog="Judge strategies: " + ", ".join(STRATEGIES) + "; or an explicit model ID.",
    )
    result.add_argument("--question", "-q", dest="prompt", help="Literal question text")
    result.add_argument("--file", "-f", help="UTF-8 context file; combines before --question")
    result.add_argument(
        "--improve",
        "--improve-prompt",
        dest="improve_prompt",
        nargs="?",
        const="",
        default=None,
        metavar="PROMPT",
        help="Send a prompt to one LLM and display numbered improved versions",
    )
    result.add_argument(
        "--prompt",
        dest="prompt_index",
        type=int,
        metavar="N",
        help="Send the Nth improved prompt from the latest --improve run",
    )
    result.set_defaults(followup_prompt=None)
    result.add_argument(
        "--followup", dest="followup_prompt", help="Ask a follow-up using the latest synthesis"
    )
    result.add_argument(
        "--export-markdown",
        action="store_true",
        help="Export stored DuckDB runs with syntheses into Markdown folders and exit",
    )
    result.add_argument(
        "--export-html",
        action="store_true",
        help=("Export stored DuckDB runs with syntheses as self-contained HTML reports and exit"),
    )
    result.add_argument(
        "--web",
        "--serve",
        dest="web",
        action="store_true",
        help="Start the interactive local HTML web interface",
    )
    result.add_argument(
        "--host",
        default=DEFAULT_WEB_HOST,
        help=f"Bind host for --web (default: {DEFAULT_WEB_HOST})",
    )
    result.add_argument(
        "--port",
        type=int,
        default=DEFAULT_WEB_PORT,
        help=f"Bind port for --web (default: {DEFAULT_WEB_PORT})",
    )
    result.add_argument(
        "--open-browser",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Open the default browser at the bound URL when starting --web",
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
    result.add_argument("--html-dir", help="HTML report directory (default: html-output)")
    result.add_argument(
        "--log-level",
        type=str.upper,
        choices=LOG_LEVELS,
        help="Log verbosity (default: INFO; overrides LOG_LEVEL from .env)",
    )
    result.add_argument(
        "--excluded-models", help="Comma-separated model IDs to skip (overrides EXCLUDED_MODELS)"
    )
    result.add_argument(
        "--gemini-models",
        help="Comma-separated Gemini model IDs to include (overrides GEMINI_MODELS)",
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


def _improved_prompts_path() -> Path:
    return Path.cwd() / ".llm-council" / "improved-prompts.json"


def _cache_improved_prompts(prompts: Sequence[str]) -> None:
    _atomic_write_text(
        _improved_prompts_path(),
        json.dumps(list(prompts), ensure_ascii=False, indent=2),
        ".improved-",
    )


def _load_improved_prompt(index: int) -> str:
    if index < 1:
        raise ConfigurationError(f"--prompt index must be at least 1 (got {index}).")
    path = _improved_prompts_path()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigurationError(
            "No improved prompts are cached. Run with --improve or --improve-prompt first."
        ) from exc
    except (OSError, UnicodeError, ValueError) as exc:
        raise ConfigurationError(f"Cannot read cached improved prompts {path}: {exc}") from exc
    if (
        not isinstance(payload, list)
        or not payload
        or any(not isinstance(item, str) or not item.strip() for item in payload)
    ):
        raise ConfigurationError(f"Cached improved prompts {path} is invalid; run --improve first.")
    if index > len(payload):
        raise ConfigurationError(
            f"--prompt {index} is out of range; {len(payload)} improved "
            f"prompt{'s are' if len(payload) != 1 else ' is'} cached (1-{len(payload)})."
        )
    return str(payload[index - 1])


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
    should_export_html = args.export_html
    should_serve_web = args.web
    improve_arg = args.improve_prompt
    prompt_index = args.prompt_index
    if should_serve_web and (
        should_export
        or should_export_html
        or followup_prompt is not None
        or args.file is not None
        or args.prompt is not None
        or improve_arg is not None
        or prompt_index is not None
    ):
        arg_parser.error(
            "--web cannot be combined with --question, --file, --followup, "
            "--improve, --prompt, --export-markdown, or --export-html"
        )
    if should_export and (
        followup_prompt is not None
        or args.file is not None
        or args.prompt is not None
        or improve_arg is not None
        or prompt_index is not None
        or args.no_save
    ):
        arg_parser.error(
            "--export-markdown cannot be combined with --question, --file, --followup, "
            "--improve, --prompt, or --no-save"
        )
    if should_export_html and (
        followup_prompt is not None
        or args.file is not None
        or args.prompt is not None
        or improve_arg is not None
        or prompt_index is not None
        or args.no_save
    ):
        arg_parser.error(
            "--export-html cannot be combined with --question, --file, --followup, "
            "--improve, --prompt, or --no-save"
        )
    if improve_arg is not None and (
        args.prompt is not None or followup_prompt is not None or prompt_index is not None
    ):
        arg_parser.error("--improve cannot be combined with --question, --followup, or --prompt")
    if prompt_index is not None and (args.prompt is not None or followup_prompt is not None):
        arg_parser.error("--prompt cannot be combined with --question or --followup")
    if followup_prompt is not None and (args.file is not None or args.prompt is not None):
        arg_parser.error("--followup cannot be combined with --question or --file")
    if improve_arg is not None and not improve_arg.strip() and args.file is None:
        arg_parser.error("provide a prompt to --improve, --file, or both")
    if (
        not should_serve_web
        and not should_export
        and not should_export_html
        and followup_prompt is None
        and improve_arg is None
        and prompt_index is None
        and args.file is None
        and args.prompt is None
    ):
        arg_parser.error("provide --question, --file, --improve, --prompt, or --web")
    try:
        options = vars(args)
        options.pop("followup_prompt")
        options.pop("export_markdown")
        options.pop("export_html")
        options.pop("web")
        web_host = options.pop("host")
        web_port = options.pop("port")
        open_browser = bool(options.pop("open_browser", False))
        options.pop("improve_prompt")
        options.pop("prompt_index")
        should_copy = options.pop("copy")
        if should_serve_web:
            options.pop("prompt")
            options.pop("file")
            no_save = bool(options.pop("no_save", False))
            if no_save:
                options["save_prompt"] = "none"
                options["save_initial_answers"] = "none"
                options["save_synthesis"] = "none"
            serve_web(host=web_host, port=web_port, open_browser=open_browser, **options)
            return 0
        if should_export or should_export_html:
            options.pop("prompt")
            options.pop("file")
            options.pop("no_save")
            written: list[Path] = []
            if should_export:
                written.extend(export_markdown(**options))
            if should_export_html:
                written.extend(export_html(**options))
            for path in written:
                print(path)
            return 0
        if improve_arg is not None:
            no_save = bool(options.pop("no_save", False))
            for unused in (
                "save_prompt",
                "save_initial_answers",
                "save_synthesis",
                "judge_prompt",
                "duckdb_path",
                "markdown_dir",
            ):
                options.pop(unused, None)
            options["prompt"] = improve_arg if improve_arg.strip() else None
            improved_versions = improve_prompt(
                **options,
                literal_prompt=args.file is None,
            )
            if not no_save:
                _cache_improved_prompts(improved_versions)
            formatted = "\n\n".join(
                f"{idx}. {version}" for idx, version in enumerate(improved_versions, 1)
            )
            print(formatted)
            if should_copy:
                _copy_to_clipboard(formatted)
            return 0
        selected_prompt = (
            _load_improved_prompt(prompt_index) if prompt_index is not None else args.prompt
        )
        prior_turns = _load_latest_conversation() if followup_prompt is not None else ()
        previous_synthesis = prior_turns[-1][1] if prior_turns else None
        if followup_prompt is not None:
            options["prompt"] = followup_prompt
        elif prompt_index is not None:
            options["prompt"] = selected_prompt
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
                else _read_prompt(selected_prompt, args.file, literal_prompt=args.file is None)[0]
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
