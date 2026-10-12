"""Command line entrypoint; stdout renders the synthesized answer as Markdown."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from rich.console import Console
from rich.markdown import Markdown

from .config import LOG_LEVELS
from .council import query
from .errors import CouncilError
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


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        answer = query(**vars(args), literal_prompt=args.file is None)
    except CouncilError as exc:
        print(f"llm-council: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("llm-council: interrupted", file=sys.stderr)
        return 130
    Console().print(Markdown(answer), markup=False)
    return 0
