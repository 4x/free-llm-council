"""A council of free OpenRouter models with configurable LLM synthesis."""

from .config import SaveMode, Settings
from .council import improve_prompt, query, query_result
from .errors import ConfigurationError, CouncilError, OpenRouterError, StorageError
from .models import CouncilResult, ModelAnswer
from .reconcile import ReconcileReport, reconcile_generations
from .report import build_report_html
from .storage import export_html, export_markdown, load_history
from .web import create_web_server, serve_web

__all__ = [
    "improve_prompt",
    "query",
    "query_result",
    "export_markdown",
    "export_html",
    "build_report_html",
    "load_history",
    "reconcile_generations",
    "ReconcileReport",
    "create_web_server",
    "serve_web",
    "Settings",
    "SaveMode",
    "CouncilResult",
    "ModelAnswer",
    "CouncilError",
    "ConfigurationError",
    "OpenRouterError",
    "StorageError",
]
