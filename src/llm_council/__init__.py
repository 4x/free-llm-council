"""A council of free OpenRouter models with configurable LLM synthesis."""

from .config import SaveMode, Settings
from .council import improve_prompt, query, query_result
from .errors import ConfigurationError, CouncilError, OpenRouterError, StorageError
from .models import CouncilResult, ModelAnswer
from .storage import export_markdown

__all__ = [
    "improve_prompt",
    "query",
    "query_result",
    "export_markdown",
    "Settings",
    "SaveMode",
    "CouncilResult",
    "ModelAnswer",
    "CouncilError",
    "ConfigurationError",
    "OpenRouterError",
    "StorageError",
]
