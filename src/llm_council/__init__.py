"""A council of free OpenRouter models with configurable LLM synthesis."""

from .config import SaveMode, Settings
from .council import query, query_result
from .errors import ConfigurationError, CouncilError, OpenRouterError, StorageError
from .models import CouncilResult, ModelAnswer

__all__ = [
    "query",
    "query_result",
    "Settings",
    "SaveMode",
    "CouncilResult",
    "ModelAnswer",
    "CouncilError",
    "ConfigurationError",
    "OpenRouterError",
    "StorageError",
]
