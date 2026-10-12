"""Public exception types; provider failures are isolated from other council members."""


class CouncilError(Exception):
    """Base class for expected errors from a council run."""


class ConfigurationError(CouncilError):
    """Invalid settings, input, or judge selection."""


class OpenRouterError(CouncilError):
    """An OpenRouter request failed or returned an invalid response."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class StorageError(CouncilError):
    """Saving requested data failed."""
