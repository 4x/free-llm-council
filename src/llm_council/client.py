"""Small synchronous, thread-safe OpenRouter client using its HTTP API."""

from __future__ import annotations

import json
import logging
import math
import threading
import time
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any

import httpx

from .config import Settings
from .errors import OpenRouterError
from .models import ModelAnswer, model_exclusion_reason

logger = logging.getLogger(__name__)

# Share of completion tokens spent on reasoning above which a truncated answer is
# attributed to reasoning; below it, more budget mainly buys visible answer text.
REASONING_DOMINATED_SHARE = 0.5


def _token_count(value: Any) -> int | None:
    """Return a non-negative integer token count, or None when unusable."""
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        return None
    return int(value)


def _describe_length_stop(usage: Any, partial: bool) -> str:
    """Explain a `length` stop by separating reasoning tokens from visible output.

    OpenRouter counts reasoning tokens as completion tokens, so a truncation with
    little or no visible output means reasoning consumed the budget, not that the
    answer was long. Providers that omit the breakdown get an honest "unknown".
    """
    suffix = " A partial answer was returned and discarded." if partial else ""
    usage = usage if isinstance(usage, dict) else {}
    details = usage.get("completion_tokens_details")
    details = details if isinstance(details, dict) else {}
    completion = _token_count(usage.get("completion_tokens"))
    reasoning = _token_count(details.get("reasoning_tokens"))
    if completion is None or reasoning is None:
        return (
            "the provider reported no reasoning-token accounting, so the cause is "
            "unknown; increase MAX_TOKENS or cap reasoning with reasoning.max_tokens." + suffix
        )
    visible = max(completion - reasoning, 0)
    if completion and reasoning >= completion:
        return (
            f"reasoning used all {completion} completion tokens, leaving no room for the "
            "answer; cap reasoning with reasoning.max_tokens or reasoning.effort, or "
            "drop this model." + suffix
        )
    if completion and reasoning / completion >= REASONING_DOMINATED_SHARE:
        return (
            f"reasoning used {reasoning} of {completion} completion tokens, leaving "
            f"{visible} for the answer; raising MAX_TOKENS mostly buys more reasoning, "
            "so cap reasoning or drop this model." + suffix
        )
    reasoning_note = f" ({reasoning} went to reasoning)" if reasoning else " (no reasoning tokens)"
    return (
        f"the answer used {visible} of {completion} completion tokens{reasoning_note}; "
        "increase MAX_TOKENS to obtain a complete answer." + suffix
    )


class OpenRouterClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.http = httpx.Client(
            base_url=settings.base_url.rstrip("/") + "/",
            headers={"Authorization": f"Bearer {settings.api_key}"},
            timeout=settings.timeout,
        )
        self._rate_lock = threading.Lock()
        self._next_request = 0.0

    def __enter__(self) -> OpenRouterClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.http.close()

    def _pace(self) -> None:
        with self._rate_lock:
            delay = self._next_request - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            self._next_request = time.monotonic() + self.settings.request_interval

    def _retry_after(self, response: httpx.Response) -> float | None:
        value = response.headers.get("Retry-After")
        if value is None:
            return None
        try:
            delay = float(value)
            return max(0.0, delay) if math.isfinite(delay) else None
        except ValueError:
            try:
                date = parsedate_to_datetime(value)
                if date.tzinfo is None:
                    date = date.replace(tzinfo=UTC)
                return max(0.0, (date - datetime.now(UTC)).total_seconds())
            except (ValueError, TypeError, OverflowError):
                return None

    def _retry_delay(self, response: httpx.Response, attempt: int) -> float:
        retry_after = self._retry_after(response)
        if retry_after is not None:
            return retry_after
        return min(2.0**attempt, self.settings.max_retry_wait)

    def _redact(self, message: str) -> str:
        """Remove the configured API key from diagnostics, including JSON-escaped keys."""
        if self.settings.api_key:
            for key in (self.settings.api_key, json.dumps(self.settings.api_key)[1:-1]):
                message = message.replace(key, "[redacted]")
        return message

    def _error_detail(self, error: Any, status: int | None = None) -> str:
        """Preserve provider messages and nested diagnostic metadata without truncation."""
        if isinstance(error, dict):
            message = str(error.get("message", "Provider error"))
            extra = {
                key: value
                for key, value in error.items()
                if key != "message" and not (key == "code" and value == status)
            }
            if extra:
                message += "; " + json.dumps(extra, ensure_ascii=False)
        else:
            message = str(error)
        return self._redact(message)

    def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        for attempt in range(self.settings.max_retries + 1):
            logger.debug("OpenRouter %s %s (attempt %d)", method, path, attempt + 1)
            if method == "POST":
                self._pace()
            try:
                response = self.http.request(method, path, **kwargs)
            except httpx.RequestError as exc:
                # A timeout may already have consumed tokens; do not blindly duplicate inference.
                raise OpenRouterError(
                    self._redact(f"OpenRouter transport failure ({type(exc).__name__}): {exc}")
                ) from exc
            try:
                payload = response.json()
            except ValueError as exc:
                raise OpenRouterError(
                    self._redact(
                        f"OpenRouter returned non-JSON data (HTTP {response.status_code}): "
                        f"{response.text}"
                    ),
                    response.status_code,
                ) from exc
            if not isinstance(payload, dict):
                raise OpenRouterError(
                    self._redact(f"OpenRouter returned an invalid response envelope: {payload}"),
                    response.status_code,
                )
            error = payload.get("error")
            if response.is_success and not error:
                return payload
            detail = error if isinstance(error, dict) else {}
            status = response.status_code
            if response.is_success:
                try:
                    status = int(detail.get("code", status))
                except (ValueError, TypeError):
                    pass
            metadata = detail.get("metadata")
            in_flight_budget = (
                isinstance(metadata, dict)
                and metadata.get("limit_source") == "openrouter_in_flight_budget"
            )
            message = self._error_detail(error if error else payload, status)
            retry_after = self._retry_after(response) if status == 402 else None
            retryable = status in {429, 500, 502, 503, 504} or (
                status == 402 and in_flight_budget and retry_after is not None
            )
            if retryable and attempt < self.settings.max_retries:
                delay = (
                    retry_after
                    if status == 402 and retry_after is not None
                    else self._retry_delay(response, attempt)
                )
                # Do not retry earlier than Retry-After, or wait unboundedly for a daily quota.
                if delay <= self.settings.max_retry_wait:
                    logger.warning(
                        "OpenRouter error %d: %s; retrying %s %s in %.1fs (retry %d/%d)",
                        status,
                        message,
                        method,
                        path,
                        delay,
                        attempt + 1,
                        self.settings.max_retries,
                    )
                    with self._rate_lock:
                        self._next_request = max(self._next_request, time.monotonic() + delay)
                    if method != "POST":
                        time.sleep(delay)
                    continue
            raise OpenRouterError(f"OpenRouter error {status}: {message}", status)
        raise AssertionError("Retry loop must return or raise")

    def list_models(self, *, free_only: bool = True, **filters: str | int) -> list[dict[str, Any]]:
        models: list[dict[str, Any]] = []
        offset = 0
        while True:
            params: dict[str, str | int] = {
                **filters,
                "input_modalities": "text",
                "output_modalities": "text",
                "max_price": 0,
                "max_output_price": 0,
                "limit": 1000,
                "offset": offset,
            }
            if not free_only:
                params.pop("max_price")
                params.pop("max_output_price")
            payload = self._request("GET", "models", params=params)
            page = payload.get("data")
            if not isinstance(page, list) or any(not isinstance(item, dict) for item in page):
                raise OpenRouterError("OpenRouter catalog is missing its model list.")
            for model in page:
                reason = model_exclusion_reason(model, self.settings.excluded_models)
                if reason is None:
                    models.append(model)
                else:
                    logger.debug("Skipping model %s: %s", model.get("id", "<missing>"), reason)
            # Pagination counts raw catalog entries, not the smaller eligible subset.
            offset += len(page)
            links = payload.get("links")
            total = payload.get("total_count")
            if not page:
                break
            if isinstance(links, dict):
                more = bool(links.get("next"))
            elif isinstance(total, int):
                more = offset < total
            else:
                more = len(page) == 1000
            if not more:
                break
            if offset > 100_000:
                raise OpenRouterError("OpenRouter catalog pagination did not terminate.")
        return models

    def complete(
        self, model: str, messages: list[dict[str, str]], *, free_only: bool = False
    ) -> ModelAnswer:
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
        }
        if self.settings.max_tokens > 0:
            body["max_tokens"] = self.settings.max_tokens
        if free_only:
            # Protect against pricing changes between discovery and generation, including routers.
            body["provider"] = {"max_price": {"prompt": 0, "completion": 0}}
        payload = self._request("POST", "chat/completions", json=body)
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise OpenRouterError(f"Model {model} returned no answer choices.")
        choice = choices[0]
        if choice.get("error"):
            detail = self._error_detail(choice["error"])
            raise OpenRouterError(f"Model {model} returned a generation error: {detail}")
        message = choice.get("message") or {}
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, list):
            content = "\n".join(
                part["text"]
                for part in content
                if isinstance(part, dict) and isinstance(part.get("text"), str)
            )
        usage = payload.get("usage")
        if choice.get("finish_reason") == "length":
            partial = isinstance(content, str) and bool(content.strip())
            raise OpenRouterError(
                f"Model {model} exhausted MAX_TOKENS: {_describe_length_stop(usage, partial)}"
            )
        if not isinstance(content, str) or not content.strip():
            raise OpenRouterError(f"Model {model} returned no text answer.")
        actual_model = payload.get("model")
        return ModelAnswer(
            requested_model=model,
            model=actual_model if isinstance(actual_model, str) else model,
            content=content,
            usage=usage if isinstance(usage, dict) else {},
        )
