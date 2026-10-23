"""Small synchronous, thread-safe OpenRouter client using its HTTP API."""

from __future__ import annotations

import json
import logging
import math
import threading
import time
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import TYPE_CHECKING, Any

import httpx

from .config import Settings, ThinkingLevel
from .errors import OpenRouterError
from .models import ModelAnswer, model_exclusion_reason

if TYPE_CHECKING:
    from google import genai

logger = logging.getLogger(__name__)

# Share of completion tokens spent on reasoning above which a truncated answer is
# attributed to reasoning; below it, more budget mainly buys visible answer text.
REASONING_DOMINATED_SHARE = 0.5
GEMINI_DEFAULT_CONTEXT_LENGTH = 1_048_576
# OpenRouter's server-side web search tool (the `web` plugin and `:online` are deprecated).
# The model decides whether to search; max_uses bounds search spend per request because
# every search is billed on top of token costs, including for free models.
WEB_SEARCH_TOOL_TYPE = "openrouter:web_search"
WEB_SEARCH_MAX_USES = 3
OPENROUTER_METADATA_HEADER = "X-OpenRouter-Metadata"
OPENROUTER_METADATA_ENABLED = "enabled"
GENERATION_ID_HEADER = "X-Generation-Id"
# Post-hoc accounting for one generation: native tokenizer counts and upstream cost.
GENERATION_PATH = "generation"
MS_PER_SECOND = 1000


def _gemini_thinking_level(level: ThinkingLevel) -> str:
    """Map a thinking level onto Gemini's accepted values.

    Gemini accepts `minimal`, `low`, `medium`, and `high`; it has no `max` level,
    so the strongest OpenRouter effort is sent as Gemini's `high`.
    """
    if level is ThinkingLevel.MAX:
        logger.debug("Gemini has no 'max' thinking level; sending 'high'")
        return ThinkingLevel.HIGH.value
    return level.value


def _web_search_tool() -> dict[str, Any]:
    """Build the server tool entry that enables OpenRouter-run web search."""
    return {
        "type": WEB_SEARCH_TOOL_TYPE,
        "parameters": {"max_uses": WEB_SEARCH_MAX_USES},
    }


def _token_count(value: Any) -> int | None:
    """Return a non-negative integer token count, or None when unusable."""
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        return None
    return int(value)


def _extract_provider_routing(payload: dict[str, Any]) -> dict[str, Any]:
    """Extract normalized provider routing fields from payload and openrouter_metadata."""
    routing: dict[str, Any] = {}
    top_provider = payload.get("provider")
    if isinstance(top_provider, str) and top_provider.strip():
        routing["provider"] = top_provider.strip()

    meta = payload.get("openrouter_metadata")
    if not isinstance(meta, dict):
        return routing

    routing["openrouter_metadata"] = meta

    selected_provider: str | None = None
    selected_endpoint: dict[str, Any] | None = None
    endpoints = meta.get("endpoints")
    if isinstance(endpoints, list):
        for ep in endpoints:
            if isinstance(ep, dict) and ep.get("selected") is True:
                selected_endpoint = ep
                prov = ep.get("provider") or ep.get("provider_name")
                if isinstance(prov, str) and prov.strip():
                    selected_provider = prov.strip()
                    break
        if selected_provider is None and len(endpoints) == 1 and isinstance(endpoints[0], dict):
            selected_endpoint = endpoints[0]
            prov = endpoints[0].get("provider") or endpoints[0].get("provider_name")
            if isinstance(prov, str) and prov.strip():
                selected_provider = prov.strip()

    if selected_endpoint is not None:
        ttft = selected_endpoint.get("ttft")
        if (
            isinstance(ttft, (int, float))
            and not isinstance(ttft, bool)
            and math.isfinite(ttft)
            and ttft >= 0
        ):
            routing["ttft"] = int(round(ttft))
        tps = selected_endpoint.get("tps")
        if (
            isinstance(tps, (int, float))
            and not isinstance(tps, bool)
            and math.isfinite(tps)
            and tps >= 0
        ):
            routing["tps"] = float(tps)

    attempts = meta.get("attempts")
    if isinstance(attempts, list) and attempts:
        routing["attempts"] = attempts
        if selected_provider is None:
            for att in reversed(attempts):
                if isinstance(att, dict):
                    status = att.get("status")
                    if status is None or status == 200 or status == "success":
                        prov = att.get("provider") or att.get("provider_name")
                        if isinstance(prov, str) and prov.strip():
                            selected_provider = prov.strip()
                            break

    if selected_provider is None:
        raw_prov = meta.get("provider") or meta.get("provider_name")
        if isinstance(raw_prov, str) and raw_prov.strip():
            selected_provider = raw_prov.strip()

    if selected_provider is not None:
        routing["provider"] = selected_provider

    attempt_num = _token_count(meta.get("attempt"))
    if attempt_num is not None and attempt_num > 0:
        routing["attempt"] = attempt_num
    elif isinstance(attempts, list) and attempts:
        routing["attempt"] = len(attempts)

    for str_key in ("strategy", "region", "summary"):
        val = meta.get(str_key)
        if isinstance(val, str) and val.strip():
            routing[str_key] = val.strip()

    gen_time = meta.get("generation_time")
    if (
        isinstance(gen_time, (int, float))
        and not isinstance(gen_time, bool)
        and math.isfinite(gen_time)
        and gen_time >= 0
    ):
        routing["generation_time"] = int(round(gen_time))

    return routing


def _extract_citations(annotations: Any) -> list[dict[str, Any]]:
    """Extract normalized web search citations from message.annotations."""
    if not isinstance(annotations, list):
        return []
    citations: list[dict[str, Any]] = []
    seen_urls: set[str] = set()
    for item in annotations:
        if not isinstance(item, dict):
            continue
        nested = item.get("url_citation")
        source = nested if isinstance(nested, dict) else item
        url = source.get("url")
        if not isinstance(url, str):
            continue
        clean_url = url.strip()
        if not (clean_url.startswith("http://") or clean_url.startswith("https://")):
            continue
        if clean_url in seen_urls:
            continue
        seen_urls.add(clean_url)
        entry: dict[str, Any] = {"url": clean_url}
        title = source.get("title")
        if isinstance(title, str) and title.strip():
            entry["title"] = title.strip()
        snippet = source.get("content")
        if isinstance(snippet, str) and snippet.strip():
            entry["content"] = snippet.strip()
        start_idx = _token_count(source.get("start_index"))
        end_idx = _token_count(source.get("end_index"))
        if start_idx is not None:
            entry["start_index"] = start_idx
        if end_idx is not None:
            entry["end_index"] = end_idx
        citations.append(entry)
    return citations


def _extract_gemini_citations(candidate: Any) -> list[dict[str, Any]]:
    """Extract web search citations from Gemini grounding_metadata if present."""
    grounding = getattr(candidate, "grounding_metadata", None)
    if grounding is None and isinstance(candidate, dict):
        grounding = candidate.get("grounding_metadata")
    if grounding is None:
        return []
    chunks = getattr(grounding, "grounding_chunks", None)
    if chunks is None and isinstance(grounding, dict):
        chunks = grounding.get("grounding_chunks")
    if not isinstance(chunks, list):
        return []
    citations: list[dict[str, Any]] = []
    seen_urls: set[str] = set()
    for chunk in chunks:
        web = getattr(chunk, "web", None)
        if web is None and isinstance(chunk, dict):
            web = chunk.get("web")
        if web is None:
            continue
        uri = getattr(web, "uri", None)
        title = getattr(web, "title", None)
        if isinstance(web, dict):
            uri = web.get("uri", uri)
            title = web.get("title", title)
        if not isinstance(uri, str):
            continue
        clean_url = uri.strip()
        if not (clean_url.startswith("http://") or clean_url.startswith("https://")):
            continue
        if clean_url in seen_urls:
            continue
        seen_urls.add(clean_url)
        entry: dict[str, Any] = {"url": clean_url}
        if isinstance(title, str) and title.strip():
            entry["title"] = title.strip()
        citations.append(entry)
    return citations


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
            headers={
                "Authorization": f"Bearer {settings.api_key}",
                OPENROUTER_METADATA_HEADER: OPENROUTER_METADATA_ENABLED,
            },
            timeout=settings.timeout,
        )
        self.gemini: genai.Client | None = None
        if settings.gemini_api_key.strip():
            from google import genai
            from google.genai import types as genai_types

            timeout_ms = max(1, int(math.ceil(settings.timeout * 1000)))
            self.gemini = genai.Client(
                api_key=settings.gemini_api_key.strip(),
                http_options=genai_types.HttpOptions(timeout=timeout_ms),
            )
        self._rate_lock = threading.Lock()
        self._next_request = 0.0

    def __enter__(self) -> OpenRouterClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.http.close()
        if self.gemini is not None:
            self.gemini.close()

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
        """Remove configured API keys from diagnostics, including JSON-escaped keys."""
        for raw_key in (self.settings.api_key, self.settings.gemini_api_key):
            if raw_key:
                for key in (raw_key, json.dumps(raw_key)[1:-1]):
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
                header_gen_id = response.headers.get(GENERATION_ID_HEADER)
                if (
                    isinstance(header_gen_id, str)
                    and header_gen_id.strip()
                    and not isinstance(payload.get("id"), str)
                ):
                    payload = {**payload, "id": header_gen_id.strip()}
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

    def get_generation(self, generation_id: str) -> dict[str, Any]:
        """Fetch post-run accounting for one generation from `GET /generation`.

        Inline metadata already reports the router's own token counts and cost. This
        endpoint adds what only the upstream provider knows: native tokenizer counts,
        `upstream_inference_cost` for BYOK keys, and the applied `cache_discount`.
        """
        clean_id = generation_id.strip()
        if not clean_id:
            raise OpenRouterError("A generation ID is required to fetch generation metadata.")
        payload = self._request("GET", GENERATION_PATH, params={"id": clean_id})
        data = payload.get("data")
        if not isinstance(data, dict):
            raise OpenRouterError(
                self._redact(
                    f"OpenRouter returned no generation metadata for {clean_id}: "
                    f"{json.dumps(payload, ensure_ascii=False)}"
                )
            )
        return data

    def _fetch_openrouter_models(
        self, *, free_only: bool = True, **filters: str | int
    ) -> list[dict[str, Any]]:
        raw_models: list[dict[str, Any]] = []
        offset = 0
        if not self.settings.api_key.strip():
            return raw_models
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
            raw_models.extend(page)
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
        return raw_models

    @staticmethod
    def _strip_gemini_prefix(model_id: str) -> str:
        stripped = model_id.strip()
        for prefix in ("google/", "models/"):
            if stripped.startswith(prefix):
                stripped = stripped[len(prefix) :]
        return stripped.removesuffix(":free")

    def _match_configured_gemini_id(self, catalog_id: str) -> str | None:
        if not self.settings.gemini_models:
            return None
        if catalog_id in self.settings.gemini_models:
            return catalog_id
        bare_catalog = self._strip_gemini_prefix(catalog_id)
        for configured in self.settings.gemini_models:
            if self._strip_gemini_prefix(configured) == bare_catalog:
                return configured
        return None

    @staticmethod
    def _as_gemini_catalog_entry(
        configured_id: str, source: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        entry: dict[str, Any] = dict(source) if source is not None else {}
        entry["id"] = configured_id
        entry["pricing"] = {"prompt": "0", "completion": "0", "output": "0", "request": "0"}
        architecture = entry.get("architecture")
        if not isinstance(architecture, dict):
            entry["architecture"] = {
                "input_modalities": ["text"],
                "output_modalities": ["text"],
            }
        if not entry.get("context_length"):
            entry["context_length"] = GEMINI_DEFAULT_CONTEXT_LENGTH
        return entry

    def list_models(self, *, free_only: bool = True, **filters: str | int) -> list[dict[str, Any]]:
        has_gemini = bool(self.settings.gemini_models)
        # When GEMINI_MODELS is configured alongside OpenRouter, query OpenRouter
        # without price caps (or via an additional unfiltered call) so paid OpenRouter
        # entries for Gemini models supply real context_length and intelligence ranking.
        raw_page = self._fetch_openrouter_models(free_only=free_only, **filters)
        unfiltered_for_gemini: list[dict[str, Any]] = []
        if has_gemini and self.settings.api_key.strip():
            if free_only:
                unfiltered_for_gemini = self._fetch_openrouter_models(free_only=False, **filters)
            else:
                unfiltered_for_gemini = raw_page

        models: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        is_ranked_query = "sort" in filters or "min_intelligence_index" in filters

        if is_ranked_query and has_gemini and unfiltered_for_gemini:
            free_ids = {str(item.get("id")) for item in raw_page if isinstance(item.get("id"), str)}
            for item in unfiltered_for_gemini:
                raw_id = item.get("id")
                if not isinstance(raw_id, str):
                    continue
                matched_gemini = self._match_configured_gemini_id(raw_id)
                if matched_gemini is not None:
                    normalized = self._as_gemini_catalog_entry(matched_gemini, item)
                    reason = model_exclusion_reason(normalized, self.settings.excluded_models)
                    if reason is None and matched_gemini not in seen_ids:
                        models.append(normalized)
                        seen_ids.add(matched_gemini)
                    elif reason is not None:
                        logger.debug("Skipping model %s: %s", matched_gemini, reason)
                    continue
                if not free_only or raw_id in free_ids:
                    reason = model_exclusion_reason(item, self.settings.excluded_models)
                    if reason is None and raw_id not in seen_ids:
                        models.append(item)
                        seen_ids.add(raw_id)
                    elif reason is not None:
                        logger.debug("Skipping model %s: %s", raw_id, reason)
            return models

        gemini_metadata: dict[str, dict[str, Any]] = {}
        for item in unfiltered_for_gemini:
            raw_id = item.get("id")
            if isinstance(raw_id, str):
                matched = self._match_configured_gemini_id(raw_id)
                if matched is not None and matched not in gemini_metadata:
                    gemini_metadata[matched] = item

        for model in raw_page:
            raw_id = model.get("id")
            matched_gemini = (
                self._match_configured_gemini_id(raw_id) if isinstance(raw_id, str) else None
            )
            if matched_gemini is not None:
                model = self._as_gemini_catalog_entry(
                    matched_gemini, gemini_metadata.get(matched_gemini, model)
                )
            reason = model_exclusion_reason(model, self.settings.excluded_models)
            if reason is None:
                model_id_str = str(model["id"])
                if model_id_str not in seen_ids:
                    models.append(model)
                    seen_ids.add(model_id_str)
            else:
                logger.debug("Skipping model %s: %s", model.get("id", "<missing>"), reason)

        if not is_ranked_query:
            for gemini_id in self.settings.gemini_models:
                if gemini_id in seen_ids:
                    continue
                gemini_model = self._as_gemini_catalog_entry(
                    gemini_id, gemini_metadata.get(gemini_id)
                )
                reason = model_exclusion_reason(gemini_model, self.settings.excluded_models)
                if reason is None:
                    models.append(gemini_model)
                    seen_ids.add(gemini_id)
                else:
                    logger.debug("Skipping model %s: %s", gemini_id, reason)
        return models

    def _is_gemini_model(self, model: str) -> bool:
        return self._match_configured_gemini_id(model) is not None

    def is_gemini_model(self, model: str) -> bool:
        """Return whether the model ID is served directly through the Gemini API."""
        return self._is_gemini_model(model)

    @staticmethod
    def _gemini_usage_dict(usage_metadata: Any) -> dict[str, Any]:
        if usage_metadata is None:
            return {}
        prompt_tokens = getattr(usage_metadata, "prompt_token_count", None)
        candidates_tokens = getattr(usage_metadata, "candidates_token_count", None)
        thoughts_tokens = getattr(usage_metadata, "thoughts_token_count", None)
        total_tokens = getattr(usage_metadata, "total_token_count", None)
        if isinstance(usage_metadata, dict):
            prompt_tokens = usage_metadata.get("prompt_token_count", prompt_tokens)
            candidates_tokens = usage_metadata.get("candidates_token_count", candidates_tokens)
            thoughts_tokens = usage_metadata.get("thoughts_token_count", thoughts_tokens)
            total_tokens = usage_metadata.get("total_token_count", total_tokens)
        usage: dict[str, Any] = {}
        p_count = _token_count(prompt_tokens)
        c_count = _token_count(candidates_tokens)
        r_count = _token_count(thoughts_tokens)
        t_count = _token_count(total_tokens)
        if p_count is not None:
            usage["prompt_tokens"] = p_count
        if c_count is not None or r_count is not None:
            completion = (c_count or 0) + (r_count or 0)
            usage["completion_tokens"] = completion
            if r_count is not None:
                usage["completion_tokens_details"] = {"reasoning_tokens": r_count}
        if t_count is not None:
            usage["total_tokens"] = t_count
        elif "prompt_tokens" in usage or "completion_tokens" in usage:
            usage["total_tokens"] = usage.get("prompt_tokens", 0) + usage.get(
                "completion_tokens", 0
            )
        return usage

    def _complete_gemini(
        self, model: str, messages: list[dict[str, str]], *, search_web: bool = False
    ) -> ModelAnswer:
        if self.gemini is None:
            raise OpenRouterError(f"GEMINI_API_KEY is required to call Gemini model {model!r}.")
        from google.genai import errors as genai_errors
        from google.genai import types as genai_types

        system_parts: list[str] = []
        contents: list[genai_types.Content] = []
        for message in messages:
            role = message.get("role", "user")
            text = message.get("content", "")
            if role == "system":
                if text:
                    system_parts.append(text)
            else:
                gemini_role = "model" if role == "assistant" else "user"
                contents.append(
                    genai_types.Content(
                        role=gemini_role,
                        parts=[genai_types.Part.from_text(text=text)],
                    )
                )
        config_kwargs: dict[str, Any] = {}
        if system_parts:
            config_kwargs["system_instruction"] = "\n\n".join(system_parts)
        if self.settings.max_tokens > 0:
            config_kwargs["max_output_tokens"] = self.settings.max_tokens
        if self.settings.thinking_level is not None:
            gemini_level = _gemini_thinking_level(self.settings.thinking_level)
            config_kwargs["thinking_config"] = genai_types.ThinkingConfig(
                thinking_level=genai_types.ThinkingLevel[gemini_level.upper()]
            )
        if search_web:
            config_kwargs["tools"] = [genai_types.Tool(google_search=genai_types.GoogleSearch())]
        gen_config = genai_types.GenerateContentConfig(**config_kwargs) if config_kwargs else None

        if not system_parts and not contents:
            contents.append(
                genai_types.Content(
                    role="user",
                    parts=[genai_types.Part.from_text(text="")],
                )
            )
        api_model_name = self._strip_gemini_prefix(model)
        for attempt in range(self.settings.max_retries + 1):
            logger.debug("Gemini generate_content %s (attempt %d)", model, attempt + 1)
            self._pace()
            call_started = time.perf_counter()
            try:
                response = self.gemini.models.generate_content(
                    model=api_model_name,
                    contents=contents,
                    config=gen_config,
                )
            except genai_errors.APIError as exc:
                status = exc.code if isinstance(exc.code, int) else None
                detail = exc.details if getattr(exc, "details", None) else (exc.message or str(exc))
                message_text = self._error_detail(detail, status)
                raw_resp = getattr(exc, "response", None)
                retryable = status in {429, 500, 502, 503, 504}
                if retryable and attempt < self.settings.max_retries:
                    delay = (
                        self._retry_delay(raw_resp, attempt)
                        if isinstance(raw_resp, httpx.Response)
                        else min(2.0**attempt, self.settings.max_retry_wait)
                    )
                    if delay <= self.settings.max_retry_wait:
                        logger.warning(
                            "Gemini error %s: %s; retrying %s in %.1fs (retry %d/%d)",
                            status,
                            message_text,
                            model,
                            delay,
                            attempt + 1,
                            self.settings.max_retries,
                        )
                        with self._rate_lock:
                            self._next_request = max(self._next_request, time.monotonic() + delay)
                        continue
                prefix = f"Gemini error {status}" if status is not None else "Gemini error"
                raise OpenRouterError(f"{prefix}: {message_text}", status) from exc
            except httpx.RequestError as exc:
                raise OpenRouterError(
                    self._redact(f"Gemini transport failure ({type(exc).__name__}): {exc}")
                ) from exc
            except Exception as exc:
                raise OpenRouterError(
                    self._redact(f"Gemini error ({type(exc).__name__}): {exc}")
                ) from exc

            candidates = getattr(response, "candidates", None)
            if not isinstance(candidates, list) or not candidates:
                feedback = getattr(response, "prompt_feedback", None)
                detail_msg = f" ({feedback})" if feedback else ""
                raise OpenRouterError(
                    self._redact(f"Model {model} returned no answer candidates.{detail_msg}")
                )
            candidate = candidates[0]
            finish_reason = getattr(candidate, "finish_reason", None)
            finish_name = getattr(finish_reason, "name", str(finish_reason or "")).upper()
            content = getattr(response, "text", None)
            usage = self._gemini_usage_dict(getattr(response, "usage_metadata", None))
            if finish_name == "MAX_TOKENS" or finish_reason == "length":
                partial = isinstance(content, str) and bool(content.strip())
                raise OpenRouterError(
                    f"Model {model} exhausted MAX_TOKENS: {_describe_length_stop(usage, partial)}"
                )
            if not isinstance(content, str) or not content.strip():
                raise OpenRouterError(f"Model {model} returned no text answer.")
            elapsed_ms = max(0, int(round((time.perf_counter() - call_started) * MS_PER_SECOND)))
            if usage:
                usage["elapsed_ms"] = elapsed_ms
            citations = _extract_gemini_citations(candidate)
            if citations:
                usage["citations"] = citations
            actual_model = getattr(response, "model_version", None)
            return ModelAnswer(
                requested_model=model,
                model=actual_model if isinstance(actual_model, str) and actual_model else model,
                content=content,
                usage=usage,
            )
        raise AssertionError("Retry loop must return or raise")

    def complete(
        self,
        model: str,
        messages: list[dict[str, str]],
        *,
        free_only: bool = False,
        search_web: bool = False,
    ) -> ModelAnswer:
        """Answer with one model, optionally enabling server-side web search."""
        if self._is_gemini_model(model):
            return self._complete_gemini(model, messages, search_web=search_web)
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
        }
        if self.settings.max_tokens > 0:
            body["max_tokens"] = self.settings.max_tokens
        if self.settings.thinking_level is not None:
            # OpenRouter normalizes effort across providers and maps unsupported values
            # down to the nearest level the selected model accepts.
            body["reasoning"] = {"effort": self.settings.thinking_level.value}
        if free_only:
            # Protect against pricing changes between discovery and generation, including routers.
            body["provider"] = {"max_price": {"prompt": 0, "completion": 0}}
        if search_web:
            # Search charges are billed separately and are not capped by provider.max_price.
            body["tools"] = [_web_search_tool()]
        call_started = time.perf_counter()
        payload = self._request(
            "POST",
            "chat/completions",
            json=body,
            headers={OPENROUTER_METADATA_HEADER: OPENROUTER_METADATA_ENABLED},
        )
        elapsed_ms = max(0, int(round((time.perf_counter() - call_started) * MS_PER_SECOND)))
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
        usage_dict: dict[str, Any] = dict(usage) if isinstance(usage, dict) else {}
        usage_dict["elapsed_ms"] = elapsed_ms
        gen_id = payload.get("id")
        if isinstance(gen_id, str) and gen_id.strip():
            usage_dict["generation_id"] = gen_id.strip()
        for reason_key in ("finish_reason", "native_finish_reason"):
            reason_val = choice.get(reason_key)
            if isinstance(reason_val, str) and reason_val.strip():
                usage_dict[reason_key] = reason_val.strip()
        service_tier = payload.get("service_tier")
        if isinstance(service_tier, str) and service_tier.strip():
            usage_dict["service_tier"] = service_tier.strip()
        routing = _extract_provider_routing(payload)
        if routing:
            usage_dict.update(routing)
        if isinstance(message, dict):
            annotations = message.get("annotations")
            citations = _extract_citations(annotations)
            if citations:
                usage_dict["citations"] = citations
                usage_dict["annotations"] = annotations
        server_tool_use = usage_dict.get("server_tool_use")
        if search_web and isinstance(server_tool_use, dict):
            logger.debug(
                "Model %s ran %s web search(es)", model, server_tool_use.get("web_search_requests")
            )
        return ModelAnswer(
            requested_model=model,
            model=actual_model if isinstance(actual_model, str) else model,
            content=content,
            usage=usage_dict,
        )
