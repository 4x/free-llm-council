from datetime import UTC, datetime, timedelta
from email.utils import format_datetime

import httpx
import pytest

from llm_council import OpenRouterError, Settings
from llm_council.client import OpenRouterClient


@pytest.fixture
def client():
    with OpenRouterClient(
        Settings().with_overrides(
            api_key="secret-key",
            request_interval=0,
            max_retries=2,
            max_retry_wait=60,
        )
    ) as instance:
        yield instance


def transport(client, handler):
    client.http.close()
    client.http = httpx.Client(
        base_url="https://mock.invalid/api/v1/", transport=httpx.MockTransport(handler)
    )


def good():
    return {
        "model": "actual/model",
        "choices": [{"message": {"content": "Answer"}, "finish_reason": "stop"}],
    }


def test_retries_429_honors_retry_after(client, monkeypatch, caplog):
    calls = []
    waits = []
    clock = [0.0]
    monkeypatch.setattr("llm_council.client.time.monotonic", lambda: clock[0])

    def sleep(delay):
        waits.append(delay)
        clock[0] += delay

    monkeypatch.setattr("llm_council.client.time.sleep", sleep)

    def handler(request):
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(
                429,
                json={"error": {"code": 429, "message": "slow down"}},
                headers={"Retry-After": "5"},
            )
        return httpx.Response(200, json=good())

    transport(client, handler)
    answer = client.complete("requested/model", [{"role": "user", "content": "Question"}])
    assert answer.model == "actual/model"
    assert len(calls) == 2 and waits == [5.0]
    assert "429: slow down" in caplog.text


def test_retry_after_over_cap_fails_without_early_retry(client):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(
            429,
            json={"error": {"code": 429, "message": "daily quota"}},
            headers={"Retry-After": "3600"},
        )

    transport(client, handler)
    with pytest.raises(OpenRouterError, match="daily quota"):
        client.complete("model", [])
    assert len(calls) == 1


def test_retry_after_http_date(client):
    date = format_datetime(datetime.now(UTC) + timedelta(seconds=20), usegmt=True)
    response = httpx.Response(429, headers={"Retry-After": date})
    assert 18 <= client._retry_delay(response, 0) <= 20


def test_retry_after_non_finite_is_ignored(client):
    response = httpx.Response(429, headers={"Retry-After": "Infinity"})
    assert client._retry_after(response) is None


def test_retries_503_honors_retry_after(client, monkeypatch):
    calls = []
    waits = []
    clock = [0.0]
    monkeypatch.setattr("llm_council.client.time.monotonic", lambda: clock[0])

    def sleep(delay):
        waits.append(delay)
        clock[0] += delay

    monkeypatch.setattr("llm_council.client.time.sleep", sleep)

    def handler(request):
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(
                503,
                json={"error": {"code": 503, "message": "unavailable"}},
                headers={"Retry-After": "4"},
            )
        return httpx.Response(200, json=good())

    transport(client, handler)
    answer = client.complete("requested/model", [{"role": "user", "content": "Question"}])
    assert answer.model == "actual/model"
    assert len(calls) == 2 and waits == [4.0]


def test_retries_in_flight_budget_402_honors_retry_after(client, monkeypatch):
    calls = []
    waits = []
    clock = [0.0]
    monkeypatch.setattr("llm_council.client.time.monotonic", lambda: clock[0])

    def sleep(delay):
        waits.append(delay)
        clock[0] += delay

    monkeypatch.setattr("llm_council.client.time.sleep", sleep)

    def handler(request):
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(
                402,
                json={
                    "error": {
                        "code": 402,
                        "message": "in-flight budget reached",
                        "metadata": {"limit_source": "openrouter_in_flight_budget"},
                    }
                },
                headers={"Retry-After": "7"},
            )
        return httpx.Response(200, json=good())

    transport(client, handler)
    answer = client.complete("requested/model", [{"role": "user", "content": "Question"}])
    assert answer.model == "actual/model"
    assert len(calls) == 2 and waits == [7.0]


def test_in_flight_budget_402_without_retry_after_is_not_retried(client):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(
            402,
            json={
                "error": {
                    "code": 402,
                    "message": "in-flight budget reached",
                    "metadata": {"limit_source": "openrouter_in_flight_budget"},
                }
            },
        )

    transport(client, handler)
    with pytest.raises(OpenRouterError, match="in-flight budget reached"):
        client.complete("model", [])
    assert len(calls) == 1


def test_other_402_with_retry_after_is_not_retried(client):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(
            402,
            json={"error": {"message": "insufficient credits"}},
            headers={"Retry-After": "1"},
        )

    transport(client, handler)
    with pytest.raises(OpenRouterError, match="insufficient credits"):
        client.complete("model", [])
    assert len(calls) == 1


def test_retry_after_over_cap_on_402_fails_without_early_retry(client):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(
            402,
            json={
                "error": {
                    "code": 402,
                    "message": "in-flight budget reached",
                    "metadata": {"limit_source": "openrouter_in_flight_budget"},
                }
            },
            headers={"Retry-After": "3600"},
        )

    transport(client, handler)
    with pytest.raises(OpenRouterError, match="in-flight budget reached"):
        client.complete("model", [])
    assert len(calls) == 1


def test_request_start_pacing(client, monkeypatch):
    client.settings = client.settings.with_overrides(request_interval=3.1)
    now = [0.0]
    waits = []
    monkeypatch.setattr("llm_council.client.time.monotonic", lambda: now[0])

    def sleep(delay):
        waits.append(delay)
        now[0] += delay

    monkeypatch.setattr("llm_council.client.time.sleep", sleep)
    transport(client, lambda _: httpx.Response(200, json=good()))
    for _ in range(3):
        client.complete("model", [])
    assert waits == [3.1, 3.1]


def test_catalog_requests_are_limited_to_free_prompt_and_output_pricing(client):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json={"data": [], "total_count": 0})

    transport(client, handler)
    client.list_models()
    assert len(requests) == 1
    assert requests[0].url.params.get("max_price") == "0"
    assert requests[0].url.params.get("max_output_price") == "0"


def test_negative_max_tokens_uses_provider_default(client):
    client.settings = client.settings.with_overrides(max_tokens=-1)
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json=good())

    transport(client, handler)
    client.complete("model", [])
    assert "max_tokens" not in requests[0].read().decode()


def test_body_error_with_200_is_not_success(client):
    transport(
        client,
        lambda _: httpx.Response(
            200, json={"error": {"code": 402, "message": "insufficient credits"}}
        ),
    )
    with pytest.raises(OpenRouterError, match="402"):
        client.complete("model", [])


@pytest.mark.parametrize("status", [400, 401, 402, 403, 404])
def test_permanent_errors_are_not_retried_and_keys_redacted(client, status):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(status, json={"error": {"message": "invalid secret-key"}})

    transport(client, handler)
    with pytest.raises(OpenRouterError) as caught:
        client.complete("model", [])
    assert "secret-key" not in str(caught.value)
    assert "[redacted]" in str(caught.value)
    assert len(calls) == 1


def test_ambiguous_timeouts_are_not_retried(client):
    calls = []

    def handler(request):
        calls.append(request)
        raise httpx.ReadTimeout("already submitted", request=request)

    transport(client, handler)
    with pytest.raises(OpenRouterError, match="ReadTimeout.*already submitted"):
        client.complete("model", [])
    assert len(calls) == 1


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {},
        {"choices": []},
        {"choices": [None]},
        {"choices": [{"message": {"content": ""}}]},
        {"choices": [{"message": {"content": "partial"}, "finish_reason": "length"}]},
        {"choices": [{"error": {"message": "failed"}}]},
    ],
)
def test_malformed_and_incomplete_responses(client, payload):
    transport(client, lambda _: httpx.Response(200, json=payload))
    with pytest.raises(OpenRouterError):
        client.complete("model", [])


def test_non_json_response(client):
    transport(client, lambda _: httpx.Response(502, text="upstream HTML"))
    with pytest.raises(OpenRouterError, match="non-JSON.*upstream HTML"):
        client.complete("model", [])


@pytest.mark.parametrize("status", [402, 429])
def test_full_provider_error_and_metadata_are_preserved_and_redacted(client, status):
    """Keep nested upstream diagnostics and redact the key in every field."""
    client.settings = client.settings.with_overrides(max_retries=0)
    transport(
        client,
        lambda _: httpx.Response(
            status,
            json={
                "error": {
                    "message": "Account cannot run this request: secret-key",
                    "metadata": {"raw": "Provider detail secret-key", "provider_name": "Fixture"},
                }
            },
        ),
    )
    with pytest.raises(OpenRouterError) as caught:
        client.complete("model", [])
    text = str(caught.value)
    assert caught.value.status_code == status
    assert "Account cannot run this request" in text
    assert "Provider detail" in text and "Fixture" in text
    assert "secret-key" not in text
    assert text.count("[redacted]") == 2


def test_generation_error_preserves_detail(client):
    """Errors inside answer choices keep their provider explanation."""
    transport(
        client,
        lambda _: httpx.Response(
            200,
            json={"choices": [{"error": {"message": "generation quota secret-key"}}]},
        ),
    )
    with pytest.raises(OpenRouterError, match="generation quota \\[redacted\\]"):
        client.complete("model", [])


def test_provider_specific_error_code_is_preserved(client):
    """HTTP status must not replace a provider's distinct diagnostic code."""
    transport(
        client,
        lambda _: httpx.Response(
            402, json={"error": {"code": "account_suspended", "message": "Contact support"}}
        ),
    )
    with pytest.raises(OpenRouterError) as caught:
        client.complete("model", [])
    assert "account_suspended" in str(caught.value)
    assert "Contact support" in str(caught.value)


def test_generation_error_code_is_preserved(client):
    """Generation errors without a message still carry useful diagnostic codes."""
    transport(
        client,
        lambda _: httpx.Response(200, json={"choices": [{"error": {"code": 429}}]}),
    )
    with pytest.raises(OpenRouterError) as caught:
        client.complete("model", [])
    assert "429" in str(caught.value)


def test_string_error_is_preserved(client):
    """Non-object provider errors should not collapse to an HTTP status."""
    transport(client, lambda _: httpx.Response(402, json={"error": "payment required"}))
    with pytest.raises(OpenRouterError, match="402: payment required"):
        client.complete("model", [])


@pytest.mark.parametrize(
    ("usage", "content", "expected"),
    [
        (
            {"completion_tokens": 2048, "completion_tokens_details": {"reasoning_tokens": 2048}},
            "",
            "reasoning used all 2048 completion tokens, leaving no room for the answer",
        ),
        (
            {"completion_tokens": 2048, "completion_tokens_details": {"reasoning_tokens": 1900}},
            "Partial",
            "reasoning used 1900 of 2048 completion tokens, leaving 148 for the answer",
        ),
        (
            {"completion_tokens": 2048, "completion_tokens_details": {"reasoning_tokens": 300}},
            "",
            "the answer used 1748 of 2048 completion tokens (300 went to reasoning)",
        ),
        (
            {"completion_tokens": 2048, "completion_tokens_details": {"reasoning_tokens": 0}},
            "",
            "the answer used 2048 of 2048 completion tokens (no reasoning tokens)",
        ),
        ({}, "", "no reasoning-token accounting"),
        (
            {"completion_tokens": 2048, "completion_tokens_details": {"reasoning_tokens": "many"}},
            "",
            "no reasoning-token accounting",
        ),
    ],
)
def test_max_tokens_failure_distinguishes_reasoning_from_visible_output(
    client, usage, content, expected
):
    """A truncated answer must say whether reasoning or visible output ate the budget."""
    transport(
        client,
        lambda _: httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": content}, "finish_reason": "length"}],
                "usage": usage,
            },
        ),
    )
    with pytest.raises(OpenRouterError) as caught:
        client.complete("model", [])
    text = str(caught.value)
    assert "exhausted MAX_TOKENS: " in text
    assert expected in text
    reasoning_dominated = "reasoning used" in expected or "reasoning-token accounting" in expected
    if reasoning_dominated:
        assert "increase MAX_TOKENS to obtain a complete answer" not in text
    if content:
        assert "A partial answer was returned and discarded." in text


def test_content_parts(client):
    payload = good()
    payload["choices"][0]["message"]["content"] = [
        {"type": "text", "text": "first"},
        {"type": "image"},
        {"type": "text", "text": "second"},
    ]
    transport(client, lambda _: httpx.Response(200, json=payload))
    assert client.complete("model", []).content == "first\nsecond"


def test_gemini_complete_converts_messages_and_usage(monkeypatch):
    from types import SimpleNamespace

    from google.genai import types as genai_types

    captured_calls = []

    class FakeModels:
        def generate_content(self, *, model, contents, config=None):
            captured_calls.append({"model": model, "contents": contents, "config": config})
            return SimpleNamespace(
                candidates=[SimpleNamespace(finish_reason=genai_types.FinishReason.STOP)],
                text="Gemini answer text",
                model_version="gemini-3.6-flash-001",
                usage_metadata=SimpleNamespace(
                    prompt_token_count=12,
                    candidates_token_count=30,
                    thoughts_token_count=8,
                    total_token_count=50,
                ),
            )

    class FakeGenaiClient:
        def __init__(self, *, api_key=None, http_options=None):
            self.api_key = api_key
            self.http_options = http_options
            self.models = FakeModels()
            self.closed = False

        def close(self):
            self.closed = True

    monkeypatch.setattr("google.genai.Client", FakeGenaiClient)

    settings = Settings().with_overrides(
        api_key="or-key",
        gemini_api_key="gem-secret-key",
        gemini_models="gemini-3.6-flash,gemini-3.7-flash",
        request_interval=0,
        max_tokens=1024,
    )
    with OpenRouterClient(settings) as instance:
        assert instance.gemini is not None
        assert instance.gemini.api_key == "gem-secret-key"
        answer = instance.complete(
            "gemini-3.6-flash",
            [
                {"role": "system", "content": "System instruction"},
                {"role": "user", "content": "First user turn"},
                {"role": "assistant", "content": "First model reply"},
                {"role": "user", "content": "Followup question"},
            ],
            free_only=True,
        )

    assert instance.gemini.closed
    assert answer.requested_model == "gemini-3.6-flash"
    assert answer.model == "gemini-3.6-flash-001"
    assert answer.content == "Gemini answer text"
    assert answer.usage["prompt_tokens"] == 12
    assert answer.usage["completion_tokens"] == 38
    assert answer.usage["completion_tokens_details"] == {"reasoning_tokens": 8}
    assert answer.usage["total_tokens"] == 50
    assert isinstance(answer.usage.get("elapsed_ms"), int)
    assert len(captured_calls) == 1
    call = captured_calls[0]
    assert call["model"] == "gemini-3.6-flash"
    assert call["config"].system_instruction == "System instruction"
    assert call["config"].max_output_tokens == 1024
    assert [(c.role, c.parts[0].text) for c in call["contents"]] == [
        ("user", "First user turn"),
        ("model", "First model reply"),
        ("user", "Followup question"),
    ]


def test_openrouter_complete_captures_router_metadata_timing_and_citations(client):
    seen_headers: list[dict[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_headers.append(dict(request.headers))
        return httpx.Response(
            200,
            headers={"X-Generation-Id": "gen-header-fallback"},
            json={
                "id": "gen-abc123xyz",
                "model": "qwen/qwen3-coder:free",
                "service_tier": "auto",
                "choices": [
                    {
                        "message": {
                            "content": "Synthesized answer with web citations.",
                            "annotations": [
                                {
                                    "type": "url_citation",
                                    "url_citation": {
                                        "url": "https://openrouter.ai/docs/guides/features/router-metadata",
                                        "title": "OpenRouter Router Metadata",
                                        "content": "Documentation on X-OpenRouter-Metadata header",
                                        "start_index": 0,
                                        "end_index": 42,
                                    },
                                },
                                {
                                    "type": "url_citation",
                                    "url_citation": {
                                        "url": "https://openrouter.ai/docs/guides/features/router-metadata",
                                        "title": "Duplicate URL is deduplicated",
                                    },
                                },
                            ],
                        },
                        "finish_reason": "stop",
                        "native_finish_reason": "STOP",
                    }
                ],
                "usage": {
                    "prompt_tokens": 120,
                    "completion_tokens": 300,
                    "total_tokens": 420,
                    "cost": 0.004,
                    "prompt_tokens_details": {"cached_tokens": 40},
                    "completion_tokens_details": {"reasoning_tokens": 180},
                    "cost_details": {"server_tool_cost": 0.004},
                    "server_tool_use": {"web_search_requests": 1},
                },
                "openrouter_metadata": {
                    "strategy": "fallback",
                    "region": "us-east",
                    "attempt": 2,
                    "generation_time": 1420,
                    "summary": "Selected Fireworks after 1 retry",
                    "endpoints": [
                        {"provider": "Together", "selected": False, "attempt": 1, "status": 429},
                        {
                            "provider": "Fireworks",
                            "selected": True,
                            "attempt": 2,
                            "status": 200,
                            "ttft": 210,
                            "tps": 85.4,
                        },
                    ],
                },
            },
        )

    transport(client, handler)
    answer = client.complete(
        "openrouter/free",
        [{"role": "user", "content": "Explain OpenRouter metadata"}],
        free_only=True,
        search_web=True,
    )

    assert seen_headers[0].get("x-openrouter-metadata") == "enabled"
    assert answer.requested_model == "openrouter/free"
    assert answer.model == "qwen/qwen3-coder:free"
    assert answer.usage["generation_id"] == "gen-abc123xyz"
    assert answer.usage["finish_reason"] == "stop"
    assert answer.usage["native_finish_reason"] == "STOP"
    assert answer.usage["service_tier"] == "auto"
    assert answer.usage["provider"] == "Fireworks"
    assert answer.usage["strategy"] == "fallback"
    assert answer.usage["region"] == "us-east"
    assert answer.usage["attempt"] == 2
    assert answer.usage["generation_time"] == 1420
    assert answer.usage["ttft"] == 210
    assert answer.usage["tps"] == 85.4
    assert isinstance(answer.usage["elapsed_ms"], int)
    assert answer.usage["completion_tokens_details"] == {"reasoning_tokens": 180}
    assert answer.usage["prompt_tokens_details"] == {"cached_tokens": 40}
    assert answer.usage["cost_details"] == {"server_tool_cost": 0.004}
    assert answer.usage["citations"] == [
        {
            "url": "https://openrouter.ai/docs/guides/features/router-metadata",
            "title": "OpenRouter Router Metadata",
            "content": "Documentation on X-OpenRouter-Metadata header",
            "start_index": 0,
            "end_index": 42,
        }
    ]


def test_openrouter_complete_falls_back_to_x_generation_id_header(client):
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"X-Generation-Id": "gen-from-header-999"},
            json={
                "model": "test/alpha:free",
                "choices": [{"message": {"content": "Hello"}, "finish_reason": "stop"}],
                "usage": {"total_tokens": 10},
            },
        )

    transport(client, handler)
    answer = client.complete("test/alpha:free", [{"role": "user", "content": "Hi"}])

    assert answer.usage["generation_id"] == "gen-from-header-999"


def test_gemini_max_tokens_truncation_raises_actionable_error(monkeypatch):
    from types import SimpleNamespace

    from google.genai import types as genai_types

    class FakeModels:
        def generate_content(self, **_kwargs):
            return SimpleNamespace(
                candidates=[SimpleNamespace(finish_reason=genai_types.FinishReason.MAX_TOKENS)],
                text="Partial Gemini answer",
                model_version="gemini-3.6-flash",
                usage_metadata=SimpleNamespace(
                    prompt_token_count=10,
                    candidates_token_count=100,
                    thoughts_token_count=1948,
                    total_token_count=2058,
                ),
            )

    class FakeGenaiClient:
        def __init__(self, **_kwargs):
            self.models = FakeModels()

        def close(self):
            pass

    monkeypatch.setattr("google.genai.Client", FakeGenaiClient)
    settings = Settings().with_overrides(
        api_key="or-key",
        gemini_api_key="gem-secret-key",
        gemini_models=("gemini-3.6-flash",),
        request_interval=0,
    )
    with OpenRouterClient(settings) as instance:
        with pytest.raises(OpenRouterError) as caught:
            instance.complete("gemini-3.6-flash", [{"role": "user", "content": "Hi"}])
    text = str(caught.value)
    assert "exhausted MAX_TOKENS" in text
    assert "reasoning used 1948 of 2048 completion tokens" in text
    assert "A partial answer was returned and discarded." in text


def test_gemini_api_error_retries_429_and_redacts_key(monkeypatch):
    from types import SimpleNamespace

    from google.genai import errors as genai_errors
    from google.genai import types as genai_types

    attempts = []

    class FakeModels:
        def generate_content(self, **_kwargs):
            attempts.append(1)
            if len(attempts) == 1:
                resp = httpx.Response(429, headers={"Retry-After": "0"})
                raise genai_errors.APIError(
                    429,
                    {
                        "error": {
                            "message": "Quota exceeded for gem-secret-key",
                            "status": "RESOURCE_EXHAUSTED",
                        }
                    },
                    resp,
                )
            return SimpleNamespace(
                candidates=[SimpleNamespace(finish_reason=genai_types.FinishReason.STOP)],
                text="Recovered Gemini answer",
                model_version="gemini-3.7-flash",
                usage_metadata=None,
            )

    class FakeGenaiClient:
        def __init__(self, **_kwargs):
            self.models = FakeModels()

        def close(self):
            pass

    monkeypatch.setattr("google.genai.Client", FakeGenaiClient)
    settings = Settings().with_overrides(
        api_key="or-key",
        gemini_api_key="gem-secret-key",
        gemini_models=("gemini-3.7-flash",),
        request_interval=0,
        max_retries=1,
    )
    with OpenRouterClient(settings) as instance:
        answer = instance.complete("gemini-3.7-flash", [{"role": "user", "content": "Hi"}])
    assert answer.content == "Recovered Gemini answer"
    assert len(attempts) == 2

    # Permanent 403 error is not retried and redacts the Gemini API key
    class FailModels:
        def generate_content(self, **_kwargs):
            resp = httpx.Response(403)
            raise genai_errors.APIError(
                403,
                {"error": {"message": "Invalid key gem-secret-key"}},
                resp,
            )

    monkeypatch.setattr(
        "google.genai.Client",
        lambda **_kw: SimpleNamespace(models=FailModels(), close=lambda: None),
    )
    with OpenRouterClient(settings) as instance:
        with pytest.raises(OpenRouterError) as caught:
            instance.complete("gemini-3.7-flash", [{"role": "user", "content": "Hi"}])
    assert "gem-secret-key" not in str(caught.value)
    assert "[redacted]" in str(caught.value)
    assert caught.value.status_code == 403
