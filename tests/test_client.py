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
