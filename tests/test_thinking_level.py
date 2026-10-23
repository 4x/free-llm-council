from __future__ import annotations

import json
import threading
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import httpx
import pytest

from llm_council import ConfigurationError, Settings, ThinkingLevel, cli, create_web_server
from llm_council.client import OpenRouterClient


def transport(client: OpenRouterClient, handler) -> None:
    client.http.close()
    client.http = httpx.Client(
        base_url="https://mock.invalid/api/v1/", transport=httpx.MockTransport(handler)
    )


def good() -> dict:
    return {
        "model": "actual/model",
        "choices": [{"message": {"content": "Answer"}, "finish_reason": "stop"}],
    }


@pytest.fixture
def client():
    with OpenRouterClient(
        Settings().with_overrides(api_key="secret-key", request_interval=0, max_retries=0)
    ) as instance:
        yield instance


@pytest.fixture
def web_url(config, router):
    """Run create_web_server on an ephemeral localhost port."""
    server = create_web_server(
        host="127.0.0.1",
        port=0,
        settings=config.with_overrides(thinking_level="low"),
    )
    host, port = server.server_address[:2]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://{host}:{port}"
    finally:
        server.shutdown()
        server.server_close()


def _parse_sse(raw_body: str) -> list[dict]:
    events: list[dict] = []
    for block in raw_body.split("\n\n"):
        for line in block.splitlines():
            if line.startswith("data:"):
                payload = line[len("data:") :].strip()
                if payload:
                    events.append(json.loads(payload))
    return events


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("minimal", ThinkingLevel.MINIMAL),
        ("LOW", ThinkingLevel.LOW),
        (" medium ", ThinkingLevel.MEDIUM),
        ("high", ThinkingLevel.HIGH),
        ("max", ThinkingLevel.MAX),
        ("", None),
        ("default", None),
        (None, None),
    ],
)
def test_thinking_level_parse(raw, expected):
    assert ThinkingLevel.parse(raw) is expected


def test_invalid_thinking_level_is_rejected():
    with pytest.raises(ConfigurationError, match="thinking level"):
        Settings().with_overrides(thinking_level="turbo")


def test_thinking_level_parses_from_env(tmp_path, monkeypatch):
    path = tmp_path / ".env"
    path.write_text("THINKING_LEVEL=HIGH\n", encoding="utf-8")
    monkeypatch.delenv("THINKING_LEVEL", raising=False)
    assert Settings.from_env(path).thinking_level is ThinkingLevel.HIGH

    monkeypatch.setenv("THINKING_LEVEL", "max")
    assert Settings.from_env(path).thinking_level is ThinkingLevel.MAX

    path.write_text("THINKING_LEVEL=\n", encoding="utf-8")
    monkeypatch.delenv("THINKING_LEVEL")
    assert Settings.from_env(path).thinking_level is None


def test_default_thinking_level_is_omitted_from_openrouter_body(client):
    bodies: list[dict] = []

    def handler(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json=good())

    transport(client, handler)
    client.complete("requested/model", [{"role": "user", "content": "Q"}])
    assert "reasoning" not in bodies[0]


@pytest.mark.parametrize("level", list(ThinkingLevel))
def test_openrouter_sends_reasoning_effort(client, level):
    bodies: list[dict] = []

    def handler(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json=good())

    transport(client, handler)
    client.settings = client.settings.with_overrides(thinking_level=level)
    client.complete("requested/model", [{"role": "user", "content": "Q"}])
    assert bodies[0]["reasoning"] == {"effort": level.value}


def _gemini_client(monkeypatch, thinking_level):
    calls: list[dict] = []

    class FakeModels:
        def generate_content(self, *, model, contents, config=None):
            calls.append({"model": model, "config": config})
            return SimpleNamespace(
                candidates=[SimpleNamespace(finish_reason="STOP")],
                text="Gemini answer",
                model_version=model,
                usage_metadata=None,
            )

    monkeypatch.setattr(
        "google.genai.Client",
        lambda **_kw: SimpleNamespace(models=FakeModels(), close=lambda: None),
    )
    settings = Settings().with_overrides(
        api_key="or-key",
        gemini_api_key="gem-key",
        gemini_models="gemini-3.6-flash",
        request_interval=0,
        thinking_level=thinking_level,
    )
    return OpenRouterClient(settings), calls


@pytest.mark.parametrize(
    "level,expected",
    [
        (ThinkingLevel.MINIMAL, "MINIMAL"),
        (ThinkingLevel.LOW, "LOW"),
        (ThinkingLevel.MEDIUM, "MEDIUM"),
        (ThinkingLevel.HIGH, "HIGH"),
        # Gemini has no 'max'; the strongest OpenRouter effort maps to 'high'.
        (ThinkingLevel.MAX, "HIGH"),
    ],
)
def test_gemini_sends_thinking_level(monkeypatch, level, expected):
    client, calls = _gemini_client(monkeypatch, level)
    with client:
        client.complete("gemini-3.6-flash", [{"role": "user", "content": "Q"}])
    assert calls[0]["config"].thinking_config.thinking_level.name == expected


def test_gemini_omits_thinking_config_by_default(monkeypatch):
    client, calls = _gemini_client(monkeypatch, None)
    with client:
        client.complete("gemini-3.6-flash", [{"role": "user", "content": "Q"}])
    assert calls[0]["config"].thinking_config is None


def test_cli_forwards_thinking_level(monkeypatch):
    captured: dict[str, object] = {}

    def fake_query(**kwargs: object) -> str:
        captured.update(kwargs)
        return "Synthesized answer"

    monkeypatch.setattr(cli, "query", fake_query)
    monkeypatch.setattr(cli, "_cache_latest_synthesis", lambda *_: None)

    assert cli.main(["-q", "Question", "--thinking-level", "MAX", "--no-save"]) == 0
    assert captured["thinking_level"] == "max"


def test_cli_rejects_unknown_thinking_level(capsys):
    with pytest.raises(SystemExit) as caught:
        cli.main(["-q", "Question", "--thinking-level", "turbo"])
    assert caught.value.code == 2
    assert "invalid choice" in capsys.readouterr().err


def test_web_config_reports_thinking_level(web_url):
    with urlopen(f"{web_url}/api/config", timeout=5) as resp:
        cfg = json.loads(resp.read().decode("utf-8"))
    assert cfg["thinking_level"] == "low"
    assert cfg["thinking_levels"] == [level.value for level in ThinkingLevel]


def test_web_query_forwards_thinking_level(router, web_url):
    req = Request(
        f"{web_url}/api/query",
        data=json.dumps(
            {
                "prompt": "Explain entropy",
                "judge": "test/beta:free",
                "thinking_level": "medium",
            }
        ).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(req, timeout=10) as resp:
        events = _parse_sse(resp.read().decode("utf-8"))

    assert events[-1]["type"] == "synthesis"
    assert router.calls
    assert all(call["reasoning"] == {"effort": "medium"} for call in router.calls)


def test_web_query_keeps_server_thinking_level_when_omitted(router, web_url):
    req = Request(
        f"{web_url}/api/query",
        data=json.dumps({"prompt": "Explain entropy", "judge": "test/beta:free"}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(req, timeout=10):
        pass
    assert all(call["reasoning"] == {"effort": "low"} for call in router.calls)


def test_web_query_rejects_unknown_thinking_level(web_url):
    req = Request(
        f"{web_url}/api/query",
        data=json.dumps({"prompt": "Question", "thinking_level": "turbo"}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with pytest.raises(HTTPError) as exc_info:
        urlopen(req, timeout=5)
    assert exc_info.value.code == 400


def test_web_ui_exposes_thinking_level_dropdown(web_url):
    with urlopen(f"{web_url}/", timeout=5) as resp:
        html = resp.read().decode("utf-8")
    assert 'id="thinking-level-select"' in html
    for level in ThinkingLevel:
        assert f'<option value="{level.value}"' in html
    assert "thinking_level: thinkingLevelSelect.value || null" in html
