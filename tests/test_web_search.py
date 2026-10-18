from __future__ import annotations

import json
from types import SimpleNamespace

import httpx
import pytest
from conftest import model

from llm_council import ConfigurationError, CouncilError, Settings, cli, query_result
from llm_council.client import OpenRouterClient
from llm_council.judges import select_judge
from llm_council.models import council_models, supports_tools

WEB_SEARCH_TOOL = {
    "type": "openrouter:web_search",
    "parameters": {"max_uses": 3},
}


def transport(client, handler):
    client.http.close()
    client.http = httpx.Client(
        base_url="https://mock.invalid/api/v1/", transport=httpx.MockTransport(handler)
    )


def good():
    return {
        "model": "actual/model",
        "choices": [{"message": {"content": "Answer"}, "finish_reason": "stop"}],
        "usage": {"total_tokens": 42, "server_tool_use": {"web_search_requests": 2}},
    }


@pytest.fixture
def client():
    with OpenRouterClient(
        Settings().with_overrides(api_key="secret-key", request_interval=0, max_retries=0)
    ) as instance:
        yield instance


def test_search_web_adds_openrouter_server_tool(client):
    calls = []

    def handler(request):
        calls.append(json.loads(request.content))
        return httpx.Response(200, json=good())

    transport(client, handler)
    answer = client.complete("requested/model", [{"role": "user", "content": "Q"}], search_web=True)
    assert answer.content == "Answer"
    assert calls[0]["tools"] == [WEB_SEARCH_TOOL]


def test_search_web_off_omits_the_tools_field(client):
    calls = []

    def handler(request):
        calls.append(json.loads(request.content))
        return httpx.Response(200, json=good())

    transport(client, handler)
    client.complete("requested/model", [{"role": "user", "content": "Q"}])
    assert "tools" not in calls[0]


def test_search_web_uses_gemini_google_search_grounding(monkeypatch):
    gemini_calls = []

    class FakeModels:
        def generate_content(self, *, model, contents, config=None):
            gemini_calls.append({"model": model, "config": config})
            return SimpleNamespace(
                candidates=[SimpleNamespace(finish_reason="STOP")],
                text="Grounded answer",
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
    )
    with OpenRouterClient(settings) as gemini_client:
        answer = gemini_client.complete(
            "gemini-3.6-flash", [{"role": "user", "content": "Q"}], search_web=True
        )
        assert answer.content == "Grounded answer"
        assert gemini_calls[0]["config"].tools[0].google_search is not None

        gemini_calls.clear()
        gemini_client.complete("gemini-3.6-flash", [{"role": "user", "content": "Q"}])
        assert not gemini_calls[0]["config"].tools


@pytest.mark.parametrize(
    "parameters,expected",
    [
        (["tools", "temperature"], True),
        (["temperature"], False),
        ([], False),
        (None, False),
        ("tools", False),
    ],
)
def test_supports_tools_requires_an_explicit_tool_capability(parameters, expected):
    entry = model("test/model:free")
    entry["supported_parameters"] = parameters
    assert supports_tools(entry) is expected


def test_council_models_can_require_tool_calling():
    tooled = model("test/tooled:free")
    tooled["supported_parameters"] = ["tools"]
    untooled = model("test/untooled:free")
    assert council_models([tooled, untooled], require_tools=True) == [tooled]
    assert council_models([tooled, untooled]) == [tooled, untooled]


def test_query_result_search_web_adds_tools_and_skips_untooled_models(config, router):
    for entry in router.catalog:
        if entry["id"] in {"test/alpha:free", "test/beta:free"}:
            entry["supported_parameters"] = ["tools"]
    router.catalog.append(model("test/no-tools:free"))

    result = query_result(
        "Question",
        settings=config.with_overrides(search_web=True),
        no_save=True,
    )
    assert result.synthesis == "Synthesized fixture answer"
    # Both tool-capable members plus the tool-capable judge; the untooled model is skipped.
    assert {call["model"] for call in router.calls} == {"test/alpha:free", "test/beta:free"}
    assert all(call["tools"] == [WEB_SEARCH_TOOL] for call in router.calls)


def test_query_result_search_web_requires_tool_capable_members(config, router):
    with pytest.raises(CouncilError, match="tool calling") as error:
        query_result(
            "Question",
            settings=config.with_overrides(search_web=True),
            no_save=True,
        )
    assert "--search-web" in str(error.value)
    assert router.calls == []


def test_search_web_restricts_automatic_judge_selection(config, router):
    router.catalog[0]["supported_parameters"] = ["tools"]
    with OpenRouterClient(config) as client:
        selected, _ = select_judge(
            config.with_overrides(search_web=True, judge="random"), router.catalog, client
        )
    assert selected == "test/alpha:free"

    with OpenRouterClient(config) as client:
        with pytest.raises(ConfigurationError, match="tool calling"):
            select_judge(
                config.with_overrides(search_web=True, judge="test/beta:free"),
                router.catalog,
                client,
            )

    with OpenRouterClient(config) as client:
        with pytest.raises(ConfigurationError, match="No eligible judge models"):
            select_judge(
                config.with_overrides(search_web=True, judge="largest-context"),
                [
                    {**entry, "supported_parameters": []}
                    for entry in router.catalog
                    if entry["id"] != "openrouter/free"
                ],
                client,
            )


def test_search_web_setting_parses_from_env(tmp_path, monkeypatch):
    path = tmp_path / ".env"
    path.write_text("SEARCH_WEB=yes\n", encoding="utf-8")
    monkeypatch.delenv("SEARCH_WEB", raising=False)
    assert Settings.from_env(path).search_web is True

    monkeypatch.setenv("SEARCH_WEB", "0")
    assert Settings.from_env(path).search_web is False

    with pytest.raises(ConfigurationError, match="search_web"):
        Settings().with_overrides(search_web="sometimes")


def test_cli_forwards_search_web_flag(monkeypatch):
    captured: dict[str, object] = {}

    def fake_query(**kwargs: object) -> str:
        captured.update(kwargs)
        return "Synthesized answer"

    monkeypatch.setattr(cli, "query", fake_query)
    monkeypatch.setattr(cli, "_cache_latest_synthesis", lambda *_: None)

    assert cli.main(["-q", "Question", "--search-web", "--no-save"]) == 0
    assert captured["search_web"] is True
