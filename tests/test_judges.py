import pytest

from llm_council import ConfigurationError, Settings, judges, query_result
from llm_council.models import council_models


@pytest.mark.parametrize(
    "strategy",
    [
        "random",
        "round-robin",
        "free-router",
        "auto",
        "most-intelligent",
        "largest-context",
        "test/beta:free",
    ],
)
def test_judge_strategies(config, router, monkeypatch, strategy):
    monkeypatch.setattr(judges.random, "choice", lambda values: values[-1])
    judges._round_robin_positions.clear()
    scope = "all" if strategy == "auto" else "free"
    result = query_result(
        "Question", settings=config, judge=strategy, judge_scope=scope, no_save=True
    )
    expected = {
        "random": "test/beta:free",
        "round-robin": "test/alpha:free",
        "free-router": "openrouter/free",
        "auto": "openrouter/auto",
        "most-intelligent": "test/beta:free",
        "largest-context": "test/beta:free",
        "test/beta:free": "test/beta:free",
    }
    assert result.judge_requested == expected[strategy]
    if strategy in {"auto", "free-router"}:
        assert result.judge == "test/alpha:free"
    if strategy == "auto":
        assert "provider" not in router.calls[-1]
    if strategy == "most-intelligent":
        assert router.gets[-1]["min_intelligence_index"] == ["0"]
        assert router.gets[-1]["sort"] == ["intelligence-high-to-low"]


def test_round_robin_rotates_within_process(config):
    judges._round_robin_positions.clear()
    results = [
        query_result("Question", settings=config, judge="round-robin", no_save=True)
        for _ in range(3)
    ]
    assert [result.judge_requested for result in results] == [
        "test/alpha:free",
        "test/beta:free",
        "test/alpha:free",
    ]


@pytest.mark.parametrize("judge", ["auto", "test/paid", "not-a-strategy"])
def test_invalid_or_paid_judge_rejected_before_member_calls(config, router, judge):
    with pytest.raises(ConfigurationError):
        query_result("Question", settings=config, judge=judge)
    assert not router.calls
    assert not config.duckdb_path.exists()


def test_paid_judge_opt_in(config, router):
    result = query_result(
        "Question", settings=config, judge="test/paid", judge_scope="all", no_save=True
    )
    assert result.judge == "test/paid"
    assert "provider" not in router.calls[-1]
    assert len(result.answers) == 2


def test_paid_judge_catalog_lookup_drops_free_price_filters(config, router):
    # The free-priced discovery query hides test/paid, so an unfiltered lookup must
    # run before the judge can be reported as paid rather than unknown.
    with pytest.raises(ConfigurationError, match="requires judge_scope"):
        query_result("Question", settings=config, judge="test/paid", no_save=True)
    assert not router.calls
    assert router.gets[0]["max_price"] == ["0"]
    assert router.gets[0]["max_output_price"] == ["0"]
    assert "max_price" not in router.gets[1]
    assert "max_output_price" not in router.gets[1]


def test_all_scope_judge_catalog_drops_free_price_filters(config, router, monkeypatch):
    monkeypatch.setattr(judges.random, "choice", lambda values: values[-1])
    result = query_result(
        "Question", settings=config, judge="random", judge_scope="all", no_save=True
    )
    # Eligible ids are sorted, so the last one is the paid fixture model.
    assert result.judge_requested == "test/paid"
    # Member discovery keeps the free-price caps; the scope='all' judge query does not.
    assert router.gets[0]["max_price"] == ["0"]
    assert router.gets[0]["max_output_price"] == ["0"]
    assert "max_price" not in router.gets[1]
    assert "max_output_price" not in router.gets[1]


def test_intelligence_ranking_can_select_paid_judge(config):
    result = query_result(
        "Question", settings=config, judge="most-intelligent", judge_scope="all", no_save=True
    )
    assert result.judge == "test/paid"


@pytest.mark.parametrize(
    "strategy,variable", [("free-router", "FREE_ROUTER_MODEL"), ("auto", "AUTO_ROUTER_MODEL")]
)
def test_router_strategy_requires_configured_id(config, router, strategy, variable):
    """No strategy supplies a built-in router model ID."""
    with pytest.raises(ConfigurationError, match=variable):
        query_result(
            "Question",
            settings=config,
            judge=strategy,
            free_router_model="",
            auto_router_model="",
            no_save=True,
        )
    assert not router.calls


@pytest.mark.parametrize("strategy", ["free-router", "auto"])
def test_router_strategy_uses_arbitrary_configured_id(config, router, strategy):
    """Routing accepts operator-provided IDs, not a fixed provider namespace."""
    result = query_result(
        "Question",
        settings=config,
        judge=strategy,
        judge_scope="all",
        free_router_model="fixture/free-routing",
        auto_router_model="fixture/auto-routing",
        no_save=True,
    )
    expected = "fixture/free-routing" if strategy == "free-router" else "fixture/auto-routing"
    assert result.judge_requested == expected
    assert router.calls[-1]["model"] == expected
    assert ("provider" in router.calls[-1]) == (strategy == "free-router")


def test_router_ids_parse_from_environment(tmp_path, monkeypatch):
    """Router settings are configurable with the standard precedence rules."""
    path = tmp_path / "settings.env"
    path.write_text(
        "FREE_ROUTER_MODEL=fixture/router\nROUTER_MODELS=fixture/a,fixture/a, fixture/b\n",
        encoding="utf-8",
    )
    for name in ("FREE_ROUTER_MODEL", "AUTO_ROUTER_MODEL", "ROUTER_MODELS"):
        monkeypatch.delenv(name, raising=False)
    settings = Settings.from_env(path)
    assert settings.configured_routers == ("fixture/a", "fixture/b", "fixture/router")
    monkeypatch.setenv("FREE_ROUTER_MODEL", "fixture/override")
    assert Settings.from_env(path).free_router_model == "fixture/override"


def test_no_model_specific_exclusions_without_configuration():
    """Previously hardcoded provider IDs are ordinary entries without configuration."""
    from conftest import model

    catalog = [model("thinkingmachines/inkling-small:free"), model("openrouter/free")]
    assert len(council_models(catalog)) == 2
    assert not Settings().configured_routers


def test_missing_intelligence_coverage_is_not_guessed(config, router):
    router.ranked = []
    with pytest.raises(ConfigurationError, match="no intelligence-ranked"):
        query_result("Question", settings=config, judge="most-intelligent")
    assert not router.calls


def test_gemini_model_selected_as_most_intelligent_and_called_via_gemini_api(
    config, router, monkeypatch
):
    """OpenRouter's paid google/gemini-* ranking selects gemini-3.8-flash as judge,
    while actual member and judge inference calls always go through the Gemini API."""
    from types import SimpleNamespace

    from conftest import model
    from google.genai import types as genai_types

    # google/gemini-3.8-flash is paid on OpenRouter, so free-only catalog queries hide it,
    # but it is the #1 ranked model in OpenRouter's intelligence-high-to-low ranking.
    gemini_or_entry = model("google/gemini-3.8-flash", price="0.002", context=2_000_000)
    router.catalog.append(gemini_or_entry)
    router.ranked = [gemini_or_entry, *router.ranked]

    gemini_calls = []

    class FakeModels:
        def generate_content(self, *, model, contents, config=None):
            gemini_calls.append({"model": model, "contents": contents, "config": config})
            is_judge = bool(config and config.system_instruction)
            text = "Gemini 3.8 Flash synthesis" if is_judge else f"Gemini member: {model}"
            return SimpleNamespace(
                candidates=[SimpleNamespace(finish_reason=genai_types.FinishReason.STOP)],
                text=text,
                model_version=model,
                usage_metadata=None,
            )

    monkeypatch.setattr(
        "google.genai.Client",
        lambda **_kw: SimpleNamespace(models=FakeModels(), close=lambda: None),
    )

    result = query_result(
        "Question",
        settings=config.with_overrides(
            gemini_api_key="gem-key",
            gemini_models="gemini-3.6-flash,gemini-3.8-flash",
            judge="most-intelligent",
            judge_scope="free",
        ),
        no_save=True,
    )
    assert result.judge_requested == "gemini-3.8-flash"
    assert result.judge == "gemini-3.8-flash"
    assert result.synthesis == "Gemini 3.8 Flash synthesis"
    # OpenRouter only received calls for its 2 free models, never for Gemini models
    assert sorted(call["model"] for call in router.calls) == [
        "test/alpha:free",
        "test/beta:free",
    ]
    # Gemini API received the 2 member calls + 1 synthesis judge call
    assert [call["model"] for call in gemini_calls] == [
        "gemini-3.6-flash",
        "gemini-3.8-flash",
        "gemini-3.8-flash",
    ]
    assert gemini_calls[-1]["config"].system_instruction is not None


def test_gemini_model_context_length_from_openrouter_selects_largest_context_judge(
    config, router, monkeypatch
):
    """OpenRouter's catalog context_length for google/gemini-* is used by largest-context."""
    from types import SimpleNamespace

    from conftest import model
    from google.genai import types as genai_types

    # Make test/beta:free huge (1,500,000), and google/gemini-3.8-flash even larger (2,000,000)
    router.catalog[1]["context_length"] = 1_500_000
    router.catalog.append(model("google/gemini-3.8-flash", price="0.002", context=2_000_000))

    gemini_calls = []

    class FakeModels:
        def generate_content(self, *, model, contents, config=None):
            gemini_calls.append({"model": model, "config": config})
            is_judge = bool(config and config.system_instruction)
            return SimpleNamespace(
                candidates=[SimpleNamespace(finish_reason=genai_types.FinishReason.STOP)],
                text="Largest context synthesis" if is_judge else f"Answer {model}",
                model_version=model,
                usage_metadata=None,
            )

    monkeypatch.setattr(
        "google.genai.Client",
        lambda **_kw: SimpleNamespace(models=FakeModels(), close=lambda: None),
    )

    result = query_result(
        "Question",
        settings=config.with_overrides(
            gemini_api_key="gem-key",
            gemini_models="gemini-3.8-flash",
            judge="largest-context",
            judge_scope="free",
        ),
        no_save=True,
    )
    assert result.judge_requested == "gemini-3.8-flash"
    assert result.synthesis == "Largest context synthesis"
    assert all("gemini" not in call["model"] for call in router.calls)
    assert [call["model"] for call in gemini_calls] == ["gemini-3.8-flash", "gemini-3.8-flash"]
