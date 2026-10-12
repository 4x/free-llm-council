import pytest

from llm_council import ConfigurationError, judges, query_result


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


def test_intelligence_ranking_can_select_paid_judge(config):
    result = query_result(
        "Question", settings=config, judge="most-intelligent", judge_scope="all", no_save=True
    )
    assert result.judge == "test/paid"


def test_missing_intelligence_coverage_is_not_guessed(config, router):
    router.ranked = []
    with pytest.raises(ConfigurationError, match="no intelligence-ranked"):
        query_result("Question", settings=config, judge="most-intelligent")
    assert not router.calls
