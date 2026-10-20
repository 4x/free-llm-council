"""Tests for post-run reconciliation of stored answers against GET /generation."""

from __future__ import annotations

import json

import duckdb
import pytest

from llm_council import ModelAnswer, Settings, reconcile_generations
from llm_council.errors import ConfigurationError, StorageError
from llm_council.models import (
    GENERATION_RECONCILED_AT_KEY,
    GENERATION_UNAVAILABLE_KEY,
    GENERATION_USAGE_KEY,
)
from llm_council.storage import RunStorage, load_generation_candidates

UPSTREAM_DATA = {
    "id": "gen-member-1",
    "model": "test/alpha:free",
    "provider_name": "FixtureProvider",
    "native_tokens_prompt": 39,
    "native_tokens_completion": 158,
    "total_cost": 0.0025,
    "upstream_inference_cost": 0.0021,
    "cache_discount": 0.0004,
    "finish_reason": "stop",
}


@pytest.fixture
def settings(router, tmp_path):
    return Settings().with_overrides(
        api_key="fixture-key",
        base_url=router.url,
        request_interval=0,
        max_retries=0,
        save_initial_answers="duckdb",
        save_synthesis="duckdb",
        duckdb_path=tmp_path / "council.duckdb",
        markdown_dir=tmp_path / "markdown",
    )


def persist(settings, usage, *, question_id="q1", model="test/alpha:free", content="Answer"):
    """Store one initial answer with the given usage JSON."""
    with RunStorage(settings, question_id, "Reconcile question") as storage:
        storage.save_question("Reconcile question", None)
        storage.save_answer(ModelAnswer(model, model, content=content, usage=usage), "initial", 1)


def stored_usage(settings, *, requested_model="test/alpha:free"):
    """Read the first stored usage mapping for a model back from DuckDB."""
    with duckdb.connect(str(settings.duckdb_path), read_only=True) as connection:
        row = connection.execute(
            "SELECT usage FROM responses WHERE requested_model = ?", [requested_model]
        ).fetchone()
    assert row is not None
    return json.loads(row[0])


def test_candidates_include_only_unreconciled_answers_with_generation_ids(settings):
    """Gemini answers and already-reconciled rows never enter the queue."""
    persist(settings, {"generation_id": "gen-pending", "total_tokens": 10})
    persist(settings, {"completion_tokens": 5}, question_id="q2")
    persist(
        settings,
        {
            "generation_id": "gen-done",
            GENERATION_USAGE_KEY: {"native_tokens_prompt": 1},
            GENERATION_RECONCILED_AT_KEY: 1_700_000_000,
        },
        question_id="q3",
    )

    candidates = load_generation_candidates(settings)

    assert [row["generation_id"] for row in candidates.pending] == ["gen-pending"]
    assert candidates.without_generation_id == 1


def test_candidates_limit_bounds_the_queue_but_not_the_skip_count(settings):
    """`limit` caps fetched rows newest first while still counting unreconcilable rows."""
    for index in range(3):
        persist(settings, {"generation_id": f"gen-{index}"}, question_id=f"q{index}")
    persist(settings, {"total_tokens": 1}, question_id="q-gemini")

    candidates = load_generation_candidates(settings, limit=2)

    assert len(candidates.pending) == 2
    assert candidates.without_generation_id == 1


def test_reconcile_merges_native_tokens_and_upstream_cost(settings, router):
    """Native tokenizer counts, BYOK cost, and cache discount are written into usage."""
    router.generations["gen-member-1"] = UPSTREAM_DATA
    persist(
        settings,
        {
            "generation_id": "gen-member-1",
            "provider": "FixtureProvider",
            "total_tokens": 200,
            "cost": 0.0025,
            "citations": [{"url": "https://example.com/a", "title": "A"}],
        },
    )

    report = reconcile_generations(settings=settings)

    assert report.updated_count == 1
    assert report.failures == ()
    assert "Reconciled 1 of 1" in report.summary()
    usage = stored_usage(settings)
    assert usage[GENERATION_USAGE_KEY] == UPSTREAM_DATA
    assert isinstance(usage[GENERATION_RECONCILED_AT_KEY], int)
    # Inline accounting survives untouched alongside the reconciled payload.
    assert usage["total_tokens"] == 200
    assert usage["cost"] == 0.0025
    assert usage["citations"] == [{"url": "https://example.com/a", "title": "A"}]
    assert router.gets[-1]["id"] == ["gen-member-1"]


def test_reconcile_is_idempotent(settings, router):
    """A second pass fetches nothing because the row is already reconciled."""
    router.generations["gen-member-1"] = UPSTREAM_DATA
    persist(settings, {"generation_id": "gen-member-1", "total_tokens": 200})
    reconcile_generations(settings=settings)
    generation_requests = len(router.gets)

    second = reconcile_generations(settings=settings)

    assert second.items == ()
    assert len(router.gets) == generation_requests
    assert "awaiting reconciliation" in second.summary()


def test_reconcile_marks_pruned_generation_and_stops_retrying(settings, router):
    """A 404 generation is recorded once, reported, and excluded from later passes."""
    persist(settings, {"generation_id": "gen-gone", "total_tokens": 7})

    report = reconcile_generations(settings=settings)

    assert report.updated_count == 0
    assert report.unavailable_count == 1
    assert report.failures == ()
    usage = stored_usage(settings)
    assert "404" in usage[GENERATION_UNAVAILABLE_KEY]
    assert GENERATION_USAGE_KEY not in usage
    assert load_generation_candidates(settings).pending == ()
    assert reconcile_generations(settings=settings).items == ()


def test_reconcile_isolates_transient_failures_and_keeps_them_queued(settings, router):
    """A 500 leaves the row untouched and available for the next pass."""
    router.generation_status["gen-flaky"] = 500
    persist(settings, {"generation_id": "gen-flaky", "total_tokens": 11})

    report = reconcile_generations(settings=settings)

    assert report.updated_count == 0
    assert len(report.failures) == 1
    assert "500" in report.failures[0].detail
    assert stored_usage(settings) == {"generation_id": "gen-flaky", "total_tokens": 11}
    assert [row["generation_id"] for row in load_generation_candidates(settings).pending] == [
        "gen-flaky"
    ]


def test_reconcile_reports_unreconcilable_answers_without_failing(settings):
    """Stored answers without a generation ID produce a clear, non-failing summary."""
    persist(settings, {"total_tokens": 3})

    report = reconcile_generations(settings=settings)

    assert report.items == ()
    assert "1 have no generation ID" in report.summary()


def test_reconcile_requires_an_api_key(settings):
    """Reconciliation talks to OpenRouter and fails loudly without credentials."""
    with pytest.raises(ConfigurationError, match="OPENROUTER_API_KEY"):
        reconcile_generations(settings=settings.with_overrides(api_key=""))


def test_reconcile_rejects_non_positive_limit(settings):
    """A zero or negative batch size is a configuration error, not a silent no-op."""
    with pytest.raises(ConfigurationError, match="at least 1"):
        reconcile_generations(settings=settings, limit=0)


def test_reconcile_requires_an_existing_database(settings):
    """Pointing at a missing DuckDB file is an actionable storage error."""
    assert not settings.duckdb_path.exists()
    with pytest.raises(StorageError, match="does not exist"):
        reconcile_generations(settings=settings)
