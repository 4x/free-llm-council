"""Post-run reconciliation of stored answers against `GET /generation?id=`.

Inline router metadata captured during a run reports the router's own accounting.
Only `GET /generation` adds what the upstream provider knows: native tokenizer
counts, BYOK `upstream_inference_cost`, and the applied `cache_discount`. This
module backfills those fields into `responses.usage` for answers already stored,
without touching prompts, content, or any other column.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from .client import OpenRouterClient
from .config import Settings
from .errors import ConfigurationError, OpenRouterError
from .models import (
    GENERATION_RECONCILED_AT_KEY,
    GENERATION_UNAVAILABLE_KEY,
    GENERATION_USAGE_KEY,
)
from .storage import load_generation_candidates, save_generation_reconciliation

logger = logging.getLogger(__name__)

# OpenRouter prunes old generations, so a missing one is permanent rather than a
# transient failure; record it so each later run does not retry the same row.
GENERATION_GONE_STATUS = 404


class ReconcileOutcome(StrEnum):
    """Result of attempting to reconcile one stored answer."""

    UPDATED = "updated"
    UNAVAILABLE = "unavailable"
    FAILED = "failed"


@dataclass(frozen=True)
class ReconcileItem:
    """One stored answer's reconciliation result."""

    response_id: str
    model: str
    generation_id: str
    outcome: ReconcileOutcome
    detail: str = ""


@dataclass(frozen=True)
class ReconcileReport:
    """Aggregate outcome of one reconciliation pass."""

    items: tuple[ReconcileItem, ...] = ()
    # Responses with no generation ID, which no amount of retrying can reconcile.
    without_generation_id: int = 0

    @property
    def updated_count(self) -> int:
        """Number of rows whose usage gained `/generation` accounting."""
        return sum(1 for item in self.items if item.outcome is ReconcileOutcome.UPDATED)

    @property
    def unavailable_count(self) -> int:
        """Number of rows whose generation OpenRouter no longer serves."""
        return sum(1 for item in self.items if item.outcome is ReconcileOutcome.UNAVAILABLE)

    @property
    def failures(self) -> tuple[ReconcileItem, ...]:
        """Rows that failed transiently and stay queued for the next pass."""
        return tuple(item for item in self.items if item.outcome is ReconcileOutcome.FAILED)

    def summary(self) -> str:
        """Return a one-paragraph, user-facing description of the pass."""
        if not self.items:
            if self.without_generation_id:
                return (
                    "No stored answers are awaiting reconciliation; "
                    f"{self.without_generation_id} have no generation ID "
                    "(Gemini answers or runs recorded before metadata capture)."
                )
            return "No stored answers are awaiting reconciliation."
        parts = [f"Reconciled {self.updated_count} of {len(self.items)} stored answer(s)."]
        if self.unavailable_count:
            parts.append(
                f"{self.unavailable_count} generation(s) are no longer available "
                "and were marked as such."
            )
        if self.failures:
            parts.append(f"{len(self.failures)} failed and will be retried on the next run.")
        if self.without_generation_id:
            parts.append(f"{self.without_generation_id} skipped without a generation ID.")
        return " ".join(parts)


def _merge_generation_usage(
    usage: dict[str, Any], data: dict[str, Any], reconciled_at: int
) -> dict[str, Any]:
    """Return stored usage with the fetched generation accounting merged in."""
    merged = dict(usage)
    # Keep the whole payload so later analysis can reach fields we do not surface.
    merged[GENERATION_USAGE_KEY] = data
    merged[GENERATION_RECONCILED_AT_KEY] = reconciled_at
    return merged


def _reconcile_rows(
    client: OpenRouterClient, pending: tuple[dict[str, Any], ...], reconciled_at: int
) -> tuple[list[ReconcileItem], list[tuple[str, dict[str, Any]]]]:
    """Fetch generation accounting for each stored row, isolating per-row failures."""
    items: list[ReconcileItem] = []
    updates: list[tuple[str, dict[str, Any]]] = []
    for row in pending:
        response_id = str(row["response_id"])
        generation_id = str(row["generation_id"])
        model = str(row["model"])
        try:
            data = client.get_generation(generation_id)
        except OpenRouterError as exc:
            if exc.status_code == GENERATION_GONE_STATUS:
                logger.warning(
                    "Generation %s for %s is no longer available: %s",
                    generation_id,
                    model,
                    exc,
                )
                unavailable = dict(row["usage"])
                unavailable[GENERATION_UNAVAILABLE_KEY] = str(exc)
                updates.append((response_id, unavailable))
                items.append(
                    ReconcileItem(
                        response_id, model, generation_id, ReconcileOutcome.UNAVAILABLE, str(exc)
                    )
                )
                continue
            logger.error("Cannot reconcile generation %s for %s: %s", generation_id, model, exc)
            items.append(
                ReconcileItem(response_id, model, generation_id, ReconcileOutcome.FAILED, str(exc))
            )
            continue
        updates.append((response_id, _merge_generation_usage(row["usage"], data, reconciled_at)))
        items.append(ReconcileItem(response_id, model, generation_id, ReconcileOutcome.UPDATED))
        logger.debug(
            "Reconciled generation %s for %s: native prompt=%s, completion=%s, upstream cost=%s",
            generation_id,
            model,
            data.get("native_tokens_prompt"),
            data.get("native_tokens_completion"),
            data.get("upstream_inference_cost"),
        )
    return items, updates


def reconcile_generations(
    *,
    env_file: str | Path | None = None,
    settings: Settings | None = None,
    client: OpenRouterClient | None = None,
    limit: int | None = None,
    **overrides: Any,
) -> ReconcileReport:
    """Backfill native token counts and upstream BYOK cost for stored answers.

    Rows are selected by generation ID, so Gemini answers and runs recorded before
    inline metadata capture are counted and skipped rather than guessed at. Each
    fetched payload lands under `usage["generation"]`, leaving the inline accounting
    untouched. Failures are isolated per row: transient errors stay queued, while a
    pruned generation is marked `generation_unavailable` and never retried.

    `client` is injectable for tests; otherwise a client is opened and closed here.
    """
    config = settings if settings is not None else Settings.from_env(env_file)
    config = config.with_overrides(**overrides)
    if limit is not None and limit < 1:
        raise ConfigurationError("Reconcile limit must be at least 1.")
    if not config.api_key.strip():
        raise ConfigurationError("Set OPENROUTER_API_KEY in .env or the environment.")
    candidates = load_generation_candidates(config, limit=limit)
    if not candidates.pending:
        logger.info("No stored answers are awaiting generation reconciliation.")
        return ReconcileReport(without_generation_id=candidates.without_generation_id)
    logger.info("Reconciling %d stored answer(s) against /generation", len(candidates.pending))
    reconciled_at = int(time.time())
    if client is not None:
        items, updates = _reconcile_rows(client, candidates.pending, reconciled_at)
    else:
        with OpenRouterClient(config) as owned:
            items, updates = _reconcile_rows(owned, candidates.pending, reconciled_at)
    written = save_generation_reconciliation(config, updates)
    logger.info("Stored reconciled usage for %d response(s)", written)
    return ReconcileReport(
        items=tuple(items), without_generation_id=candidates.without_generation_id
    )
