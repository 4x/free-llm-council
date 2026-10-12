"""Judge policies; ranking uses OpenRouter's server-side intelligence index."""

from __future__ import annotations

import random
import threading
from typing import Any

from .client import OpenRouterClient
from .config import Settings
from .errors import ConfigurationError
from .models import (
    HARNESS_ONLY_MODELS,
    ROUTERS,
    council_models,
    is_free_model,
    model_exclusion_reason,
)

STRATEGIES = ("random", "round-robin", "free-router", "auto", "most-intelligent", "largest-context")
_round_robin_lock = threading.Lock()
_round_robin_positions: dict[tuple[str, ...], int] = {}


def select_judge(
    settings: Settings, catalog: list[dict[str, Any]], client: OpenRouterClient
) -> tuple[str, bool]:
    strategy = settings.judge.strip()
    normalized = strategy.lower()
    requested_id = {"free-router": "openrouter/free", "auto": "openrouter/auto"}.get(
        normalized, strategy
    )
    if requested_id in HARNESS_ONLY_MODELS or requested_id in settings.excluded_models:
        raise ConfigurationError(
            f"Judge model {requested_id!r} is excluded from plain-chat queries."
        )
    if normalized in {"free-router", "openrouter/free"}:
        return "openrouter/free", True
    if normalized in {"auto", "openrouter/auto"}:
        if settings.judge_scope != "all":
            raise ConfigurationError(
                "The auto judge may use paid models; set judge_scope='all' "
                "or use 'free-router' for free routing."
            )
        return "openrouter/auto", False
    eligible = (
        council_models(catalog, settings.excluded_models)
        if settings.judge_scope == "free"
        else [
            model
            for model in catalog
            if model_exclusion_reason(model, settings.excluded_models) is None
            and model["id"] not in ROUTERS
        ]
    )
    ids = sorted({model["id"] for model in eligible})
    if normalized not in STRATEGIES:
        matches = [model for model in catalog if model.get("id") == strategy]
        if not matches or model_exclusion_reason(matches[0], settings.excluded_models) is not None:
            raise ConfigurationError(f"Unknown judge strategy or text model: {strategy!r}.")
        free = is_free_model(matches[0])
        if settings.judge_scope == "free" and not free:
            raise ConfigurationError("A paid judge requires judge_scope='all'.")
        return strategy, free
    if not ids:
        raise ConfigurationError("No eligible judge models were discovered.")
    if normalized == "random":
        selected = random.choice(ids)
    elif normalized == "round-robin":
        key = tuple(ids)
        with _round_robin_lock:
            position = _round_robin_positions.get(key, 0)
            selected = ids[position % len(ids)]
            _round_robin_positions[key] = position + 1
    elif normalized == "largest-context":
        selected = max(
            eligible, key=lambda model: (int(model.get("context_length") or 0), model["id"])
        )["id"]
    else:
        # Exclude unranked models; never silently substitute context length for intelligence.
        ranked = client.list_models(sort="intelligence-high-to-low", min_intelligence_index=0)
        selected = next((model["id"] for model in ranked if model.get("id") in ids), "")
        if not selected:
            raise ConfigurationError(
                "OpenRouter returned no intelligence-ranked judge in the chosen scope; "
                "use an explicit judge, largest-context, or judge_scope='all'."
            )
    selected_model = next(model for model in eligible if model["id"] == selected)
    return selected, is_free_model(selected_model)
