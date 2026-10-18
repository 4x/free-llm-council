"""Catalog model selection and structured council results."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

TEXT_CHARGES = (
    "prompt",
    "completion",
    "request",
    "internal_reasoning",
    "input_cache_read",
    "input_cache_write",
)


def is_text_model(model: dict[str, Any]) -> bool:
    architecture = model.get("architecture")
    if not isinstance(architecture, dict):
        return False
    # Older catalog entries may use only the explicit 'text+image->text' summary.
    modality = architecture.get("modality")
    summary = modality.split("->") if isinstance(modality, str) else []
    for index, key in enumerate(("input_modalities", "output_modalities")):
        values = architecture.get(key)
        if key not in architecture and len(summary) == 2:
            values = [part.strip() for part in summary[index].split("+")]
        if not isinstance(values, list) or any(not isinstance(item, str) for item in values):
            return False
        if "text" not in values:
            return False
    return True


def model_exclusion_reason(
    model: dict[str, Any], excluded_models: tuple[str, ...] = ()
) -> str | None:
    model_id = model.get("id")
    if not isinstance(model_id, str) or not model_id:
        return "missing model ID"
    if model_id in excluded_models:
        return "configured exclusion"
    if not is_text_model(model):
        return "text input/output not confirmed"
    return None


def supports_tools(model: dict[str, Any]) -> bool:
    """Return whether the model advertises tool calling.

    OpenRouter's server-side `openrouter:web_search` tool is written into the request's
    `tools` array, so only models that accept tool calls can decide to search.
    """
    parameters = model.get("supported_parameters")
    return isinstance(parameters, list) and "tools" in parameters


def is_free_model(model: dict[str, Any]) -> bool:
    pricing = model.get("pricing") or {}
    if "prompt" not in pricing or "completion" not in pricing:
        return False
    # Conditional prices cannot safely be assumed free for an arbitrary-length prompt.
    if pricing.get("overrides"):
        return False
    try:
        charges = [pricing.get(key, "0") for key in TEXT_CHARGES]
        if "output" in pricing:
            charges.append(pricing["output"])
        return all(Decimal(str(charge)) == 0 for charge in charges)
    except (InvalidOperation, ValueError):
        return False


def council_models(
    catalog: list[dict[str, Any]],
    excluded_models: tuple[str, ...] = (),
    router_models: tuple[str, ...] = (),
    require_tools: bool = False,
) -> list[dict[str, Any]]:
    """Return free text models, optionally limited to tool-calling models."""
    unique: dict[str, dict[str, Any]] = {}
    for model in catalog:
        model_id = model.get("id")
        if not isinstance(model_id, str) or model_id in router_models:
            continue
        if model_exclusion_reason(model, excluded_models) is not None:
            continue
        if not is_free_model(model):
            continue
        if require_tools and not supports_tools(model):
            logger.debug("Skipping model %s: tool calling not supported", model_id)
            continue
        unique[model_id] = model
    return [unique[key] for key in sorted(unique)]


@dataclass(frozen=True)
class ModelAnswer:
    requested_model: str
    model: str
    content: str | None = None
    error: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CouncilResult:
    question_id: str
    prompt: str
    answers: tuple[ModelAnswer, ...]
    synthesis: str
    judge: str
    judge_requested: str
    synthesis_usage: dict[str, Any]
    markdown_path: Path | None = None

    @property
    def failures(self) -> tuple[ModelAnswer, ...]:
        return tuple(answer for answer in self.answers if answer.error is not None)
