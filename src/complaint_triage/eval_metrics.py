"""The metrics this task is judged on.

Registered with the shared ``eval-harness`` package rather than defined inside
it. Micro-F1 over a label set means nothing to a retrieval agent, and a harness
that shipped it would be pretending otherwise.

Note what these metrics require that a simple average cannot give: Micro-F1 is
computed over the whole prediction set at once, not averaged per case. The
harness supports that because a metric is any ``Sequence[Observation] -> float``,
not only a field average.

Registration is an explicit call, not an import side effect. The registry is
global and shared with sibling projects; a module that registered itself on
import would mean any transitive import silently changed what every other
consumer sees.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Final

from eval_harness import Direction
from eval_harness.metrics import Observation, register_metric

from complaint_triage.categories import UNPARSED

MICRO_F1: Final = "micro_f1"
MICRO_F1_RAW: Final = "micro_f1_raw"
UNPARSEABLE_RATE: Final = "unparseable_rate"
NORMALIZATION_GAIN: Final = "normalization_gain"
SUMMARY_LENGTH_RATIO: Final = "summary_length_ratio"

CLASSIFICATION_METRICS: Final = (
    MICRO_F1,
    MICRO_F1_RAW,
    NORMALIZATION_GAIN,
    UNPARSEABLE_RATE,
    "p50_latency_seconds",
    "p95_latency_seconds",
)

SUMMARIZATION_METRICS: Final = (
    SUMMARY_LENGTH_RATIO,
    "p50_latency_seconds",
    "p95_latency_seconds",
)

# Held in a dict rather than a module global so no `global` statement is
# needed; the flag exists only to make registration idempotent.
_STATE: dict[str, bool] = {"registered": False}


def _micro_f1(predictions: Sequence[Any], references: Sequence[Any]) -> float | None:
    from sklearn.metrics import f1_score

    if not references:
        return None
    cleaned = [p if p else UNPARSED for p in predictions]
    return float(f1_score(list(references), cleaned, average="micro"))


def micro_f1(observations: Sequence[Observation]) -> float | None:
    """Micro-F1 over parsed predictions. Unreadable output counts as wrong."""
    usable = [o for o in observations if o.get("reference") is not None]
    return _micro_f1(
        [o.get("prediction") for o in usable], [o["reference"] for o in usable]
    )


def micro_f1_raw(observations: Sequence[Observation]) -> float | None:
    """Micro-F1 scoring the raw generation as a string, with no parsing.

    The control. This is what the same model scores when nothing reads its
    output for it, and the gap between this and :func:`micro_f1` is the whole
    argument for the normalization layer.
    """
    usable = [o for o in observations if o.get("reference") is not None]
    return _micro_f1(
        [o.get("raw_normalized") for o in usable], [o["reference"] for o in usable]
    )


def normalization_gain(observations: Sequence[Observation]) -> float | None:
    """Micro-F1 with parsing minus Micro-F1 without it.

    Reported as its own metric so a change that quietly erodes the parser shows
    up directly, rather than having to be inferred from two numbers moving
    together.
    """
    parsed, raw = micro_f1(observations), micro_f1_raw(observations)
    if parsed is None or raw is None:
        return None
    return parsed - raw


def unparseable_rate(observations: Sequence[Observation]) -> float | None:
    """Share of generations the parser could not read.

    A harness-health metric, not a quality one. If it rises, the model changed
    its output shape and the parser has not caught up.
    """
    if not observations:
        return None
    return sum(o.get("prediction") in (None, "") for o in observations) / len(
        observations
    )


def summary_length_ratio(observations: Sequence[Observation]) -> float | None:
    """Mean summary length as a fraction of source length.

    A guard, not a quality score. Reference-based summarization metrics reward
    copying, so a model that returns the complaint verbatim can post a strong
    score while having summarized nothing. A ratio drifting towards 1.0 is that
    failure, and no similarity metric will show it.
    """
    ratios = [
        o["summary_words"] / o["source_words"]
        for o in observations
        if o.get("source_words")
    ]
    return sum(ratios) / len(ratios) if ratios else None


def register_task_metrics(*, replace: bool = False) -> tuple[str, ...]:
    """Register this task's metrics and return the classification set.

    Idempotent, so calling it from the CLI and from a notebook in the same
    process is safe.
    """
    if _STATE["registered"] and not replace:
        return CLASSIFICATION_METRICS

    register_metric(
        MICRO_F1,
        micro_f1,
        direction=Direction.HIGHER_IS_BETTER,
        description="Micro-F1 over parsed predictions.",
        replace=True,
    )
    register_metric(
        MICRO_F1_RAW,
        micro_f1_raw,
        direction=Direction.HIGHER_IS_BETTER,
        description="Micro-F1 scoring raw generations, with no parsing. The control.",
        replace=True,
    )
    register_metric(
        NORMALIZATION_GAIN,
        normalization_gain,
        direction=Direction.HIGHER_IS_BETTER,
        description="How much measured F1 the normalization layer is worth.",
        replace=True,
    )
    register_metric(
        UNPARSEABLE_RATE,
        unparseable_rate,
        direction=Direction.LOWER_IS_BETTER,
        description="Share of generations the parser could not read.",
        replace=True,
    )
    register_metric(
        SUMMARY_LENGTH_RATIO,
        summary_length_ratio,
        direction=Direction.LOWER_IS_BETTER,
        description=(
            "Mean summary length over source length. A guard against a model "
            "that copies its input and scores well for it."
        ),
        replace=True,
    )
    # Latency percentiles are built into the harness; nothing to register.
    _STATE["registered"] = True
    return CLASSIFICATION_METRICS
