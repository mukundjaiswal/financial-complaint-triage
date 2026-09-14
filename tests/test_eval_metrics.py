import pytest
from eval_harness import Direction
from eval_harness.metrics import get_metric

from complaint_triage.eval_metrics import (
    MICRO_F1,
    MICRO_F1_RAW,
    NORMALIZATION_GAIN,
    SUMMARY_LENGTH_RATIO,
    UNPARSEABLE_RATE,
    register_task_metrics,
)


def obs(prediction, raw_normalized, reference):
    return {
        "prediction": prediction,
        "raw_normalized": raw_normalized,
        "reference": reference,
    }


def test_registration_is_idempotent():
    assert register_task_metrics() == register_task_metrics()


def test_quality_metrics_are_higher_is_better():
    assert get_metric(MICRO_F1).direction is Direction.HIGHER_IS_BETTER
    assert get_metric(NORMALIZATION_GAIN).direction is Direction.HIGHER_IS_BETTER


def test_failure_metrics_are_lower_is_better():
    assert get_metric(UNPARSEABLE_RATE).direction is Direction.LOWER_IS_BETTER
    assert get_metric(SUMMARY_LENGTH_RATIO).direction is Direction.LOWER_IS_BETTER


def test_parsing_scores_higher_than_raw_on_messy_output():
    """The project's central claim, as an assertion."""
    observations = [
        obs("credit_card", r"category: credit\_card", "credit_card"),
        obs("debt_collection", "this is debt_collection.", "debt_collection"),
    ]
    assert get_metric(MICRO_F1)(observations) == pytest.approx(1.0)
    assert get_metric(MICRO_F1_RAW)(observations) == pytest.approx(0.0)
    assert get_metric(NORMALIZATION_GAIN)(observations) == pytest.approx(1.0)


def test_unreadable_output_counts_as_wrong_not_skipped():
    """Dropping it would flatter the score by shrinking the denominator."""
    observations = [
        obs(None, "no idea", "credit_card"),
        obs("credit_card", "credit_card", "credit_card"),
    ]
    assert get_metric(MICRO_F1)(observations) == pytest.approx(0.5)


def test_unparseable_rate_counts_them():
    observations = [obs(None, "x", "a"), obs("credit_card", "credit_card", "a")]
    assert get_metric(UNPARSEABLE_RATE)(observations) == pytest.approx(0.5)


def test_cases_without_a_reference_are_excluded():
    observations = [obs("credit_card", "credit_card", None)]
    assert get_metric(MICRO_F1)(observations) is None


def test_length_ratio_flags_a_model_that_copies_its_input():
    """Similarity metrics reward copying; this is what catches it."""
    copier = [{"summary_words": 100, "source_words": 100}]
    real = [{"summary_words": 15, "source_words": 100}]
    assert get_metric(SUMMARY_LENGTH_RATIO)(copier) == pytest.approx(1.0)
    assert get_metric(SUMMARY_LENGTH_RATIO)(real) == pytest.approx(0.15)


def test_metrics_of_nothing_are_none_not_zero():
    assert get_metric(MICRO_F1)([]) is None
    assert get_metric(UNPARSEABLE_RATE)([]) is None
