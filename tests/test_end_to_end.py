"""The full pipeline through the shared harness, driven by a fake model.

This is the test that proves the cross-repo seam: a task defined here, scored
by metrics registered here, run and gated entirely by eval-harness.
"""

from eval_harness import EvalCase, EvaluationRunner, RegressionCheck

from complaint_triage.eval_metrics import CLASSIFICATION_METRICS, register_task_metrics
from complaint_triage.llm.fake import ScriptedChatModel
from complaint_triage.tasks import Classifier

CASES = [
    EvalCase(input="my card was charged twice", reference="credit_card"),
    EvalCase(input="a collector keeps calling", reference="debt_collection"),
]


def runner_for(replies):
    classifier = Classifier(ScriptedChatModel(replies))

    def task(case):
        return {**classifier.predict(case.input), "reference": case.reference}

    return EvaluationRunner(task, metrics=CLASSIFICATION_METRICS, seeds=[0])


def test_a_clean_run_scores_perfectly():
    register_task_metrics(replace=True)
    report = runner_for(["Category: credit_card", "Category: debt_collection"]).run(
        CASES, dataset="v1.jsonl"
    )
    assert report.metric("micro_f1").mean == 1.0
    assert report.metric("unparseable_rate").mean == 0.0


def test_the_report_shows_what_normalization_is_worth():
    register_task_metrics(replace=True)
    report = runner_for([r"Category: credit\_card", "Category: debt_reporting"]).run(
        CASES, dataset="v1.jsonl"
    )
    assert report.metric("micro_f1").mean == 1.0
    assert report.metric("micro_f1_raw").mean == 0.0
    assert report.metric("normalization_gain").mean == 1.0


def test_the_gate_catches_a_parser_that_stopped_working():
    """A model changing its output shape must fail the build, not pass quietly."""
    register_task_metrics(replace=True)
    good = runner_for(["Category: credit_card", "Category: debt_collection"]).run(
        CASES, dataset="v1.jsonl"
    )
    broken = runner_for(["I think maybe", "not sure"]).run(CASES, dataset="v1.jsonl")

    result = RegressionCheck(tolerance=0.02).compare(
        broken, good.to_dict(include_observations=False)
    )
    assert not result.passed
    names = {c.name for c in result.regressions}
    assert "micro_f1" in names
    assert "unparseable_rate" in names


def test_latency_is_reported_without_the_task_measuring_it():
    register_task_metrics(replace=True)
    report = runner_for(["Category: credit_card"]).run(CASES, dataset="v1.jsonl")
    assert report.metric("p95_latency_seconds").mean is not None
