"""Command-line interface.

complaint-triage classify "my card was charged twice"
complaint-triage evaluate --dataset evaluation/datasets/v1.jsonl --shots 0
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Annotated

import typer

from complaint_triage import __version__
from complaint_triage.settings import get_settings

app = typer.Typer(
    name="complaint-triage",
    help="Route and summarize financial complaints with an open-weight model.",
    no_args_is_help=True,
    add_completion=False,
)


@app.command()
def version() -> None:
    """Print the package version."""
    typer.echo(__version__)


@app.command()
def config() -> None:
    """Print the resolved configuration, with secrets masked."""
    typer.echo(json.dumps(json.loads(get_settings().model_dump_json()), indent=2))


@app.command()
def classify(
    complaint: Annotated[str, typer.Argument(help="The complaint text.")],
    show_raw: Annotated[
        bool, typer.Option(help="Also print the unparsed generation.")
    ] = False,
) -> None:
    """Classify a single complaint."""
    from complaint_triage.llm.gguf import GgufChatModel
    from complaint_triage.tasks import Classifier

    result = Classifier(GgufChatModel()).predict(complaint)
    typer.echo(result["prediction"] or "unparseable")
    if show_raw:
        typer.echo(f"\nraw: {result['raw']!r}", err=True)


@app.command()
def summarize(
    complaint: Annotated[str, typer.Argument(help="The complaint text.")],
) -> None:
    """Summarize a single complaint."""
    from complaint_triage.llm.gguf import GgufChatModel
    from complaint_triage.tasks import Summarizer

    typer.echo(Summarizer(GgufChatModel()).predict(complaint)["summary"])


@app.command()
def evaluate(
    dataset: Annotated[Path, typer.Option(help="JSON Lines evaluation set.")],
    task: Annotated[str, typer.Option(help="classify or summarize.")] = "classify",
    shots: Annotated[
        int, typer.Option(help="In-context examples, taken from the head of the set.")
    ] = 0,
    out: Annotated[Path, typer.Option(help="Where to write the report.")] = Path(
        "evaluation/results/report.json"
    ),
    baseline: Annotated[
        Path | None, typer.Option(help="Committed baseline to compare against.")
    ] = None,
    seeds: Annotated[str, typer.Option(help="Comma-separated seeds.")] = "0",
    tolerance: Annotated[
        float, typer.Option(help="Allowed drift before a metric counts as regressed.")
    ] = 0.02,
) -> None:
    """Run the evaluation suite and optionally gate on a baseline.

    Scoring, aggregation and the gate come from the shared ``eval-harness``
    package. This command supplies only what is specific to this task.
    """
    from collections.abc import Callable, Sequence

    from eval_harness import EvalCase, EvaluationRunner, RegressionCheck, load_cases
    from eval_harness.report import load_baseline

    from complaint_triage.eval_metrics import (
        CLASSIFICATION_METRICS,
        SUMMARIZATION_METRICS,
        register_task_metrics,
    )
    from complaint_triage.llm.gguf import GgufChatModel
    from complaint_triage.tasks import Classifier, Summarizer

    register_task_metrics()
    model = GgufChatModel()

    # The dataset keeps its documented "narrative"/"product" keys rather than
    # being rewritten to suit the shared package.
    cases = load_cases(dataset, input_key="narrative", reference_key="product")

    run: Callable[[EvalCase], dict[str, object]]
    metrics: Sequence[str]

    if task == "classify":
        examples = [(c.input, c.reference or "") for c in cases[:shots]]
        scored = cases[shots:]
        classifier = Classifier(model, examples=tuple(examples))

        def run(case: EvalCase) -> dict[str, object]:
            return {**classifier.predict(case.input), "reference": case.reference}

        metrics = CLASSIFICATION_METRICS
    elif task == "summarize":
        scored = cases
        summarizer = Summarizer(model)

        def run(case: EvalCase) -> dict[str, object]:
            return {**summarizer.predict(case.input), "reference": case.reference}

        metrics = SUMMARIZATION_METRICS
    else:
        typer.echo(f"Unknown task {task!r}. Use 'classify' or 'summarize'.", err=True)
        sys.exit(2)

    report = EvaluationRunner(
        run,
        metrics=metrics,
        seeds=[int(s) for s in seeds.split(",") if s.strip()],
    ).run(scored, dataset=dataset)

    report.write(out)
    typer.echo(report.summary())
    typer.echo(f"\nWrote {out}")

    if baseline is not None:
        result = RegressionCheck(tolerance=tolerance).compare(
            report, load_baseline(baseline)
        )
        typer.echo("")
        typer.echo(result.report())
        if not result.passed:
            typer.echo(
                f"\n{len(result.regressions)} metric(s) regressed past "
                f"tolerance {tolerance}.",
                err=True,
            )
            sys.exit(1)
        typer.echo("\nNo regression against baseline.")


if __name__ == "__main__":
    app()
