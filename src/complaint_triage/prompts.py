"""Prompt templates, in the instruction format the model was trained on.

The classification prompt demands a fixed output shape. It is not obeyed every
time, which is exactly why :mod:`complaint_triage.postprocess` exists: a prompt
is a request, not a guarantee, and a pipeline that assumes otherwise measures
its own parser.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Final

from complaint_triage.categories import CATEGORIES

_LABELS: Final = ", ".join(CATEGORIES)

CLASSIFY_SYSTEM_MESSAGE: Final = f"""\
You are a financial complaint routing assistant.
Classify the customer complaint into exactly one product category.

Valid categories: {_LABELS}

Rules:
- Choose exactly one category from the list.
- Respond in the form: Category: <category>
- Do not explain your reasoning.
"""

SUMMARIZE_SYSTEM_MESSAGE: Final = """\
You are a financial complaint summarization assistant.
Summarize the customer's complaint clearly and accurately.

Rules:
- Summarize only information present in the complaint.
- Do not add assumptions or outside information.
- Keep the summary succinct but complete.
"""

INSTRUCT_TEMPLATE: Final = "<s>[INST]\n{system_message}\n\n{user_input}\n[/INST]"

_SHOT_TEMPLATE: Final = (
    "<s>[INST]\nCustomer complaint: {complaint}\n[/INST]\nCategory: {label}</s>"
)


def classify_prompt(complaint: str) -> str:
    """Zero-shot classification prompt."""
    return INSTRUCT_TEMPLATE.format(
        system_message=CLASSIFY_SYSTEM_MESSAGE,
        user_input=f"Customer complaint: {complaint}",
    )


def few_shot_prefix(examples: Iterable[tuple[str, str]]) -> str:
    """Build the in-context example block from ``(complaint, label)`` pairs."""
    return "".join(
        _SHOT_TEMPLATE.format(complaint=complaint, label=label)
        for complaint, label in examples
    )


def few_shot_classify_prompt(
    complaint: str, examples: Iterable[tuple[str, str]]
) -> str:
    """Few-shot classification prompt.

    Measured worse than zero-shot on this task (0.74 against 0.93). Kept
    because a negative result that is reproducible is worth more than a
    deleted experiment.
    """
    return few_shot_prefix(examples) + classify_prompt(complaint)


def summarize_prompt(complaint: str) -> str:
    """Summarization prompt."""
    return INSTRUCT_TEMPLATE.format(
        system_message=SUMMARIZE_SYSTEM_MESSAGE,
        user_input=f"Customer complaint: {complaint}",
    )
