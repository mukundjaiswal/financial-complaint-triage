"""The two tasks, each returning an observation the harness can score.

Every task returns both the **raw** generation and the **parsed** result. That
is deliberate and it is the core claim of this project: the report can then
show what the same model scores with and without the normalization layer, in
the same run, rather than asking anyone to take 0.667 against 0.933 on trust.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from complaint_triage import prompts
from complaint_triage.interfaces import ChatModel
from complaint_triage.postprocess import extract_category, normalize


@dataclass(frozen=True, slots=True)
class Classifier:
    """Routes a complaint to a product category."""

    model: ChatModel
    examples: Sequence[tuple[str, str]] = ()

    def predict(self, complaint: str) -> dict[str, Any]:
        """Classify ``complaint`` and report raw and parsed output."""
        prompt = (
            prompts.few_shot_classify_prompt(complaint, self.examples)
            if self.examples
            else prompts.classify_prompt(complaint)
        )
        raw = self.model.complete(prompt)
        return {
            "raw": raw,
            "raw_normalized": normalize(raw),
            "prediction": extract_category(raw),
            "shots": len(self.examples),
        }


@dataclass(frozen=True, slots=True)
class Summarizer:
    """Summarizes a complaint."""

    model: ChatModel

    def predict(self, complaint: str) -> dict[str, Any]:
        """Summarize ``complaint``."""
        summary = self.model.complete(prompts.summarize_prompt(complaint)).strip()
        return {
            "summary": summary,
            "summary_words": len(summary.split()),
            "source_words": len(complaint.split()),
        }
