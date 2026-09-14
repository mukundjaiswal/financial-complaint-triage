"""Shared fixtures. Every one is offline — no weights, no GPU, no download."""

from __future__ import annotations

import pytest

from complaint_triage.eval_metrics import register_task_metrics
from complaint_triage.llm.fake import ScriptedChatModel


@pytest.fixture(autouse=True)
def _metrics_registered():
    """Register this task's metrics before each test.

    The registry is shared with sibling projects, so tests register explicitly
    rather than relying on an import having done it.
    """
    register_task_metrics(replace=True)


@pytest.fixture
def model() -> ScriptedChatModel:
    """A model that always answers with a well-formed label."""
    return ScriptedChatModel(["Category: credit_card"])
