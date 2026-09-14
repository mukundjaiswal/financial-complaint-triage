"""Deterministic model for tests.

Exists so the full pipeline runs in CI with no weights, no GPU and no download.
A test that needs a 4GB model file is a test that stops being run.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable


class ScriptedChatModel:
    """Returns pre-set replies in order, then repeats the last one."""

    def __init__(
        self,
        replies: Iterable[str] | None = None,
        *,
        responder: Callable[[str], str] | None = None,
    ) -> None:
        """Configure a fixed reply sequence or a prompt-driven responder."""
        self._replies = list(replies or [])
        self._responder = responder
        self._index = 0
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> str:
        """Record ``prompt`` and return the next scripted reply."""
        self.prompts.append(prompt)
        if self._responder is not None:
            return self._responder(prompt)
        if not self._replies:
            return ""
        reply = self._replies[min(self._index, len(self._replies) - 1)]
        self._index += 1
        return reply

    @property
    def call_count(self) -> int:
        """Number of completions requested so far."""
        return len(self.prompts)
