"""Structural interfaces for the one external dependency this project has.

A :class:`typing.Protocol`, not a base class: nothing inherits from it, and a
class satisfies it by shape. That is what lets the entire pipeline run in tests
against a scripted fake, with no model file, no GPU and no download.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class ChatModel(Protocol):
    """A text-in, text-out instruction model."""

    def complete(self, prompt: str) -> str:
        """Return the model's completion for ``prompt``."""
        ...
