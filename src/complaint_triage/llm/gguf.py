"""Local GGUF model adapter.

A quantized open-weight model running on CPU. The point of the choice is cost:
the task is high-volume routing, and a hosted frontier model priced per token
is the wrong tool for a decision this narrow.

The runtime is imported inside the constructor so that importing this module
costs nothing and needs no model file.
"""

from __future__ import annotations

from typing import Any

from complaint_triage.exceptions import ModelError
from complaint_triage.settings import Settings, get_settings


class GgufChatModel:
    """Adapter around a local llama.cpp GGUF model."""

    def __init__(self, settings: Settings | None = None) -> None:
        """Load the model, failing early if the weights are missing."""
        from llama_cpp import Llama

        self._settings = settings or get_settings()
        self._llm: Any = Llama(
            model_path=str(self._settings.require_model_path()),
            n_ctx=self._settings.context_length,
            seed=self._settings.seed,
            verbose=False,
        )

    def complete(self, prompt: str) -> str:
        """Return the model's completion for ``prompt``."""
        settings = self._settings
        try:
            response = self._llm(
                prompt=prompt,
                max_tokens=settings.max_tokens,
                temperature=settings.temperature,
                top_p=settings.top_p,
                top_k=settings.top_k,
                repeat_penalty=settings.repeat_penalty,
                stop=["</s>"],
                echo=False,
            )
        except Exception as exc:
            message = "Local model failed to generate"
            raise ModelError(message) from exc

        choices = response.get("choices") or []
        if not choices:
            message = "Local model returned no choices"
            raise ModelError(message)
        return str(choices[0]["text"]).strip()
