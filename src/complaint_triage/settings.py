"""Typed, validated, environment-driven configuration."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from complaint_triage.exceptions import ConfigurationError


class Settings(BaseSettings):
    """Runtime configuration, loaded from the environment or a ``.env`` file."""

    model_config = SettingsConfigDict(
        env_prefix="COMPLAINT_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    model_path: Path = Path("./models/model.gguf")
    hf_token: SecretStr | None = None

    # Temperature 0: a routing decision that varies between runs of the same
    # input is not a routing decision.
    max_tokens: int = Field(default=1200, ge=1, le=8192)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    top_k: int = Field(default=50, ge=1)
    repeat_penalty: float = Field(default=1.2, ge=0.0)
    context_length: int = Field(default=4096, ge=512)

    seed: int = 42
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    def require_model_path(self) -> Path:
        """Return the model path, or fail with an actionable message."""
        if not self.model_path.exists():
            message = (
                f"Model not found at {self.model_path}. Set COMPLAINT_MODEL_PATH "
                "in .env to a local GGUF instruction model."
            )
            raise ConfigurationError(message)
        return self.model_path


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings instance."""
    return Settings()
