"""Central configuration: environment loading, validation, and paths.

All secrets come from the environment (a local ``.env`` file). Nothing is
hard-coded here. Import ``settings`` from this module
wherever configuration is needed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the project root (one level above this file's ``src`` dir).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# ---- Paths ----------------------------------------------------------------
DATA_PATH = PROJECT_ROOT / "data" / "BonBon FAQ.pdf"
CHROMA_DIR = PROJECT_ROOT / ".chroma"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
COLLECTION_NAME = "bonbon_faq"


class ConfigError(RuntimeError):
    """Raised when a required environment variable is missing."""


def _require(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ConfigError(
            f"Required environment variable '{name}' is not set. "
            f"Copy .env.example to .env and fill in your Azure OpenAI values."
        )
    return value


@dataclass(frozen=True)
class Settings:
    """Resolved configuration pulled from the environment."""

    azure_api_key: str
    azure_endpoint: str
    azure_chat_deployment: str
    azure_api_version: str
    azure_embedding_deployment: str

    langsmith_api_key: str | None = None
    langsmith_project: str = "finding-assistant-assignment"

    # Retrieval / chunking defaults (overridable by callers).
    chunk_size: int = 1000
    chunk_overlap: int = 150
    retrieval_k: int = 4

    @property
    def langsmith_enabled(self) -> bool:
        return bool(self.langsmith_api_key)


def load_settings() -> Settings:
    """Build a validated :class:`Settings` from the current environment.

    Raises :class:`ConfigError` with a legible message if a mandatory Azure
    variable is absent, so misconfiguration fails fast rather than crashing
    deep inside a client call.
    """

    settings = Settings(
        azure_api_key=_require("AZURE_OPENAI_API_KEY"),
        azure_endpoint=_require("AZURE_OPENAI_ENDPOINT"),
        azure_chat_deployment=_require("AZURE_OPENAI_DEPLOYMENT_NAME"),
        azure_api_version=_require("AZURE_OPENAI_API_VERSION"),
        azure_embedding_deployment=_require("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
        langsmith_api_key=os.getenv("LANGSMITH_API_KEY") or None,
        langsmith_project=os.getenv("LANGSMITH_PROJECT")
        or "finding-assistant-assignment",
    )

    # Enable LangSmith tracing automatically when a key is present.
    if settings.langsmith_enabled:
        os.environ.setdefault("LANGSMITH_TRACING", "true")
        os.environ.setdefault("LANGSMITH_PROJECT", settings.langsmith_project)

    return settings
