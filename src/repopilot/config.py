"""Configuration helpers for RepoPilot.

The app works without paid services:
- `offline` provider returns a grounded extractive response from retrieved chunks.
- `gemini` uses Google's Gemini API when GEMINI_API_KEY is available.
- `ollama` uses a local Ollama server.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional in production
    load_dotenv = None


if load_dotenv:
    load_dotenv()


@dataclass
class Settings:
    llm_provider: str = os.getenv("LLM_PROVIDER", "offline").lower()
    gemini_api_key: Optional[str] = os.getenv("GEMINI_API_KEY") or None
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.1")
    github_token: Optional[str] = os.getenv("GITHUB_TOKEN") or None
    chroma_dir: Path = Path(os.getenv("CHROMA_DIR", "data/chroma"))


def get_settings(**overrides) -> Settings:
    """Return settings with optional runtime overrides from the Streamlit sidebar."""
    base = Settings()
    for key, value in overrides.items():
        if value not in (None, "") and hasattr(base, key):
            setattr(base, key, value)
    base.llm_provider = (base.llm_provider or "offline").lower()
    base.chroma_dir = Path(base.chroma_dir)
    return base
