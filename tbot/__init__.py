"""LLM-driven Telegram persona bot package."""

from __future__ import annotations

from typing import Any

from .config import BotConfig, ConfigManager
from .llm_client import LLMClient
from .memory import MemoryManager

__all__ = [
    "BotConfig",
    "ConfigManager",
    "MemoryManager",
    "LLMClient",
]


def __getattr__(name: str) -> Any:
    if name == "create_application":  # pragma: no cover - import side effect
        from .bot import create_application as _create_application

        return _create_application
    raise AttributeError(name)
