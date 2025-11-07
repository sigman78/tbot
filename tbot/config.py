"""Configuration helpers for the Telegram persona bot."""
from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path
from typing import Any, Dict


def _clamp(value: float, *, minimum: float, maximum: float, field_name: str) -> float:
    if not minimum <= value <= maximum:
        raise ValueError(
            f"{field_name} must be between {minimum} and {maximum}, received {value}."
        )
    return value


def _ensure_int_in_range(value: int, *, minimum: int, maximum: int, field_name: str) -> int:
    if not minimum <= value <= maximum:
        raise ValueError(
            f"{field_name} must be between {minimum} and {maximum}, received {value}."
        )
    return value


@dataclass(slots=True)
class BotConfig:
    """Runtime configuration for the bot with lightweight validation."""

    response_frequency: float = 0.4
    persona: str = "An affable research assistant who enjoys casual conversation."
    system_prompt: str = (
        "You are role-playing as the configured persona. Maintain continuity, "
        "be helpful, and keep a light conversational tone while referencing "
        "stored memories when relevant."
    )
    llm_model: str = "openai/gpt-4o-mini"
    max_context_messages: int = 12

    # Auto-summarization settings
    auto_summarize_enabled: bool = True
    summarize_threshold: int = 18
    summarize_batch_size: int = 10

    # Reaction settings
    reactions_enabled: bool = True
    reaction_frequency: float = 0.3

    def __post_init__(self) -> None:
        self.persona = self.persona.strip()
        self.system_prompt = self.system_prompt.strip()
        self.response_frequency = _clamp(
            float(self.response_frequency),
            minimum=0.0,
            maximum=1.0,
            field_name="response_frequency",
        )
        self.max_context_messages = _ensure_int_in_range(
            int(self.max_context_messages),
            minimum=4,
            maximum=50,
            field_name="max_context_messages",
        )
        self.summarize_threshold = _ensure_int_in_range(
            int(self.summarize_threshold),
            minimum=10,
            maximum=100,
            field_name="summarize_threshold",
        )
        self.summarize_batch_size = _ensure_int_in_range(
            int(self.summarize_batch_size),
            minimum=5,
            maximum=50,
            field_name="summarize_batch_size",
        )
        self.reaction_frequency = _clamp(
            float(self.reaction_frequency),
            minimum=0.0,
            maximum=1.0,
            field_name="reaction_frequency",
        )
        if not self.llm_model:
            raise ValueError("llm_model must be a non-empty string.")

        # Auto-fix common model name mistakes
        if self.llm_model.startswith("openrouter/"):
            # Remove the openrouter/ prefix - OpenRouter API expects just provider/model
            self.llm_model = self.llm_model.replace("openrouter/", "", 1)

    def model_dump(self) -> Dict[str, Any]:
        return {
            "response_frequency": self.response_frequency,
            "persona": self.persona,
            "system_prompt": self.system_prompt,
            "llm_model": self.llm_model,
            "max_context_messages": self.max_context_messages,
            "auto_summarize_enabled": self.auto_summarize_enabled,
            "summarize_threshold": self.summarize_threshold,
            "summarize_batch_size": self.summarize_batch_size,
            "reactions_enabled": self.reactions_enabled,
            "reaction_frequency": self.reaction_frequency,
        }

    def model_dump_json(self, indent: int | None = None) -> str:
        return json.dumps(self.model_dump(), ensure_ascii=False, indent=indent)

    def model_copy(self, *, update: Dict[str, Any] | None = None, validate: bool = True) -> "BotConfig":
        data = self.model_dump()
        if update:
            data.update(update)
        new_config = replace(self, **data)
        if validate:
            # ``replace`` already triggered ``__post_init__`` validation.
            pass
        return new_config

    @classmethod
    def model_validate_json(cls, json_data: str) -> "BotConfig":
        try:
            data = json.loads(json_data)
        except json.JSONDecodeError as exc:  # pragma: no cover - defensive
            raise ValueError("Configuration file is not valid JSON") from exc
        if not isinstance(data, dict):  # pragma: no cover - defensive
            raise ValueError("Configuration file must contain a JSON object")
        return cls(**data)


class ConfigManager:
    """Handle loading and storing configuration on disk."""

    def __init__(self, path: str | Path | None = None) -> None:
        self._path = Path(path or Path.home() / ".tbot-config.json")
        self._config = BotConfig()
        if self._path.exists():
            self.load()

    @property
    def config(self) -> BotConfig:
        return self._config

    def load(self) -> BotConfig:
        try:
            data = self._path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return self._config
        self._config = BotConfig.model_validate_json(data)
        return self._config

    def save(self) -> None:
        self._path.write_text(self._config.model_dump_json(indent=2), encoding="utf-8")

    def update(self, **kwargs: Any) -> BotConfig:
        updated = self._config.model_copy(update=kwargs, validate=True)
        self._config = updated
        self.save()
        return self._config

    def set_field(self, field: str, value: Any) -> BotConfig:
        if not hasattr(self._config, field):
            raise KeyError(f"Unknown configuration field: {field}")
        return self.update(**{field: value})

