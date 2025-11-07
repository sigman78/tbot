from __future__ import annotations

from pathlib import Path

from tbot.config import BotConfig, ConfigManager


def test_config_manager_roundtrip(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    manager = ConfigManager(path=config_path)
    manager.set_field("persona", "A curious cat.")
    manager.set_field("response_frequency", 0.9)

    loaded = ConfigManager(path=config_path).config
    assert loaded.persona == "A curious cat."
    assert loaded.response_frequency == 0.9


def test_bot_config_validation() -> None:
    config = BotConfig(persona="  Explorer  ")
    assert config.persona == "Explorer"
    assert config.max_context_messages == 12


def test_model_name_auto_fix() -> None:
    """Test that incorrect openrouter/ prefix is automatically removed."""
    config = BotConfig(llm_model="openrouter/openai/gpt-4o-mini")
    assert config.llm_model == "openai/gpt-4o-mini"


def test_model_name_without_prefix_unchanged() -> None:
    """Test that correct model names are not modified."""
    config = BotConfig(llm_model="openai/gpt-4o-mini")
    assert config.llm_model == "openai/gpt-4o-mini"


def test_model_name_other_providers() -> None:
    """Test that other provider names work correctly."""
    config = BotConfig(llm_model="anthropic/claude-3-sonnet")
    assert config.llm_model == "anthropic/claude-3-sonnet"


def test_summarization_config_defaults() -> None:
    """Test default values for summarization settings."""
    config = BotConfig()
    assert config.auto_summarize_enabled is True
    assert config.summarize_threshold == 18
    assert config.summarize_batch_size == 10


def test_summarization_config_custom_values() -> None:
    """Test custom summarization settings."""
    config = BotConfig(
        auto_summarize_enabled=False,
        summarize_threshold=25,
        summarize_batch_size=15,
    )
    assert config.auto_summarize_enabled is False
    assert config.summarize_threshold == 25
    assert config.summarize_batch_size == 15


def test_summarization_threshold_validation() -> None:
    """Test that summarization threshold is validated."""
    # Too low
    try:
        BotConfig(summarize_threshold=5)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "summarize_threshold" in str(e)

    # Too high
    try:
        BotConfig(summarize_threshold=150)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "summarize_threshold" in str(e)


def test_summarization_batch_size_validation() -> None:
    """Test that batch size is validated."""
    # Too low
    try:
        BotConfig(summarize_batch_size=2)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "summarize_batch_size" in str(e)

    # Too high
    try:
        BotConfig(summarize_batch_size=100)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "summarize_batch_size" in str(e)


def test_config_serialization_with_summarization() -> None:
    """Test that summarization config is saved and loaded."""
    config = BotConfig(
        auto_summarize_enabled=False,
        summarize_threshold=20,
        summarize_batch_size=12,
    )
    dumped = config.model_dump()

    assert dumped["auto_summarize_enabled"] is False
    assert dumped["summarize_threshold"] == 20
    assert dumped["summarize_batch_size"] == 12

