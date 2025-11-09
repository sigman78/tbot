"""Tests for Telegram type abstraction layer."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from tbot.telegram_types import (
    extract_emoji_from_reaction,
    extract_emojis_from_reactions,
)


@pytest.fixture
def mock_emoji_reaction():
    """Create a mock ReactionTypeEmoji."""
    from telegram import ReactionTypeEmoji

    return ReactionTypeEmoji(emoji="👍")


@pytest.fixture
def mock_custom_emoji_reaction():
    """Create a mock ReactionTypeCustomEmoji."""
    # Create a mock since we can't easily instantiate the real class
    mock = MagicMock()
    mock.__class__.__name__ = "ReactionTypeCustomEmoji"
    return mock


def test_extract_emoji_from_standard_reaction(mock_emoji_reaction):
    """Test extracting emoji from a standard emoji reaction."""
    emoji = extract_emoji_from_reaction(mock_emoji_reaction)
    assert emoji == "👍"


def test_extract_emoji_from_custom_reaction_returns_none(mock_custom_emoji_reaction):
    """Test that custom emoji reactions return None."""
    emoji = extract_emoji_from_reaction(mock_custom_emoji_reaction)
    assert emoji is None


def test_extract_emojis_from_empty_list():
    """Test extracting emojis from empty list."""
    emojis = extract_emojis_from_reactions([])
    assert emojis == []


def test_extract_emojis_from_none():
    """Test extracting emojis from None."""
    emojis = extract_emojis_from_reactions(None)
    assert emojis == []


def test_extract_emojis_from_mixed_reactions(
    mock_emoji_reaction, mock_custom_emoji_reaction
):
    """Test extracting emojis from a mix of reaction types."""
    from telegram import ReactionTypeEmoji

    reactions = [
        ReactionTypeEmoji(emoji="👍"),
        mock_custom_emoji_reaction,
        ReactionTypeEmoji(emoji="❤️"),
    ]

    emojis = extract_emojis_from_reactions(reactions)

    # Should only extract standard emojis, not custom ones
    assert len(emojis) == 2
    assert "👍" in emojis
    assert "❤️" in emojis
