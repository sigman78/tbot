"""Abstraction layer for Telegram-specific types.

This module provides type-safe wrappers and extractors for Telegram types,
preventing Telegram type leakage into business logic.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from telegram import ReactionType

logger = logging.getLogger(__name__)


def extract_emoji_from_reaction(reaction: "ReactionType") -> str | None:
    """Extract emoji string from a Telegram reaction in a type-safe way.

    Handles all Telegram reaction types:
    - ReactionTypeEmoji: Standard emoji reactions
    - ReactionTypeCustomEmoji: Custom emoji reactions (returns None)
    - ReactionTypePaid: Paid reactions (returns None)
    - Any future types: Logs warning and returns None

    Args:
        reaction: A Telegram ReactionType object

    Returns:
        Emoji string if it's a standard emoji reaction, None otherwise
    """
    # Import here to avoid circular dependencies and keep imports clean
    from telegram import ReactionTypeEmoji

    # Type-safe extraction using pattern matching
    match reaction:
        case ReactionTypeEmoji(emoji=emoji):
            return emoji
        case _:
            # Log when we encounter non-emoji reactions (useful for debugging)
            reaction_type = type(reaction).__name__
            logger.debug(f"Ignoring non-emoji reaction type: {reaction_type}")
            return None


def extract_emojis_from_reactions(
    reactions: list["ReactionType"] | None,
) -> list[str]:
    """Extract all emoji strings from a list of Telegram reactions.

    This is a convenience function that filters out non-emoji reactions
    and returns only the emoji strings.

    Args:
        reactions: List of Telegram ReactionType objects, or None

    Returns:
        List of emoji strings (may be empty)
    """
    if not reactions:
        return []

    emojis = []
    for reaction in reactions:
        emoji = extract_emoji_from_reaction(reaction)
        if emoji is not None:
            emojis.append(emoji)

    return emojis


def has_positive_reaction(
    reactions: list["ReactionType"] | None,
    positive_emojis: set[str],
) -> bool:
    """Check if any reaction in the list is a positive emoji reaction.

    Args:
        reactions: List of Telegram ReactionType objects, or None
        positive_emojis: Set of emoji strings considered positive

    Returns:
        True if any reaction is a positive emoji, False otherwise
    """
    if not reactions:
        return False

    from telegram import ReactionTypeEmoji

    for reaction in reactions:
        match reaction:
            case ReactionTypeEmoji(emoji=emoji) if emoji in positive_emojis:
                return True

    return False
