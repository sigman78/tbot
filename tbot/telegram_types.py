"""Abstraction layer for Telegram-specific types.

This module provides type-safe wrappers and extractors for Telegram types,
preventing Telegram type leakage into business logic.
"""

from __future__ import annotations

import logging
from typing import List

from telegram import ReactionType, ReactionTypeEmoji

logger = logging.getLogger(__name__)


def extract_emoji_from_reaction(reaction: ReactionType) -> str | None:
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
    reactions: List[ReactionType] | None,
) -> List[str]:
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
