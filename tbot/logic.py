"""Utility functions for bot decision making."""

from __future__ import annotations


def should_respond(
    *,
    random_value: float,
    response_frequency: float,
    replied_to_bot: bool,
    is_private_chat: bool = False,
    mentioned_bot: bool = False,
) -> bool:
    """Determine whether the bot should reply to a message.

    Args:
        random_value: Random value between 0.0 and 1.0 for probabilistic replies
        response_frequency: Configured frequency (0.0-1.0) for replying to messages
        replied_to_bot: Whether the message is a direct reply to the bot
        is_private_chat: Whether the chat is a private 1-on-1 conversation
        mentioned_bot: Whether the bot was mentioned in the message

    Returns:
        True if the bot should respond, False otherwise
    """
    # Always respond in private chats
    if is_private_chat:
        return True

    # Always respond to direct replies
    if replied_to_bot:
        return True

    # Always respond when mentioned
    if mentioned_bot:
        return True

    # In group chats, use the configured frequency
    frequency = min(max(response_frequency, 0.0), 1.0)
    return random_value <= frequency
