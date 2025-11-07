"""Tests for bot decision logic and message handling."""
from __future__ import annotations

from tbot.logic import should_respond


def test_private_chat_always_responds_low_frequency():
    """Test that private chats always respond even with low frequency."""
    # With very low frequency and high random value, should still respond in private chat
    assert should_respond(
        random_value=0.99,
        response_frequency=0.01,
        replied_to_bot=False,
        is_private_chat=True,
    )


def test_private_chat_overrides_zero_frequency():
    """Test that private chat overrides zero frequency setting."""
    assert should_respond(
        random_value=0.5,
        response_frequency=0.0,
        replied_to_bot=False,
        is_private_chat=True,
    )


def test_group_chat_respects_low_frequency():
    """Test that group chats respect low frequency settings."""
    # With low frequency, high random value should not respond
    assert not should_respond(
        random_value=0.9,
        response_frequency=0.1,
        replied_to_bot=False,
        is_private_chat=False,
    )


def test_group_chat_responds_when_random_below_frequency():
    """Test that group chats respond when random value is below frequency."""
    assert should_respond(
        random_value=0.05,
        response_frequency=0.3,
        replied_to_bot=False,
        is_private_chat=False,
    )


def test_reply_to_bot_overrides_frequency_in_group():
    """Test that replying to bot always triggers response in groups."""
    assert should_respond(
        random_value=0.99,
        response_frequency=0.01,
        replied_to_bot=True,
        is_private_chat=False,
    )


def test_reply_to_bot_works_in_private_chat():
    """Test that replying to bot works in private chats too."""
    assert should_respond(
        random_value=0.5,
        response_frequency=0.5,
        replied_to_bot=True,
        is_private_chat=True,
    )


def test_boundary_case_frequency_equals_random():
    """Test boundary case where frequency equals random value."""
    # When random_value <= frequency, should respond
    assert should_respond(
        random_value=0.5,
        response_frequency=0.5,
        replied_to_bot=False,
        is_private_chat=False,
    )


def test_supergroup_treated_as_group():
    """Test that supergroups are treated like regular groups."""
    # Supergroups should not have is_private_chat=True
    assert not should_respond(
        random_value=0.9,
        response_frequency=0.1,
        replied_to_bot=False,
        is_private_chat=False,  # Supergroup is not private
    )
