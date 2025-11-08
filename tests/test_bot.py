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


def test_opt_in_prevents_message_processing():
    """Test that opt-in requirement prevents message processing for non-opted-in users."""
    # This test verifies the core logic that prevents message storage/processing
    # when explicit_optin is enabled and user hasn't opted in

    from tbot.memory import MemoryManager

    memory_manager = MemoryManager(auto_save=False)

    # Simulate the opt-in check logic from the bot
    chat_id = 12345
    user_id = 67890
    chat_type = "group"
    explicit_optin_enabled = True

    # User has not opted in
    assert not memory_manager.is_user_opted_in(chat_id, user_id)

    # Simulate the opt-in check from maybe_reply function
    is_group_chat = chat_type in ["group", "supergroup"]
    should_process = True

    if explicit_optin_enabled and is_group_chat:
        if not memory_manager.is_user_opted_in(chat_id, user_id):
            should_process = False

    # Message should not be processed due to opt-in requirement
    assert not should_process

    # Now test with opted-in user
    memory_manager.add_optin_user(chat_id, user_id)
    assert memory_manager.is_user_opted_in(chat_id, user_id)

    # Reset the logic check
    should_process = True
    if explicit_optin_enabled and is_group_chat:
        if not memory_manager.is_user_opted_in(chat_id, user_id):
            should_process = False

    # Message should be processed since user is opted in
    assert should_process
