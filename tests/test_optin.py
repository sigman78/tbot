"""Tests for opt-in functionality to ensure privacy compliance."""

from __future__ import annotations

from pathlib import Path

from tbot.memory import MemoryManager

from .utils import next_timestamp_generator


def test_memory_manager_optin_functions():
    """Test the MemoryManager opt-in functionality directly."""
    memory_manager = MemoryManager(auto_save=False)

    chat_id = 12345
    user_id_1 = 111
    user_id_2 = 222

    # Initially, no users should be opted in
    assert not memory_manager.is_user_opted_in(chat_id, user_id_1)
    assert not memory_manager.is_user_opted_in(chat_id, user_id_2)
    assert len(memory_manager.get_optin_users(chat_id)) == 0

    # Add first user to opt-in
    memory_manager.add_optin_user(chat_id, user_id_1)
    assert memory_manager.is_user_opted_in(chat_id, user_id_1)
    assert not memory_manager.is_user_opted_in(chat_id, user_id_2)
    assert len(memory_manager.get_optin_users(chat_id)) == 1
    assert user_id_1 in memory_manager.get_optin_users(chat_id)

    # Add second user to opt-in
    memory_manager.add_optin_user(chat_id, user_id_2)
    assert memory_manager.is_user_opted_in(chat_id, user_id_1)
    assert memory_manager.is_user_opted_in(chat_id, user_id_2)
    assert len(memory_manager.get_optin_users(chat_id)) == 2
    assert user_id_1 in memory_manager.get_optin_users(chat_id)
    assert user_id_2 in memory_manager.get_optin_users(chat_id)

    # Test different chat - should be separate
    other_chat_id = 99999
    assert not memory_manager.is_user_opted_in(other_chat_id, user_id_1)
    assert not memory_manager.is_user_opted_in(other_chat_id, user_id_2)
    assert len(memory_manager.get_optin_users(other_chat_id)) == 0

    # Add user to different chat
    memory_manager.add_optin_user(other_chat_id, user_id_1)
    assert memory_manager.is_user_opted_in(other_chat_id, user_id_1)
    assert not memory_manager.is_user_opted_in(other_chat_id, user_id_2)
    assert len(memory_manager.get_optin_users(other_chat_id)) == 1

    # Original chat should be unchanged
    assert memory_manager.is_user_opted_in(chat_id, user_id_1)
    assert memory_manager.is_user_opted_in(chat_id, user_id_2)
    assert len(memory_manager.get_optin_users(chat_id)) == 2


def test_optin_message_id_tracking():
    """Test that opt-in message IDs are properly tracked."""
    memory_manager = MemoryManager(auto_save=False)

    chat_id = 12345
    message_id = 98765

    # Initially, no message ID should be stored
    assert memory_manager.get_optin_message_id(chat_id) is None

    # Set message ID
    memory_manager.set_optin_message_id(chat_id, message_id)
    assert memory_manager.get_optin_message_id(chat_id) == message_id

    # Test different chat - should be separate
    other_chat_id = 99999
    assert memory_manager.get_optin_message_id(other_chat_id) is None

    # Clear opt-in data
    memory_manager.clear_optin_data(chat_id)
    assert memory_manager.get_optin_message_id(chat_id) is None
    assert len(memory_manager.get_optin_users(chat_id)) == 0


def test_privacy_enforcement_rejects_non_opted_in_messages():
    """Test that MemoryManager rejects messages from non-opted-in users when explicit_optin_mode is enabled."""
    # Create manager with explicit opt-in mode enabled
    memory_manager = MemoryManager(auto_save=False, explicit_optin_mode=True)

    chat_id = 12345
    user_id_opted_in = 111
    user_id_not_opted_in = 222

    # Add only one user to opt-in list
    memory_manager.add_optin_user(chat_id, user_id_opted_in)

    # Try to store message from opted-in user in group chat
    result1 = memory_manager.append_history(
        chat_id,
        "User: Hello from opted-in user",
        user_id=user_id_opted_in,
        is_group_chat=True,
    )
    assert result1 is True

    # Try to store message from non-opted-in user in group chat
    result2 = memory_manager.append_history(
        chat_id,
        "User: Hello from non-opted-in user",
        user_id=user_id_not_opted_in,
        is_group_chat=True,
    )
    assert result2 is False  # Should be rejected

    # Verify only opted-in user's message was stored
    history = memory_manager.get_history(chat_id)
    assert len(history) == 1
    assert "opted-in user" in history[0]
    assert "non-opted-in user" not in history[0]


def test_privacy_enforcement_allows_bot_messages():
    """Test that bot messages (user_id=None) are always stored regardless of opt-in."""
    memory_manager = MemoryManager(auto_save=False, explicit_optin_mode=True)

    chat_id = 12345

    # No users opted in
    assert len(memory_manager.get_optin_users(chat_id)) == 0

    # Bot message should be stored even with no opted-in users
    result = memory_manager.append_history(
        chat_id,
        "Bot: Hello!",
        user_id=None,  # Bot message
        is_group_chat=True,
    )
    assert result is True

    history = memory_manager.get_history(chat_id)
    assert len(history) == 1
    assert "Bot: Hello!" in history[0]


def test_privacy_enforcement_private_chats_bypass_optin():
    """Test that private chats bypass opt-in enforcement."""
    memory_manager = MemoryManager(auto_save=False, explicit_optin_mode=True)

    chat_id = 12345
    user_id = 111

    # User not opted in
    assert not memory_manager.is_user_opted_in(chat_id, user_id)

    # But message should still be stored in private chat
    result = memory_manager.append_history(
        chat_id,
        "User: Hello in private chat",
        user_id=user_id,
        is_group_chat=False,  # Private chat
    )
    assert result is True

    history = memory_manager.get_history(chat_id)
    assert len(history) == 1


def test_privacy_enforcement_disabled_allows_all():
    """Test that when explicit_optin_mode is disabled, all messages are stored."""
    memory_manager = MemoryManager(auto_save=False, explicit_optin_mode=False)

    chat_id = 12345
    user_id = 111

    # User not opted in
    assert not memory_manager.is_user_opted_in(chat_id, user_id)

    # But message should still be stored when opt-in mode is disabled
    result = memory_manager.append_history(
        chat_id,
        "User: Hello",
        user_id=user_id,
        is_group_chat=True,
    )
    assert result is True

    history = memory_manager.get_history(chat_id)
    assert len(history) == 1


def test_privacy_enforcement_filters_history_retrieval():
    """Test that get_history filters out non-opted-in users' messages."""
    memory_manager = MemoryManager(auto_save=False, explicit_optin_mode=True)

    chat_id = 12345
    user1 = 111
    user2 = 222

    # Both users opt in initially
    memory_manager.add_optin_user(chat_id, user1)
    memory_manager.add_optin_user(chat_id, user2)

    # Store messages from both users
    memory_manager.append_history(
        chat_id, "User1: Hello", user_id=user1, is_group_chat=True
    )
    memory_manager.append_history(
        chat_id, "User2: Hi", user_id=user2, is_group_chat=True
    )
    memory_manager.append_history(
        chat_id, "Bot: Hey!", user_id=None, is_group_chat=True
    )

    # All 3 messages visible
    history = memory_manager.get_history(chat_id, is_group_chat=True)
    assert len(history) == 3

    # Now remove user2 from opt-in
    memory_manager.remove_user_from_optin(chat_id, user2)

    # get_history should now filter out user2's messages
    history = memory_manager.get_history(chat_id, is_group_chat=True)
    assert len(history) == 2
    assert any("User1" in msg for msg in history)
    assert any("Bot" in msg for msg in history)
    assert not any("User2" in msg for msg in history)


def test_privacy_enforcement_user_summary_rejection():
    """Test that user summaries are rejected for non-opted-in users."""
    memory_manager = MemoryManager(auto_save=False, explicit_optin_mode=True)

    chat_id = 12345
    user_id_opted_in = 111
    user_id_not_opted_in = 222
    next_ts = next_timestamp_generator()

    # Add only one user to opt-in
    memory_manager.add_optin_user(chat_id, user_id_opted_in)

    # Try to store summary for opted-in user
    result1 = memory_manager.add_user_summary(
        chat_id,
        user_id=user_id_opted_in,
        username="Alice",
        summary="Alice discussed Python",
        last_active=next_ts(),
        is_group_chat=True,
    )
    assert result1 is True

    # Try to store summary for non-opted-in user
    result2 = memory_manager.add_user_summary(
        chat_id,
        user_id=user_id_not_opted_in,
        username="Bob",
        summary="Bob asked questions",
        last_active=next_ts(),
        is_group_chat=True,
    )
    assert result2 is False  # Should be rejected

    # Verify only opted-in user's summary was stored
    summaries = memory_manager.get_user_summaries(chat_id)
    assert len(summaries) == 1
    assert summaries[0].username == "Alice"


def test_remove_user_purges_all_data():
    """Test that remove_user_from_optin purges all user data."""
    memory_manager = MemoryManager(auto_save=False, explicit_optin_mode=True)

    chat_id = 12345
    user_id = 111
    next_ts = next_timestamp_generator()

    # User opts in
    memory_manager.add_optin_user(chat_id, user_id)

    # Store messages and summary
    memory_manager.append_history(
        chat_id, "User: Message 1", user_id=user_id, is_group_chat=True
    )
    memory_manager.append_history(
        chat_id, "Bot: Reply 1", user_id=None, is_group_chat=True
    )
    memory_manager.append_history(
        chat_id, "User: Message 2", user_id=user_id, is_group_chat=True
    )
    memory_manager.add_user_summary(
        chat_id,
        user_id=user_id,
        username="Alice",
        summary="Alice discussed topics",
        last_active=next_ts(),
        is_group_chat=True,
    )

    # Verify data exists
    history = memory_manager.get_history(chat_id, is_group_chat=True)
    assert len(history) == 3
    summaries = memory_manager.get_user_summaries(chat_id)
    assert len(summaries) == 1

    # Remove user from opt-in
    memory_manager.remove_user_from_optin(chat_id, user_id)

    # Verify user is no longer opted in
    assert not memory_manager.is_user_opted_in(chat_id, user_id)

    # Verify user's messages were purged (only bot message remains)
    history = memory_manager.get_history(chat_id, is_group_chat=True)
    assert len(history) == 1
    assert "Bot:" in history[0]

    # Verify user's summary was purged
    summaries = memory_manager.get_user_summaries(chat_id)
    assert len(summaries) == 0


def test_privacy_enforcement_persistence(tmp_path: Path):
    """Test that privacy-enforced data persists correctly."""
    storage_path = tmp_path / "test_privacy.json"

    # Create manager with opt-in mode
    manager1 = MemoryManager(
        storage_path=storage_path,
        auto_save=True,
        explicit_optin_mode=True,
    )

    chat_id = 12345
    user_id = 111

    # User opts in and stores message
    manager1.add_optin_user(chat_id, user_id)
    manager1.append_history(chat_id, "User: Hello", user_id=user_id, is_group_chat=True)

    # Load in new manager with opt-in mode
    manager2 = MemoryManager(
        storage_path=storage_path,
        auto_save=False,
        explicit_optin_mode=True,
    )

    # Verify opt-in status persisted
    assert manager2.is_user_opted_in(chat_id, user_id)

    # Verify history persisted with user_id metadata
    history = manager2.get_history(chat_id, is_group_chat=True)
    assert len(history) == 1
    assert "Hello" in history[0]
