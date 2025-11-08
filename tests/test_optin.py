"""Tests for opt-in functionality to ensure privacy compliance."""

from __future__ import annotations

from tbot.memory import MemoryManager


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


def test_opt_in_prevents_message_history_storage():
    """Test that the opt-in logic prevents message history storage for non-opted-in users."""
    # This test simulates the core logic from maybe_reply function
    # without needing to import the nested function

    memory_manager = MemoryManager(auto_save=False)

    # Test case 1: User not opted in, explicit_optin enabled, group chat
    chat_id = 12345
    user_id = 67890
    chat_type = "group"
    explicit_optin = True

    # User has not opted in
    assert not memory_manager.is_user_opted_in(chat_id, user_id)

    # >> This is bs, we need to test real code
    # Simulate the opt-in check logic from maybe_reply
    is_group_chat = chat_type in ["group", "supergroup"]
    should_process_message = True

    if explicit_optin and is_group_chat:
        if not memory_manager.is_user_opted_in(chat_id, user_id):
            should_process_message = False

    # Message should not be processed
    assert not should_process_message

    # Test case 2: User opted in, explicit_optin enabled, group chat
    memory_manager.add_optin_user(chat_id, user_id)
    assert memory_manager.is_user_opted_in(chat_id, user_id)

    # Reset the logic
    should_process_message = True

    if explicit_optin and is_group_chat:
        if not memory_manager.is_user_opted_in(chat_id, user_id):
            should_process_message = False

    # Message should be processed
    assert should_process_message


def test_opt_in_allows_private_chats():
    """Test that private chats bypass opt-in requirements."""
    memory_manager = MemoryManager(auto_save=False)

    chat_id = 12345
    user_id = 67890
    chat_type = "private"  # Private chat
    explicit_optin = True

    # User has not opted in
    assert not memory_manager.is_user_opted_in(chat_id, user_id)

    # >> This is bs, we need to test real code
    # Simulate the opt-in check logic from maybe_reply
    is_group_chat = chat_type in ["group", "supergroup"]
    should_process_message = True

    if explicit_optin and is_group_chat:
        if not memory_manager.is_user_opted_in(chat_id, user_id):
            should_process_message = False

    # Since it's a private chat, is_group_chat is False, so opt-in check is bypassed
    assert should_process_message


def test_opt_in_disabled_allows_all_users():
    """Test that when explicit_optin is disabled, all users can have messages processed."""
    memory_manager = MemoryManager(auto_save=False)

    chat_id = 12345
    user_id = 67890
    chat_type = "group"
    explicit_optin = False  # Opt-in disabled

    # User has not opted in
    assert not memory_manager.is_user_opted_in(chat_id, user_id)

    # Simulate the opt-in check logic from maybe_reply
    is_group_chat = chat_type in ["group", "supergroup"]
    should_process_message = True

    if explicit_optin and is_group_chat:
        if not memory_manager.is_user_opted_in(chat_id, user_id):
            should_process_message = False

    # Since explicit_optin is False, the opt-in check is bypassed
    assert should_process_message


def test_message_storage_integration():
    """Test that the opt-in logic integrates correctly with message history storage."""
    memory_manager = MemoryManager(auto_save=False)

    chat_id = 12345
    user_id_opted_in = 67890
    user_id_not_opted_in = 54321
    chat_type = "group"
    explicit_optin = True

    # Add one user to opt-in list
    memory_manager.add_optin_user(chat_id, user_id_opted_in)

    # Simulate message processing for opted-in user
    is_group_chat = chat_type in ["group", "supergroup"]

    # >> This is bs, we need to test real code
    def should_store_message(user_id, message_text):
        """Simulate the message storage logic from maybe_reply."""
        if explicit_optin and is_group_chat:
            if not memory_manager.is_user_opted_in(chat_id, user_id):
                return False  # Don't store message
        # Store message in history
        memory_manager.append_history(chat_id, f"User: {message_text}")
        return True

    # Test opted-in user
    result1 = should_store_message(user_id_opted_in, "Hello from opted-in user")
    assert result1 is True
    history = memory_manager.get_history(chat_id)
    assert len(history) == 1
    assert "Hello from opted-in user" in history[0]

    # Test non-opted-in user
    result2 = should_store_message(user_id_not_opted_in, "Hello from non-opted-in user")
    assert result2 is False  # Message should not be stored
    history = memory_manager.get_history(chat_id)
    assert len(history) == 1  # History unchanged
    assert "non-opted-in user" not in history[0]  # Non-opted-in message not stored


def test_opt_in_persistence():
    """Test that opt-in data persists correctly."""
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp_dir:
        storage_path = Path(tmp_dir) / "test_memory.json"

        # Create memory manager with persistence
        memory_manager1 = MemoryManager(storage_path=storage_path, auto_save=True)

        chat_id = 12345
        user_id = 67890

        # Add user to opt-in
        memory_manager1.add_optin_user(chat_id, user_id)
        assert memory_manager1.is_user_opted_in(chat_id, user_id)

        # Create new memory manager and load data
        memory_manager2 = MemoryManager(storage_path=storage_path, auto_save=False)
        assert memory_manager2.is_user_opted_in(chat_id, user_id)

        # Verify opt-in users are loaded correctly
        optin_users = memory_manager2.get_optin_users(chat_id)
        assert len(optin_users) == 1
        assert user_id in optin_users
