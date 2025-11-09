"""Tests for conversation context builder."""
from __future__ import annotations

from datetime import datetime

from tbot.config import BotConfig
from tbot.context import ConversationContextBuilder
from tbot.memory import MemoryEntry


def test_format_user_message() -> None:
    """Test formatting user messages for history storage."""
    result = ConversationContextBuilder.format_user_message("John", "Hello there!")
    assert result == "John: Hello there!"


def test_format_bot_message() -> None:
    """Test formatting bot messages for history storage."""
    result = ConversationContextBuilder.format_bot_message("Hi!")
    assert result == "Bot: Hi!"


def test_strip_bot_prefix() -> None:
    """Test stripping 'Bot: ' prefix from messages."""
    assert ConversationContextBuilder.strip_bot_prefix("Bot: Hello") == "Hello"
    assert ConversationContextBuilder.strip_bot_prefix("Hello") == "Hello"
    assert ConversationContextBuilder.strip_bot_prefix("Bot:No space") == "Bot:No space"


def test_private_chat_strips_user_prefix() -> None:
    """Test that user names are stripped for private chats."""
    builder = ConversationContextBuilder(is_group_chat=False)

    history = [
        "User: Hello!",
        "Bot: Hi there!",
        "User: How are you?",
    ]

    config = BotConfig()
    memories: list[MemoryEntry] = []

    messages = builder.build_messages(
        config=config,
        history=history,
        memories=memories,
        current_message="What's up?",
    )

    # System prompt (1) + history (3) + current message (1) = 5
    # Note: memory message is only added when there are memories
    assert len(messages) == 5

    # Check user messages have prefix stripped (indices: 0=system, 1-4=history+current)
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == "Hello!"

    # Check bot messages have prefix stripped
    assert messages[2]["role"] == "assistant"
    # TODO: Too strict, look up later
    assert messages[2]["content"] == "Hi there!"  # pyright: ignore[reportTypedDictNotRequiredAccess]

    # Check second user message
    assert messages[3]["role"] == "user"
    assert messages[3]["content"] == "How are you?"


def test_group_chat_preserves_user_names() -> None:
    """Test that user names are preserved for group chats.

    This is critical for the LLM to understand who said what in a group conversation.
    """
    builder = ConversationContextBuilder(is_group_chat=True)

    history = [
        "anton: Hey how are you?",
        "Bot: Im good, you?",
        "eugene: Hi guys, whats up!",
        "Bot: Good good!",
    ]

    config = BotConfig()
    memories: list[MemoryEntry] = []

    messages = builder.build_messages(
        config=config,
        history=history,
        memories=memories,
        current_message="Great to see you all!",
    )

    # System prompt (1) + history (4) + current message (1) = 6
    # Note: memory message is only added when there are memories
    assert len(messages) == 6

    # Check first user message preserves name
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == "anton: Hey how are you?"

    # Check bot message has prefix stripped
    assert messages[2]["role"] == "assistant"
    assert messages[2]["content"] == "Im good, you?"  # pyright: ignore[reportTypedDictNotRequiredAccess]

    # Check second user message preserves name
    assert messages[3]["role"] == "user"
    assert messages[3]["content"] == "eugene: Hi guys, whats up!"

    # Check second bot message
    assert messages[4]["role"] == "assistant"
    assert messages[4]["content"] == "Good good!"  # pyright: ignore[reportTypedDictNotRequiredAccess]


def test_context_includes_system_prompt_and_persona() -> None:
    """Test that system prompt and persona are included in context."""
    builder = ConversationContextBuilder(is_group_chat=False)

    config = BotConfig(
        persona="A friendly assistant",
        system_prompt="You are helpful and kind.",
    )
    memories: list[MemoryEntry] = []

    messages = builder.build_messages(
        config=config,
        history=[],
        memories=memories,
        current_message="Hi!",
    )

    # System prompt (1) + current message (1) = 2
    assert len(messages) == 2

    # Check system prompt includes persona
    assert messages[0]["role"] == "system"
    assert "You are helpful and kind." in messages[0]["content"]
    assert "A friendly assistant" in messages[0]["content"]


def test_context_includes_memories() -> None:
    """Test that memories are included in context."""
    builder = ConversationContextBuilder(is_group_chat=False)

    config = BotConfig()
    memories = [
        MemoryEntry(chat_id=123, text="User likes pizza", created_at=datetime.utcnow()),
        MemoryEntry(chat_id=123, text="User is a developer", created_at=datetime.utcnow()),
    ]

    messages = builder.build_messages(
        config=config,
        history=[],
        memories=memories,
        current_message="Hi!",
    )

    # System prompt (1) + memories (1) + current message (1) = 3
    assert len(messages) == 3

    # Check memories are included
    assert messages[1]["role"] == "system"
    assert "User likes pizza" in messages[1]["content"]
    assert "User is a developer" in messages[1]["content"]


def test_context_with_no_memories() -> None:
    """Test that context works correctly with no memories."""
    builder = ConversationContextBuilder(is_group_chat=False)

    config = BotConfig()
    memories: list[MemoryEntry] = []

    messages = builder.build_messages(
        config=config,
        history=["User: Hello"],
        memories=memories,
        current_message="Hi!",
    )

    # System prompt (1) + history (1) + current message (1) = 3
    assert len(messages) == 3

    # No memory message should be added when memories list is empty
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert messages[2]["role"] == "user"


def test_current_message_is_last() -> None:
    """Test that current message is always added last."""
    builder = ConversationContextBuilder(is_group_chat=False)

    config = BotConfig()

    messages = builder.build_messages(
        config=config,
        history=["User: Previous message"],
        memories=[],
        current_message="Current message",
    )

    # Last message should be the current message
    assert messages[-1]["role"] == "user"
    assert messages[-1]["content"] == "Current message"


def test_group_chat_sample_conversation() -> None:
    """Test the exact sample conversation from the requirements.

    Sample conversation:
    anton: Hey how are you?
    bot: Im good, you?
    eugene: Hi guys, whats up!
    bot: Good good!
    """
    builder = ConversationContextBuilder(is_group_chat=True)

    history = [
        "anton: Hey how are you?",
        "Bot: Im good, you?",
        "eugene: Hi guys, whats up!",
        "Bot: Good good!",
    ]

    config = BotConfig()

    messages = builder.build_messages(
        config=config,
        history=history,
        memories=[],
        current_message="Let's grab lunch!",
    )

    # Verify the conversation is properly formatted for the LLM
    # Find the user messages in the output
    user_messages = [m for m in messages if m["role"] == "user"]

    # Should have 3 user messages: 2 from history + 1 current
    assert len(user_messages) == 3

    # First user message should preserve "anton:" prefix
    assert user_messages[0]["content"] == "anton: Hey how are you?"

    # Second user message should preserve "eugene:" prefix
    assert user_messages[1]["content"] == "eugene: Hi guys, whats up!"

    # Current message (without prefix)
    assert user_messages[2]["content"] == "Let's grab lunch!"

    # Verify bot messages have prefix stripped
    bot_messages = [m for m in messages if m["role"] == "assistant"]
    assert len(bot_messages) == 2
    assert bot_messages[0]["content"] == "Im good, you?"
    assert bot_messages[1]["content"] == "Good good!"
