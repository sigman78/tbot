from __future__ import annotations

from tbot.logic import should_respond


def test_should_respond_respects_frequency() -> None:
    """Test that response frequency is respected in group chats."""
    assert should_respond(
        random_value=0.1, response_frequency=0.2, replied_to_bot=False, is_private_chat=False
    )
    assert not should_respond(
        random_value=0.5, response_frequency=0.2, replied_to_bot=False, is_private_chat=False
    )


def test_should_respond_prioritises_replies() -> None:
    """Test that direct replies to the bot always get responses."""
    assert should_respond(
        random_value=0.9, response_frequency=0.1, replied_to_bot=True, is_private_chat=False
    )


def test_should_respond_always_in_private_chat() -> None:
    """Test that bot always responds in private 1-on-1 chats."""
    # Should respond regardless of random value or frequency
    assert should_respond(
        random_value=0.99, response_frequency=0.01, replied_to_bot=False, is_private_chat=True
    )
    assert should_respond(
        random_value=1.0, response_frequency=0.0, replied_to_bot=False, is_private_chat=True
    )


def test_should_respond_private_chat_overrides_frequency() -> None:
    """Test that private chat setting takes precedence over frequency."""
    # Even with 0% frequency, should respond in private chat
    assert should_respond(
        random_value=0.5, response_frequency=0.0, replied_to_bot=False, is_private_chat=True
    )


def test_should_respond_group_chat_respects_frequency() -> None:
    """Test that group chats still use frequency-based responses."""
    # In group chat (not private), frequency should apply
    assert not should_respond(
        random_value=0.9, response_frequency=0.1, replied_to_bot=False, is_private_chat=False
    )
    assert should_respond(
        random_value=0.05, response_frequency=0.1, replied_to_bot=False, is_private_chat=False
    )

