"""Tests for bot utility functions."""
from __future__ import annotations

from tbot.bot import _truncate_text, MAX_MESSAGE_LENGTH


def test_truncate_text_short_message() -> None:
    """Test that short messages are not truncated."""
    text = "This is a short message"
    result = _truncate_text(text)
    assert result == text
    assert len(result) <= MAX_MESSAGE_LENGTH


def test_truncate_text_exact_limit() -> None:
    """Test message exactly at the limit."""
    text = "x" * MAX_MESSAGE_LENGTH
    result = _truncate_text(text)
    assert result == text
    assert len(result) == MAX_MESSAGE_LENGTH


def test_truncate_text_long_message() -> None:
    """Test that long messages are truncated."""
    text = "x" * (MAX_MESSAGE_LENGTH + 1000)
    result = _truncate_text(text)

    # Should be truncated
    assert len(result) < len(text)
    assert len(result) <= MAX_MESSAGE_LENGTH

    # Should have truncation indicator
    assert "[Message truncated]" in result


def test_truncate_text_very_long_message() -> None:
    """Test truncation with very long message."""
    text = "Lorem ipsum dolor sit amet. " * 1000  # Much longer than limit
    result = _truncate_text(text)

    assert len(result) <= MAX_MESSAGE_LENGTH
    assert result.endswith("... [Message truncated]")


def test_truncate_text_custom_max_length() -> None:
    """Test truncation with custom max length."""
    text = "x" * 200
    result = _truncate_text(text, max_length=100)

    assert len(result) <= 100
    assert "[Message truncated]" in result


def test_truncate_text_preserves_beginning() -> None:
    """Test that truncation preserves the beginning of the message."""
    text = "IMPORTANT: " + ("x" * 10000)
    result = _truncate_text(text)

    # Should start with the important part
    assert result.startswith("IMPORTANT:")
    assert len(result) <= MAX_MESSAGE_LENGTH


def test_truncate_text_empty_string() -> None:
    """Test truncation with empty string."""
    result = _truncate_text("")
    assert result == ""


def test_truncate_text_multiline() -> None:
    """Test truncation with multiline text."""
    lines = ["Line " + str(i) for i in range(1000)]
    text = "\n".join(lines)

    result = _truncate_text(text)

    assert len(result) <= MAX_MESSAGE_LENGTH
    assert result.startswith("Line 0")
    assert "[Message truncated]" in result
