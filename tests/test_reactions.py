"""Tests for emoji reaction functionality."""
from __future__ import annotations

import asyncio
import pytest
from unittest.mock import MagicMock

from tbot.llm_client import LLMClient, COMMON_REACTIONS


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client for testing."""
    client = MagicMock()
    return client


@pytest.fixture
def llm_client(mock_openai_client):
    """Create LLMClient with mocked OpenAI client."""
    return LLMClient(client=mock_openai_client)


def test_suggest_reaction_returns_emoji(llm_client, mock_openai_client):
    """Test that suggest_reaction returns an emoji."""
    # Mock the API response
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "👍"
    mock_openai_client.chat.completions.create.return_value = mock_response

    reaction = asyncio.run(llm_client.suggest_reaction(
        message="Great job!",
        persona="A friendly assistant",
        model="openai/gpt-4o-mini",
    ))

    assert reaction == "👍"
    mock_openai_client.chat.completions.create.assert_called_once()


def test_suggest_reaction_returns_none_when_not_needed(llm_client, mock_openai_client):
    """Test that suggest_reaction returns None when LLM says NONE."""
    # Mock the API response
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "NONE"
    mock_openai_client.chat.completions.create.return_value = mock_response

    reaction = asyncio.run(llm_client.suggest_reaction(
        message="Hello",
        persona="A friendly assistant",
        model="openai/gpt-4o-mini",
    ))

    assert reaction is None


def test_suggest_reaction_handles_empty_message(llm_client):
    """Test that suggest_reaction handles empty messages."""
    reaction = asyncio.run(llm_client.suggest_reaction(
        message="",
        persona="A friendly assistant",
        model="openai/gpt-4o-mini",
    ))

    assert reaction is None


def test_suggest_reaction_handles_api_error(llm_client, mock_openai_client):
    """Test that suggest_reaction handles API errors gracefully."""
    from openai import OpenAIError

    # Mock an API error
    mock_openai_client.chat.completions.create.side_effect = OpenAIError("API error")

    reaction = asyncio.run(llm_client.suggest_reaction(
        message="Test message",
        persona="A friendly assistant",
        model="openai/gpt-4o-mini",
    ))

    # Should return None on error instead of raising
    assert reaction is None


def test_suggest_reaction_handles_none_content(llm_client, mock_openai_client):
    """Test that suggest_reaction handles None content from API."""
    # Mock the API response with None content
    mock_response = MagicMock()
    mock_response.choices[0].message.content = None
    mock_openai_client.chat.completions.create.return_value = mock_response

    reaction = asyncio.run(llm_client.suggest_reaction(
        message="Test message",
        persona="A friendly assistant",
        model="openai/gpt-4o-mini",
    ))

    assert reaction is None


def test_suggest_reaction_uses_correct_parameters(llm_client, mock_openai_client):
    """Test that suggest_reaction uses correct LLM parameters."""
    # Mock the API response
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "👍"
    mock_openai_client.chat.completions.create.return_value = mock_response

    asyncio.run(llm_client.suggest_reaction(
        message="Great!",
        persona="A cheerful bot",
        model="test-model",
    ))

    # Verify the call was made with correct parameters
    call_args = mock_openai_client.chat.completions.create.call_args
    assert call_args.kwargs["model"] == "test-model"
    assert call_args.kwargs["temperature"] == 0.5
    assert call_args.kwargs["max_tokens"] == 10


def test_suggest_reaction_includes_persona_in_prompt(llm_client, mock_openai_client):
    """Test that suggest_reaction includes persona in the system prompt."""
    # Mock the API response
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "❤️"
    mock_openai_client.chat.completions.create.return_value = mock_response

    asyncio.run(llm_client.suggest_reaction(
        message="I love this!",
        persona="A passionate developer",
        model="test-model",
    ))

    # Verify the persona is in the system prompt
    call_args = mock_openai_client.chat.completions.create.call_args
    messages = call_args.kwargs["messages"]
    system_message = messages[0]["content"]
    assert "A passionate developer" in system_message


def test_common_reactions_list():
    """Test that COMMON_REACTIONS contains valid emojis."""
    assert len(COMMON_REACTIONS) > 0
    assert "👍" in COMMON_REACTIONS
    assert "❤️" in COMMON_REACTIONS
    assert "🔥" in COMMON_REACTIONS
    # All should be strings
    assert all(isinstance(r, str) for r in COMMON_REACTIONS)


def test_suggest_reaction_with_whitespace_response(llm_client, mock_openai_client):
    """Test that suggest_reaction handles whitespace in response."""
    # Mock the API response with leading/trailing whitespace
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "  👍  "
    mock_openai_client.chat.completions.create.return_value = mock_response

    reaction = asyncio.run(llm_client.suggest_reaction(
        message="Good work",
        persona="A friendly assistant",
        model="openai/gpt-4o-mini",
    ))

    # Should strip whitespace
    assert reaction == "👍"


def test_suggest_reaction_case_insensitive_none(llm_client, mock_openai_client):
    """Test that suggest_reaction handles 'none' in any case."""
    test_cases = ["NONE", "none", "None", "nOnE"]

    for none_value in test_cases:
        # Mock the API response
        mock_response = MagicMock()
        mock_response.choices[0].message.content = none_value
        mock_openai_client.chat.completions.create.return_value = mock_response

        reaction = asyncio.run(llm_client.suggest_reaction(
            message="Test",
            persona="A friendly assistant",
            model="openai/gpt-4o-mini",
        ))

        assert reaction is None, f"Failed for case: {none_value}"
