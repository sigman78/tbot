"""OpenAI-compatible client for generating replies via OpenRouter."""

from __future__ import annotations

import asyncio
import datetime
import json
import logging
from pathlib import Path
from typing import Iterable, List, Optional

from openai import OpenAI, OpenAIError
from openai.types.chat import ChatCompletionMessageParam

from .config import BotConfig
from .const import TG_REACTIONS as COMMON_REACTIONS
from .context import ConversationContextBuilder
from .memory import MemoryEntry, UserSummary

logger = logging.getLogger(__name__)

# Constants for LLM parameters
DEFAULT_TEMPERATURE = 0.8
DEFAULT_MAX_TOKENS = 512
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"

# Request logging for debug purposes (temporary)
ENABLE_REQUEST_LOGGING = True
REQUEST_LOG_FILE = Path("llm_requests.log")


class LLMClient:
    """Wrapper around the OpenAI client to talk to OpenRouter."""

    @classmethod
    def fromParams(
        cls, api_key: Optional[str] = None, base_url: Optional[str] = DEFAULT_BASE_URL
    ) -> "LLMClient":
        """Create an LLM client from parameters.

        Args:
            api_key: OpenRouter-compatible API key
            base_url: Base URL for the API (defaults to OpenRouter)

        Returns:
            An initialized LLM client.
        """
        return LLMClient(client=OpenAI(api_key=api_key, base_url=base_url))

    def __init__(self, client: OpenAI) -> None:
        """Initialize the LLM client.
        Args:
            client: OpenAI client instance
        """
        self._client = client

    def _log_request(self, endpoint: str, request_data: dict) -> None:
        """Log raw LLM request for debug purposes (temporary solution).

        Args:
            endpoint: API endpoint being called
            request_data: Request payload to log
        """
        if not ENABLE_REQUEST_LOGGING:
            return

        try:
            log_entry = {
                "timestamp": datetime.datetime.now().isoformat(),
                "endpoint": endpoint,
                "request": request_data,
            }

            with REQUEST_LOG_FILE.open("a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

        except Exception as e:
            # Never let logging errors break the main functionality
            logger.warning(f"Failed to log request: {e}")

    async def generate_reply(
        self,
        config: BotConfig,
        history: Iterable[str],
        memories: Iterable[MemoryEntry],
        user_message: str,
        is_group_chat: bool = False,
        user_summaries: Iterable[UserSummary] | None = None,
    ) -> str:
        """Generate a reply using the configured LLM.

        Args:
            config: Bot configuration containing model and prompts
            history: Recent conversation history
            memories: Stored memories for the persona
            user_message: The user's current message
            is_group_chat: Whether this is a group chat (affects user name handling)
            user_summaries: Per-user conversation summaries (optional)

        Returns:
            Generated reply text

        Raises:
            OpenAIError: If the API call fails
            ValueError: If the response is invalid or empty
        """
        if not user_message or not user_message.strip():
            raise ValueError("user_message cannot be empty")

        # Use the centralized context builder
        context_builder = ConversationContextBuilder(is_group_chat=is_group_chat)
        messages = context_builder.build_messages(
            config=config,
            history=list(history),
            memories=list(memories),
            current_message=user_message,
            user_summaries=list(user_summaries) if user_summaries else None,
        )

        logger.debug(
            f"Generating reply with model {config.llm_model}, "
            f"{len(messages)} messages in context"
        )

        # Log the request for debug purposes
        request_data = {
            "model": config.llm_model,
            "messages": messages,
            "temperature": DEFAULT_TEMPERATURE,
            "max_tokens": DEFAULT_MAX_TOKENS,
        }
        self._log_request("chat.completions.create", request_data)

        def _call() -> str:
            try:
                response = self._client.chat.completions.create(
                    model=config.llm_model,
                    messages=messages,
                    temperature=DEFAULT_TEMPERATURE,
                    max_tokens=DEFAULT_MAX_TOKENS,
                )
                content = response.choices[0].message.content
                if content is None:
                    raise ValueError("LLM returned empty response")
                return content.strip()
            except OpenAIError as e:
                error_msg = str(e)
                logger.error(
                    f"API error while generating reply with model '{config.llm_model}': {error_msg}"
                )
                # Provide helpful error messages for common issues
                if "Bad request" in error_msg or "400" in error_msg:
                    logger.error(
                        f"Bad request error. Check that model name '{config.llm_model}' is valid. "
                        "For OpenRouter, use format 'provider/model' (e.g., 'openai/gpt-4o-mini')"
                    )
                raise
            except (IndexError, AttributeError) as e:
                logger.error(f"Invalid response structure from LLM: {e}")
                raise ValueError("Invalid response from LLM") from e

        try:
            reply = await asyncio.to_thread(_call)
            logger.debug(f"Successfully generated reply of length {len(reply)}")
            return reply
        except Exception as e:
            logger.error(f"Failed to generate reply: {e}")
            raise

    async def generate_summary(
        self,
        messages: List[str],
        persona: str,
        model: str,
    ) -> str:
        """Generate a concise summary of conversation history.

        Args:
            messages: List of chat messages to summarize
            persona: The bot's persona for context
            model: LLM model to use for summarization

        Returns:
            Concise summary of the conversation

        Raises:
            OpenAIError: If the API call fails
            ValueError: If the response is invalid or empty
        """
        if not messages:
            raise ValueError("Cannot summarize empty message list")

        # Create a specialized prompt for summarization
        messages_text = "\n".join(messages)
        system_prompt = (
            "You are a helpful assistant that creates concise summaries of chat conversations. "
            f"The bot's persona is: {persona}. "
            "Summarize the key points, topics discussed, and important information from the conversation. "
            "Focus on facts, decisions, and context that would be useful to remember in future conversations. "
            "Keep the summary concise (2-4 sentences)."
        )

        summary_messages: List[ChatCompletionMessageParam] = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"Please summarize the following conversation:\n\n{messages_text}",
            },
        ]

        logger.debug(f"Generating summary for {len(messages)} messages")

        # Log the request for debug purposes
        request_data = {
            "model": model,
            "messages": summary_messages,
            "temperature": 0.3,
            "max_tokens": 256,
        }
        self._log_request("chat.completions.create", request_data)

        def _call() -> str:
            try:
                response = self._client.chat.completions.create(
                    model=model,
                    messages=summary_messages,
                    temperature=0.3,  # Lower temperature for more focused summaries
                    max_tokens=256,  # Shorter response for summaries
                )
                content = response.choices[0].message.content
                if content is None:
                    raise ValueError("LLM returned empty summary")
                return content.strip()
            except OpenAIError as e:
                error_msg = str(e)
                logger.error(
                    f"API error while generating summary with model '{model}': {error_msg}"
                )
                raise
            except (IndexError, AttributeError) as e:
                logger.error(f"Invalid response structure from LLM: {e}")
                raise ValueError("Invalid response from LLM") from e

        try:
            summary = await asyncio.to_thread(_call)
            logger.info(f"Successfully generated summary of length {len(summary)}")
            return summary
        except Exception as e:
            logger.error(f"Failed to generate summary: {e}")
            raise

    async def generate_user_summaries(
        self,
        messages: List[str],
        persona: str,
        model: str,
    ) -> dict[str, str]:
        """Generate per-user conversation summaries.

        Args:
            messages: List of chat messages to summarize (in format "Username: message")
            persona: The bot's persona for context
            model: LLM model to use for summarization

        Returns:
            Dictionary mapping username to their conversation summary

        Raises:
            OpenAIError: If the API call fails
            ValueError: If the response is invalid or empty
        """
        if not messages:
            raise ValueError("Cannot summarize empty message list")

        # Group messages by user
        user_messages: dict[str, List[str]] = {}
        for msg in messages:
            # Skip bot messages
            if msg.startswith("Bot: "):
                continue

            # Parse user messages in format "Username: message"
            if ": " in msg:
                parts = msg.split(": ", 1)
                if len(parts) == 2:
                    username, message = parts
                    if username not in user_messages:
                        user_messages[username] = []
                    user_messages[username].append(message)

        if not user_messages:
            logger.warning("No user messages found to summarize")
            return {}

        # Create a specialized prompt for per-user summarization
        messages_text = "\n".join(messages)
        user_list = ", ".join(user_messages.keys())

        system_prompt = (
            "You are a helpful assistant that creates concise per-user conversation summaries. "
            f"The bot's persona is: {persona}. "
            "For each user mentioned in the conversation, create a brief summary of their key points and contributions. "
            "Format your response as one line per user in the format: 'Username: summary text' "
            "Focus on facts, topics discussed, and context that would be useful to remember. "
            "Keep each user's summary concise (1-2 sentences)."
        )

        summary_messages: List[ChatCompletionMessageParam] = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"Please create per-user summaries for the following conversation. Users: {user_list}\n\nConversation:\n{messages_text}",
            },
        ]

        logger.debug(f"Generating per-user summaries for {len(user_messages)} users")

        # Log the request for debug purposes
        request_data = {
            "model": model,
            "messages": summary_messages,
            "temperature": 0.3,
            "max_tokens": 512,
        }
        self._log_request("chat.completions.create", request_data)

        def _call() -> dict[str, str]:
            try:
                response = self._client.chat.completions.create(
                    model=model,
                    messages=summary_messages,
                    temperature=0.3,  # Lower temperature for more focused summaries
                    max_tokens=512,  # Enough for multiple user summaries
                )
                content = response.choices[0].message.content
                if content is None:
                    raise ValueError("LLM returned empty summary")

                # Parse the response into username: summary pairs
                result = {}
                for line in content.strip().split("\n"):
                    line = line.strip()
                    if not line:
                        continue
                    if ": " in line:
                        parts = line.split(": ", 1)
                        if len(parts) == 2:
                            username, summary = parts
                            result[username.strip()] = summary.strip()

                return result
            except OpenAIError as e:
                error_msg = str(e)
                logger.error(
                    f"API error while generating user summaries with model '{model}': {error_msg}"
                )
                raise
            except (IndexError, AttributeError) as e:
                logger.error(f"Invalid response structure from LLM: {e}")
                raise ValueError("Invalid response from LLM") from e

        try:
            summaries = await asyncio.to_thread(_call)
            logger.info(f"Successfully generated summaries for {len(summaries)} users")
            return summaries
        except Exception as e:
            logger.error(f"Failed to generate user summaries: {e}")
            raise

    async def suggest_reaction(
        self,
        message: str,
        persona: str,
        model: str,
    ) -> str | None:
        """Suggest an appropriate emoji reaction for a message.

        Args:
            message: The user's message to react to
            persona: The bot's persona for context
            model: LLM model to use

        Returns:
            An emoji reaction string, or None if no reaction is needed

        Raises:
            OpenAIError: If the API call fails
        """
        if not message or not message.strip():
            return None

        # Create a specialized prompt for reaction selection
        reactions_list = ",".join(COMMON_REACTIONS[:20])  # Use top 20 most common
        system_prompt = (
            f"You are a helpful assistant that suggests emoji reactions. "
            f"The bot's persona is: {persona}. "
            f"Based on the message, suggest ONE emoji reaction that would be appropriate, "
            f"or respond with 'NONE' if no reaction is needed. "
            f"Available reactions: {reactions_list}. "
            f"Respond with ONLY the emoji or 'NONE', nothing else."
        )

        reaction_messages: List[ChatCompletionMessageParam] = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"Message: {message}\n\nSuggest reaction:",
            },
        ]

        logger.debug(f"Requesting reaction suggestion for message: {message[:50]}...")

        # Log the request for debug purposes
        request_data = {
            "model": model,
            "messages": reaction_messages,
            "temperature": 0.5,
            "max_tokens": 10,
        }
        self._log_request("chat.completions.create", request_data)

        def _call() -> str | None:
            try:
                response = self._client.chat.completions.create(
                    model=model,
                    messages=reaction_messages,
                    temperature=0.5,  # Lower temperature for more consistent reactions
                    max_tokens=10,  # Very short response
                )
                content = response.choices[0].message.content
                if content is None:
                    return None
                content = content.strip()

                # Check if LLM suggested no reaction
                if content.upper() == "NONE" or not content:
                    return None

                # Return the suggested emoji
                return content
            except OpenAIError as e:
                logger.error(f"API error while suggesting reaction: {e}")
                return None  # Fail gracefully for reactions
            except Exception as e:
                logger.error(f"Error suggesting reaction: {e}")
                return None

        try:
            reaction = await asyncio.to_thread(_call)
            if reaction:
                logger.debug(f"Suggested reaction: {reaction}")
            return reaction
        except Exception as e:
            logger.error(f"Failed to suggest reaction: {e}")
            return None  # Fail gracefully
