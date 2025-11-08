"""Conversation context builder for LLM API calls.

This module centralizes the logic for composing conversation history
into the format expected by the LLM API.
"""

from __future__ import annotations

from typing import List

from openai.types.chat import ChatCompletionMessageParam

from .config import BotConfig
from .memory import MemoryEntry


class ConversationContextBuilder:
    """Builds conversation context for LLM API calls.

    Handles proper formatting of messages for both private and group chats,
    ensuring user names are preserved where needed.
    """

    def __init__(self, *, is_group_chat: bool = False):
        """Initialize the context builder.

        Args:
            is_group_chat: Whether this is a group chat (affects user name handling)
        """
        self.is_group_chat = is_group_chat

    def build_messages(
        self,
        *,
        config: BotConfig,
        history: List[str],
        memories: List[MemoryEntry],
        current_message: str,
    ) -> List[ChatCompletionMessageParam]:
        """Build the complete message list for LLM API.

        Args:
            config: Bot configuration containing persona and prompts
            history: Recent conversation history (with user name prefixes)
            memories: Stored memories for context
            current_message: The current user message (without prefix)

        Returns:
            List of messages formatted for OpenAI-compatible API
        """
        messages: List[ChatCompletionMessageParam] = []

        # Add system prompt with persona
        system_content = f"{config.system_prompt}\nPersona: {config.persona}"
        messages.append({"role": "system", "content": system_content})

        # Add memories if available
        if memories:
            memory_lines = [f"- {entry.text}" for entry in memories]
            memory_blob = "\n".join(memory_lines)
            messages.append({
                "role": "system",
                "content": f"Relevant persona memories (optional):\n{memory_blob}",
            })

        # Add conversation history
        for item in history:
            messages.append(self._parse_history_item(item))

        # Add current message
        messages.append({"role": "user", "content": current_message})

        return messages

    def _parse_history_item(self, item: str) -> ChatCompletionMessageParam:
        """Parse a history item into an API message.

        History items are stored in format:
        - "Bot: <message>" for bot responses
        - "<user_name>: <message>" for user messages

        For group chats, we preserve user names in the message content.
        For private chats, we strip the "User: " prefix.

        Args:
            item: History item string

        Returns:
            Formatted message for API
        """
        if item.startswith("Bot: "):
            # Bot message - always strip the prefix
            content = item[5:]  # Remove "Bot: "
            return {"role": "assistant", "content": content}

        # User message
        if ": " in item:
            parts = item.split(": ", 1)
            if len(parts) == 2:
                user_name, message = parts

                # For group chats, preserve user names in content
                # For private chats, strip the "User: " prefix
                if self.is_group_chat:
                    # Keep the full format for group chats
                    content = item
                else:
                    # Strip prefix for private chats (assume single user)
                    content = message

                return {"role": "user", "content": content}

        # Fallback: return as-is if format is unexpected
        return {"role": "user", "content": item}

    @staticmethod
    def format_user_message(user_name: str, message: str) -> str:
        """Format a user message for storage in history.

        Args:
            user_name: Name/handle of the user
            message: The message text

        Returns:
            Formatted message for history storage
        """
        return f"{user_name}: {message}"

    @staticmethod
    def format_bot_message(message: str) -> str:
        """Format a bot message for storage in history.

        Args:
            message: The bot's response text

        Returns:
            Formatted message for history storage
        """
        return f"Bot: {message}"

    @staticmethod
    def strip_bot_prefix(message: str) -> str:
        """Strip 'Bot: ' prefix from a message if present.

        Args:
            message: Message that may have bot prefix

        Returns:
            Message without bot prefix
        """
        if message.startswith("Bot: "):
            return message[5:]
        return message
