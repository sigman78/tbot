"""Simple in-memory store for persona memories and chat history."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


@dataclass
class MemoryEntry:
    chat_id: int
    text: str
    created_at: datetime

    def to_dict(self) -> dict:
        """Serialize to dictionary for JSON storage."""
        return {
            "chat_id": self.chat_id,
            "text": self.text,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MemoryEntry":
        """Deserialize from dictionary."""
        return cls(
            chat_id=data["chat_id"],
            text=data["text"],
            created_at=datetime.fromisoformat(data["created_at"]),
        )


@dataclass
class UserSummary:
    """Per-user conversation summary."""

    username: str
    summary: str
    last_active: datetime

    def to_dict(self) -> dict:
        """Serialize to dictionary for JSON storage."""
        return {
            "username": self.username,
            "summary": self.summary,
            "last_active": self.last_active.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "UserSummary":
        """Deserialize from dictionary."""
        return cls(
            username=data["username"],
            summary=data["summary"],
            last_active=datetime.fromisoformat(data["last_active"]),
        )


@dataclass
class ChatStatistics:
    """Runtime statistics for a chat."""

    replies: int = 0
    reactions: int = 0
    llm_calls: int = 0
    tokens_sent: int = 0
    tokens_received: int = 0

    def to_dict(self) -> dict:
        """Serialize to dictionary for JSON storage."""
        return {
            "replies": self.replies,
            "reactions": self.reactions,
            "llm_calls": self.llm_calls,
            "tokens_sent": self.tokens_sent,
            "tokens_received": self.tokens_received,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ChatStatistics":
        """Deserialize from dictionary."""
        return cls(
            replies=data.get("replies", 0),
            reactions=data.get("reactions", 0),
            llm_calls=data.get("llm_calls", 0),
            tokens_sent=data.get("tokens_sent", 0),
            tokens_received=data.get("tokens_received", 0),
        )


class MemoryManager:
    """Store persona memories and recent chat messages per chat."""

    def __init__(
        self,
        history_size: int = 20,
        storage_path: Path | str | None = None,
        auto_save: bool = True,
        max_summarized_users: int = 10,
    ) -> None:
        """Initialize the memory manager.

        Args:
            history_size: Maximum number of messages to keep in history
            storage_path: Path to persistence file (default: ~/.tbot-data.json)
            auto_save: Whether to auto-save after changes
            max_summarized_users: Maximum number of users to keep summaries for
        """
        self._memories: Dict[int, List[MemoryEntry]] = {}
        self._history: Dict[int, List[str]] = {}
        self._history_size = history_size
        self._summarization_count: Dict[int, int] = {}
        self._user_summaries: Dict[
            int, Dict[str, UserSummary]
        ] = {}  # chat_id -> {username -> UserSummary}
        self._statistics: Dict[int, ChatStatistics] = {}  # chat_id -> ChatStatistics
        self._optin_lists: Dict[int, set[int]] = {}  # chat_id -> set of user_ids
        self._optin_message_ids: Dict[
            int, int
        ] = {}  # chat_id -> message_id of opt-in request
        self._max_summarized_users = max_summarized_users
        self._storage_path = Path(storage_path or Path.home() / ".tbot-data.json")
        self._auto_save = auto_save
        self._dirty = False  # Track if data has changed since last save

        # Load existing data if available
        if self._storage_path.exists():
            self.load()

    def add_memory(self, chat_id: int, text: str) -> MemoryEntry:
        entry = MemoryEntry(
            chat_id=chat_id, text=text.strip(), created_at=datetime.utcnow()
        )
        self._memories.setdefault(chat_id, []).append(entry)
        self._mark_dirty()
        return entry

    def get_memories(self, chat_id: int) -> List[MemoryEntry]:
        return list(self._memories.get(chat_id, []))

    def clear_memories(self, chat_id: int) -> None:
        self._memories.pop(chat_id, None)
        self._mark_dirty()

    def append_history(self, chat_id: int, message: str) -> None:
        history = self._history.setdefault(chat_id, [])
        history.append(message)
        if len(history) > self._history_size:
            del history[: len(history) - self._history_size]
        self._mark_dirty()

    def get_history(self, chat_id: int, limit: int | None = None) -> List[str]:
        history = self._history.get(chat_id, [])
        if limit is None:
            return list(history)
        return history[-limit:]

    def should_summarize(self, chat_id: int, threshold: int) -> bool:
        """Check if chat history has reached the summarization threshold.

        Args:
            chat_id: The chat to check
            threshold: Number of messages that triggers summarization

        Returns:
            True if summarization should be triggered
        """
        history = self._history.get(chat_id, [])
        return len(history) >= threshold

    def get_messages_for_summary(
        self, chat_id: int, batch_size: int
    ) -> Tuple[List[str], int]:
        """Get the oldest messages for summarization.

        Args:
            chat_id: The chat to get messages from
            batch_size: Number of messages to summarize

        Returns:
            Tuple of (messages to summarize, total history size)
        """
        history = self._history.get(chat_id, [])
        if not history:
            return [], 0

        # Get the oldest batch_size messages
        messages_to_summarize = history[:batch_size]
        return messages_to_summarize, len(history)

    def clear_summarized_messages(self, chat_id: int, count: int) -> None:
        """Remove the oldest messages from history after they've been summarized.

        Args:
            chat_id: The chat to clear messages from
            count: Number of messages to remove from the beginning
        """
        history = self._history.get(chat_id, [])
        if not history:
            return

        # Remove the oldest 'count' messages
        del history[:count]
        logger.info(f"Cleared {count} summarized messages from chat {chat_id}")

        # Track summarization
        self._summarization_count[chat_id] = (
            self._summarization_count.get(chat_id, 0) + 1
        )
        self._mark_dirty()

    def get_summarization_count(self, chat_id: int) -> int:
        """Get the number of times history has been summarized for a chat.

        Args:
            chat_id: The chat to check

        Returns:
            Number of summarizations performed
        """
        return self._summarization_count.get(chat_id, 0)

    def get_history_size(self, chat_id: int) -> int:
        """Get the current size of history for a chat.

        Args:
            chat_id: The chat to check

        Returns:
            Number of messages in history
        """
        return len(self._history.get(chat_id, []))

    def add_user_summary(
        self,
        chat_id: int,
        username: str,
        summary: str,
        last_active: datetime,
    ) -> None:
        """Add or update a user's conversation summary.

        Args:
            chat_id: The chat this summary belongs to
            username: The user's name
            summary: The summary text
            last_active: Timestamp for when the user was last active.
                        If None, uses current UTC time.
        """
        if chat_id not in self._user_summaries:
            self._user_summaries[chat_id] = {}

        self._user_summaries[chat_id][username] = UserSummary(
            username=username,
            summary=summary,
            last_active=last_active,
        )

        # Enforce max user limit
        self._cleanup_old_user_summaries(chat_id)
        self._mark_dirty()

    def get_user_summaries(self, chat_id: int) -> List[UserSummary]:
        """Get all user summaries for a chat, ordered by last active.

        Args:
            chat_id: The chat to get summaries for

        Returns:
            List of user summaries, most recently active first
        """
        summaries = self._user_summaries.get(chat_id, {})
        return sorted(
            summaries.values(),
            key=lambda s: s.last_active,
            reverse=True,
        )

    def clear_user_summaries(self, chat_id: int) -> None:
        """Clear all user summaries for a chat.

        Args:
            chat_id: The chat to clear summaries for
        """
        self._user_summaries.pop(chat_id, None)
        self._mark_dirty()

    def _cleanup_old_user_summaries(self, chat_id: int) -> None:
        """Remove oldest user summaries if limit exceeded.

        Args:
            chat_id: The chat to clean up
        """
        if chat_id not in self._user_summaries:
            return

        summaries = self._user_summaries[chat_id]
        if len(summaries) <= self._max_summarized_users:
            return

        # Sort by last_active, keep only the most recent N users
        sorted_summaries = sorted(
            summaries.items(),
            key=lambda item: item[1].last_active,
            reverse=True,
        )

        # Keep only the top N users
        self._user_summaries[chat_id] = dict(
            sorted_summaries[: self._max_summarized_users]
        )

        logger.info(
            f"Cleaned up user summaries for chat {chat_id}, "
            f"kept {self._max_summarized_users} most recent users"
        )

    def increment_reply_count(self, chat_id: int) -> None:
        """Increment the reply count for a chat.

        Args:
            chat_id: The chat to increment
        """
        stats = self._statistics.setdefault(chat_id, ChatStatistics())
        stats.replies += 1
        self._mark_dirty()

    def increment_reaction_count(self, chat_id: int) -> None:
        """Increment the reaction count for a chat.

        Args:
            chat_id: The chat to increment
        """
        stats = self._statistics.setdefault(chat_id, ChatStatistics())
        stats.reactions += 1
        self._mark_dirty()

    def increment_llm_call_count(
        self, chat_id: int, tokens_sent: int = 0, tokens_received: int = 0
    ) -> None:
        """Increment the LLM call count and token counts for a chat.

        Args:
            chat_id: The chat to increment
            tokens_sent: Number of tokens sent in the request
            tokens_received: Number of tokens received in the response
        """
        stats = self._statistics.setdefault(chat_id, ChatStatistics())
        stats.llm_calls += 1
        stats.tokens_sent += tokens_sent
        stats.tokens_received += tokens_received
        self._mark_dirty()

    def get_statistics(self, chat_id: int) -> ChatStatistics:
        """Get statistics for a chat.

        Args:
            chat_id: The chat to get statistics for

        Returns:
            Chat statistics object
        """
        return self._statistics.get(chat_id, ChatStatistics())

    def clear_statistics(self, chat_id: int) -> None:
        """Clear all statistics for a chat.

        Args:
            chat_id: The chat to clear statistics for
        """
        self._statistics.pop(chat_id, None)
        self._mark_dirty()

    def add_optin_user(self, chat_id: int, user_id: int) -> None:
        """Add a user to the opt-in list for a chat.

        Args:
            chat_id: The chat to add the user to
            user_id: The user ID to add
        """
        if chat_id not in self._optin_lists:
            self._optin_lists[chat_id] = set()
        self._optin_lists[chat_id].add(user_id)
        self._mark_dirty()
        logger.info(f"User {user_id} opted in to chat {chat_id}")

    def is_user_opted_in(self, chat_id: int, user_id: int) -> bool:
        """Check if a user has opted in to a chat.

        Args:
            chat_id: The chat to check
            user_id: The user ID to check

        Returns:
            True if the user is opted in
        """
        return user_id in self._optin_lists.get(chat_id, set())

    def get_optin_users(self, chat_id: int) -> set[int]:
        """Get all users who have opted in to a chat.

        Args:
            chat_id: The chat to get users for

        Returns:
            Set of user IDs
        """
        return self._optin_lists.get(chat_id, set()).copy()

    def set_optin_message_id(self, chat_id: int, message_id: int) -> None:
        """Store the message ID of the opt-in request.

        Args:
            chat_id: The chat the message belongs to
            message_id: The message ID to store
        """
        self._optin_message_ids[chat_id] = message_id
        self._mark_dirty()
        logger.info(f"Set opt-in message ID {message_id} for chat {chat_id}")

    def get_optin_message_id(self, chat_id: int) -> int | None:
        """Get the message ID of the opt-in request.

        Args:
            chat_id: The chat to get the message ID for

        Returns:
            The message ID, or None if not set
        """
        return self._optin_message_ids.get(chat_id)

    def clear_optin_data(self, chat_id: int) -> None:
        """Clear all opt-in data for a chat.

        Args:
            chat_id: The chat to clear data for
        """
        self._optin_lists.pop(chat_id, None)
        self._optin_message_ids.pop(chat_id, None)
        self._mark_dirty()

    def _mark_dirty(self) -> None:
        """Mark data as changed and trigger auto-save if enabled."""
        self._dirty = True
        if self._auto_save:
            self.save()

    def save(self) -> None:
        """Save all data to disk.

        Persists memories, history, summarization counts, and user summaries for all chats.
        """
        try:
            # Convert memories to serializable format
            memories_data = {}
            for chat_id, entries in self._memories.items():
                memories_data[str(chat_id)] = [entry.to_dict() for entry in entries]

            # Convert history to serializable format (already strings)
            history_data = {str(k): v for k, v in self._history.items()}

            # Convert summarization counts to serializable format
            summarization_data = {
                str(k): v for k, v in self._summarization_count.items()
            }

            # Convert user summaries to serializable format
            user_summaries_data = {}
            for chat_id, summaries in self._user_summaries.items():
                user_summaries_data[str(chat_id)] = {
                    username: summary.to_dict()
                    for username, summary in summaries.items()
                }

            # Convert statistics to serializable format
            statistics_data = {}
            for chat_id, stats in self._statistics.items():
                statistics_data[str(chat_id)] = stats.to_dict()

            # Convert opt-in lists to serializable format
            optin_lists_data = {}
            for chat_id, user_ids in self._optin_lists.items():
                optin_lists_data[str(chat_id)] = list(user_ids)

            # Convert opt-in message IDs to serializable format
            optin_message_ids_data = {
                str(k): v for k, v in self._optin_message_ids.items()
            }

            data = {
                "version": 4,  # Increment version for opt-in support
                "memories": memories_data,
                "history": history_data,
                "summarization_count": summarization_data,
                "user_summaries": user_summaries_data,
                "statistics": statistics_data,
                "optin_lists": optin_lists_data,
                "optin_message_ids": optin_message_ids_data,
            }

            # Write atomically using a temp file
            temp_path = self._storage_path.with_suffix(".tmp")
            temp_path.write_text(
                json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            temp_path.replace(self._storage_path)

            self._dirty = False
            logger.debug(f"Saved data to {self._storage_path}")
        except Exception as e:
            logger.error(f"Failed to save data: {e}", exc_info=True)

    def load(self) -> None:
        """Load all data from disk.

        Loads memories, history, summarization counts, and user summaries for all chats.
        """
        try:
            if not self._storage_path.exists():
                logger.debug("No saved data found")
                return

            data = json.loads(self._storage_path.read_text(encoding="utf-8"))

            # Validate version (for future migrations)
            version = data.get("version", 1)
            if version not in [1, 2, 3, 4]:
                logger.warning(f"Unknown data version {version}, skipping load")
                return

            # Load memories
            self._memories = {}
            for chat_id_str, entries_data in data.get("memories", {}).items():
                chat_id = int(chat_id_str)
                self._memories[chat_id] = [
                    MemoryEntry.from_dict(entry) for entry in entries_data
                ]

            # Load history
            self._history = {}
            for chat_id_str, messages in data.get("history", {}).items():
                self._history[int(chat_id_str)] = messages

            # Load summarization counts
            self._summarization_count = {}
            for chat_id_str, count in data.get("summarization_count", {}).items():
                self._summarization_count[int(chat_id_str)] = count

            # Load user summaries (version 2+)
            self._user_summaries = {}
            if version >= 2:
                for chat_id_str, summaries_data in data.get(
                    "user_summaries", {}
                ).items():
                    chat_id = int(chat_id_str)
                    self._user_summaries[chat_id] = {
                        username: UserSummary.from_dict(summary_data)
                        for username, summary_data in summaries_data.items()
                    }

            # Load statistics (version 3+)
            self._statistics = {}
            if version >= 3:
                for chat_id_str, stats_data in data.get("statistics", {}).items():
                    chat_id = int(chat_id_str)
                    self._statistics[chat_id] = ChatStatistics.from_dict(stats_data)

            # Load opt-in lists (version 4+)
            self._optin_lists = {}
            if version >= 4:
                for chat_id_str, user_ids in data.get("optin_lists", {}).items():
                    chat_id = int(chat_id_str)
                    self._optin_lists[chat_id] = set(user_ids)

            # Load opt-in message IDs (version 4+)
            self._optin_message_ids = {}
            if version >= 4:
                for chat_id_str, message_id in data.get(
                    "optin_message_ids", {}
                ).items():
                    self._optin_message_ids[int(chat_id_str)] = message_id

            self._dirty = False
            logger.info(
                f"Loaded data from {self._storage_path} (version {version}): "
                f"{len(self._memories)} chats with memories, "
                f"{len(self._history)} chats with history, "
                f"{len(self._user_summaries)} chats with user summaries, "
                f"{len(self._statistics)} chats with statistics, "
                f"{len(self._optin_lists)} chats with opt-in lists"
            )
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse saved data: {e}", exc_info=True)
        except Exception as e:
            logger.error(f"Failed to load data: {e}", exc_info=True)
