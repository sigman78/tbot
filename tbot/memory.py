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


class MemoryManager:
    """Store persona memories and recent chat messages per chat."""

    def __init__(
        self,
        history_size: int = 20,
        storage_path: Path | str | None = None,
        auto_save: bool = True,
    ) -> None:
        """Initialize the memory manager.

        Args:
            history_size: Maximum number of messages to keep in history
            storage_path: Path to persistence file (default: ~/.tbot-data.json)
            auto_save: Whether to auto-save after changes
        """
        self._memories: Dict[int, List[MemoryEntry]] = {}
        self._history: Dict[int, List[str]] = {}
        self._history_size = history_size
        self._summarization_count: Dict[int, int] = {}
        self._storage_path = Path(storage_path or Path.home() / ".tbot-data.json")
        self._auto_save = auto_save
        self._dirty = False  # Track if data has changed since last save

        # Load existing data if available
        if self._storage_path.exists():
            self.load()

    def add_memory(self, chat_id: int, text: str) -> MemoryEntry:
        entry = MemoryEntry(chat_id=chat_id, text=text.strip(), created_at=datetime.utcnow())
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

    def _mark_dirty(self) -> None:
        """Mark data as changed and trigger auto-save if enabled."""
        self._dirty = True
        if self._auto_save:
            self.save()

    def save(self) -> None:
        """Save all data to disk.

        Persists memories, history, and summarization counts for all chats.
        """
        try:
            # Convert memories to serializable format
            memories_data = {}
            for chat_id, entries in self._memories.items():
                memories_data[str(chat_id)] = [entry.to_dict() for entry in entries]

            # Convert history to serializable format (already strings)
            history_data = {str(k): v for k, v in self._history.items()}

            # Convert summarization counts to serializable format
            summarization_data = {str(k): v for k, v in self._summarization_count.items()}

            data = {
                "version": 1,  # For future format migrations
                "memories": memories_data,
                "history": history_data,
                "summarization_count": summarization_data,
            }

            # Write atomically using a temp file
            temp_path = self._storage_path.with_suffix(".tmp")
            temp_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
            temp_path.replace(self._storage_path)

            self._dirty = False
            logger.debug(f"Saved data to {self._storage_path}")
        except Exception as e:
            logger.error(f"Failed to save data: {e}", exc_info=True)

    def load(self) -> None:
        """Load all data from disk.

        Loads memories, history, and summarization counts for all chats.
        """
        try:
            if not self._storage_path.exists():
                logger.debug("No saved data found")
                return

            data = json.loads(self._storage_path.read_text(encoding="utf-8"))

            # Validate version (for future migrations)
            version = data.get("version", 1)
            if version != 1:
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

            self._dirty = False
            logger.info(
                f"Loaded data from {self._storage_path}: "
                f"{len(self._memories)} chats with memories, "
                f"{len(self._history)} chats with history"
            )
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse saved data: {e}", exc_info=True)
        except Exception as e:
            logger.error(f"Failed to load data: {e}", exc_info=True)

