from __future__ import annotations

from pathlib import Path

from tbot.memory import MemoryManager


def test_memory_manager_stores_entries(tmp_path: Path) -> None:
    storage_path = tmp_path / "test_data.json"
    manager = MemoryManager(history_size=3, storage_path=storage_path, auto_save=False)
    entry = manager.add_memory(1, "Loves hiking")
    assert entry.text == "Loves hiking"
    assert manager.get_memories(1)[0].text == "Loves hiking"

    manager.append_history(1, "User: Hello")
    manager.append_history(1, "Bot: Hi")
    manager.append_history(1, "User: How are you?")
    manager.append_history(1, "Bot: Great!")

    history = manager.get_history(1)
    assert history == [
        "Bot: Hi",
        "User: How are you?",
        "Bot: Great!",
    ]


def test_memory_manager_history_per_chat(tmp_path: Path) -> None:
    """Test that history is stored separately per chat."""
    storage_path = tmp_path / "test_data.json"
    manager = MemoryManager(storage_path=storage_path, auto_save=False)

    manager.append_history(111, "User: Hello from chat 1")
    manager.append_history(222, "User: Hello from chat 2")
    manager.append_history(111, "Bot: Hi chat 1!")

    assert len(manager.get_history(111)) == 2
    assert len(manager.get_history(222)) == 1
    assert "chat 1" in manager.get_history(111)[0]
    assert "chat 2" in manager.get_history(222)[0]


def test_should_summarize(tmp_path: Path) -> None:
    """Test threshold detection for summarization."""
    storage_path = tmp_path / "test_data.json"
    manager = MemoryManager(storage_path=storage_path, auto_save=False)
    chat_id = 100

    # Add messages below threshold
    for i in range(15):
        manager.append_history(chat_id, f"Message {i}")

    assert not manager.should_summarize(chat_id, threshold=18)

    # Add more to reach threshold
    for i in range(15, 18):
        manager.append_history(chat_id, f"Message {i}")

    assert manager.should_summarize(chat_id, threshold=18)


def test_get_messages_for_summary(tmp_path: Path) -> None:
    """Test retrieving oldest messages for summarization."""
    storage_path = tmp_path / "test_data.json"
    manager = MemoryManager(storage_path=storage_path, auto_save=False)
    chat_id = 100

    # Add 20 messages
    for i in range(20):
        manager.append_history(chat_id, f"Message {i}")

    messages, total = manager.get_messages_for_summary(chat_id, batch_size=10)

    assert len(messages) == 10
    assert total == 20
    assert messages[0] == "Message 0"  # Oldest message first
    assert messages[9] == "Message 9"


def test_clear_summarized_messages(tmp_path: Path) -> None:
    """Test clearing summarized messages from history."""
    storage_path = tmp_path / "test_data.json"
    manager = MemoryManager(storage_path=storage_path, auto_save=False)
    chat_id = 100

    # Add 20 messages
    for i in range(20):
        manager.append_history(chat_id, f"Message {i}")

    # Clear first 10 messages
    manager.clear_summarized_messages(chat_id, count=10)

    history = manager.get_history(chat_id)
    assert len(history) == 10
    assert history[0] == "Message 10"  # Now starts from message 10
    assert history[-1] == "Message 19"

    # Check summarization was tracked
    assert manager.get_summarization_count(chat_id) == 1


def test_get_history_size(tmp_path: Path) -> None:
    """Test getting current history size."""
    storage_path = tmp_path / "test_data.json"
    manager = MemoryManager(storage_path=storage_path, auto_save=False)
    chat_id = 100

    assert manager.get_history_size(chat_id) == 0

    for i in range(15):
        manager.append_history(chat_id, f"Message {i}")

    assert manager.get_history_size(chat_id) == 15


def test_summarization_count_per_chat(tmp_path: Path) -> None:
    """Test that summarization count is tracked per chat."""
    storage_path = tmp_path / "test_data.json"
    manager = MemoryManager(storage_path=storage_path, auto_save=False)

    # Add and summarize for chat 1
    for i in range(20):
        manager.append_history(111, f"Message {i}")
    manager.clear_summarized_messages(111, 10)

    # Add and summarize for chat 2
    for i in range(30):
        manager.append_history(222, f"Message {i}")
    manager.clear_summarized_messages(222, 15)
    manager.clear_summarized_messages(222, 15)  # Second summarization

    assert manager.get_summarization_count(111) == 1
    assert manager.get_summarization_count(222) == 2
    assert manager.get_summarization_count(333) == 0  # No summarizations


def test_memories_separate_per_chat(tmp_path: Path) -> None:
    """Test that memories are stored separately per chat."""
    storage_path = tmp_path / "test_data.json"
    manager = MemoryManager(storage_path=storage_path, auto_save=False)

    memory1 = manager.add_memory(111, "Important fact for chat 1")
    memory2 = manager.add_memory(222, "Important fact for chat 2")
    memory3 = manager.add_memory(111, "Another fact for chat 1")

    chat1_memories = manager.get_memories(111)
    chat2_memories = manager.get_memories(222)

    assert len(chat1_memories) == 2
    assert len(chat2_memories) == 1
    assert memory1 in chat1_memories
    assert memory3 in chat1_memories
    assert memory2 in chat2_memories


def test_persistence_save_and_load(tmp_path: Path) -> None:
    """Test that data is saved and loaded correctly."""
    storage_path = tmp_path / "test_data.json"

    # Create manager and add data
    manager1 = MemoryManager(storage_path=storage_path, auto_save=False)
    manager1.add_memory(100, "User likes Python")
    manager1.add_memory(100, "User prefers dark mode")
    manager1.add_memory(200, "Different chat memory")
    manager1.append_history(100, "User: Hello")
    manager1.append_history(100, "Bot: Hi there!")
    manager1.append_history(200, "User: Test")

    # Save data
    manager1.save()
    assert storage_path.exists()

    # Create new manager and load data
    manager2 = MemoryManager(storage_path=storage_path, auto_save=False)

    # Verify memories loaded
    chat100_memories = manager2.get_memories(100)
    assert len(chat100_memories) == 2
    assert chat100_memories[0].text == "User likes Python"
    assert chat100_memories[1].text == "User prefers dark mode"

    chat200_memories = manager2.get_memories(200)
    assert len(chat200_memories) == 1
    assert chat200_memories[0].text == "Different chat memory"

    # Verify history loaded
    chat100_history = manager2.get_history(100)
    assert len(chat100_history) == 2
    assert chat100_history[0] == "User: Hello"
    assert chat100_history[1] == "Bot: Hi there!"

    chat200_history = manager2.get_history(200)
    assert len(chat200_history) == 1
    assert chat200_history[0] == "User: Test"


def test_persistence_auto_save(tmp_path: Path) -> None:
    """Test that auto-save works correctly."""
    storage_path = tmp_path / "test_data.json"

    # Create manager with auto-save enabled
    manager = MemoryManager(storage_path=storage_path, auto_save=True)
    manager.add_memory(100, "Auto-saved memory")

    # File should exist after adding memory
    assert storage_path.exists()

    # Load in new manager
    manager2 = MemoryManager(storage_path=storage_path, auto_save=False)
    assert len(manager2.get_memories(100)) == 1
    assert manager2.get_memories(100)[0].text == "Auto-saved memory"


def test_persistence_handles_missing_file(tmp_path: Path) -> None:
    """Test that manager handles missing data file gracefully."""
    storage_path = tmp_path / "nonexistent.json"

    # Should not raise an error
    manager = MemoryManager(storage_path=storage_path, auto_save=False)
    assert manager.get_history_size(100) == 0
    assert len(manager.get_memories(100)) == 0


def test_persistence_summarization_counts(tmp_path: Path) -> None:
    """Test that summarization counts persist correctly."""
    storage_path = tmp_path / "test_data.json"

    # Create manager and track summarizations
    manager1 = MemoryManager(storage_path=storage_path, auto_save=False)
    for i in range(20):
        manager1.append_history(100, f"Message {i}")
    manager1.clear_summarized_messages(100, 10)
    manager1.save()

    # Load in new manager
    manager2 = MemoryManager(storage_path=storage_path, auto_save=False)
    assert manager2.get_summarization_count(100) == 1
    assert manager2.get_history_size(100) == 10
