# TBot Modularity & Interface Improvement Plan

## Executive Summary

This document analyzes the tbot codebase and provides recommendations for improving modularity, enforcing clearer interfaces, and strengthening type safety. While the codebase already demonstrates good separation of concerns and comprehensive type annotations, there are opportunities to formalize interfaces using Python Protocols, reduce coupling, and improve testability and extensibility.

**Current State**: ✅ Good separation of concerns, ✅ Type annotations, ⚠️ Tight coupling, ⚠️ No formal protocols
**Target State**: ✅ Protocol-based interfaces, ✅ Dependency injection, ✅ Modular architecture, ✅ Enhanced testability

---

## Table of Contents

1. [Current Architecture Analysis](#current-architecture-analysis)
2. [Key Issues Identified](#key-issues-identified)
3. [Proposed Interface Definitions](#proposed-interface-definitions)
4. [Module Boundary Improvements](#module-boundary-improvements)
5. [Implementation Roadmap](#implementation-roadmap)
6. [Benefits & Trade-offs](#benefits--trade-offs)

---

## Current Architecture Analysis

### Module Dependency Graph

```
┌─────────────────────────────────────────────────────────────┐
│                        main.py                              │
│                    (CLI Entry Point)                        │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                      bot.py (800 lines)                     │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ • Application wiring                                  │  │
│  │ • 13 command handlers (inline)                        │  │
│  │ • Message processing logic                            │  │
│  │ • Error handling                                      │  │
│  │ • Permission checks                                   │  │
│  │ • Reaction handling                                   │  │
│  └──────────────────────────────────────────────────────┘  │
└─────┬────────┬──────────┬────────────┬────────────┬────────┘
      │        │          │            │            │
      ▼        ▼          ▼            ▼            ▼
┌──────────┐ ┌──────┐ ┌────────┐ ┌─────────┐ ┌──────────┐
│ config.py│ │logic.│ │context.│ │ memory. │ │llm_client│
│          │ │  py  │ │   py   │ │   py    │ │   .py    │
│          │ │      │ │        │ │(576 ln) │ │(466 ln)  │
└──────────┘ └──────┘ └────────┘ └─────────┘ └──────────┘
```

### Current Interfaces (Implicit)

The codebase currently uses **concrete classes** without formal interface definitions:

| Module | Public API | Type Safety | Issues |
|--------|-----------|-------------|---------|
| `config.py` | `BotConfig`, `ConfigManager` | ✅ Full type hints | No interface abstraction |
| `memory.py` | `MemoryManager` | ✅ Full type hints | God class, 576 lines |
| `llm_client.py` | `LLMClient` | ✅ Full type hints | Hard-coded to OpenAI SDK |
| `context.py` | `ConversationContextBuilder` | ✅ Full type hints | Tightly coupled to memory types |
| `logic.py` | `should_respond()` | ✅ Full type hints | Pure function (good) |
| `bot.py` | `create_application()` | ✅ Full type hints | Too large, too many concerns |

**Key Observation**: While type hints are present, there are no **Protocol** definitions to enable dependency inversion and pluggability.

---

## Key Issues Identified

### 1. ⚠️ No Formal Protocol Definitions

**Problem**: Direct coupling to concrete implementations prevents:
- Easy mocking in tests
- Swapping implementations (e.g., different LLM providers, storage backends)
- Clear contractual interfaces

**Example** (`bot.py:162-178`):
```python
def create_application(
    token: str,
    *,
    api_key: str | None = None,
    config_manager: ConfigManager | None = None,  # ⚠️ Concrete class
    memory_manager: MemoryManager | None = None,  # ⚠️ Concrete class
    llm_client: LLMClient | None = None,          # ⚠️ Concrete class
) -> Application:
    # Creates concrete instances if None
    config_manager = config_manager or ConfigManager()
    memory_manager = memory_manager or MemoryManager(...)
    llm_client = llm_client or LLMClient.fromParams(...)
```

**Impact**:
- Hard to test with mocks
- Can't easily switch to PostgreSQL storage, Redis cache, or different LLM provider
- Violates Dependency Inversion Principle

---

### 2. ⚠️ bot.py God Module (800 lines)

**Problem**: `bot.py` has too many responsibilities:

```python
bot.py:162-791 (800 lines):
├─ Application factory function
├─ 13 inline command handlers:
│  ├─ handle_persona()
│  ├─ handle_config()
│  ├─ handle_set()
│  ├─ handle_summary()
│  ├─ handle_forget()
│  ├─ handle_stat()
│  ├─ handle_memory()
│  ├─ handle_bot_added_to_group()
│  ├─ handle_reaction()
│  ├─ handle_ask_optin()
│  ├─ handle_help()
│  ├─ maybe_reply()           # Core message handler
│  └─ error_handler()
├─ Auto-summarization logic
├─ Permission checking
├─ Reaction suggestion logic
└─ Message truncation utilities
```

**Impact**:
- Hard to test individual handlers
- Difficult to navigate and maintain
- Violates Single Responsibility Principle
- No clear separation between routing and business logic

---

### 3. ⚠️ MemoryManager God Class (576 lines)

**Problem**: `MemoryManager` handles too many concerns:

```python
memory.py (MemoryManager):
├─ Long-term memories (add, get, clear)
├─ Conversation history (append, get, summarization triggers)
├─ Per-user summaries (add, get, cleanup with LRU)
├─ Chat statistics (replies, reactions, LLM calls, tokens)
├─ Opt-in user lists (add, check, get)
├─ Opt-in message ID tracking
├─ JSON persistence (save, load, versioned format)
└─ Dirty tracking & auto-save
```

**Impact**:
- Violates Single Responsibility Principle
- Changes to any concern require modifying this large class
- Hard to test individual concerns in isolation

**Recommendation**: Split into separate, focused components:
- `HistoryStore` - Conversation history
- `MemoryStore` - Long-term memories
- `SummaryStore` - User summaries
- `StatisticsCollector` - Chat statistics
- `OptInManager` - Opt-in tracking
- `DataPersistence` - JSON serialization

---

### 4. ⚠️ Missing Domain-Specific Exceptions

**Problem**: Generic exceptions used throughout:

```python
# llm_client.py:154
raise ValueError("LLM returned empty response")

# llm_client.py:201
raise ValueError("Cannot summarize empty message list")

# config.py:13
raise ValueError("response_frequency must be between ...")
```

**Impact**:
- Consumers can't distinguish between error types
- No structured error handling
- Poor error messages for API consumers

**Recommendation**: Define domain exception hierarchy:

```python
class TBotError(Exception):
    """Base exception for tbot errors"""

class ConfigurationError(TBotError):
    """Configuration validation failed"""

class LLMError(TBotError):
    """LLM API call failed"""

class LLMEmptyResponseError(LLMError):
    """LLM returned empty response"""

class StorageError(TBotError):
    """Data persistence failed"""
```

---

### 5. ⚠️ Context Builder Tightly Coupled to Memory Types

**Problem** (`context.py:32-40`):
```python
def build_messages(
    self,
    *,
    config: BotConfig,              # ⚠️ Concrete type
    history: List[str],
    memories: List[MemoryEntry],    # ⚠️ Concrete type
    current_message: str,
    user_summaries: List[UserSummary] | None = None,  # ⚠️ Concrete type
) -> List[ChatCompletionMessageParam]:
```

**Impact**:
- Can't use alternative memory representations
- Hard to test with minimal data structures

**Recommendation**: Use Protocol types or generic interfaces

---

### 6. ⚠️ LLMClient Hard-Coded to OpenAI SDK

**Problem** (`llm_client.py:42-65`):
```python
class LLMClient:
    def __init__(self, client: OpenAI) -> None:  # ⚠️ Hard-coded to OpenAI
        self._client = client

    # All methods call self._client.chat.completions.create()
```

**Impact**:
- Can't easily support Anthropic, Gemini, local models
- Hard to create fake LLM for tests
- Violates Open/Closed Principle

**Recommendation**: Define `LLMProvider` protocol

---

## Proposed Interface Definitions

### Architecture: Protocol-Based Interfaces

```python
# New file: tbot/interfaces.py
from __future__ import annotations

from typing import Protocol, List, Iterable, runtime_checkable
from datetime import datetime
from openai.types.chat import ChatCompletionMessageParam

# ============================================================================
# 1. Configuration Interfaces
# ============================================================================

@runtime_checkable
class ConfigStore(Protocol):
    """Interface for configuration persistence."""

    @property
    def config(self) -> BotConfig:
        """Get current configuration."""
        ...

    def load(self) -> BotConfig:
        """Load configuration from storage."""
        ...

    def save(self) -> None:
        """Save configuration to storage."""
        ...

    def update(self, **kwargs) -> BotConfig:
        """Update and save configuration."""
        ...


# ============================================================================
# 2. Memory & Storage Interfaces
# ============================================================================

@runtime_checkable
class HistoryStore(Protocol):
    """Interface for conversation history management."""

    def append(self, chat_id: int, message: str) -> None:
        """Append message to history."""
        ...

    def get(self, chat_id: int, limit: int | None = None) -> List[str]:
        """Get conversation history."""
        ...

    def get_size(self, chat_id: int) -> int:
        """Get current history size."""
        ...

    def clear_oldest(self, chat_id: int, count: int) -> None:
        """Remove oldest messages."""
        ...


@runtime_checkable
class MemoryStore(Protocol):
    """Interface for long-term memory storage."""

    def add(self, chat_id: int, text: str) -> MemoryEntry:
        """Add a memory entry."""
        ...

    def get(self, chat_id: int) -> List[MemoryEntry]:
        """Get all memories for a chat."""
        ...

    def clear(self, chat_id: int) -> None:
        """Clear all memories for a chat."""
        ...


@runtime_checkable
class SummaryStore(Protocol):
    """Interface for user summary storage."""

    def add(self, chat_id: int, username: str, summary: str) -> None:
        """Add or update user summary."""
        ...

    def get(self, chat_id: int) -> List[UserSummary]:
        """Get all user summaries, ordered by recency."""
        ...

    def clear(self, chat_id: int) -> None:
        """Clear all summaries for a chat."""
        ...


@runtime_checkable
class StatisticsCollector(Protocol):
    """Interface for chat statistics tracking."""

    def increment_replies(self, chat_id: int) -> None:
        """Increment reply count."""
        ...

    def increment_reactions(self, chat_id: int) -> None:
        """Increment reaction count."""
        ...

    def increment_llm_calls(
        self,
        chat_id: int,
        tokens_sent: int = 0,
        tokens_received: int = 0
    ) -> None:
        """Increment LLM call count and token usage."""
        ...

    def get(self, chat_id: int) -> ChatStatistics:
        """Get statistics for a chat."""
        ...


# ============================================================================
# 3. LLM Provider Interface
# ============================================================================

@runtime_checkable
class LLMProvider(Protocol):
    """Interface for LLM API providers."""

    async def generate_reply(
        self,
        messages: List[ChatCompletionMessageParam],
        model: str,
        temperature: float = 0.8,
        max_tokens: int = 512,
    ) -> str:
        """Generate a reply using the LLM.

        Args:
            messages: Conversation messages in OpenAI format
            model: Model identifier
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            Generated reply text

        Raises:
            LLMError: If generation fails
        """
        ...

    async def generate_summary(
        self,
        messages: List[ChatCompletionMessageParam],
        model: str,
    ) -> str:
        """Generate a concise summary.

        Args:
            messages: Messages to summarize
            model: Model identifier

        Returns:
            Summary text

        Raises:
            LLMError: If generation fails
        """
        ...

    async def suggest_reaction(
        self,
        message: str,
        persona: str,
        model: str,
    ) -> str | None:
        """Suggest an emoji reaction.

        Args:
            message: User message
            persona: Bot persona
            model: Model identifier

        Returns:
            Emoji string or None
        """
        ...


# ============================================================================
# 4. Context Building Interface
# ============================================================================

@runtime_checkable
class ContextBuilder(Protocol):
    """Interface for building LLM conversation context."""

    def build_messages(
        self,
        config: BotConfig,
        history: Iterable[str],
        memories: Iterable[MemoryEntry],
        current_message: str,
        user_summaries: Iterable[UserSummary] | None = None,
    ) -> List[ChatCompletionMessageParam]:
        """Build message list for LLM API.

        Args:
            config: Bot configuration
            history: Conversation history
            memories: Long-term memories
            current_message: Current user message
            user_summaries: Per-user summaries

        Returns:
            Formatted messages for LLM API
        """
        ...


# ============================================================================
# 5. Decision Logic Interface
# ============================================================================

@runtime_checkable
class ResponseDecider(Protocol):
    """Interface for deciding when to respond."""

    def should_respond(
        self,
        random_value: float,
        response_frequency: float,
        replied_to_bot: bool,
        is_private_chat: bool = False,
        mentioned_bot: bool = False,
    ) -> bool:
        """Determine if bot should respond.

        Args:
            random_value: Random value [0.0, 1.0]
            response_frequency: Configured frequency
            replied_to_bot: Whether message replies to bot
            is_private_chat: Whether in private chat
            mentioned_bot: Whether bot was mentioned

        Returns:
            True if should respond
        """
        ...


# ============================================================================
# 6. Opt-In Management Interface
# ============================================================================

@runtime_checkable
class OptInManager(Protocol):
    """Interface for managing user opt-in consent."""

    def add_user(self, chat_id: int, user_id: int) -> None:
        """Add user to opt-in list."""
        ...

    def is_opted_in(self, chat_id: int, user_id: int) -> bool:
        """Check if user opted in."""
        ...

    def get_users(self, chat_id: int) -> set[int]:
        """Get all opted-in users."""
        ...

    def set_optin_message(self, chat_id: int, message_id: int) -> None:
        """Store opt-in request message ID."""
        ...

    def get_optin_message(self, chat_id: int) -> int | None:
        """Get opt-in request message ID."""
        ...
```

---

## Module Boundary Improvements

### Proposed New Architecture

```
tbot/
├── __init__.py                 # Public API exports
├── main.py                     # CLI entry point
│
├── interfaces.py               # ✨ NEW: Protocol definitions
├── exceptions.py               # ✨ NEW: Domain exception hierarchy
│
├── core/                       # ✨ NEW: Core business logic
│   ├── __init__.py
│   ├── config.py              # ConfigManager (implements ConfigStore)
│   ├── decision.py            # ResponseDecider implementation
│   └── context_builder.py    # ContextBuilder implementation
│
├── storage/                    # ✨ NEW: Data storage layer
│   ├── __init__.py
│   ├── history.py             # HistoryStore implementation
│   ├── memory.py              # MemoryStore implementation
│   ├── summary.py             # SummaryStore implementation
│   ├── statistics.py          # StatisticsCollector implementation
│   ├── optin.py               # OptInManager implementation
│   └── persistence.py         # JSON file persistence
│
├── llm/                        # ✨ NEW: LLM provider layer
│   ├── __init__.py
│   ├── openai_provider.py     # OpenAI implementation of LLMProvider
│   ├── mock_provider.py       # Mock for testing
│   └── client.py              # High-level LLM client (uses LLMProvider)
│
├── bot/                        # ✨ NEW: Telegram bot layer
│   ├── __init__.py
│   ├── application.py         # Application factory
│   ├── handlers/              # ✨ NEW: Command handlers
│   │   ├── __init__.py
│   │   ├── config_handlers.py    # /config, /set, /persona
│   │   ├── memory_handlers.py    # /memory, /forget, /summary
│   │   ├── stats_handlers.py     # /stat
│   │   ├── optin_handlers.py     # /ask_optin
│   │   ├── help_handler.py       # /help
│   │   └── message_handler.py    # maybe_reply logic
│   ├── utils.py               # Message truncation, parsing
│   └── error_handler.py       # Error handling
│
├── models.py                   # Data models (MemoryEntry, UserSummary, etc.)
└── const.py                    # Constants
```

### Key Changes

1. **Separation of Concerns**:
   - `core/` - Business logic (decision making, context building, config)
   - `storage/` - Data persistence (abstracted behind protocols)
   - `llm/` - LLM provider abstraction
   - `bot/` - Telegram-specific wiring

2. **Smaller, Focused Modules**:
   - `bot.py` (800 lines) → Split into `bot/application.py` + `bot/handlers/*`
   - `memory.py` (576 lines) → Split into `storage/{history,memory,summary,statistics,optin}.py`

3. **Protocol-Based Interfaces**:
   - All major components implement protocols from `interfaces.py`
   - Easy to swap implementations
   - Clear contractual APIs

---

## Implementation Roadmap

### Phase 1: Foundation (Non-Breaking)

**Goal**: Introduce protocols and exceptions without breaking existing code

✅ **Tasks**:
1. Create `tbot/interfaces.py` with all Protocol definitions
2. Create `tbot/exceptions.py` with domain exception hierarchy
3. Create `tbot/models.py` and move dataclasses (MemoryEntry, UserSummary, etc.)
4. Update existing classes to explicitly satisfy protocols (no API changes yet)
5. Add Protocol conformance tests

**Duration**: 2-3 days
**Risk**: Low (backward compatible)

---

### Phase 2: Extract Storage Components

**Goal**: Split MemoryManager into focused components

✅ **Tasks**:
1. Create `tbot/storage/` package
2. Extract `HistoryStore` implementation → `storage/history.py`
3. Extract `MemoryStore` implementation → `storage/memory.py`
4. Extract `SummaryStore` implementation → `storage/summary.py`
5. Extract `StatisticsCollector` implementation → `storage/statistics.py`
6. Extract `OptInManager` implementation → `storage/optin.py`
7. Create `storage/persistence.py` for JSON serialization
8. Create `storage/memory_manager.py` as a facade that composes all stores
9. Add comprehensive tests for each component
10. Update `create_application()` to use new components

**Duration**: 3-5 days
**Risk**: Medium (requires careful data migration)

**Migration Strategy**:
```python
# Old way (still supported via facade)
memory_manager = MemoryManager()

# New way (direct component access)
history_store = JsonHistoryStore()
memory_store = JsonMemoryStore()
summary_store = JsonSummaryStore()
stats_collector = JsonStatisticsCollector()
optin_manager = JsonOptInManager()

# Or use facade for backward compatibility
memory_manager = CompositeMemoryManager(
    history=history_store,
    memory=memory_store,
    summary=summary_store,
    statistics=stats_collector,
    optin=optin_manager,
)
```

---

### Phase 3: Extract Bot Handlers

**Goal**: Split bot.py into modular handlers

✅ **Tasks**:
1. Create `tbot/bot/` package
2. Create `bot/handlers/` package
3. Extract command handlers:
   - `config_handlers.py` - handle_persona, handle_config, handle_set
   - `memory_handlers.py` - handle_memory, handle_forget, handle_summary
   - `stats_handlers.py` - handle_stat
   - `optin_handlers.py` - handle_ask_optin, handle_bot_added_to_group, handle_reaction
   - `help_handler.py` - handle_help
   - `message_handler.py` - maybe_reply, _maybe_auto_summarize
4. Create `bot/utils.py` - _truncate_text, _parse_argument, _get_message
5. Create `bot/error_handler.py` - error_handler
6. Create `bot/application.py` - create_application with dependency injection
7. Update tests to use handler modules

**Duration**: 2-3 days
**Risk**: Low (pure refactoring)

**Example Handler Structure**:
```python
# tbot/bot/handlers/config_handlers.py
from telegram import Update
from telegram.ext import ContextTypes

from ...interfaces import ConfigStore

class ConfigHandlers:
    def __init__(self, config_store: ConfigStore):
        self.config_store = config_store

    async def handle_config(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """List all current tunable parameters."""
        # Implementation here

    async def handle_set(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Set a configuration parameter."""
        # Implementation here
```

---

### Phase 4: Abstract LLM Provider

**Goal**: Support multiple LLM backends

✅ **Tasks**:
1. Create `tbot/llm/` package
2. Define `LLMProvider` protocol (already in `interfaces.py`)
3. Create `llm/openai_provider.py` - OpenAI implementation
4. Create `llm/mock_provider.py` - Testing mock
5. Refactor `LLMClient` to use `LLMProvider` protocol
6. Update `create_application()` to accept `LLMProvider`
7. Add tests with mock provider

**Duration**: 2-3 days
**Risk**: Low

**Example**:
```python
# tbot/llm/openai_provider.py
from openai import OpenAI
from ..interfaces import LLMProvider
from ..exceptions import LLMError, LLMEmptyResponseError

class OpenAIProvider:
    """OpenAI implementation of LLMProvider protocol."""

    def __init__(self, client: OpenAI):
        self._client = client

    async def generate_reply(
        self,
        messages: List[ChatCompletionMessageParam],
        model: str,
        temperature: float = 0.8,
        max_tokens: int = 512,
    ) -> str:
        try:
            response = await asyncio.to_thread(
                self._client.chat.completions.create,
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            content = response.choices[0].message.content
            if content is None:
                raise LLMEmptyResponseError("LLM returned empty response")
            return content.strip()
        except OpenAIError as e:
            raise LLMError(f"OpenAI API error: {e}") from e


# Easy to add new providers:
class AnthropicProvider:
    """Anthropic implementation of LLMProvider protocol."""
    # Implementation using anthropic SDK
    ...

class LocalLlamaProvider:
    """Local Llama implementation via llama.cpp."""
    # Implementation using local inference
    ...
```

---

### Phase 5: Dependency Injection

**Goal**: Clean dependency injection in `create_application()`

✅ **Tasks**:
1. Update `create_application()` to accept protocol types
2. Create factory functions for default implementations
3. Add dependency container (optional: use `dependency-injector` library)
4. Update tests to inject mocks easily

**Duration**: 1-2 days
**Risk**: Low

**Example**:
```python
# tbot/bot/application.py
from ..interfaces import (
    ConfigStore,
    HistoryStore,
    MemoryStore,
    SummaryStore,
    StatisticsCollector,
    OptInManager,
    LLMProvider,
    ContextBuilder,
)

def create_application(
    token: str,
    *,
    config_store: ConfigStore | None = None,
    history_store: HistoryStore | None = None,
    memory_store: MemoryStore | None = None,
    summary_store: SummaryStore | None = None,
    statistics: StatisticsCollector | None = None,
    optin_manager: OptInManager | None = None,
    llm_provider: LLMProvider | None = None,
    context_builder: ContextBuilder | None = None,
) -> Application:
    """Create Telegram application with dependency injection.

    All dependencies are optional and will use default implementations if not provided.
    This allows easy testing with mocks and swapping implementations.
    """
    # Use defaults if not provided
    config_store = config_store or create_default_config_store()
    history_store = history_store or create_default_history_store()
    memory_store = memory_store or create_default_memory_store()
    summary_store = summary_store or create_default_summary_store()
    statistics = statistics or create_default_statistics_collector()
    optin_manager = optin_manager or create_default_optin_manager()
    llm_provider = llm_provider or create_default_llm_provider()
    context_builder = context_builder or create_default_context_builder()

    # Wire up handlers with dependencies
    config_handlers = ConfigHandlers(config_store)
    memory_handlers = MemoryHandlers(memory_store, summary_store)
    message_handler = MessageHandler(
        config_store=config_store,
        history_store=history_store,
        memory_store=memory_store,
        summary_store=summary_store,
        statistics=statistics,
        llm_provider=llm_provider,
        context_builder=context_builder,
    )

    # Create and configure application
    application = Application.builder().token(token).build()

    # Register handlers
    application.add_handler(CommandHandler("config", config_handlers.handle_config))
    application.add_handler(CommandHandler("set", config_handlers.handle_set))
    # ... etc

    return application
```

**Testing becomes trivial**:
```python
# tests/test_bot.py
from unittest.mock import Mock

def test_message_handler_with_mock_llm():
    # Create mocks
    mock_llm = Mock(spec=LLMProvider)
    mock_llm.generate_reply.return_value = "Hello!"

    mock_history = Mock(spec=HistoryStore)
    mock_history.get.return_value = []

    # Inject mocks
    app = create_application(
        token="test",
        llm_provider=mock_llm,
        history_store=mock_history,
    )

    # Test behavior
    # ...
```

---

## Benefits & Trade-offs

### Benefits

#### ✅ 1. Improved Testability
- Easy to inject mocks and stubs
- Test components in isolation
- Faster unit tests (no I/O dependencies)

**Example**:
```python
# Before: Hard to test without real OpenAI client
def test_bot_reply():
    llm_client = LLMClient(OpenAI(api_key="fake"))  # ❌ Requires API key
    # ...

# After: Easy with protocol-based mocking
def test_bot_reply():
    mock_llm = Mock(spec=LLMProvider)
    mock_llm.generate_reply.return_value = "Test response"  # ✅ No API needed
    # ...
```

#### ✅ 2. Enhanced Extensibility
- Support multiple LLM providers (OpenAI, Anthropic, local models)
- Support multiple storage backends (JSON, SQLite, PostgreSQL, Redis)
- Easy to add new features without modifying existing code

**Example**:
```python
# Easy to add PostgreSQL storage
class PostgresHistoryStore:
    """PostgreSQL implementation of HistoryStore protocol."""
    def __init__(self, db_url: str):
        self.engine = create_engine(db_url)

    def append(self, chat_id: int, message: str) -> None:
        # SQL INSERT

    def get(self, chat_id: int, limit: int | None = None) -> List[str]:
        # SQL SELECT

# Use in production
history = PostgresHistoryStore("postgresql://...")
app = create_application(token="...", history_store=history)
```

#### ✅ 3. Better Code Organization
- Smaller, focused modules (easier to understand)
- Clear separation of concerns
- Reduced cognitive load

**Metrics**:
- `bot.py`: 800 lines → ~100 lines (application.py)
- `memory.py`: 576 lines → 6 files × ~100 lines each
- Command handlers: Inline → 6 focused modules

#### ✅ 4. Clearer Contracts
- Protocol definitions document expected behavior
- Type checkers (mypy, pyright) can verify implementations
- IDE autocomplete works better

#### ✅ 5. Easier Onboarding
- New developers can understand components independently
- Clear module boundaries
- Better documentation via protocols

### Trade-offs

#### ⚠️ 1. Increased Complexity (Initially)
- More files and directories
- More abstraction layers
- Learning curve for protocols

**Mitigation**:
- Good documentation
- Keep default implementations simple
- Gradual migration (backward compatible)

#### ⚠️ 2. More Boilerplate
- Protocol definitions require duplication
- Factory functions for defaults
- More imports

**Mitigation**:
- Use type aliases where appropriate
- Provide convenience functions
- Good IDE support reduces friction

#### ⚠️ 3. Performance Overhead (Minimal)
- Extra indirection through protocols
- Dependency injection overhead

**Impact**: Negligible for this application (I/O-bound, not CPU-bound)

#### ⚠️ 4. Migration Effort
- Requires refactoring existing code
- Need to update tests
- Risk of introducing bugs

**Mitigation**:
- Incremental migration (phase by phase)
- Maintain backward compatibility
- Comprehensive test coverage
- Review each phase carefully

---

## Appendix A: Type Safety Enhancements

### Current State

The codebase already has excellent type coverage:
- ✅ Function signatures typed
- ✅ Return types specified
- ✅ Union types using `|` syntax
- ✅ Generic types (List, Dict, Iterable)

### Recommended Enhancements

#### 1. Enable Strict Type Checking

Add `pyproject.toml`:
```toml
[tool.mypy]
python_version = "3.11"
strict = true
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
disallow_any_generics = true
```

#### 2. Add Runtime Type Checking (Optional)

For critical APIs, consider runtime validation:
```python
from typing import runtime_checkable

@runtime_checkable
class LLMProvider(Protocol):
    ...

# Can check at runtime
assert isinstance(my_provider, LLMProvider)
```

#### 3. Use NewType for Domain Types

Prevent mixing incompatible IDs:
```python
from typing import NewType

ChatId = NewType('ChatId', int)
UserId = NewType('UserId', int)
MessageId = NewType('MessageId', int)

def get_history(chat_id: ChatId) -> List[str]:
    ...

# Type checker will catch this:
user_id: UserId = UserId(123)
get_history(user_id)  # ❌ Type error
```

#### 4. Use Literal Types for Constants

```python
from typing import Literal

ChatType = Literal["private", "group", "supergroup"]

def handle_message(chat_type: ChatType) -> None:
    if chat_type == "private":  # ✅ Type safe
        ...
    elif chat_type == "grup":  # ❌ Type checker catches typo
        ...
```

---

## Appendix B: Testing Strategy

### Current State
- ✅ Comprehensive test suite (1481 lines)
- ✅ Good coverage of core functionality

### Enhanced Testing with Protocols

#### 1. Mock Providers

```python
# tests/mocks.py
class MockLLMProvider:
    """Test double for LLM provider."""

    def __init__(self):
        self.replies = []
        self.call_count = 0

    async def generate_reply(self, messages, model, **kwargs) -> str:
        self.call_count += 1
        if self.replies:
            return self.replies.pop(0)
        return "Mock response"

    def expect_reply(self, reply: str) -> None:
        """Queue a reply for next call."""
        self.replies.append(reply)


# tests/test_bot.py
async def test_bot_generates_reply():
    mock_llm = MockLLMProvider()
    mock_llm.expect_reply("Hello, user!")

    app = create_application(token="test", llm_provider=mock_llm)

    # Send message and verify
    # ...

    assert mock_llm.call_count == 1
```

#### 2. In-Memory Storage for Tests

```python
# tests/storage.py
class InMemoryHistoryStore:
    """In-memory history store for testing."""

    def __init__(self):
        self._data: Dict[int, List[str]] = {}

    def append(self, chat_id: int, message: str) -> None:
        self._data.setdefault(chat_id, []).append(message)

    def get(self, chat_id: int, limit: int | None = None) -> List[str]:
        history = self._data.get(chat_id, [])
        if limit is None:
            return history[:]
        return history[-limit:]
```

#### 3. Protocol Conformance Tests

```python
# tests/test_protocols.py
from tbot.interfaces import LLMProvider, HistoryStore
from tbot.llm.openai_provider import OpenAIProvider
from tbot.storage.history import JsonHistoryStore

def test_openai_provider_conforms_to_protocol():
    """Verify OpenAIProvider implements LLMProvider protocol."""
    assert isinstance(OpenAIProvider(...), LLMProvider)

def test_json_history_store_conforms_to_protocol():
    """Verify JsonHistoryStore implements HistoryStore protocol."""
    assert isinstance(JsonHistoryStore(), HistoryStore)
```

---

## Appendix C: Migration Checklist

### Pre-Migration
- [ ] Review and approve this plan
- [ ] Set up feature branch
- [ ] Ensure test coverage is adequate
- [ ] Create backup of current codebase

### Phase 1: Foundation
- [ ] Create `tbot/interfaces.py`
- [ ] Create `tbot/exceptions.py`
- [ ] Create `tbot/models.py`
- [ ] Update existing classes to satisfy protocols
- [ ] Add protocol conformance tests
- [ ] Update documentation
- [ ] Code review and merge

### Phase 2: Storage
- [ ] Create `tbot/storage/` package
- [ ] Implement `HistoryStore`
- [ ] Implement `MemoryStore`
- [ ] Implement `SummaryStore`
- [ ] Implement `StatisticsCollector`
- [ ] Implement `OptInManager`
- [ ] Implement `JsonPersistence`
- [ ] Create `CompositeMemoryManager` facade
- [ ] Update `create_application()`
- [ ] Migrate tests
- [ ] Code review and merge

### Phase 3: Bot Handlers
- [ ] Create `tbot/bot/` package
- [ ] Extract command handlers to separate modules
- [ ] Extract utilities to `bot/utils.py`
- [ ] Extract error handler
- [ ] Create `bot/application.py`
- [ ] Update imports
- [ ] Migrate tests
- [ ] Code review and merge

### Phase 4: LLM Provider
- [ ] Create `tbot/llm/` package
- [ ] Implement `OpenAIProvider`
- [ ] Implement `MockLLMProvider` for tests
- [ ] Refactor `LLMClient` to use provider
- [ ] Update `create_application()`
- [ ] Migrate tests
- [ ] Code review and merge

### Phase 5: Dependency Injection
- [ ] Update `create_application()` signature
- [ ] Create factory functions for defaults
- [ ] Update all handler constructors
- [ ] Update tests to use DI
- [ ] Code review and merge

### Post-Migration
- [ ] Update README with new architecture
- [ ] Update documentation
- [ ] Performance testing
- [ ] Deploy to staging
- [ ] Monitor for issues
- [ ] Deploy to production

---

## Conclusion

This plan provides a roadmap for improving tbot's modularity, type safety, and extensibility while maintaining backward compatibility. The phased approach minimizes risk and allows for incremental progress.

**Key Takeaways**:
1. **Protocols over Concrete Classes**: Use `typing.Protocol` to define interfaces
2. **Smaller, Focused Modules**: Split large files into cohesive components
3. **Dependency Injection**: Inject dependencies via protocols for testability
4. **Domain-Specific Exceptions**: Create exception hierarchy for better error handling
5. **Incremental Migration**: Implement changes phase by phase to reduce risk

**Next Steps**:
1. Review this plan and gather feedback
2. Prioritize phases based on business needs
3. Begin with Phase 1 (Foundation) to establish protocols
4. Iterate through subsequent phases

**Estimated Total Effort**: 10-15 days of focused development + testing + review

---

*Document created: 2025-11-08*
*Author: Claude (Sonnet 4.5)*
*Project: tbot - LLM-driven Telegram Persona Bot*
