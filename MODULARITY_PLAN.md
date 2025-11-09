# TBot Modularity Improvement Plan

## Status
- Current: ✅ Type hints, ✅ Tests | ⚠️ No protocols, ⚠️ Tight coupling, ⚠️ Large modules
- Target: Protocol-based interfaces, dependency injection, modular architecture

## Issues

### 1. No Protocol Interfaces
- All dependencies are concrete classes (ConfigManager, MemoryManager, LLMClient)
- Hard to mock, swap implementations, or extend
- Files: `bot.py:162-178`, all modules

### 2. Large Modules
- `bot.py`: 800 lines (13 handlers, wiring, error handling, permissions)
- `memory.py`: 576 lines (history, memories, summaries, stats, opt-in, persistence)

### 3. Tight Coupling
- `LLMClient` hard-coded to OpenAI SDK
- `ConversationContextBuilder` coupled to `MemoryEntry`, `UserSummary`
- No abstraction layers

### 4. Missing Domain Exceptions
- Generic `ValueError`, `OpenAIError` used throughout
- Can't distinguish error types

## Proposed Protocols

```python
# tbot/interfaces.py

from typing import Protocol, List, runtime_checkable

@runtime_checkable
class ConfigStore(Protocol):
    @property
    def config(self) -> BotConfig: ...
    def load(self) -> BotConfig: ...
    def save(self) -> None: ...
    def update(self, **kwargs) -> BotConfig: ...

@runtime_checkable
class HistoryStore(Protocol):
    def append(self, chat_id: int, message: str) -> None: ...
    def get(self, chat_id: int, limit: int | None = None) -> List[str]: ...
    def clear_oldest(self, chat_id: int, count: int) -> None: ...

@runtime_checkable
class MemoryStore(Protocol):
    def add(self, chat_id: int, text: str) -> MemoryEntry: ...
    def get(self, chat_id: int) -> List[MemoryEntry]: ...
    def clear(self, chat_id: int) -> None: ...

@runtime_checkable
class SummaryStore(Protocol):
    def add(self, chat_id: int, username: str, summary: str) -> None: ...
    def get(self, chat_id: int) -> List[UserSummary]: ...
    def clear(self, chat_id: int) -> None: ...

@runtime_checkable
class StatisticsCollector(Protocol):
    def increment_replies(self, chat_id: int) -> None: ...
    def increment_reactions(self, chat_id: int) -> None: ...
    def increment_llm_calls(self, chat_id: int, tokens_sent: int = 0, tokens_received: int = 0) -> None: ...
    def get(self, chat_id: int) -> ChatStatistics: ...

@runtime_checkable
class OptInManager(Protocol):
    def add_user(self, chat_id: int, user_id: int) -> None: ...
    def is_opted_in(self, chat_id: int, user_id: int) -> bool: ...
    def set_optin_message(self, chat_id: int, message_id: int) -> None: ...

@runtime_checkable
class LLMProvider(Protocol):
    async def generate_reply(self, messages: List[ChatCompletionMessageParam], model: str, **kwargs) -> str: ...
    async def generate_summary(self, messages: List[ChatCompletionMessageParam], model: str) -> str: ...
    async def suggest_reaction(self, message: str, persona: str, model: str) -> str | None: ...
```

## Proposed Structure

```
tbot/
├── interfaces.py           # Protocol definitions
├── exceptions.py           # TBotError, ConfigError, LLMError, StorageError
├── models.py              # MemoryEntry, UserSummary, ChatStatistics
├── const.py
│
├── core/
│   ├── config.py          # ConfigManager (implements ConfigStore)
│   ├── decision.py        # should_respond()
│   └── context_builder.py # ContextBuilder
│
├── storage/
│   ├── history.py         # JsonHistoryStore
│   ├── memory.py          # JsonMemoryStore
│   ├── summary.py         # JsonSummaryStore
│   ├── statistics.py      # JsonStatisticsCollector
│   ├── optin.py           # JsonOptInManager
│   └── persistence.py     # File I/O utilities
│
├── llm/
│   ├── openai_provider.py # OpenAIProvider
│   ├── mock_provider.py   # MockLLMProvider (testing)
│   └── client.py          # High-level LLM client
│
└── bot/
    ├── application.py     # create_application() with DI
    ├── handlers/
    │   ├── config_handlers.py  # /config, /set, /persona
    │   ├── memory_handlers.py  # /memory, /forget, /summary
    │   ├── stats_handlers.py   # /stat
    │   ├── optin_handlers.py   # /ask_optin, reaction handler
    │   ├── help_handler.py     # /help
    │   └── message_handler.py  # maybe_reply
    ├── utils.py           # Truncation, parsing
    └── error_handler.py
```

## Refactoring Map

| Current | New | Change |
|---------|-----|--------|
| `bot.py` (800L) | `bot/application.py` (100L) + `bot/handlers/*` (6 files) | Split handlers |
| `memory.py` (576L) | `storage/{history,memory,summary,statistics,optin}.py` (5 files) | Split concerns |
| `llm_client.py` | `llm/openai_provider.py` + `llm/client.py` | Abstract provider |
| `config.py` | `core/config.py` | Move to core/ |
| `context.py` | `core/context_builder.py` | Move to core/ |
| `logic.py` | `core/decision.py` | Move to core/ |
| - | `interfaces.py` | New |
| - | `exceptions.py` | New |
| - | `models.py` | Extract dataclasses |

## Implementation Phases

### Phase 1: Foundation (2-3 days) ✅ Non-breaking
- [ ] Create `interfaces.py`, `exceptions.py`, `models.py`
- [ ] Add protocol conformance tests
- [ ] Documentation

### Phase 2: Storage (3-5 days)
- [ ] Create `storage/` package
- [ ] Extract 5 store implementations
- [ ] Create `CompositeMemoryManager` facade (backward compat)
- [ ] Update `create_application()`
- [ ] Migrate tests

### Phase 3: Bot Handlers (2-3 days)
- [ ] Create `bot/` package
- [ ] Extract 6 handler modules
- [ ] Extract utils, error handler
- [ ] Update imports

### Phase 4: LLM Provider (2-3 days)
- [ ] Create `llm/` package
- [ ] Implement `OpenAIProvider`
- [ ] Implement `MockLLMProvider`
- [ ] Refactor client to use provider

### Phase 5: Dependency Injection (1-2 days)
- [ ] Update `create_application()` to accept protocols
- [ ] Create factory functions
- [ ] Update handler constructors
- [ ] Update tests

**Total: 10-15 days**

## Benefits

- **Testability**: Mock via protocols, no I/O in unit tests
- **Extensibility**: Swap backends (PostgreSQL, Redis, Anthropic, local LLMs)
- **Maintainability**: 100-150 line modules vs 800 line files
- **Type Safety**: Protocol contracts verified by mypy/pyright

## Migration Strategy

Backward compatible via facade pattern:
```python
# Old code still works
memory_manager = MemoryManager()

# New code uses DI
app = create_application(
    token="...",
    history_store=PostgresHistoryStore(),
    llm_provider=AnthropicProvider(),
)
```

## Type Safety Enhancements

```toml
# pyproject.toml
[tool.mypy]
strict = true
disallow_untyped_defs = true
```

```python
# Use NewType for domain types
ChatId = NewType('ChatId', int)
UserId = NewType('UserId', int)

# Use Literal for constants
ChatType = Literal["private", "group", "supergroup"]
```

## Testing Strategy

```python
# Mock providers for testing
mock_llm = MockLLMProvider()
mock_llm.expect_reply("Hello!")

app = create_application(
    token="test",
    llm_provider=mock_llm,
    history_store=InMemoryHistoryStore(),
)
```

## Key Files to Modify

- `tbot/bot.py` → Split into `bot/application.py` + `bot/handlers/*`
- `tbot/memory.py` → Split into `storage/*`
- `tbot/llm_client.py` → Refactor to use `LLMProvider` protocol
- `tbot/__init__.py` → Update exports

## Risks & Mitigation

| Risk | Mitigation |
|------|------------|
| Breaking changes | Maintain facade for backward compat |
| Migration bugs | Comprehensive test coverage, incremental phases |
| Performance | Minimal overhead (I/O-bound app) |
| Complexity | Good docs, gradual rollout |

---

**Est. Effort**: 10-15 days | **Risk**: Low-Medium | **Impact**: High
