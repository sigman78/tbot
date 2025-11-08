# Telegram Persona Bot

A lightweight Python project showcasing an LLM-driven Telegram bot that role-plays as a configurable persona. The bot talks through [OpenRouter](https://openrouter.ai) using an OpenAI-compatible API and keeps short-term memories for each chat.

## Features

- **Smart reply logic**: Always responds in private 1-on-1 chats, uses configurable frequency in groups
- **Emoji reactions**: LLM-powered emoji reactions to messages based on persona and context
- **Auto-summarization**: Automatically summarizes old conversations and stores per-user summaries
- **Runtime statistics**: Tracks replies, reactions, LLM calls, and token usage per chat
- **Persistent storage**: Memories, history, summaries, and statistics saved to disk and restored on restart
- **Unified configuration**: Clean CLI with click, comprehensive Telegram commands, and JSON config
- Per-chat memory and history tracking with automatic management
- Extensible LLM wrapper for OpenRouter-compatible models
- Async implementation powered by `python-telegram-bot` v20
- Comprehensive error handling and debug logging

## Setup

1. Install dependencies (preferably in a virtual environment):

   ```bash
   cd /path/to/repo/tbot
   pip install -r requirements-dev.txt
   ```

2. Export your credentials:

   ```bash
   export TELEGRAM_BOT_TOKEN="<telegram-token>"
   export API_KEY="<openrouter-api-key>"
   ```

3. Run the bot:

   ```bash
   python -m tbot --token "$TELEGRAM_BOT_TOKEN" --api-key "$API_KEY"
   ```

   CLI options:
   - `--token` - Telegram bot token (or use `TELEGRAM_BOT_TOKEN` env var)
   - `--api-key` - OpenRouter API key (or use `API_KEY` env var)
   - `--verbose` / `-v` - Enable verbose DEBUG logging
   - `--debug` - Enable raw LLM request logging to `llm_requests.log`

The bot stores its configuration in `~/.tbot-config.json` by default.

## Behavior

- **Private chats**: The bot always responds to messages in private 1-on-1 conversations
- **Group chats**: The bot uses the configured `response_frequency` to decide whether to respond
- **Direct replies**: The bot always responds when you reply to one of its messages (in any chat type)
- **Mentions**: The bot always responds when it's mentioned in a message (by @username or first name)

### Auto-summarization

The bot automatically manages conversation history per chat:

- **History tracking**: Each chat maintains its own separate conversation history
- **Automatic summarization**: When history reaches the threshold (default: 18 messages), the bot:
  1. Takes the oldest messages (default: 10 messages)
  2. Uses the LLM to generate a concise summary
  3. Stores the summary as a memory
  4. Removes the summarized messages from history to save space
- **Memory integration**: Summaries are automatically included in context for future conversations
- **Configuration**: Adjust `summarize_threshold` and `summarize_batch_size` in config, or disable with `auto_summarize_enabled: false`

This ensures the bot can maintain long-term context while keeping the conversation history manageable.

### Data Persistence

All chat data is automatically persisted to disk:

- **Storage location**: `~/.tbot-data.json` (configurable)
- **What's saved**: Memories, conversation history, per-user summaries, and runtime statistics for all chats
- **Auto-save**: Changes are immediately saved to disk after each modification
- **Auto-load**: Data is automatically loaded when the bot starts
- **Atomic writes**: Uses temporary files to prevent data corruption
- **Per-chat isolation**: Each chat's data is stored and restored independently

The bot maintains continuity across restarts, remembering past conversations, accumulated memories, and usage statistics.

## Configuration

The bot stores its configuration in `~/.tbot-config.json`. Here are the available settings:

| Setting | Default | Description |
|---------|---------|-------------|
| `response_frequency` | 0.4 | Probability (0.0-1.0) of responding in group chats |
| `persona` | "An affable research assistant..." | The bot's persona description |
| `system_prompt` | "You are role-playing..." | System prompt for the LLM |
| `llm_model` | "openai/gpt-4o-mini" | OpenRouter model to use |
| `max_context_messages` | 12 | Number of recent messages to include in LLM context |
| `auto_summarize_enabled` | true | Enable/disable automatic summarization |
| `summarize_threshold` | 18 | Number of messages that triggers summarization |
| `summarize_batch_size` | 10 | Number of oldest messages to summarize at once |
| `max_summarized_users` | 10 | Maximum number of users to keep summaries for |
| `reactions_enabled` | true | Enable/disable emoji reactions to messages |
| `reaction_frequency` | 0.3 | Probability (0.0-1.0) of adding a reaction to user messages |

You can edit the config file directly or use Telegram commands to update settings.

## Telegram Commands

Configure and monitor the bot directly from Telegram:

- `/persona` - Show current persona description
- `/config` - List all tunable parameters and their current values
- `/set <param> <value>` - Set any configuration parameter (e.g., `/set response_frequency 0.5`)
- `/summary` - Display current chat conversation summaries
- `/forget` - Reset chat memory, history, and summaries
- `/stat` - Show runtime statistics (replies, reactions, LLM calls, tokens used)
- `/memory add|list|clear` - Manage long-term memories
- `/help` - Show command help

