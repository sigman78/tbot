# Telegram Persona Bot

Simple Telegram bot that role-plays as a configurable persona via OpenAI API compatible service.

## Features

- Smart replies in private/group chats with configurable frequency
- Privacy mode: explicit opt-in required for group interactions
- LLM-powered emoji reactions based on context
- Auto-summarization of conversation history
- Persistent memory and statistics across restarts
- Comprehensive Telegram commands for configuration

## Quick Start

```bash
# Install dependencies
uv sync

# Run bot (choose one method)
uv run tbot-cli --token <telegram-bot-token> --api-key <openai-api-key>
# OR with environment variables
export TELEGRAM_BOT_TOKEN="<token>" && export API_KEY="<key>" && uv run tbot-cli
# OR create .env file with TELEGRAM_BOT_TOKEN and API_KEY variables
```

## Development

```bash
# Setup dev environment
uv sync --dev

# Run tests
uv run pytest

# Run bot with debug logging
uv run tbot-cli --token <token> --api-key <key> --verbose
```

## Configuration

Config stored in `~/.tbot-config.json`. Key settings:
- `response_frequency`: Reply probability in groups (0.0-1.0)
- `persona`: Bot personality description
- `llm_model`: OpenRouter model to use
- `explicit_optin`: Privacy mode requiring user opt-in

## Telegram Commands

`/config` - Show settings
`/set <param> <value>` - Update setting
`/persona` - Show persona
`/summary` - Chat summaries
`/forget` - Reset memory
`/stat` - Usage statistics
`/help` - Command help

## Contributing

1. Fork and clone
2. `uv sync --dev`
3. Make changes
4. `uv run pytest`
5. Submit PR
