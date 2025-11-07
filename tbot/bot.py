"""Telegram bot wiring for the persona simulator."""

from __future__ import annotations

import asyncio
import logging
import os
import random

from telegram import Message, Update
from telegram.constants import ChatAction, ParseMode
from telegram.ext import (
    Application,
    CallbackContext,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from .config import BotConfig, ConfigManager
from .llm_client import LLMClient
from .logic import should_respond
from .memory import MemoryManager

logger = logging.getLogger(__name__)

# Telegram message length limit
MAX_MESSAGE_LENGTH = 4096


def _get_message(update: Update) -> Message | None:
    """Return the effective message for an update when available."""

    return update.effective_message


def _truncate_text(text: str, max_length: int = MAX_MESSAGE_LENGTH) -> str:
    """Truncate text to fit Telegram's message length limit.

    Args:
        text: Text to truncate
        max_length: Maximum length (default: 4096 for Telegram)

    Returns:
        Truncated text with ellipsis if needed
    """
    if len(text) <= max_length:
        return text

    # Leave room for truncation indicator
    truncate_msg = "\n\n... [Message truncated]"
    max_content = max_length - len(truncate_msg)
    return text[:max_content] + truncate_msg


async def _reply_with_config(update: Update, config: BotConfig) -> None:
    message = _get_message(update)
    if message is None:
        return

    # Truncate persona if it's too long
    persona_display = config.persona
    if len(persona_display) > 500:
        persona_display = persona_display[:500] + "..."

    text = (
        "<b>Persona bot configuration</b>\n"
        f"Persona: {persona_display}\n"
        f"Response frequency: {config.response_frequency:.2f}\n"
        f"Model: {config.llm_model}\n"
        f"Max context messages: {config.max_context_messages}\n"
        f"Auto-summarization: {'enabled' if config.auto_summarize_enabled else 'disabled'}\n"
        f"Summarize threshold: {config.summarize_threshold}\n"
    )
    text = _truncate_text(text)
    await message.reply_text(text, parse_mode=ParseMode.HTML)


def _parse_argument(update: Update) -> str:
    message = _get_message(update)
    if not message or not message.text:
        return ""
    parts = message.text.split(" ", 1)
    if len(parts) == 1:
        return ""
    return parts[1].strip()


async def _maybe_auto_summarize(
    *,
    chat_id: int,
    config: BotConfig,
    memory_manager: MemoryManager,
    llm_client: LLMClient,
) -> None:
    """Check if history should be summarized and perform summarization if needed.

    Args:
        chat_id: The chat to check
        config: Bot configuration
        memory_manager: Memory manager instance
        llm_client: LLM client for generating summaries
    """
    # Check if auto-summarization is enabled
    if not config.auto_summarize_enabled:
        return

    # Check if threshold is reached
    if not memory_manager.should_summarize(chat_id, config.summarize_threshold):
        return

    logger.info(
        f"Chat {chat_id} reached threshold ({config.summarize_threshold}), "
        "triggering auto-summarization"
    )

    try:
        # Get messages to summarize
        messages_to_summarize, total_size = memory_manager.get_messages_for_summary(
            chat_id, config.summarize_batch_size
        )

        if not messages_to_summarize:
            logger.warning(f"No messages to summarize for chat {chat_id}")
            return

        logger.debug(
            f"Summarizing {len(messages_to_summarize)} oldest messages "
            f"(out of {total_size} total)"
        )

        # Generate summary using LLM
        summary = await llm_client.generate_summary(
            messages=messages_to_summarize,
            persona=config.persona,
            model=config.llm_model,
        )

        # Add summary to memories - ensure no unexpected formatting issues
        clean_summary = summary
        if clean_summary.startswith("Bot: "):
            clean_summary = clean_summary[5:]  # Remove any unexpected "Bot:" prefix
        memory_manager.add_memory(chat_id, f"[Auto-summary]: {clean_summary}")

        # Clear the summarized messages from history
        memory_manager.clear_summarized_messages(chat_id, len(messages_to_summarize))

        logger.info(
            f"Successfully summarized and stored {len(messages_to_summarize)} messages "
            f"for chat {chat_id}. Summary: {summary[:100]}..."
        )
    except Exception as e:
        # Log error but don't fail the whole conversation
        logger.error(f"Failed to auto-summarize chat {chat_id}: {e}", exc_info=True)


def create_application(
    token: str,
    *,
    api_key: str | None = None,
    config_manager: ConfigManager | None = None,
    memory_manager: MemoryManager | None = None,
    llm_client: LLMClient | None = None,
) -> Application:
    """Create the Telegram application with handlers wired in."""

    config_manager = config_manager or ConfigManager()
    memory_manager = memory_manager or MemoryManager()
    resolved_api_key = api_key or os.getenv("API_KEY")
    llm_client = llm_client or LLMClient.fromParams(api_key=resolved_api_key)

    application = Application.builder().token(token).build()

    async def handle_persona(
        update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        argument = _parse_argument(update)
        message = _get_message(update)
        if message is None:
            return
        if not argument:
            await message.reply_text("Usage: /persona <description>")
            return
        config_manager.set_field("persona", argument)
        await message.reply_text("Persona updated.")

    async def handle_frequency(
        update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        argument = _parse_argument(update)
        message = _get_message(update)
        if message is None:
            return
        try:
            value = float(argument)
        except ValueError:
            await message.reply_text("Usage: /frequency <0.0-1.0>")
            return
        config_manager.set_field("response_frequency", value)
        await message.reply_text(f"Response frequency set to {value:.2f}.")

    async def handle_prompt(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        argument = _parse_argument(update)
        message = _get_message(update)
        if message is None:
            return
        if not argument:
            await message.reply_text("Usage: /prompt <system prompt>")
            return
        config_manager.set_field("system_prompt", argument)
        await message.reply_text("System prompt updated.")

    async def handle_model(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        argument = _parse_argument(update)
        message = _get_message(update)
        if message is None:
            return
        if not argument:
            await message.reply_text("Usage: /model <model name>")
            return
        config_manager.set_field("llm_model", argument)
        await message.reply_text(f"Model set to {argument}.")

    async def handle_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await _reply_with_config(update, config_manager.config)

    async def handle_memory(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        argument = _parse_argument(update)
        message = _get_message(update)
        if message is None or update.effective_chat is None:
            return
        if not argument:
            await message.reply_text("Usage: /memory <add|clear|list> [text]")
            return
        chat_id = update.effective_chat.id
        parts = argument.split(" ", 1)
        action = parts[0].lower()
        payload = parts[1].strip() if len(parts) > 1 else ""
        if action == "add" and payload:
            entry = memory_manager.add_memory(chat_id, payload)
            await message.reply_text(f"Stored memory at {entry.created_at.isoformat()}")
        elif action == "clear":
            memory_manager.clear_memories(chat_id)
            await message.reply_text("Cleared memories for this chat.")
        elif action == "list":
            memories = memory_manager.get_memories(chat_id)
            if not memories:
                await message.reply_text("No memories stored yet.")
            else:
                lines = [f"- {m.text} ({m.created_at:%Y-%m-%d})" for m in memories]
                text = "\n".join(lines)
                text = _truncate_text(text)
                await message.reply_text(text)
        else:
            await message.reply_text("Usage: /memory <add|clear|list> [text]")

    async def handle_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        message = _get_message(update)
        if message is None:
            return
        await message.reply_text(
            "Available commands:\n"
            "/persona <text> - set persona\n"
            "/frequency <0-1> - adjust reply probability\n"
            "/prompt <text> - set system prompt\n"
            "/model <name> - set OpenRouter model\n"
            "/memory add|list|clear - manage memories\n"
            "/status - show configuration"
        )

    async def maybe_reply(update: Update, context: CallbackContext) -> None:
        message = _get_message(update)
        if message is None or update.effective_chat is None:
            return
        chat_id = update.effective_chat.id
        text = message.text or ""
        if not text:
            return

        # Ensure consistent formatting for user messages in history
        user_name = "User"
        if update.effective_user and update.effective_user.first_name:
            user_name = update.effective_user.first_name
        memory_manager.append_history(
            chat_id,
            f"{user_name}: {text}",
        )

        config = config_manager.config
        bot_user = context.bot if hasattr(context, "bot") else None
        replied_to_bot = False
        mentioned_bot = False

        if message.reply_to_message and bot_user:
            replied_to = message.reply_to_message.from_user
            replied_to_bot = replied_to.id == bot_user.id if replied_to else False

        # Check if the bot is mentioned in the message
        if bot_user:
            bot_username = bot_user.username
            bot_first_name = getattr(bot_user, "first_name", "")

            # Check for @username mention or first name mention
            if (bot_username and f"@{bot_username}" in text) or (
                bot_first_name and bot_first_name in text
            ):
                mentioned_bot = True
                logger.debug(f"Bot was mentioned in chat {chat_id}")

        # Detect if this is a private 1-on-1 chat
        is_private_chat = update.effective_chat.type == "private"

        # Try to add a reaction if enabled
        if config.reactions_enabled and random.random() <= config.reaction_frequency:
            try:
                # Check if we're in a group chat and have necessary permissions
                can_set_reactions = True
                if update.effective_chat.type in ["group", "supergroup"]:
                    try:
                        bot_member = await context.bot.get_chat_member(
                            chat_id, context.bot.id
                        )
                        # Different Telegram versions have different permission structures
                        # Try to be as permissive as possible to avoid errors
                        if (
                            hasattr(bot_member, "can_send_messages")
                            and not bot_member.can_send_messages
                        ) or (
                            hasattr(bot_member, "status")
                            and bot_member.status not in ["administrator", "creator"]
                        ):
                            can_set_reactions = False
                            logger.debug(
                                f"Bot lacks permissions to set reactions in chat {chat_id}"
                            )
                    except Exception as perm_error:
                        logger.debug(
                            f"Failed to check permissions in chat {chat_id}: {perm_error}"
                        )
                        can_set_reactions = False

                if can_set_reactions:
                    reaction = await llm_client.suggest_reaction(
                        message=text,
                        persona=config.persona,
                        model=config.llm_model,
                    )
                    if reaction:
                        await message.set_reaction(reaction)
                        logger.debug(
                            f"Set reaction {reaction} on message in chat {chat_id}"
                        )
            except Exception as e:
                # Reactions are non-critical, just log and continue
                logger.debug(f"Failed to set reaction in chat {chat_id}: {e}")

        should_reply = should_respond(
            random_value=random.random(),
            response_frequency=config.response_frequency,
            replied_to_bot=replied_to_bot,
            is_private_chat=is_private_chat,
            mentioned_bot=mentioned_bot,
        )

        if not should_reply:
            return

        # Notify - something brewing
        await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
        history = memory_manager.get_history(chat_id, config.max_context_messages)
        memories = memory_manager.get_memories(chat_id)

        try:
            reply = await llm_client.generate_reply(
                config=config,
                history=history,
                memories=memories,
                user_message=text,
            )
            # Check if we can send messages in this chat (for groups)
            can_send_messages = True
            if update.effective_chat.type in ["group", "supergroup"]:
                try:
                    bot_member = await context.bot.get_chat_member(
                        chat_id, context.bot.id
                    )
                    if (
                        hasattr(bot_member, "can_send_messages")
                        and not bot_member.can_send_messages
                    ):
                        can_send_messages = False
                        logger.warning(
                            f"Bot doesn't have permission to send messages in chat {chat_id}"
                        )
                except Exception as perm_error:
                    logger.debug(
                        f"Failed to check send permissions in chat {chat_id}: {perm_error}"
                    )

            if can_send_messages:
                # Make sure the response doesn't contain the "Bot:" prefix
                clean_reply = reply
                if clean_reply.startswith("Bot: "):
                    clean_reply = clean_reply[5:]  # Remove "Bot: " prefix

                # Send the clean reply
                await message.reply_text(
                    _truncate_text(clean_reply),
                    parse_mode=ParseMode.MARKDOWN,
                )

                # Store full reply in history with prefix (for proper history processing)
                memory_manager.append_history(chat_id, f"Bot: {clean_reply}")
                logger.info(f"Successfully replied to message in chat {chat_id}")
            else:
                logger.warning(
                    f"Skipped sending reply in chat {chat_id} due to insufficient permissions"
                )

            # Check if auto-summarization should be triggered
            await _maybe_auto_summarize(
                chat_id=chat_id,
                config=config,
                memory_manager=memory_manager,
                llm_client=llm_client,
            )
        except Exception as e:
            logger.error(f"Failed to generate reply for chat {chat_id}: {e}")
            error_message = (
                "Sorry, I encountered an error generating a response. "
                "Please try again later."
            )
            await message.reply_text(error_message)

    async def error_handler(update: object, context: CallbackContext) -> None:
        """Handle errors in the telegram bot."""
        # Get detailed error information
        error = context.error
        error_str = str(error)

        # More detailed logging for common Telegram API errors
        if "Bad Request" in error_str:
            if "Not enough rights" in error_str or "permission" in error_str.lower():
                logger.error(
                    f"Permission error: {error_str}. Bot likely lacks necessary permissions in the chat."
                )
            elif "bot was blocked" in error_str.lower():
                logger.error(f"Bot was blocked by the user: {error_str}")
            elif "Chat not found" in error_str:
                logger.error(f"Chat not found error: {error_str}")
            elif "message is not modified" in error_str.lower():
                logger.debug(f"Harmless error - message not modified: {error_str}")
                return  # Skip user notification for this error
            else:
                logger.error(f"Telegram API error: {error_str}")
        else:
            logger.error(f"Update {update} caused error {error}", exc_info=error)

        # Try to notify the user if possible and appropriate
        if isinstance(update, Update) and update.effective_message:
            try:
                # Check if we can reply in this chat before attempting
                can_reply = True
                if update.effective_chat and update.effective_chat.type in [
                    "group",
                    "supergroup",
                ]:
                    try:
                        bot_member = await context.bot.get_chat_member(
                            update.effective_chat.id, context.bot.id
                        )
                        if (
                            hasattr(bot_member, "can_send_messages")
                            and not bot_member.can_send_messages
                        ):
                            can_reply = False
                    except Exception:
                        can_reply = False

                if can_reply:
                    error_text = (
                        "Sorry, an error occurred while processing your request."
                    )
                    await update.effective_message.reply_text(error_text)
            except Exception as notify_error:
                logger.debug(f"Failed to send error notification: {notify_error}")

    # Log bot info on startup to help with troubleshooting
    async def log_bot_info(application: Application) -> None:
        """Log information about the bot on startup."""
        try:
            bot = application.bot
            bot_info = await bot.get_me()
            logger.info(f"Bot initialized: @{bot_info.username} (ID: {bot_info.id})")
            logger.info(f"Bot name: {bot_info.first_name}")

            # Log warning about group privacy mode
            logger.info(
                "IMPORTANT: To work properly in groups, this bot should have Group Privacy Mode disabled "
                "in BotFather settings (/mybots → Select bot → Bot Settings → Group Privacy)"
            )
        except Exception as e:
            logger.warning(f"Failed to log bot information: {e}")

    # Register the startup info logger
    application.post_init = log_bot_info

    application.add_handler(CommandHandler("persona", handle_persona))
    application.add_handler(CommandHandler("frequency", handle_frequency))
    application.add_handler(CommandHandler("prompt", handle_prompt))
    application.add_handler(CommandHandler("model", handle_model))
    application.add_handler(CommandHandler("status", handle_status))
    application.add_handler(CommandHandler("memory", handle_memory))
    application.add_handler(CommandHandler("help", handle_help))
    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, maybe_reply)
    )

    # Add error handler
    application.add_error_handler(error_handler)

    return application


def run(token: str, *, api_key: str | None = None) -> None:
    """Helper entry point for manual runs."""
    application = create_application(token, api_key=api_key)
    if application.updater is None:
        raise RuntimeError("Application was created without an updater")
    application.run_polling()


async def run_polling(token: str, *, api_key: str | None = None) -> None:
    """Helper entry point for manual runs."""
    application = create_application(token, api_key=api_key)
    if application.updater is None:
        raise RuntimeError("Application was created without an updater")

    async with application:
        await application.start()
        await application.updater.start_polling()
        try:
            await asyncio.Event().wait()
        finally:
            await application.updater.stop()
            await application.stop()
