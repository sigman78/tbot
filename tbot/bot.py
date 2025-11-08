"""Telegram bot wiring for the persona simulator."""

from __future__ import annotations

import asyncio
import logging
import os
import random

from telegram import Message, Update
from telegram.constants import ChatAction, ChatMemberStatus, ParseMode
from telegram.ext import (
    Application,
    CallbackContext,
    ChatMemberHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    MessageReactionHandler,
    filters,
)

from .config import BotConfig, ConfigManager
from .const import POSITIVE_REACTIONS
from .context import ConversationContextBuilder
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

        # Generate per-user summaries using LLM
        user_summaries = await llm_client.generate_user_summaries(
            messages=messages_to_summarize,
            persona=config.persona,
            model=config.llm_model,
        )

        # Store each user's summary
        for username, summary in user_summaries.items():
            clean_summary = ConversationContextBuilder.strip_bot_prefix(summary)
            memory_manager.add_user_summary(chat_id, username, clean_summary)
            logger.debug(f"Stored summary for user {username}: {clean_summary[:50]}...")

        # Clear the summarized messages from history
        memory_manager.clear_summarized_messages(chat_id, len(messages_to_summarize))

        logger.info(
            f"Successfully summarized and stored {len(messages_to_summarize)} messages "
            f"for chat {chat_id}. Generated summaries for {len(user_summaries)} users."
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
    # Create memory manager with max_summarized_users from config
    if memory_manager is None:
        config = config_manager.config
        memory_manager = MemoryManager(max_summarized_users=config.max_summarized_users)
    resolved_api_key = api_key or os.getenv("API_KEY")
    llm_client = llm_client or LLMClient.fromParams(api_key=resolved_api_key)

    application = Application.builder().token(token).build()

    async def handle_persona(
        update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Display the first part of the persona card."""
        message = _get_message(update)
        if message is None:
            return
        config = config_manager.config
        # Show first 500 characters of persona
        persona_preview = config.persona
        if len(persona_preview) > 500:
            persona_preview = persona_preview[:500] + "..."

        text = f"<b>Current Persona:</b>\n{persona_preview}"
        text = _truncate_text(text)
        await message.reply_text(text, parse_mode=ParseMode.HTML)

    async def handle_config(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """List all current tunable parameters."""
        message = _get_message(update)
        if message is None:
            return
        config = config_manager.config

        text = (
            "<b>Current Configuration:</b>\n"
            f"• response_frequency: {config.response_frequency:.2f}\n"
            f"• llm_model: {config.llm_model}\n"
            f"• max_context_messages: {config.max_context_messages}\n"
            f"• auto_summarize_enabled: {config.auto_summarize_enabled}\n"
            f"• summarize_threshold: {config.summarize_threshold}\n"
            f"• summarize_batch_size: {config.summarize_batch_size}\n"
            f"• max_summarized_users: {config.max_summarized_users}\n"
            f"• reactions_enabled: {config.reactions_enabled}\n"
            f"• reaction_frequency: {config.reaction_frequency:.2f}\n"
            f"• explicit_optin: {config.explicit_optin}\n"
            "\nUse /set &lt;param&gt; &lt;value&gt; to change a parameter."
        )
        text = _truncate_text(text)
        await message.reply_text(text, parse_mode=ParseMode.HTML)

    async def handle_set(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Set a configuration parameter."""
        argument = _parse_argument(update)
        message = _get_message(update)
        if message is None:
            return

        if not argument:
            await message.reply_text(
                "Usage: /set <param> <value>\n"
                "Available parameters: response_frequency, llm_model, max_context_messages, "
                "auto_summarize_enabled, summarize_threshold, summarize_batch_size, "
                "max_summarized_users, reactions_enabled, reaction_frequency, "
                "explicit_optin, persona, system_prompt"
            )
            return

        parts = argument.split(None, 1)
        if len(parts) < 2:
            await message.reply_text("Usage: /set <param> <value>")
            return

        param, value_str = parts

        # Convert value to appropriate type
        try:
            if param in ["response_frequency", "reaction_frequency"]:
                value = float(value_str)
            elif param in [
                "max_context_messages",
                "summarize_threshold",
                "summarize_batch_size",
                "max_summarized_users",
            ]:
                value = int(value_str)
            elif param in ["auto_summarize_enabled", "reactions_enabled", "explicit_optin"]:
                value = value_str.lower() in ["true", "1", "yes", "on"]
            elif param in ["persona", "system_prompt", "llm_model"]:
                value = value_str
            else:
                await message.reply_text(f"Unknown parameter: {param}")
                return

            config_manager.set_field(param, value)
            await message.reply_text(f"✓ Set {param} = {value}")
        except (ValueError, KeyError) as e:
            await message.reply_text(f"Error setting {param}: {e}")

    async def handle_summary(
        update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Show current chat summary (user summaries)."""
        message = _get_message(update)
        if message is None or update.effective_chat is None:
            return
        chat_id = update.effective_chat.id

        user_summaries = memory_manager.get_user_summaries(chat_id)
        if not user_summaries:
            await message.reply_text("No conversation summaries yet.")
            return

        lines = ["<b>Conversation Summaries:</b>"]
        for summary in user_summaries:
            lines.append(f"\n<b>{summary.username}:</b> {summary.summary}")

        text = "\n".join(lines)
        text = _truncate_text(text)
        await message.reply_text(text, parse_mode=ParseMode.HTML)

    async def handle_forget(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Reset chat memory and summaries."""
        message = _get_message(update)
        if message is None or update.effective_chat is None:
            return
        chat_id = update.effective_chat.id

        # Clear history, memories, and summaries
        memory_manager._history.pop(chat_id, None)
        memory_manager.clear_memories(chat_id)
        memory_manager.clear_user_summaries(chat_id)
        memory_manager._mark_dirty()

        await message.reply_text("✓ Chat memory and summaries have been reset.")

    async def handle_stat(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Show runtime statistics for the current chat."""
        message = _get_message(update)
        if message is None or update.effective_chat is None:
            return
        chat_id = update.effective_chat.id

        stats = memory_manager.get_statistics(chat_id)

        text = (
            "<b>Chat Statistics:</b>\n"
            f"• Replies: {stats.replies}\n"
            f"• Reactions: {stats.reactions}\n"
            f"• LLM calls: {stats.llm_calls}\n"
            f"• Tokens sent: {stats.tokens_sent}\n"
            f"• Tokens received: {stats.tokens_received}\n"
        )
        if stats.tokens_sent + stats.tokens_received > 0:
            text += f"• Total tokens: {stats.tokens_sent + stats.tokens_received}"

        text = _truncate_text(text)
        await message.reply_text(text, parse_mode=ParseMode.HTML)

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

    async def handle_bot_added_to_group(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle when bot is added to a group chat."""
        if update.my_chat_member is None or update.effective_chat is None:
            return

        chat_id = update.effective_chat.id
        config = config_manager.config

        # Only process for group chats
        if update.effective_chat.type not in ["group", "supergroup"]:
            return

        # Check if bot was added to the group
        new_status = update.my_chat_member.new_chat_member.status
        old_status = update.my_chat_member.old_chat_member.status

        # Bot was just added (transition from not-member to member/admin)
        if old_status in [ChatMemberStatus.LEFT, ChatMemberStatus.BANNED] and \
           new_status in [ChatMemberStatus.MEMBER, ChatMemberStatus.ADMINISTRATOR]:

            # If explicit_optin is enabled, send opt-in request
            if config.explicit_optin:
                try:
                    optin_message = await context.bot.send_message(
                        chat_id=chat_id,
                        text=(
                            "👋 Hello! I've been added to this group.\n\n"
                            "React to this message with a positive reaction (👍, ❤️, etc.) to opt-in to interactions with me.\n\n"
                            "⚠️ Privacy: I will have access only to messages of users who agreed to opt-in. "
                            "Messages from users who don't opt-in will not be read, stored, or processed by me."
                        )
                    )

                    # Store the message ID for tracking reactions
                    memory_manager.set_optin_message_id(chat_id, optin_message.message_id)
                    logger.info(f"Bot added to group {chat_id}, sent opt-in request")
                except Exception as e:
                    logger.error(f"Failed to send opt-in message in chat {chat_id}: {e}")

    async def handle_reaction(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle message reactions to track opt-ins."""
        if update.message_reaction is None or update.effective_chat is None:
            return

        chat_id = update.effective_chat.id
        config = config_manager.config

        # Only process reactions if explicit_optin is enabled
        if not config.explicit_optin:
            return

        # Only process reactions in group chats
        if update.effective_chat.type not in ["group", "supergroup"]:
            return

        message_id = update.message_reaction.message_id
        user = update.message_reaction.user

        # Check if this is a reaction to the opt-in message
        optin_message_id = memory_manager.get_optin_message_id(chat_id)
        if optin_message_id is None or message_id != optin_message_id:
            return

        # Check if user reacted with a positive reaction
        new_reactions = update.message_reaction.new_reaction
        if not new_reactions:
            return

        # Check if any of the new reactions are positive
        for reaction in new_reactions:
            # Handle both emoji reactions and custom emoji reactions
            emoji = None
            if hasattr(reaction, "emoji"):
                emoji = reaction.emoji
            elif hasattr(reaction, "type") and reaction.type == "emoji":
                emoji = getattr(reaction, "emoji", None)

            if emoji and emoji in POSITIVE_REACTIONS:
                # Add user to opt-in list
                if user and user.id:
                    memory_manager.add_optin_user(chat_id, user.id)
                    logger.info(f"User {user.id} ({user.first_name}) opted in to chat {chat_id}")
                break

    async def handle_ask_optin(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Request users to opt-in to bot interactions."""
        message = _get_message(update)
        if message is None or update.effective_chat is None:
            return

        chat_id = update.effective_chat.id
        config = config_manager.config

        # Only works in group chats
        if update.effective_chat.type not in ["group", "supergroup"]:
            await message.reply_text(
                "This command is only available in group chats."
            )
            return

        # Send opt-in request message
        optin_message = await message.reply_text(
            "👋 React to this message with a positive reaction (👍, ❤️, etc.) to opt-in to interactions with this bot.\n\n"
            "⚠️ Privacy: I will have access only to messages of users who agreed to opt-in. "
            "Messages from users who don't opt-in will not be read, stored, or processed by me."
        )

        # Store the message ID for tracking reactions
        memory_manager.set_optin_message_id(chat_id, optin_message.message_id)
        logger.info(f"Sent opt-in request in chat {chat_id}")

    async def handle_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        message = _get_message(update)
        if message is None:
            return

        config = config_manager.config
        help_text = (
            "<b>Available Commands:</b>\n\n"
            "/persona - Show current persona\n"
            "/config - List all tunable parameters\n"
            "/set &lt;param&gt; &lt;value&gt; - Set a config parameter\n"
            "/summary - Show current chat summaries\n"
            "/forget - Reset chat memory and summaries\n"
            "/stat - Show runtime statistics\n"
            "/memory add|list|clear - Manage long-term memories\n"
        )

        # Add opt-in command to help if explicit_optin is enabled
        if config.explicit_optin:
            help_text += "/ask_optin - Request users to opt-in\n"

        help_text += "/help - Show this help message"

        await message.reply_text(help_text, parse_mode=ParseMode.HTML)

    async def maybe_reply(update: Update, context: CallbackContext) -> None:
        message = _get_message(update)
        if message is None or update.effective_chat is None:
            return
        chat_id = update.effective_chat.id
        text = message.text or ""
        if not text:
            return

        config = config_manager.config

        # Check explicit opt-in if enabled in group chats
        is_group_chat = update.effective_chat.type in ["group", "supergroup"]
        if config.explicit_optin and is_group_chat:
            # Check if user has opted in
            user_id = update.effective_user.id if update.effective_user else None
            if user_id and not memory_manager.is_user_opted_in(chat_id, user_id):
                # User has not opted in, don't process their message
                logger.debug(f"User {user_id} has not opted in to chat {chat_id}, ignoring message")
                return

        # Ensure consistent formatting for user messages in history
        user_name = "User"
        if update.effective_user and update.effective_user.first_name:
            user_name = update.effective_user.first_name
        memory_manager.append_history(
            chat_id,
            ConversationContextBuilder.format_user_message(user_name, text),
        )
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
                        memory_manager.increment_reaction_count(chat_id)
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
        user_summaries = memory_manager.get_user_summaries(chat_id)

        try:
            reply = await llm_client.generate_reply(
                config=config,
                history=history,
                memories=memories,
                user_message=text,
                is_group_chat=not is_private_chat,
                user_summaries=user_summaries,
            )
            # Track LLM call in statistics
            memory_manager.increment_llm_call_count(chat_id)

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
                clean_reply = ConversationContextBuilder.strip_bot_prefix(reply)

                # Send the clean reply
                await message.reply_text(
                    _truncate_text(clean_reply),
                    parse_mode=ParseMode.MARKDOWN,
                )

                # Store full reply in history with prefix (for proper history processing)
                memory_manager.append_history(
                    chat_id,
                    ConversationContextBuilder.format_bot_message(clean_reply),
                )
                # Track reply in statistics
                memory_manager.increment_reply_count(chat_id)
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

    # Register command handlers
    application.add_handler(CommandHandler("persona", handle_persona))
    application.add_handler(CommandHandler("config", handle_config))
    application.add_handler(CommandHandler("set", handle_set))
    application.add_handler(CommandHandler("summary", handle_summary))
    application.add_handler(CommandHandler("forget", handle_forget))
    application.add_handler(CommandHandler("stat", handle_stat))
    application.add_handler(CommandHandler("memory", handle_memory))
    application.add_handler(CommandHandler("ask_optin", handle_ask_optin))
    application.add_handler(CommandHandler("help", handle_help))

    # Register reaction handler for opt-in tracking
    application.add_handler(MessageReactionHandler(handle_reaction))

    # Register chat member handler for bot being added to groups
    application.add_handler(ChatMemberHandler(handle_bot_added_to_group, ChatMemberHandler.MY_CHAT_MEMBER))

    # Register message handler
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
