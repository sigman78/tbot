"""CLI entry point for running the Telegram persona bot."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for the application.

    Args:
        verbose: If True, set log level to DEBUG, otherwise INFO
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(description="Run the persona Telegram bot")
    parser.add_argument(
        "--token",
        default=os.getenv("TELEGRAM_BOT_TOKEN"),
        help="Telegram bot token (or set TELEGRAM_BOT_TOKEN env var)",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("API_KEY"),
        help="OpenRouter-compatible API key (or set API_KEY env var)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose (DEBUG) logging",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point for the bot application."""
    from .bot import run, run_polling

    args = parse_args()
    setup_logging(verbose=args.verbose)

    logger = logging.getLogger(__name__)
    logger.info("Starting Telegram persona bot...")

    if not args.token:
        raise SystemExit(
            "Telegram bot token is required. Use --token or TELEGRAM_BOT_TOKEN."
        )
    if not args.api_key:
        raise SystemExit("API key is required. Use --api-key or API_KEY.")

    try:
        # asyncio.run(run_polling(args.token, api_key=args.api_key))
        run(args.token, api_key=args.api_key)
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Bot crashed with error: {e}", exc_info=True)
        raise SystemExit(1) from e


if __name__ == "__main__":
    main()
