"""CLI entry point for running the Telegram persona bot."""

from __future__ import annotations

import logging
import os
import sys

import click


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


@click.command()
@click.option(
    "--token",
    envvar="TELEGRAM_BOT_TOKEN",
    required=True,
    help="Telegram bot token (or set TELEGRAM_BOT_TOKEN env var)",
)
@click.option(
    "--api-key",
    envvar="API_KEY",
    required=True,
    help="OpenRouter-compatible API key (or set API_KEY env var)",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose (DEBUG) logging",
)
@click.option(
    "--debug",
    is_flag=True,
    help="Enable raw LLM request logging to llm_requests.log",
)
def main(token: str, api_key: str, verbose: bool, debug: bool) -> None:
    """Run the persona Telegram bot."""
    from .bot import run
    from .llm_client import set_debug_logging

    setup_logging(verbose=verbose)

    # Configure LLM debug logging
    set_debug_logging(debug)

    logger = logging.getLogger(__name__)
    logger.info("Starting Telegram persona bot...")
    if debug:
        logger.info("LLM request logging enabled (llm_requests.log)")

    try:
        run(token, api_key=api_key)
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Bot crashed with error: {e}", exc_info=True)
        raise SystemExit(1) from e


if __name__ == "__main__":
    main()
