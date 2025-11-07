import sys

from tbot import main


def test_parse_args_accepts_cli_api_key(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "")
    monkeypatch.setenv("API_KEY", "")
    monkeypatch.setattr(sys, "argv", ["prog", "--token", "abc", "--api-key", "secret"])

    args = main.parse_args()

    assert args.token == "abc"
    assert args.api_key == "secret"


def test_parse_args_env_fallback(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "telegram-token")
    monkeypatch.setenv("API_KEY", "preferred-key")
    monkeypatch.setattr(sys, "argv", ["prog"])

    args = main.parse_args()

    assert args.token == "telegram-token"
    assert args.api_key == "preferred-key"
