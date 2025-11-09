FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim
COPY . /app
WORKDIR /app
RUN uv sync --frozen --no-dev
CMD ["python", "-m", "tbot", "--token", "${TELEGRAM_BOT_TOKEN}", "--api-key", "${API_KEY}"]
