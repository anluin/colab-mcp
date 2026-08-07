#!/usr/bin/env sh
set -eu

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required: https://docs.astral.sh/uv/getting-started/installation/" >&2
  exit 1
fi

cd "$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
exec uv run --locked colab-mcp setup codex "$@"
