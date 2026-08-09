#!/usr/bin/env sh
set -eu

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required: https://docs.astral.sh/uv/getting-started/installation/" >&2
  exit 1
fi

# Default client is Codex. Pass clients as arguments, e.g.:
#   sh scripts/install.sh grok
#   sh scripts/install.sh codex grok
cd "$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
if [ "$#" -eq 0 ]; then
  set -- codex
fi
exec uv run --locked colab-mcp setup "$@"
