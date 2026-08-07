$ErrorActionPreference = "Stop"

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw "uv is required: https://docs.astral.sh/uv/getting-started/installation/"
}

$ProjectDirectory = Split-Path -Parent $PSScriptRoot
& uv --directory $ProjectDirectory run --locked colab-mcp setup codex @args
exit $LASTEXITCODE
