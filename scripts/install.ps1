$ErrorActionPreference = "Stop"

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw "uv is required: https://docs.astral.sh/uv/getting-started/installation/"
}

# Default client is Codex. Pass clients as arguments, e.g.:
#   .\scripts\install.ps1 grok
#   .\scripts\install.ps1 codex grok
$Clients = if ($args.Count -gt 0) { $args } else { @("codex") }
$ProjectDirectory = Split-Path -Parent $PSScriptRoot
& uv --directory $ProjectDirectory run --locked colab-mcp setup @Clients
exit $LASTEXITCODE
