"""Small cross-platform management CLI for Colab MCP."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Any

from .auth_setup import main as authenticate
from .manager import ColabManager
from .version import COLAB_CLI_VERSION, COLAB_MCP_VERSION


def server_command(uv: str, project: Path) -> list[str]:
    return [uv, "--directory", str(project), "run", "--locked", "colab-mcp", "serve"]


def isolated_server_command(uv: str, project: Path) -> list[str]:
    """Run without sharing the project's environment with another active MCP client."""
    return [
        uv,
        "--directory",
        str(project),
        "run",
        "--isolated",
        "--locked",
        "colab-mcp",
        "serve",
    ]


def install_command(client: str, executable: str, uv: str, project: Path, name: str) -> list[str]:
    # Grok shares the host with other MCP clients (often Codex). Use an isolated
    # uv environment so a long-running project-env session cannot lock Windows
    # console-script entry points and break Grok startup.
    if client == "grok":
        server = isolated_server_command(uv, project)
        return [
            executable,
            "mcp",
            "add",
            "--scope",
            "user",
            name,
            "-e",
            "COLAB_MCP_AUTH=oauth2",
            "--",
            *server,
        ]
    server = server_command(uv, project)
    if client == "codex":
        return [
            executable,
            "mcp",
            "add",
            name,
            "--env",
            "COLAB_MCP_AUTH=oauth2",
            "--",
            *server,
        ]
    if client == "claude":
        return [
            executable,
            "mcp",
            "add",
            "--transport",
            "stdio",
            "--scope",
            "user",
            "--env",
            "COLAB_MCP_AUTH=oauth2",
            name,
            "--",
            *server,
        ]
    raise ValueError(f"Unsupported client: {client}")


def config_json(uv: str, project: Path, name: str = "colab") -> str:
    command = server_command(uv, project)
    return json.dumps(
        {
            "mcpServers": {
                name: {
                    "command": command[0],
                    "args": command[1:],
                    "env": {"COLAB_MCP_AUTH": "oauth2"},
                }
            }
        },
        indent=2,
    )


def _run(command: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, text=True, check=check)


def _project() -> Path:
    return Path(__file__).resolve().parents[1]


def _client_has_server(executable: str, client: str, name: str) -> bool:
    """Return True when the named MCP server is already registered with the client."""
    if client == "grok":
        # Grok has no `mcp get`; use machine-readable list instead.
        listed = subprocess.run(
            [executable, "mcp", "list", "--json"],
            text=True,
            capture_output=True,
            check=False,
        )
        if listed.returncode != 0:
            return False
        try:
            payload = json.loads(listed.stdout)
        except json.JSONDecodeError:
            return False
        if not isinstance(payload, list):
            return False
        return any(isinstance(item, dict) and item.get("name") == name for item in payload)
    existing = _run([executable, "mcp", "get", name], check=False)
    return existing.returncode == 0


def claude_desktop_config_path(
    system: str | None = None,
    home: Path | None = None,
    appdata: str | None = None,
    localappdata: str | None = None,
) -> Path:
    system = system or platform.system()
    home = home or Path.home()
    if system == "Windows":
        local_value = localappdata or os.environ.get("LOCALAPPDATA")
        if local_value:
            packages = Path(local_value) / "Packages"
            packaged_configs = sorted(
                packages.glob("Claude_*/LocalCache/Roaming/Claude/claude_desktop_config.json")
            )
            if packaged_configs:
                return max(packaged_configs, key=lambda path: path.stat().st_mtime)
        appdata_value = appdata or os.environ.get("APPDATA")
        root = Path(appdata_value) if appdata_value else home / "AppData" / "Roaming"
        return root / "Claude" / "claude_desktop_config.json"
    if system == "Darwin":
        return home / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json"
    return home / ".config" / "Claude" / "claude_desktop_config.json"


def install_claude_desktop(name: str, force: bool) -> None:
    path = claude_desktop_config_path()
    if path.exists():
        try:
            config = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise SystemExit(f"Refusing to modify invalid Claude Desktop JSON: {path}") from error
        if not isinstance(config, dict):
            raise SystemExit(f"Refusing to modify non-object Claude Desktop config: {path}")
    else:
        config = {}
    servers = config.setdefault("mcpServers", {})
    if not isinstance(servers, dict):
        raise SystemExit(f"Claude Desktop mcpServers must be an object: {path}")
    if name in servers and not force:
        print(f"MCP server {name!r} is already registered with Claude Desktop; nothing changed.")
        return
    command = isolated_server_command(_uv(), _project())
    servers[name] = {
        "command": command[0],
        "args": command[1:],
        "env": {"COLAB_MCP_AUTH": "oauth2"},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    print(f"Registered MCP server {name!r} with Claude Desktop. Restart Claude Desktop.")


def _uv() -> str:
    executable = shutil.which("uv")
    if not executable:
        raise SystemExit("uv was not found: https://docs.astral.sh/uv/")
    return executable


def install(client: str, name: str, force: bool) -> None:
    if client == "claude-desktop":
        install_claude_desktop(name, force)
        return
    uv = _uv()
    project = _project()
    if client == "json":
        print(config_json(uv, project, name))
        return
    executable = shutil.which(client)
    if not executable:
        raise SystemExit(f"{client!r} was not found on PATH")
    if _client_has_server(executable, client, name):
        if not force:
            print(f"MCP server {name!r} is already registered with {client}; nothing changed.")
            return
        _run([executable, "mcp", "remove", name])
    _run(install_command(client, executable, uv, project, name))
    restart_hint = {
        "grok": "Restart Grok, open /mcps, and call colab_health.",
        "codex": "Restart the client, then run /mcp.",
        "claude": "Restart the client, then run /mcp.",
    }.get(client, "Restart the client, then run /mcp.")
    print(f"Registered MCP server {name!r} with {client}. {restart_hint}")


def doctor(live: bool = False) -> None:
    manager = ColabManager()
    report: dict[str, Any] = {
        "oauth_token_present": manager.authenticated,
        "uv": shutil.which("uv"),
        "codex": shutil.which("codex"),
        "claude": shutil.which("claude"),
        "grok": shutil.which("grok"),
        "server_command": server_command(_uv(), _project()),
        "isolated_server_command": isolated_server_command(_uv(), _project()),
        "colab_mcp_version": COLAB_MCP_VERSION,
        "google_colab_cli_version": COLAB_CLI_VERSION,
        "python_compatible": True,
        "python": platform.python_version(),
        "operating_system": platform.system(),
    }
    if live:
        try:
            report["live_assignment_count"] = len(manager.client().list_assignments())
            report["colab_api_reachable"] = True
        except Exception as error:
            report["colab_api_reachable"] = False
            report["colab_api_error"] = str(error)
    print(json.dumps(report, indent=2))
    if not manager.authenticated:
        raise SystemExit("Authentication is missing. Run `colab-mcp auth`.")


def main() -> None:
    parser = argparse.ArgumentParser(prog="colab-mcp")
    parser.add_argument("--version", action="version", version=f"%(prog)s {COLAB_MCP_VERSION}")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("auth", help="Complete or refresh interactive Google Colab OAuth")
    doctor_parser = commands.add_parser(
        "doctor", help="Check authentication and local client availability"
    )
    doctor_parser.add_argument(
        "--live", action="store_true", help="Also verify the authenticated Colab assignments API"
    )
    commands.add_parser("serve", help="Run the non-interactive MCP stdio server")

    install_parser = commands.add_parser("install", help="Register with an MCP client")
    install_parser.add_argument(
        "client", choices=["codex", "claude", "claude-desktop", "grok", "json"]
    )
    install_parser.add_argument("--name", default="colab")
    install_parser.add_argument("--force", action="store_true")

    setup_parser = commands.add_parser(
        "setup", help="Authenticate and register one or more clients"
    )
    setup_parser.add_argument(
        "clients", nargs="+", choices=["codex", "claude", "claude-desktop", "grok"]
    )
    setup_parser.add_argument("--name", default="colab")
    setup_parser.add_argument("--force", action="store_true")

    args = parser.parse_args()
    if args.command == "auth":
        authenticate()
    elif args.command == "doctor":
        doctor(args.live)
    elif args.command == "serve":
        from .server import main as serve

        serve()
    elif args.command == "install":
        install(args.client, args.name, args.force)
    elif args.command == "setup":
        authenticate()
        for client in args.clients:
            install(client, args.name, args.force)


if __name__ == "__main__":
    main()
