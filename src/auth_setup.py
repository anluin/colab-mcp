"""One-time interactive OAuth bootstrap, deliberately separate from MCP stdio."""

import os

from colab_cli.auth import AuthProvider, get_credentials


def main() -> None:
    provider = AuthProvider(os.environ.get("COLAB_MCP_AUTH", "oauth2"))
    config = os.environ.get("COLAB_MCP_OAUTH_CONFIG")
    get_credentials(config, provider)
    print("Colab authentication is ready.")


if __name__ == "__main__":
    main()
