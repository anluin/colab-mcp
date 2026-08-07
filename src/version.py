"""Package and pinned-upstream version reporting."""

from importlib.metadata import PackageNotFoundError, version


def package_version(distribution: str = "colab-mcp") -> str:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return "0+unknown"


COLAB_MCP_VERSION = package_version()
COLAB_CLI_VERSION = package_version("google-colab-cli")
