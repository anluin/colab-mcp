# Contributing

Use Python 3.12+ and `uv`. Keep the MCP workload-agnostic and isolate all Colab-specific behavior
behind the manager/remote adapter boundary.

```bash
uv sync --locked --all-groups
uv run ruff format .
uv run ruff check .
uv run mypy src
uv run pytest -q
uv run --with bandit bandit -q -lll -r src scripts
uv run --with pip-audit pip-audit
uv build
uv run twine check dist/*
```

Add happy-path and failure-path tests for behavior changes. Tests must be platform-neutral. A
Colab-facing change also needs the smallest practical live CPU/GPU validation, with release in a
`finally` block and confirmation that the allocated endpoint disappeared. Never commit OAuth
credentials, session state, endpoint URLs, runtime artifacts, or build outputs.

Workspace-sync changes must run `uv run python scripts/live_workspace_probe.py`. The probe verifies
a nested multi-file initial push, a changed/new/unchanged delta, multi-chunk binary transfer,
destination-only preservation, built-in VCS exclusions, pull-back SHA-256 equality, and release.

Use conventional commits and keep each commit cohesive. Update the README when a public tool,
schema, error, setup step, or limitation changes. Git history and release notes are the changelog.
