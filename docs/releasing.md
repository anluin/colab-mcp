# Releasing

1. Confirm `main` is clean, current with `origin/main`, and CI passes on Ubuntu, macOS, and Windows.
2. Run every command in `CONTRIBUTING.md` from `uv.lock` and perform the documented live CPU/GPU
   validation with zero project-owned assignments left afterward.
3. Audit tracked and ignored files for credentials, state, endpoints, build outputs, and temporary
   artifacts.
4. Update the project version and lockfile; Git history and release notes are the changelog.
5. Commit `chore(release): vX.Y.Z`, create annotated tag `vX.Y.Z`, then push the commit and tag.
6. Build from the tag, validate wheel and source distribution, and put an evidence-backed readiness
   report directly in the release notes covering platforms, test counts, live operations,
   limitations, and upstream risks. Do not call a release production-ready when a criterion is
   merely assumed.
