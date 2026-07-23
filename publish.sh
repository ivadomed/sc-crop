#!/usr/bin/env bash
# Publish sc-crop to PyPI.
#
# Usage:
#   TWINE_TOKEN=pypi-xxx bash publish.sh
#
# The token is read from the TWINE_TOKEN environment variable — never pass it
# as a positional argument or store it in this file.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_DIR"

# ── version consistency check ────────────────────────────────────────────────
TOML_VERSION=$(grep '^version' pyproject.toml | head -1 | sed 's/.*= *"\(.*\)"/\1/')
INIT_VERSION=$(grep '^__version__' sc_crop/__init__.py | sed 's/.*"\(.*\)".*/\1/')

if [[ "$TOML_VERSION" != "$INIT_VERSION" ]]; then
    echo "ERROR: version mismatch — pyproject.toml=$TOML_VERSION, __init__.py=$INIT_VERSION"
    exit 1
fi

echo "Publishing sc-crop v${TOML_VERSION}"

# ── build (from a clean git archive, not the working directory) ─────────────
# The working directory can contain gitignored-but-present local files (e.g.
# cached model weights used for dev/testing) that `python -m build` would
# happily bundle into the wheel since it doesn't consult .gitignore. Building
# from `git archive HEAD` guarantees the artifact only ever contains what's
# actually committed, regardless of what else happens to sit in this checkout.
CLEAN_SRC="$(mktemp -d)"
trap 'rm -rf "$CLEAN_SRC"' EXIT
git archive HEAD | tar -x -C "$CLEAN_SRC"

rm -rf dist/
python -m build --outdir "$REPO_DIR/dist" "$CLEAN_SRC"

# ── check ────────────────────────────────────────────────────────────────────
twine check dist/*

# ── upload ───────────────────────────────────────────────────────────────────
if [[ -z "${TWINE_TOKEN:-}" ]]; then
    echo "ERROR: TWINE_TOKEN is not set. Run: TWINE_TOKEN=pypi-xxx bash publish.sh"
    exit 1
fi

TWINE_USERNAME=__token__ TWINE_PASSWORD="$TWINE_TOKEN" twine upload dist/*

echo "Done — sc-crop v${TOML_VERSION} published."
