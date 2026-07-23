#!/usr/bin/env bash
# Publish sc-crop to PyPI.
#
# Usage:
#   bash publish.sh
#
# Credentials are read from ~/.pypirc (section [pypi], username=__token__),
# same as twine does by default -- nothing to pass on the command line.

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
if [[ ! -f "$HOME/.pypirc" ]] || ! grep -q '^\[pypi\]' "$HOME/.pypirc"; then
    echo "ERROR: ~/.pypirc with a [pypi] section is required (username=__token__, password=<your PyPI token>)."
    exit 1
fi

twine upload dist/*

echo "Done — sc-crop v${TOML_VERSION} published."
