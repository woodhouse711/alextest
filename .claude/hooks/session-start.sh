#!/bin/bash
set -euo pipefail

# Only run in Claude Code on the web (remote environment)
if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

REPO_DIR="${CLAUDE_PROJECT_DIR:-$(git -C "$(dirname "$0")" rev-parse --show-toplevel)}"
REQUIREMENTS="$REPO_DIR/field_atlas/requirements.txt"

echo "Installing Field Atlas dependencies..."
pip install --quiet -r "$REQUIREMENTS"

echo "Installing dev tools..."
pip install --quiet pytest flake8

echo "Setup complete."
