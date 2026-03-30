#!/usr/bin/env bash
# Remove all Python and pytest cache artefacts from the project tree.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "Cleaning cache in: $PROJECT_ROOT"

find "$PROJECT_ROOT" -type d -name "__pycache__"  -exec rm -rf {} + 2>/dev/null || true
find "$PROJECT_ROOT" -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
find "$PROJECT_ROOT" -type d -name ".mypy_cache"   -exec rm -rf {} + 2>/dev/null || true
find "$PROJECT_ROOT" -type d -name ".ruff_cache"   -exec rm -rf {} + 2>/dev/null || true
find "$PROJECT_ROOT" -type f -name "*.pyc"         -delete 2>/dev/null || true
find "$PROJECT_ROOT" -type f -name "*.pyo"         -delete 2>/dev/null || true

echo "Done."
