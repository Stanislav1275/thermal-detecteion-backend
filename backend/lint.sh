#!/bin/bash
# Lint script for the project
# Runs ruff, black, and mypy checks
#
# Usage:
#   ./lint.sh          # Check only (no fixes)
#   ./lint.sh --fix    # Auto-fix issues where possible

set -e

FIX_MODE=false
if [ "$1" == "--fix" ] || [ "$1" == "-f" ]; then
    FIX_MODE=true
fi

if [ "$FIX_MODE" = true ]; then
    echo "🔧 Running ruff with auto-fix..."
    ruff check --fix backend/
    echo "✨ Auto-formatting code with black..."
    black backend/
else
    echo "🔍 Running ruff check..."
    ruff check backend/
    echo "✨ Checking code formatting with black..."
    black --check backend/
fi

echo "🔎 Running mypy type check..."
mypy backend/app --ignore-missing-imports

if [ "$FIX_MODE" = true ]; then
    echo "✅ Lint checks passed and auto-fixes applied!"
else
    echo "✅ All lint checks passed!"
    echo ""
    echo "💡 Tip: Run './lint.sh --fix' to automatically fix issues where possible"
fi

