#!/bin/bash
# Lint script for the project
# Runs ruff, black, and mypy checks

set -e

echo "🔍 Running ruff check..."
ruff check backend/

echo "✨ Checking code formatting with black..."
black --check backend/

echo "🔎 Running mypy type check..."
mypy backend/app --ignore-missing-imports

echo "✅ All lint checks passed!"

