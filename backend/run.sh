#!/bin/bash
set -e

if [ -z "$CONDA_DEFAULT_ENV" ] || [ "$CONDA_DEFAULT_ENV" != "thermal-detection" ]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate thermal-detection
fi

cd "$(dirname "$0")"

API_HOST=${API_HOST:-0.0.0.0}
API_PORT=${API_PORT:-8000}

echo "🚀 Запуск FastAPI сервера..."
echo "   API: http://localhost:${API_PORT}"
echo "   Docs: http://localhost:${API_PORT}/docs"
echo ""

uvicorn app.main:app --reload --host ${API_HOST} --port ${API_PORT}

