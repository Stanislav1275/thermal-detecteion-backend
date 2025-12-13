#!/bin/bash
set -e

if [ -z "$CONDA_DEFAULT_ENV" ] || [ "$CONDA_DEFAULT_ENV" != "thermal-detection" ]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate thermal-detection
fi

cd "$(dirname "$0")"

echo "🚀 Запуск FastAPI сервера..."
echo "   API: http://localhost:8000"
echo "   Docs: http://localhost:8000/docs"
echo ""

uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

