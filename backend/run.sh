#!/bin/bash
# Скрипт для запуска Backend API

set -e

# Активация conda окружения
if [ -z "$CONDA_DEFAULT_ENV" ] || [ "$CONDA_DEFAULT_ENV" != "thermal-detection" ]; then
    echo "🔄 Активация conda окружения..."
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate thermal-detection
fi

# Проверка модели
MODEL_PATH="training/models/best.pt"
if [ ! -f "$MODEL_PATH" ]; then
    echo "⚠️  Предупреждение: Модель не найдена по пути $MODEL_PATH"
    echo "   Убедитесь, что модель обучена перед запуском API"
    echo ""
    read -p "Продолжить запуск без модели? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Переход в директорию backend
cd "$(dirname "$0")"

# Запуск FastAPI
echo "🚀 Запуск FastAPI сервера..."
echo "   API будет доступен на http://localhost:8000"
echo "   Документация: http://localhost:8000/docs"
echo ""

uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

