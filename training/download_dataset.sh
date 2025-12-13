#!/bin/bash
# Простой скрипт для скачивания датасета

set -e

echo "📥 Скачивание FLIR ADAS датасета..."

# Установка kagglehub если нужно
if ! python -c "import kagglehub" 2>/dev/null; then
    echo "📦 Установка kagglehub..."
    pip install kagglehub
fi

# Скачивание датасета
echo "⬇️  Загрузка датасета (это может занять несколько минут)..."
python -c "import kagglehub; print('Путь:', kagglehub.dataset_download('deepnewbie/flir-thermal-images-dataset'))"

echo ""
echo "✅ Датасет скачан!"
echo "   Путь: ~/.cache/kagglehub/datasets/deepnewbie/flir-thermal-images-dataset/versions/1"

