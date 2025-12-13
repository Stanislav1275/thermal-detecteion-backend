#!/bin/bash
# Подготовка датасета в YOLO формат

set -e

DATASET_ROOT="$HOME/.cache/kagglehub/datasets/deepnewbie/flir-thermal-images-dataset/versions/1"

echo "🔄 Подготовка датасета в YOLO формат..."

if [ ! -d "$DATASET_ROOT" ]; then
    echo "❌ Датасет не найден по пути: $DATASET_ROOT"
    echo "   Сначала выполните: ./download_dataset.sh"
    exit 1
fi

python prepare_dataset.py \
    --dataset-root "$DATASET_ROOT" \
    --output-root ./datasets/yolo \
    --splits train val

echo ""
echo "✅ Датасет подготовлен!"
echo "   Структура: ./datasets/yolo/"

